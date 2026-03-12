"""
Build training dataset: 1 sample per second with ~212 features + target.

For each second T (with 120s lookback, 15s forward):
  - Computes 212 features from raw market data
  - Computes target: 5-class price movement in next 15 seconds
  - Saves per-day parquet files

Usage:
  python -m src.training.build_dataset --start 2026-03-01 --end 2026-03-02
  python -m src.training.build_dataset --start 2025-11-11 --end 2026-03-11 --workers 8
  python -m src.training.build_dataset --validate --start 2026-03-01 --end 2026-03-02
"""

import argparse
import multiprocessing as mp
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.feature_engine import (
    FEATURE_COLUMNS,
    DayData,
    compute_features,
    load_day_data,
    _last_before,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset"

# Target thresholds (basis points)
T1 = 1.5   # FLAT boundary
T2 = 5.0   # STRONG boundary

TARGET_NAMES = {0: "STRONG_DOWN", 1: "DOWN", 2: "FLAT", 3: "UP", 4: "STRONG_UP"}

# All output columns
ALL_COLUMNS = ["timestamp_ms", "target", "change_bps"] + FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def classify_target(change_bps):
    """Map change in bps to target class 0-4."""
    if change_bps < -T2:
        return 0  # STRONG_DOWN
    elif change_bps < -T1:
        return 1  # DOWN
    elif change_bps <= T1:
        return 2  # FLAT
    elif change_bps <= T2:
        return 3  # UP
    else:
        return 4  # STRONG_UP


# ---------------------------------------------------------------------------
# Process a single day
# ---------------------------------------------------------------------------

def process_day(date_str, data_dir=None, output_dir=None):
    """
    Process one day: compute features + target for every second.
    Returns (date_str, n_rows, n_skipped, elapsed_sec) or raises on error.
    """
    data_dir   = Path(data_dir)   if data_dir   else DATA_DIR
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_path = output_dir / f"{date_str}.parquet"

    if out_path.exists():
        n = pq.read_metadata(out_path).num_rows
        return (date_str, n, 0, 0.0, "skipped")

    t0 = time.time()

    # Load day
    day = load_day_data(date_str, data_dir=data_dir)

    # Day boundaries (midnight UTC)
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_start_ms = int(dt.timestamp() * 1000)
    day_end_ms   = day_start_ms + 86_400_000

    T_start = day_start_ms + 120_000   # need 120s lookback
    T_end   = day_end_ms   - 15_000    # need 15s forward for target

    # Pre-allocate output array
    max_rows = (T_end - T_start) // 1000 + 1
    n_cols   = len(ALL_COLUMNS)
    data     = np.full((max_rows, n_cols), np.nan, dtype=np.float64)

    row = 0
    skipped = 0

    for T_ms in range(T_start, T_end, 1000):
        # Midprice at T (futures bookticker)
        idx_T = _last_before(day.bf_ts, T_ms)
        if idx_T < 0 or (T_ms - day.bf_ts[idx_T]) > 5000:
            skipped += 1
            continue
        mid_T = day.bf_mid[idx_T]

        # Midprice at T+15s
        T15 = T_ms + 15_000
        idx_T15 = _last_before(day.bf_ts, T15)
        if idx_T15 < 0 or (T15 - day.bf_ts[idx_T15]) > 5000:
            skipped += 1
            continue
        mid_T15 = day.bf_mid[idx_T15]

        # Target
        change_bps = (mid_T15 - mid_T) / mid_T * 10_000
        target = classify_target(change_bps)

        # Features
        feats = compute_features(day, T_ms)

        # Store row
        data[row, 0] = T_ms
        data[row, 1] = target
        data[row, 2] = change_bps
        data[row, 3:] = [feats[c] for c in FEATURE_COLUMNS]
        row += 1

        # Progress every 10,000 rows
        if row % 10_000 == 0:
            elapsed = time.time() - t0
            pct = (T_ms - T_start) / (T_end - T_start) * 100
            rate = row / elapsed
            eta = (max_rows - row) / rate if rate > 0 else 0
            print(f"  {date_str}: {row:,} rows ({pct:.0f}%) "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    # Trim and save
    data = data[:row]
    df = pd.DataFrame(data, columns=ALL_COLUMNS)
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    df["target"] = df["target"].astype(np.int8)

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    elapsed = time.time() - t0
    return (date_str, row, skipped, elapsed, "done")


def _process_day_wrapper(args):
    """Wrapper for multiprocessing (unpacks tuple args)."""
    date_str, data_dir, output_dir = args
    try:
        return process_day(date_str, data_dir, output_dir)
    except Exception as e:
        return (date_str, 0, 0, 0.0, f"ERROR: {e}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_day(date_str, output_dir=None):
    """Validate a single day's dataset file."""
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    path = output_dir / f"{date_str}.parquet"

    if not path.exists():
        return (date_str, "MISSING")

    df = pq.read_table(path).to_pandas()
    issues = []

    # Row count
    if len(df) < 80_000:
        issues.append(f"low rows: {len(df):,}")

    # Column check
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"missing cols: {missing_cols[:5]}")

    # NaN check
    nan_pct = df[FEATURE_COLUMNS].isna().mean()
    high_nan = nan_pct[nan_pct > 0.01]
    if len(high_nan) > 0:
        issues.append(f"{len(high_nan)} cols with >1% NaN")

    # Target distribution
    counts = df["target"].value_counts().sort_index()
    total = len(df)
    dist_str = " | ".join(f"{TARGET_NAMES.get(i,'?')}:{c:,}({c/total*100:.1f}%)"
                          for i, c in counts.items())

    if issues:
        return (date_str, f"ISSUES: {'; '.join(issues)} | {dist_str}")
    else:
        return (date_str, f"OK {len(df):,} rows | {dist_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1)")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Validate existing dataset only")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end,   "%Y-%m-%d")
    dates = []
    cur = start
    while cur < end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    data_dir   = args.data_dir
    output_dir = args.output_dir

    if args.validate:
        print(f"Validating {len(dates)} days...")
        for d in dates:
            result = validate_day(d, output_dir)
            print(f"  {result[0]}: {result[1]}")
        return

    print(f"Building dataset: {len(dates)} days, {args.workers} worker(s)")
    print(f"Output: {output_dir or OUTPUT_DIR}")
    print(f"Target thresholds: T1={T1} bps, T2={T2} bps")
    print()

    t_total = time.time()
    total_rows = 0
    total_skipped = 0

    if args.workers > 1:
        jobs = [(d, data_dir, output_dir) for d in dates]
        with mp.Pool(args.workers) as pool:
            for result in pool.imap(_process_day_wrapper, jobs):
                date_str, n_rows, n_skip, elapsed, status = result
                total_rows += n_rows
                total_skipped += n_skip
                if status == "skipped":
                    print(f"  {date_str}: skipped (already exists, {n_rows:,} rows)")
                elif status == "done":
                    print(f"  {date_str}: {n_rows:,} rows, {n_skip:,} skipped, {elapsed:.1f}s")
                else:
                    print(f"  {date_str}: {status}")
    else:
        for d in dates:
            try:
                result = process_day(d, data_dir, output_dir)
                date_str, n_rows, n_skip, elapsed, status = result
                total_rows += n_rows
                total_skipped += n_skip
                if status == "skipped":
                    print(f"  {date_str}: skipped (already exists, {n_rows:,} rows)")
                elif status == "done":
                    print(f"  {date_str}: {n_rows:,} rows, {n_skip:,} skipped, {elapsed:.1f}s")
                else:
                    print(f"  {date_str}: {status}")
            except Exception as e:
                print(f"  {d}: ERROR - {e}")

    total_time = time.time() - t_total
    print(f"\nDone. {total_rows:,} total rows, {total_skipped:,} skipped samples, {total_time:.1f}s")

    # Auto-validate
    print("\nValidation:")
    for d in dates:
        result = validate_day(d, output_dir)
        print(f"  {result[0]}: {result[1]}")

    sys.exit(0)


if __name__ == "__main__":
    main()

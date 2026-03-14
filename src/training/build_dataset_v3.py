"""
Build V3 training dataset: 1 sample per second within each 5-minute block.

For each block (multiple of 300s):
  - Fixes open_ref = last ref_price <= block_start
  - Fixes close_ref = last ref_price <= block_end
  - Target: 1 if close_ref >= open_ref (Up), 0 if not (Down)
  - Generates 1 row every 1s (300 rows per block)
  - Computes ~154 features per row

Loads 30 min from previous day + 5 min from next day for border blocks.

Usage:
  python -m src.training.build_dataset_v3 --start 2026-03-01 --end 2026-03-02
  python -m src.training.build_dataset_v3 --start 2025-11-11 --end 2026-03-12 --workers 8
  python -m src.training.build_dataset_v3 --validate --start 2026-03-01 --end 2026-03-02
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

from src.features.feature_engine_v3 import (
    FEATURE_COLUMNS_V3,
    DayData,
    build_ref_price,
    compute_features_v3,
    load_day_data,
    _last_before,
    _safe_div,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_v3"

BLOCK_DURATION_MS = 300_000  # 5 minutes
SAMPLE_INTERVAL_MS = 1_000   # 1 row every 1 second
ROWS_PER_BLOCK = BLOCK_DURATION_MS // SAMPLE_INTERVAL_MS  # 300

# Output columns
# seconds_to_expiry is already in FEATURE_COLUMNS_V3, so exclude from meta
META_COLUMNS = ["block_start_ms", "timestamp_ms", "target", "terminal_return_bps"]
ALL_COLUMNS = META_COLUMNS + FEATURE_COLUMNS_V3


# ---------------------------------------------------------------------------
# Merge day data (for border overlap)
# ---------------------------------------------------------------------------

def _merge_day_data(main_day, extra_day, prepend=True):
    """
    Merge arrays from extra_day into main_day.
    If prepend=True, extra data goes before main (previous day).
    If prepend=False, extra data goes after main (next day).
    """
    if extra_day is None:
        return main_day

    def _concat(a, b):
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a
        return np.concatenate([a, b]) if prepend else np.concatenate([b, a])

    # We need to be careful: prepend means extra comes first
    if prepend:
        first, second = extra_day, main_day
    else:
        first, second = main_day, extra_day

    # Trades futures
    main_day.tf_ts    = np.concatenate([first.tf_ts, second.tf_ts])
    main_day.tf_price = np.concatenate([first.tf_price, second.tf_price])
    main_day.tf_qty   = np.concatenate([first.tf_qty, second.tf_qty])
    main_day.tf_ibm   = np.concatenate([first.tf_ibm, second.tf_ibm])

    # Trades spot
    main_day.ts_ts    = np.concatenate([first.ts_ts, second.ts_ts])
    main_day.ts_price = np.concatenate([first.ts_price, second.ts_price])
    main_day.ts_qty   = np.concatenate([first.ts_qty, second.ts_qty])
    main_day.ts_ibm   = np.concatenate([first.ts_ibm, second.ts_ibm])

    # Bookticker futures
    main_day.bf_ts  = np.concatenate([first.bf_ts, second.bf_ts])
    main_day.bf_bid = np.concatenate([first.bf_bid, second.bf_bid])
    main_day.bf_ask = np.concatenate([first.bf_ask, second.bf_ask])
    main_day.bf_mid = np.concatenate([first.bf_mid, second.bf_mid])

    # Bookticker spot
    main_day.bs_ts  = np.concatenate([first.bs_ts, second.bs_ts])
    main_day.bs_bid = np.concatenate([first.bs_bid, second.bs_bid])
    main_day.bs_ask = np.concatenate([first.bs_ask, second.bs_ask])
    main_day.bs_mid = np.concatenate([first.bs_mid, second.bs_mid])

    # Orderbook futures
    for key in first.ob_fut:
        main_day.ob_fut[key] = np.concatenate([first.ob_fut[key], second.ob_fut[key]])

    # Orderbook spot
    for key in first.ob_spot:
        main_day.ob_spot[key] = np.concatenate([first.ob_spot[key], second.ob_spot[key]])

    # Mark price
    main_day.mp_ts      = np.concatenate([first.mp_ts, second.mp_ts])
    main_day.mp_mark    = np.concatenate([first.mp_mark, second.mp_mark])
    main_day.mp_index   = np.concatenate([first.mp_index, second.mp_index])
    main_day.mp_funding = np.concatenate([first.mp_funding, second.mp_funding])
    main_day.mp_next_ms = np.concatenate([first.mp_next_ms, second.mp_next_ms])

    # Liquidations
    main_day.lq_ts     = np.concatenate([first.lq_ts, second.lq_ts])
    main_day.lq_is_buy = np.concatenate([first.lq_is_buy, second.lq_is_buy])
    main_day.lq_qty    = np.concatenate([first.lq_qty, second.lq_qty])

    # Metrics
    main_day.mt_ts       = np.concatenate([first.mt_ts, second.mt_ts])
    main_day.mt_ls_ratio = np.concatenate([first.mt_ls_ratio, second.mt_ls_ratio])
    main_day.mt_top_ls   = np.concatenate([first.mt_top_ls, second.mt_top_ls])
    main_day.mt_taker_ls = np.concatenate([first.mt_taker_ls, second.mt_taker_ls])
    main_day.mt_oi       = np.concatenate([first.mt_oi, second.mt_oi])

    # Cross-exchange: Coinbase quotes
    if hasattr(first, 'cb_ts') and hasattr(second, 'cb_ts'):
        main_day.cb_ts  = np.concatenate([first.cb_ts, second.cb_ts])
        main_day.cb_bid = np.concatenate([first.cb_bid, second.cb_bid])
        main_day.cb_ask = np.concatenate([first.cb_ask, second.cb_ask])
        main_day.cb_mid = np.concatenate([first.cb_mid, second.cb_mid])

    # Cross-exchange: Bybit quotes
    if hasattr(first, 'bb_ts') and hasattr(second, 'bb_ts'):
        main_day.bb_ts  = np.concatenate([first.bb_ts, second.bb_ts])
        main_day.bb_bid = np.concatenate([first.bb_bid, second.bb_bid])
        main_day.bb_ask = np.concatenate([first.bb_ask, second.bb_ask])
        main_day.bb_mid = np.concatenate([first.bb_mid, second.bb_mid])

    # Cross-exchange: Coinbase trades
    if hasattr(first, 'ct_ts') and hasattr(second, 'ct_ts'):
        main_day.ct_ts    = np.concatenate([first.ct_ts, second.ct_ts])
        main_day.ct_price = np.concatenate([first.ct_price, second.ct_price])
        main_day.ct_qty   = np.concatenate([first.ct_qty, second.ct_qty])
        main_day.ct_ibm   = np.concatenate([first.ct_ibm, second.ct_ibm])

    # Cross-exchange: Bybit trades
    if hasattr(first, 'bt_ts') and hasattr(second, 'bt_ts'):
        main_day.bt_ts    = np.concatenate([first.bt_ts, second.bt_ts])
        main_day.bt_price = np.concatenate([first.bt_price, second.bt_price])
        main_day.bt_qty   = np.concatenate([first.bt_qty, second.bt_qty])
        main_day.bt_ibm   = np.concatenate([first.bt_ibm, second.bt_ibm])

    # Cross-exchange: Coinbase orderbook
    if hasattr(first, 'ob_cb') and hasattr(second, 'ob_cb'):
        for key in first.ob_cb:
            main_day.ob_cb[key] = np.concatenate([first.ob_cb[key], second.ob_cb[key]])

    # Cross-exchange: Bybit orderbook
    if hasattr(first, 'ob_bb') and hasattr(second, 'ob_bb'):
        for key in first.ob_bb:
            main_day.ob_bb[key] = np.concatenate([first.ob_bb[key], second.ob_bb[key]])

    return main_day


def _load_day_safe(date_str, data_dir, time_range=None, lightweight=False):
    """Load day data, return None if directory doesn't exist."""
    d = Path(data_dir) / date_str
    if not d.exists():
        return None
    try:
        return load_day_data(date_str, data_dir=data_dir, time_range=time_range,
                             lightweight=lightweight)
    except Exception:
        return None


def _load_windowed(date_strs, data_dir, window_start_ms, window_end_ms):
    """Load and merge data from multiple dates, filtered to a time window.

    Tries each date in date_strs; merges all that have data within the window.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    merged = None
    for ds in date_strs:
        day = _load_day_safe(ds, data_dir, time_range=(window_start_ms, window_end_ms))
        if day is None:
            continue
        # Skip if no data in window (e.g. all timestamps filtered out)
        if len(day.tf_ts) == 0 and len(day.bf_ts) == 0:
            continue
        if merged is None:
            merged = day
        else:
            merged = _merge_day_data(merged, day, prepend=False)
    return merged


# ---------------------------------------------------------------------------
# Process a single day
# ---------------------------------------------------------------------------

CHUNK_SIZE_BLOCKS = 24  # Process 24 blocks (2 hours) at a time
LOOKBACK_MS = 300_000   # Max feature lookback: 5 minutes


def process_day(date_str, data_dir=None, output_dir=None):
    """
    Process one day: compute features + target for every 1s within each block.

    Uses chunked loading: only keeps ~10-35 min of market data in RAM at a
    time instead of the full 24h. This reduces per-worker memory from ~2 GB
    to ~200 MB, allowing 8+ parallel workers.

    Returns (date_str, n_rows, n_blocks, n_skipped, elapsed_sec, status).
    """
    data_dir   = Path(data_dir)   if data_dir   else DATA_DIR
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_path = output_dir / f"{date_str}.parquet"

    if out_path.exists():
        n = pq.read_metadata(out_path).num_rows
        return (date_str, n, 0, 0, 0.0, "skipped")

    t0 = time.time()

    # Day boundaries (midnight UTC)
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_start_ms = int(dt.timestamp() * 1000)
    day_end_ms   = day_start_ms + 86_400_000

    # --- Step 1: Build ref_price + keep lightweight data ---
    # Load only mark_price + metrics + liquidations (~5 MB instead of ~2 GB).
    # ref_price only needs mark_price (index_price). Heavy streams (trades,
    # orderbooks, booktickers) are loaded per-chunk in Step 3.
    day_full = load_day_data(date_str, data_dir=data_dir, lightweight=True)
    prev_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    prev_day = _load_day_safe(prev_date, data_dir, lightweight=True)
    if prev_day is not None:
        cutoff = day_start_ms - 1_800_000
        _filter_day_after(prev_day, cutoff)
        day_full = _merge_day_data(day_full, prev_day, prepend=True)
    next_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = _load_day_safe(next_date, data_dir, lightweight=True)
    if next_day is not None:
        cutoff = day_end_ms + 300_000
        _filter_day_before(next_day, cutoff)
        day_full = _merge_day_data(day_full, next_day, prepend=False)
    ref = build_ref_price(day_full)

    # Keep lightweight arrays that features need for the full day
    # (metrics, liquidations, mark_price — all tiny)
    _kept_mt_ts       = day_full.mt_ts
    _kept_mt_ls_ratio = day_full.mt_ls_ratio
    _kept_mt_top_ls   = day_full.mt_top_ls
    _kept_mt_taker_ls = day_full.mt_taker_ls
    _kept_mt_oi       = day_full.mt_oi
    _kept_lq_ts       = day_full.lq_ts
    _kept_lq_is_buy   = day_full.lq_is_buy
    _kept_lq_qty      = day_full.lq_qty
    _kept_mp_ts       = day_full.mp_ts
    _kept_mp_mark     = day_full.mp_mark
    _kept_mp_index    = day_full.mp_index
    _kept_mp_funding  = day_full.mp_funding
    _kept_mp_next_ms  = day_full.mp_next_ms
    del day_full, prev_day, next_day

    if len(ref['ts']) == 0:
        return (date_str, 0, 0, 0, time.time() - t0, "ERROR: no ref_price data")

    # --- Step 2: Enumerate blocks ---
    first_block = (day_start_ms // BLOCK_DURATION_MS) * BLOCK_DURATION_MS
    if first_block < day_start_ms:
        first_block += BLOCK_DURATION_MS

    blocks = []
    bs = first_block
    while bs + BLOCK_DURATION_MS <= day_end_ms:
        blocks.append(bs)
        bs += BLOCK_DURATION_MS

    # Pre-allocate output
    max_rows = len(blocks) * ROWS_PER_BLOCK
    n_cols = len(ALL_COLUMNS)
    data = np.full((max_rows, n_cols), np.nan, dtype=np.float64)

    row = 0
    n_blocks_ok = 0
    n_skipped_blocks = 0
    block_results_history = []

    # --- Step 3: Process blocks in chunks ---
    # Each chunk loads only the time window it needs from disk.
    # Adjacent dates are included in the window when needed (border chunks).
    all_dates = [prev_date, date_str, next_date]

    for chunk_start_idx in range(0, len(blocks), CHUNK_SIZE_BLOCKS):
        chunk_blocks = blocks[chunk_start_idx:chunk_start_idx + CHUNK_SIZE_BLOCKS]

        # Time window this chunk needs: lookback before first block to end of last block
        window_start = chunk_blocks[0] - LOOKBACK_MS
        window_end = chunk_blocks[-1] + BLOCK_DURATION_MS

        # Load data for this window (may span prev/next day)
        day = _load_windowed(all_dates, data_dir, window_start, window_end)

        # Inject full-day lightweight data (metrics, liquidations, mark_price)
        # These are tiny and must be consistent across all chunks.
        if day is not None:
            day.mt_ts       = _kept_mt_ts
            day.mt_ls_ratio = _kept_mt_ls_ratio
            day.mt_top_ls   = _kept_mt_top_ls
            day.mt_taker_ls = _kept_mt_taker_ls
            day.mt_oi       = _kept_mt_oi
            day.lq_ts       = _kept_lq_ts
            day.lq_is_buy   = _kept_lq_is_buy
            day.lq_qty      = _kept_lq_qty
            day.mp_ts       = _kept_mp_ts
            day.mp_mark     = _kept_mp_mark
            day.mp_index    = _kept_mp_index
            day.mp_funding  = _kept_mp_funding
            day.mp_next_ms  = _kept_mp_next_ms

        for block_idx_in_chunk, block_start_ms in enumerate(chunk_blocks):
            block_idx = chunk_start_idx + block_idx_in_chunk
            block_end_ms = block_start_ms + BLOCK_DURATION_MS

            # open_ref: last ref_price <= block_start
            open_idx = _last_before(ref['ts'], block_start_ms)
            if open_idx < 0 or np.isnan(ref['price'][open_idx]):
                n_skipped_blocks += 1
                continue
            open_ref = ref['price'][open_idx]
            open_ref_age_ms = block_start_ms - ref['ts'][open_idx]

            # close_ref: last ref_price <= block_end
            close_idx = _last_before(ref['ts'], block_end_ms)
            if close_idx < 0 or np.isnan(ref['price'][close_idx]):
                n_skipped_blocks += 1
                continue
            close_ref = ref['price'][close_idx]

            # Target
            target = 1 if close_ref >= open_ref else 0
            terminal_return_bps = _safe_div(close_ref - open_ref, open_ref) * 10_000

            # Generate rows every 1s within block
            n_blocks_ok += 1
            block_cache = {}
            for sample_offset_ms in range(0, BLOCK_DURATION_MS, SAMPLE_INTERVAL_MS):
                T_ms = block_start_ms + sample_offset_ms

                if T_ms < day_start_ms or T_ms >= day_end_ms:
                    continue

                feats = compute_features_v3(
                    day, ref, T_ms, block_start_ms, open_ref,
                    open_ref_age_ms=open_ref_age_ms,
                    block_results=block_results_history,
                    block_cache=block_cache,
                )

                data[row, 0] = block_start_ms
                data[row, 1] = T_ms
                data[row, 2] = target
                data[row, 3] = terminal_return_bps
                for ci, col in enumerate(FEATURE_COLUMNS_V3):
                    data[row, 4 + ci] = feats.get(col, np.nan)
                row += 1

            block_results_history.insert(0, {
                'return_bps': terminal_return_bps,
                'result': target,
            })
            if len(block_results_history) > 6:
                block_results_history = block_results_history[:6]

            # Progress
            if (block_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                pct = (block_idx + 1) / len(blocks) * 100
                rate = row / elapsed if elapsed > 0 else 0
                eta = (max_rows - row) / rate if rate > 0 else 0
                print(f"  {date_str}: block {block_idx+1}/{len(blocks)} ({pct:.0f}%), "
                      f"{row:,} rows [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        del day  # Free chunk data

    # Trim and save
    data = data[:row]
    df = pd.DataFrame(data, columns=ALL_COLUMNS)
    df["block_start_ms"] = df["block_start_ms"].astype(np.int64)
    df["timestamp_ms"]   = df["timestamp_ms"].astype(np.int64)
    df["target"]         = df["target"].astype(np.int8)

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    elapsed = time.time() - t0
    return (date_str, row, n_blocks_ok, n_skipped_blocks, elapsed, "done")


def _filter_day_after(day, cutoff_ms):
    """Filter day data to only keep timestamps >= cutoff_ms (in-place)."""
    def _filter_1d(ts_arr, *arrays):
        mask = ts_arr >= cutoff_ms
        results = [ts_arr[mask]]
        for a in arrays:
            results.append(a[mask])
        return results

    r = _filter_1d(day.tf_ts, day.tf_price, day.tf_qty, day.tf_ibm)
    day.tf_ts, day.tf_price, day.tf_qty, day.tf_ibm = r

    r = _filter_1d(day.ts_ts, day.ts_price, day.ts_qty, day.ts_ibm)
    day.ts_ts, day.ts_price, day.ts_qty, day.ts_ibm = r

    r = _filter_1d(day.bf_ts, day.bf_bid, day.bf_ask, day.bf_mid)
    day.bf_ts, day.bf_bid, day.bf_ask, day.bf_mid = r

    r = _filter_1d(day.bs_ts, day.bs_bid, day.bs_ask, day.bs_mid)
    day.bs_ts, day.bs_bid, day.bs_ask, day.bs_mid = r

    # Orderbooks
    mask = day.ob_fut['ts'] >= cutoff_ms
    for k in day.ob_fut:
        day.ob_fut[k] = day.ob_fut[k][mask]
    mask = day.ob_spot['ts'] >= cutoff_ms
    for k in day.ob_spot:
        day.ob_spot[k] = day.ob_spot[k][mask]

    r = _filter_1d(day.mp_ts, day.mp_mark, day.mp_index, day.mp_funding, day.mp_next_ms)
    day.mp_ts, day.mp_mark, day.mp_index, day.mp_funding, day.mp_next_ms = r

    if len(day.lq_ts) > 0:
        r = _filter_1d(day.lq_ts, day.lq_is_buy, day.lq_qty)
        day.lq_ts, day.lq_is_buy, day.lq_qty = r

    if len(day.mt_ts) > 0:
        r = _filter_1d(day.mt_ts, day.mt_ls_ratio, day.mt_top_ls, day.mt_taker_ls, day.mt_oi)
        day.mt_ts, day.mt_ls_ratio, day.mt_top_ls, day.mt_taker_ls, day.mt_oi = r


def _filter_day_before(day, cutoff_ms):
    """Filter day data to only keep timestamps <= cutoff_ms (in-place)."""
    def _filter_1d(ts_arr, *arrays):
        mask = ts_arr <= cutoff_ms
        results = [ts_arr[mask]]
        for a in arrays:
            results.append(a[mask])
        return results

    r = _filter_1d(day.tf_ts, day.tf_price, day.tf_qty, day.tf_ibm)
    day.tf_ts, day.tf_price, day.tf_qty, day.tf_ibm = r

    r = _filter_1d(day.ts_ts, day.ts_price, day.ts_qty, day.ts_ibm)
    day.ts_ts, day.ts_price, day.ts_qty, day.ts_ibm = r

    r = _filter_1d(day.bf_ts, day.bf_bid, day.bf_ask, day.bf_mid)
    day.bf_ts, day.bf_bid, day.bf_ask, day.bf_mid = r

    r = _filter_1d(day.bs_ts, day.bs_bid, day.bs_ask, day.bs_mid)
    day.bs_ts, day.bs_bid, day.bs_ask, day.bs_mid = r

    mask = day.ob_fut['ts'] <= cutoff_ms
    for k in day.ob_fut:
        day.ob_fut[k] = day.ob_fut[k][mask]
    mask = day.ob_spot['ts'] <= cutoff_ms
    for k in day.ob_spot:
        day.ob_spot[k] = day.ob_spot[k][mask]

    r = _filter_1d(day.mp_ts, day.mp_mark, day.mp_index, day.mp_funding, day.mp_next_ms)
    day.mp_ts, day.mp_mark, day.mp_index, day.mp_funding, day.mp_next_ms = r

    if len(day.lq_ts) > 0:
        r = _filter_1d(day.lq_ts, day.lq_is_buy, day.lq_qty)
        day.lq_ts, day.lq_is_buy, day.lq_qty = r

    if len(day.mt_ts) > 0:
        r = _filter_1d(day.mt_ts, day.mt_ls_ratio, day.mt_top_ls, day.mt_taker_ls, day.mt_oi)
        day.mt_ts, day.mt_ls_ratio, day.mt_top_ls, day.mt_taker_ls, day.mt_oi = r


def _process_day_wrapper(args):
    """Wrapper for multiprocessing."""
    date_str, data_dir, output_dir = args
    try:
        return process_day(date_str, data_dir, output_dir)
    except Exception as e:
        import traceback
        return (date_str, 0, 0, 0, 0.0, f"ERROR: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_day(date_str, output_dir=None):
    """Validate a single day's V3 dataset file."""
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    path = output_dir / f"{date_str}.parquet"

    if not path.exists():
        return (date_str, "MISSING")

    df = pq.read_table(path).to_pandas()
    issues = []

    # Row count (288 blocks * 60 rows = 17,280 expected)
    if len(df) < 15_000:
        issues.append(f"low rows: {len(df):,}")

    # Column check
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"missing cols: {missing_cols[:5]}")

    # NaN check in features
    nan_pct = df[FEATURE_COLUMNS_V3].isna().mean()
    high_nan = nan_pct[nan_pct > 0.01]
    if len(high_nan) > 0:
        issues.append(f"{len(high_nan)} cols with >1% NaN")

    # Target distribution
    n_up   = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    total  = len(df)
    up_pct = n_up / total * 100 if total > 0 else 0

    # Block count
    n_blocks = df["block_start_ms"].nunique()

    # Block integrity: all rows in same block have same target
    block_target_check = df.groupby("block_start_ms")["target"].nunique()
    bad_blocks = (block_target_check > 1).sum()
    if bad_blocks > 0:
        issues.append(f"LEAKAGE: {bad_blocks} blocks with mixed targets!")

    dist_str = f"Up:{n_up:,}({up_pct:.1f}%) | Down:{n_down:,}({100-up_pct:.1f}%) | {n_blocks} blocks"

    if issues:
        return (date_str, f"ISSUES: {'; '.join(issues)} | {dist_str}")
    else:
        return (date_str, f"OK {len(df):,} rows | {dist_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build V3 training dataset (5-min blocks)")
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

    print(f"Building V3 dataset: {len(dates)} days, {args.workers} worker(s)")
    print(f"Output: {output_dir or OUTPUT_DIR}")
    print(f"Block duration: {BLOCK_DURATION_MS // 1000}s, sample every {SAMPLE_INTERVAL_MS // 1000}s")
    print(f"Expected: ~{ROWS_PER_BLOCK} rows/block, ~288 blocks/day, ~{288 * ROWS_PER_BLOCK:,} rows/day")
    print()

    t_total = time.time()
    total_rows = 0
    total_blocks = 0
    total_skipped = 0

    if args.workers > 1:
        jobs = [(d, data_dir, output_dir) for d in dates]
        with mp.Pool(args.workers) as pool:
            for result in pool.imap(_process_day_wrapper, jobs):
                date_str, n_rows, n_blk, n_skip, elapsed, status = result
                total_rows += n_rows
                total_blocks += n_blk
                total_skipped += n_skip
                if status == "skipped":
                    print(f"  {date_str}: skipped (already exists, {n_rows:,} rows)")
                elif status == "done":
                    print(f"  {date_str}: {n_rows:,} rows, {n_blk} blocks, "
                          f"{n_skip} skipped, {elapsed:.1f}s")
                else:
                    print(f"  {date_str}: {status}")
    else:
        for d in dates:
            try:
                result = process_day(d, data_dir, output_dir)
                date_str, n_rows, n_blk, n_skip, elapsed, status = result
                total_rows += n_rows
                total_blocks += n_blk
                total_skipped += n_skip
                if status == "skipped":
                    print(f"  {date_str}: skipped (already exists, {n_rows:,} rows)")
                elif status == "done":
                    print(f"  {date_str}: {n_rows:,} rows, {n_blk} blocks, "
                          f"{n_skip} skipped, {elapsed:.1f}s")
                else:
                    print(f"  {date_str}: {status}")
            except Exception as e:
                print(f"  {d}: ERROR - {e}")

    total_time = time.time() - t_total
    print(f"\nDone. {total_rows:,} total rows, {total_blocks:,} blocks, "
          f"{total_skipped:,} skipped blocks, {total_time:.1f}s")

    # Auto-validate
    print("\nValidation:")
    for d in dates:
        result = validate_day(d, output_dir)
        print(f"  {result[0]}: {result[1]}")

    sys.exit(0)


if __name__ == "__main__":
    main()

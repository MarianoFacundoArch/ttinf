"""
Unified downloader: Tardis (8 datasets) + Binance DV (metrics).

Downloads all data needed for training, with parallel execution,
progress tracking, ETA, and post-download validation.

Usage:
  python -m src.data.download_all --start 2025-11-11 --end 2026-03-11
  python -m src.data.download_all --days 30
  python -m src.data.download_all --start 2025-11-11 --end 2026-03-11 --workers 6
  python -m src.data.download_all --validate  # validate existing data only
  python -m src.data.download_all --status    # show download status
"""

import argparse
import gzip
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARDIS_API_KEY = os.environ.get("TARDIS_API_KEY", "")
TARDIS_BASE = "https://datasets.tardis.dev/v1"
BINANCE_BASE = "https://data.binance.vision/data"
SYMBOL = "BTCUSDT"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Expected streams for a complete day
EXPECTED_STREAMS = [
    "trades_futures",
    "trades_spot",
    "bookticker_futures",
    "bookticker_spot",
    "orderbook_futures",
    "orderbook_spot",
    "mark_price",
    "liquidations",
    "metrics",
]

# ---------------------------------------------------------------------------
# Thread-safe progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total_jobs: int):
        self.total = total_jobs
        self.completed = 0
        self.skipped = 0
        self.failed = 0
        self.errors: list[str] = []
        self.lock = Lock()
        self.start_time = time.time()
        self.bytes_downloaded = 0

    def record(self, status: str, msg: str = "", size_bytes: int = 0,
               label: str = ""):
        with self.lock:
            if status == "done":
                self.completed += 1
                self.bytes_downloaded += size_bytes
            elif status == "skip":
                self.completed += 1
                self.skipped += 1
            elif status == "fail":
                self.completed += 1
                self.failed += 1
                self.errors.append(msg)

            done = self.completed
            total = self.total
            elapsed = time.time() - self.start_time
            new_downloads = done - self.skipped - self.failed
            remaining = total - done

            if new_downloads > 0:
                rate = elapsed / new_downloads
                # Estimate remaining: assume same skip ratio going forward
                skip_ratio = self.skipped / done if done > 0 else 0
                remaining_actual = remaining * (1 - skip_ratio)
                eta_sec = rate * remaining_actual
            elif done < total:
                eta_sec = 0
            else:
                eta_sec = 0

            pct = done / total * 100 if total > 0 else 0
            eta_str = _fmt_duration(eta_sec) if eta_sec > 0 else "--:--"
            elapsed_str = _fmt_duration(elapsed)
            mb = self.bytes_downloaded / (1024 * 1024)

            bar_len = 30
            filled = int(bar_len * done / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)

            # Print completed download on its own line
            if status == "done" and label:
                sys.stdout.write(f"\r\033[K  + {label} ({size_bytes / (1024*1024):.1f} MB)\n")

            sys.stdout.write(
                f"\r  [{bar}] {done}/{total} ({pct:.0f}%) | "
                f"{elapsed_str} elapsed | ETA {eta_str} | "
                f"{mb:.0f} MB dl | {self.failed} err"
            )
            sys.stdout.flush()

    def summary(self):
        elapsed = time.time() - self.start_time
        mb = self.bytes_downloaded / (1024 * 1024)
        print(f"\n\n{'='*60}")
        print(f"  Download complete in {_fmt_duration(elapsed)}")
        print(f"  Downloaded: {self.completed - self.skipped - self.failed} new")
        print(f"  Skipped (existing): {self.skipped}")
        print(f"  Failed: {self.failed}")
        print(f"  Total size: {mb:.1f} MB")
        if self.errors:
            print(f"\n  Errors:")
            for e in self.errors[:20]:
                print(f"    - {e}")
            if len(self.errors) > 20:
                print(f"    ... and {len(self.errors) - 20} more")
        print(f"{'='*60}")


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Tardis download + processing (reused from download_tardis.py)
# ---------------------------------------------------------------------------

TARDIS_DATASETS = {
    "fut_trades": {
        "exchange": "binance-futures",
        "data_type": "trades",
        "output_stream": "trades_futures",
        "column_map": {
            "timestamp": "timestamp_ms", "id": "agg_trade_id",
            "price": "price", "amount": "qty", "side": "_side",
        },
        "post_process": "trades",
    },
    "spot_trades": {
        "exchange": "binance",
        "data_type": "trades",
        "output_stream": "trades_spot",
        "column_map": {
            "timestamp": "timestamp_ms", "id": "agg_trade_id",
            "price": "price", "amount": "qty", "side": "_side",
        },
        "post_process": "trades",
    },
    "fut_bookticker": {
        "exchange": "binance-futures",
        "data_type": "book_ticker",
        "output_stream": "bookticker_futures",
        "column_map": {
            "timestamp": "timestamp_ms", "bid_price": "best_bid_price",
            "bid_amount": "best_bid_qty", "ask_price": "best_ask_price",
            "ask_amount": "best_ask_qty",
        },
    },
    "spot_bookticker": {
        "exchange": "binance",
        "data_type": "book_ticker",
        "output_stream": "bookticker_spot",
        "column_map": {
            "timestamp": "timestamp_ms", "bid_price": "best_bid_price",
            "bid_amount": "best_bid_qty", "ask_price": "best_ask_price",
            "ask_amount": "best_ask_qty",
        },
    },
    "fut_book25": {
        "exchange": "binance-futures",
        "data_type": "book_snapshot_25",
        "output_stream": "orderbook_futures",
        "post_process": "book_snapshot",
    },
    "spot_book25": {
        "exchange": "binance",
        "data_type": "book_snapshot_25",
        "output_stream": "orderbook_spot",
        "post_process": "book_snapshot",
    },
    "fut_deriv": {
        "exchange": "binance-futures",
        "data_type": "derivative_ticker",
        "output_stream": "mark_price",
        "column_map": {
            "timestamp": "timestamp_ms", "mark_price": "mark_price",
            "index_price": "index_price", "funding_rate": "funding_rate",
            "funding_timestamp": "next_funding_time_ms",
            "open_interest": "open_interest",
        },
    },
    "fut_liquidations": {
        "exchange": "binance-futures",
        "data_type": "liquidations",
        "output_stream": "liquidations",
        "column_map": {
            "timestamp": "timestamp_ms", "side": "side",
            "price": "price", "amount": "qty",
        },
    },
}


def _tardis_download_csv(exchange: str, data_type: str, date_str: str) -> pd.DataFrame | None:
    year, month, day = date_str.split("-")
    url = f"{TARDIS_BASE}/{exchange}/{data_type}/{year}/{month}/{day}/{SYMBOL}.csv.gz"
    headers = {"Authorization": f"Bearer {TARDIS_API_KEY}"} if TARDIS_API_KEY else {}

    resp = requests.get(url, headers=headers, timeout=180)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    raw = gzip.decompress(resp.content)
    return pd.read_csv(io.BytesIO(raw)), len(resp.content)


def _apply_column_map(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    df = df.drop(columns=["exchange", "symbol", "local_timestamp"], errors="ignore")
    rename = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename)
    keep = [v for v in column_map.values()]
    available = [c for c in keep if c in df.columns]
    return df[available]


def _convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["timestamp_ms", "next_funding_time_ms"]:
        if col in df.columns and len(df) > 0 and df[col].iloc[0] > 1e15:
            df[col] = df[col] // 1000
    return df


def _process_trades(df: pd.DataFrame) -> pd.DataFrame:
    df["is_buyer_maker"] = df["_side"] == "sell"
    df = df.drop(columns=["_side"], errors="ignore")
    df["agg_trade_id"] = df["agg_trade_id"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["qty"] = df["qty"].astype("float64")
    return df


def _process_book_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    result = {}
    result["timestamp_ms"] = df["timestamp"].values // 1000
    result["recv_ts"] = result["timestamp_ms"]
    for i in range(20):
        result[f"bid_price_{i}"] = df[f"bids[{i}].price"].values
        result[f"bid_qty_{i}"] = df[f"bids[{i}].amount"].values
        result[f"ask_price_{i}"] = df[f"asks[{i}].price"].values
        result[f"ask_qty_{i}"] = df[f"asks[{i}].amount"].values
    return pd.DataFrame(result)


def _save_parquet(df: pd.DataFrame, date_str: str, stream: str) -> int:
    out_path = OUTPUT_DIR / date_str / stream / "full_day.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    return out_path.stat().st_size


def download_tardis_job(ds_name: str, date_str: str) -> tuple[str, int, int]:
    """Download one tardis dataset for one day. Returns (status, rows, bytes)."""
    ds = TARDIS_DATASETS[ds_name]
    out_path = OUTPUT_DIR / date_str / ds["output_stream"] / "full_day.parquet"
    if out_path.exists():
        return "skip", 0, 0

    result = _tardis_download_csv(ds["exchange"], ds["data_type"], date_str)
    if result is None:
        return "404", 0, 0
    df, dl_bytes = result

    if len(df) == 0:
        return "empty", 0, 0

    if "column_map" in ds:
        df = _apply_column_map(df, ds["column_map"])

    df = _convert_timestamps(df)

    post = ds.get("post_process")
    if post == "trades":
        df = _process_trades(df)
    elif post == "book_snapshot":
        df = _process_book_snapshot(df)

    file_size = _save_parquet(df, date_str, ds["output_stream"])
    return "done", len(df), dl_bytes


# ---------------------------------------------------------------------------
# Binance DV download (metrics only)
# ---------------------------------------------------------------------------

def download_binance_metrics_job(date_str: str) -> tuple[str, int, int]:
    """Download metrics from Binance DV for one day. Returns (status, rows, bytes)."""
    out_path = OUTPUT_DIR / date_str / "metrics" / "full_day.parquet"
    if out_path.exists():
        return "skip", 0, 0

    filename = f"{SYMBOL}-metrics-{date_str}.zip"
    url = f"{BINANCE_BASE}/futures/um/daily/metrics/{SYMBOL}/{filename}"

    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return "404", 0, 0
    resp.raise_for_status()
    dl_bytes = len(resp.content)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        raw = zf.read(csv_name)

    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]

    df = pd.read_csv(io.BytesIO(raw))
    if len(df) == 0:
        return "empty", 0, 0

    file_size = _save_parquet(df, date_str, "metrics")
    return "done", len(df), dl_bytes


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

def create_jobs(start_date: datetime, end_date: datetime) -> list[tuple]:
    """Create list of (source, dataset_name, date_str) jobs."""
    jobs = []
    current = start_date
    while current < end_date:
        date_str = current.strftime("%Y-%m-%d")

        # 8 Tardis datasets
        for ds_name in TARDIS_DATASETS:
            jobs.append(("tardis", ds_name, date_str))

        # 1 Binance DV metrics
        jobs.append(("binance", "metrics", date_str))

        current += timedelta(days=1)
    return jobs


def execute_job(job: tuple) -> tuple[str, str, str, str, int, int]:
    """Execute a single download job. Returns (source, ds_name, date, status, rows, bytes)."""
    source, ds_name, date_str = job

    try:
        if source == "tardis":
            status, rows, dl_bytes = download_tardis_job(ds_name, date_str)
        else:
            status, rows, dl_bytes = download_binance_metrics_job(date_str)
        return source, ds_name, date_str, status, rows, dl_bytes
    except Exception as e:
        return source, ds_name, date_str, f"error:{e}", 0, 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

EXPECTED_SCHEMAS = {
    "trades_futures":      {"min_cols": 5, "required": ["timestamp_ms", "price", "qty"]},
    "trades_spot":         {"min_cols": 5, "required": ["timestamp_ms", "price", "qty"]},
    "bookticker_futures":  {"min_cols": 5, "required": ["timestamp_ms", "best_bid_price"]},
    "bookticker_spot":     {"min_cols": 5, "required": ["timestamp_ms", "best_bid_price"]},
    "orderbook_futures":   {"min_cols": 80, "required": ["timestamp_ms", "bid_price_0", "ask_price_0"]},
    "orderbook_spot":      {"min_cols": 80, "required": ["timestamp_ms", "bid_price_0", "ask_price_0"]},
    "mark_price":          {"min_cols": 5, "required": ["timestamp_ms", "mark_price", "funding_rate"]},
    "liquidations":        {"min_cols": 3, "required": ["timestamp_ms", "price"]},
    "metrics":             {"min_cols": 6, "required": ["create_time"]},
}


def validate_day(date_str: str) -> list[str]:
    """Validate a downloaded day. Returns list of issues."""
    issues = []
    day_dir = OUTPUT_DIR / date_str

    if not day_dir.exists():
        return [f"{date_str}: directory missing"]

    for stream in EXPECTED_STREAMS:
        path = day_dir / stream / "full_day.parquet"
        if not path.exists():
            issues.append(f"{date_str}/{stream}: missing")
            continue

        try:
            t = pq.read_metadata(path)
            cols = pq.read_schema(path).names

            schema = EXPECTED_SCHEMAS.get(stream, {})
            min_cols = schema.get("min_cols", 1)
            required = schema.get("required", [])

            if t.num_rows == 0:
                issues.append(f"{date_str}/{stream}: 0 rows")
            if len(cols) < min_cols:
                issues.append(f"{date_str}/{stream}: only {len(cols)} cols (need {min_cols})")
            for req_col in required:
                if req_col not in cols:
                    issues.append(f"{date_str}/{stream}: missing column '{req_col}'")
        except Exception as e:
            issues.append(f"{date_str}/{stream}: corrupt - {e}")

    return issues


def validate_range(start_date: datetime, end_date: datetime):
    """Validate all days in range."""
    current = start_date
    total_days = (end_date - start_date).days
    all_issues = []
    complete_days = 0

    print(f"\nValidating {total_days} days...")
    while current < end_date:
        date_str = current.strftime("%Y-%m-%d")
        issues = validate_day(date_str)
        if issues:
            all_issues.extend(issues)
        else:
            complete_days += 1
        current += timedelta(days=1)

    print(f"\n  Complete days: {complete_days}/{total_days}")
    if all_issues:
        print(f"  Issues found: {len(all_issues)}")
        for issue in all_issues[:30]:
            print(f"    - {issue}")
        if len(all_issues) > 30:
            print(f"    ... and {len(all_issues) - 30} more")
    else:
        print("  All data validated OK!")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status(start_date: datetime, end_date: datetime):
    """Show download status for the date range."""
    current = start_date
    total_days = (end_date - start_date).days
    total_streams = len(EXPECTED_STREAMS)

    complete = 0
    partial = 0
    missing = 0
    total_size = 0

    print(f"\nStatus for {start_date.date()} to {end_date.date()} ({total_days} days, {total_streams} streams each)\n")

    while current < end_date:
        date_str = current.strftime("%Y-%m-%d")
        day_dir = OUTPUT_DIR / date_str

        if not day_dir.exists():
            missing += 1
            current += timedelta(days=1)
            continue

        found = 0
        day_size = 0
        for stream in EXPECTED_STREAMS:
            path = day_dir / stream / "full_day.parquet"
            if path.exists():
                found += 1
                day_size += path.stat().st_size

        total_size += day_size

        if found == total_streams:
            complete += 1
        elif found > 0:
            partial += 1
            streams_missing = [s for s in EXPECTED_STREAMS
                               if not (day_dir / s / "full_day.parquet").exists()]
            print(f"  {date_str}: {found}/{total_streams} (missing: {', '.join(streams_missing)})")
        else:
            missing += 1

        current += timedelta(days=1)

    gb = total_size / (1024**3)
    print(f"\n  Complete: {complete}/{total_days} days")
    print(f"  Partial:  {partial}/{total_days} days")
    print(f"  Missing:  {missing}/{total_days} days")
    print(f"  Total size: {gb:.2f} GB")
    print(f"  Need to download: {partial + missing} days × {total_streams} streams")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download all training data (Tardis + Binance DV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.download_all --start 2025-11-11 --end 2026-03-11
  python -m src.data.download_all --days 120 --workers 6
  python -m src.data.download_all --start 2025-11-11 --end 2026-03-11 --status
  python -m src.data.download_all --start 2025-11-11 --end 2026-03-11 --validate
        """,
    )
    parser.add_argument("--days", type=int, help="Download last N days")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate downloaded data only")
    parser.add_argument("--status", action="store_true",
                        help="Show download status only")
    args = parser.parse_args()

    # Date range
    if args.days:
        end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=args.days)
    elif args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = (
            datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if args.end
            else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        )
    else:
        parser.print_help()
        return

    # Status only
    if args.status:
        show_status(start_date, end_date)
        return

    # Validate only
    if args.validate:
        validate_range(start_date, end_date)
        return

    # Check tardis API key
    if not TARDIS_API_KEY:
        print("ERROR: Set TARDIS_API_KEY environment variable")
        print("  export TARDIS_API_KEY=your_key_here")
        return

    # Create jobs
    all_jobs = create_jobs(start_date, end_date)
    total_days = (end_date - start_date).days

    print(f"{'='*60}")
    print(f"  Download: {start_date.date()} to {end_date.date()} ({total_days} days)")
    print(f"  Datasets: 8 Tardis + 1 Binance DV = 9 per day")
    print(f"  Total jobs: {len(all_jobs)}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    tracker = ProgressTracker(len(all_jobs))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_job, job): job for job in all_jobs}

        for future in as_completed(futures):
            source, ds_name, date_str, status, rows, dl_bytes = future.result()
            label = f"{date_str} {ds_name} ({rows:,} rows)" if rows else ""

            if status == "skip":
                tracker.record("skip")
            elif status == "done":
                tracker.record("done", size_bytes=dl_bytes, label=label)
            elif status == "404":
                tracker.record("fail", msg=f"{ds_name} {date_str}: 404")
            elif status == "empty":
                tracker.record("fail", msg=f"{ds_name} {date_str}: empty")
            elif status.startswith("error"):
                tracker.record("fail", msg=f"{ds_name} {date_str}: {status}")

    tracker.summary()

    # Auto-validate
    print("\nRunning validation...")
    validate_range(start_date, end_date)


if __name__ == "__main__":
    main()
    sys.exit(0)

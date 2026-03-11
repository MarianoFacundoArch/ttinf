"""
Download historical BTCUSDT data from Binance Data Vision (data.binance.vision).

Downloads daily ZIP files, extracts CSVs, converts to parquet, and saves them
into data/raw/YYYY-MM-DD/<stream>/ matching the same layout as websocket_capture.

Usage:
  python -m src.data.download_binance --days 30             # last 30 days
  python -m src.data.download_binance --start 2026-02-01 --end 2026-03-01
  python -m src.data.download_binance --datasets aggTrades_spot,bookTicker
  python -m src.data.download_binance --list                 # show available datasets
"""

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://data.binance.vision/data"
SYMBOL = "BTCUSDT"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    "aggTrades_spot": {
        "url_path": f"spot/daily/aggTrades/{SYMBOL}",
        "file_pattern": f"{SYMBOL}-aggTrades-{{date}}.zip",
        "output_stream": "trades_spot",
        "columns": [
            "agg_trade_id", "price", "qty", "first_trade_id",
            "last_trade_id", "timestamp_ms", "is_buyer_maker", "is_best_match",
        ],
        "rename": {"transact_time": "timestamp_ms", "quantity": "qty"},
        "keep_cols": ["timestamp_ms", "agg_trade_id", "price", "qty", "is_buyer_maker"],
        "dtypes": {
            "timestamp_ms": "int64",
            "agg_trade_id": "int64",
            "price": "float64",
            "qty": "float64",
            "is_buyer_maker": "bool",
        },
    },
    "aggTrades_futures": {
        "url_path": f"futures/um/daily/aggTrades/{SYMBOL}",
        "file_pattern": f"{SYMBOL}-aggTrades-{{date}}.zip",
        "output_stream": "trades_futures",
        "columns": [
            "agg_trade_id", "price", "qty", "first_trade_id",
            "last_trade_id", "timestamp_ms", "is_buyer_maker",
        ],
        "rename": {"transact_time": "timestamp_ms", "quantity": "qty"},
        "keep_cols": ["timestamp_ms", "agg_trade_id", "price", "qty", "is_buyer_maker"],
        "dtypes": {
            "timestamp_ms": "int64",
            "agg_trade_id": "int64",
            "price": "float64",
            "qty": "float64",
            "is_buyer_maker": "bool",
        },
    },
    "bookTicker": {
        "url_path": f"futures/um/daily/bookTicker/{SYMBOL}",
        "file_pattern": f"{SYMBOL}-bookTicker-{{date}}.zip",
        "output_stream": "bookticker_futures",
        "columns": [
            "timestamp_ms", "best_bid_price", "best_bid_qty",
            "best_ask_price", "best_ask_qty",
        ],
        "keep_cols": None,  # keep all
        "dtypes": {
            "timestamp_ms": "int64",
            "best_bid_price": "float64",
            "best_bid_qty": "float64",
            "best_ask_price": "float64",
            "best_ask_qty": "float64",
        },
    },
    "metrics": {
        "url_path": f"futures/um/daily/metrics/{SYMBOL}",
        "file_pattern": f"{SYMBOL}-metrics-{{date}}.zip",
        "output_stream": "metrics",
        "columns": None,  # has header row
        "keep_cols": None,
        "dtypes": None,  # handled specially
    },
    "markPriceKlines": {
        "url_path": f"futures/um/daily/markPriceKlines/{SYMBOL}/1m",
        "file_pattern": f"{SYMBOL}-markPriceKlines-1m-{{date}}.zip",
        "output_stream": "mark_price_klines",
        "columns": [
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
        "keep_cols": [
            "open_time", "open", "high", "low", "close", "close_time",
        ],
        "dtypes": {
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "close_time": "int64",
        },
    },
    "premiumIndexKlines": {
        "url_path": f"futures/um/daily/premiumIndexKlines/{SYMBOL}/1m",
        "file_pattern": f"{SYMBOL}-premiumIndexKlines-1m-{{date}}.zip",
        "output_stream": "premium_index_klines",
        "columns": [
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
        "keep_cols": [
            "open_time", "open", "high", "low", "close", "close_time",
        ],
        "dtypes": {
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "close_time": "int64",
        },
    },
    "bookDepth": {
        "url_path": f"futures/um/daily/bookDepth/{SYMBOL}",
        "file_pattern": f"{SYMBOL}-bookDepth-{{date}}.zip",
        "output_stream": "book_depth_futures",
        "columns": None,  # has header row
        "keep_cols": None,
        "dtypes": None,
    },
    "klines_spot": {
        "url_path": f"spot/daily/klines/{SYMBOL}/1m",
        "file_pattern": f"{SYMBOL}-1m-{{date}}.zip",
        "output_stream": "klines_spot",
        "columns": [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
        "keep_cols": [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume",
        ],
        "dtypes": {
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "close_time": "int64",
            "quote_volume": "float64",
            "count": "int64",
            "taker_buy_volume": "float64",
            "taker_buy_quote_volume": "float64",
        },
    },
    "klines_futures": {
        "url_path": f"futures/um/daily/klines/{SYMBOL}/1m",
        "file_pattern": f"{SYMBOL}-1m-{{date}}.zip",
        "output_stream": "klines_futures",
        "columns": [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ],
        "keep_cols": [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume",
        ],
        "dtypes": {
            "open_time": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "close_time": "int64",
            "quote_volume": "float64",
            "count": "int64",
            "taker_buy_volume": "float64",
            "taker_buy_quote_volume": "float64",
        },
    },
}


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------

def download_zip(url: str) -> bytes | None:
    """Download a ZIP file. Returns bytes or None if 404."""
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def parse_csv_from_zip(
    zip_bytes: bytes,
    columns: list[str] | None,
    keep_cols: list[str] | None,
    dtypes: dict | None,
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Extract CSV from ZIP and parse into DataFrame."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        raw = zf.read(csv_name)

    # Strip BOM if present
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]

    text = raw.decode("utf-8")
    buf = io.StringIO(text)

    # Detect if first row is a header
    first_line = text.split("\n", 1)[0].strip()
    first_val = first_line.split(",")[0].strip('"').strip()
    has_header = not first_val.lstrip("-").replace(".", "", 1).isdigit()

    if has_header:
        # CSV has its own header
        df = pd.read_csv(buf)
    elif columns is not None:
        df = pd.read_csv(buf, header=None, names=columns)
    else:
        df = pd.read_csv(buf)

    # Rename columns to standardized names
    if rename:
        df = df.rename(columns=rename)

    # Keep only relevant columns
    if keep_cols:
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]

    # Cast dtypes
    if dtypes:
        for col, dtype in dtypes.items():
            if col in df.columns:
                if dtype == "bool":
                    df[col] = df[col].astype(bool)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    return df


def save_parquet(df: pd.DataFrame, date_str: str, stream_name: str):
    """Save DataFrame as a single parquet file for the day."""
    day_dir = OUTPUT_DIR / date_str / stream_name
    day_dir.mkdir(parents=True, exist_ok=True)

    filepath = day_dir / f"full_day.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, filepath, compression="snappy")
    return filepath


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_dataset(dataset_name: str, date_str: str) -> bool:
    """Download a single dataset for a single date. Returns True if successful."""
    ds = DATASETS[dataset_name]
    filename = ds["file_pattern"].format(date=date_str)
    url = f"{BASE_URL}/{ds['url_path']}/{filename}"

    # Check if already downloaded
    out_path = OUTPUT_DIR / date_str / ds["output_stream"] / "full_day.parquet"
    if out_path.exists():
        return True  # skip

    zip_bytes = download_zip(url)
    if zip_bytes is None:
        return False  # not available

    df = parse_csv_from_zip(
        zip_bytes,
        columns=ds["columns"],
        keep_cols=ds["keep_cols"],
        dtypes=ds["dtypes"],
        rename=ds.get("rename"),
    )

    filepath = save_parquet(df, date_str, ds["output_stream"])
    print(f"  {dataset_name} {date_str}: {len(df)} rows -> {filepath.name}")
    return True


def download_range(
    datasets: list[str],
    start_date: datetime,
    end_date: datetime,
):
    """Download all specified datasets for a date range."""
    current = start_date
    total_days = (end_date - start_date).days
    day_num = 0

    while current < end_date:
        day_num += 1
        date_str = current.strftime("%Y-%m-%d")
        print(f"\n[{day_num}/{total_days}] {date_str}")

        for ds_name in datasets:
            try:
                success = download_dataset(ds_name, date_str)
                if not success:
                    print(f"  {ds_name} {date_str}: not available (404)")
            except Exception as e:
                print(f"  {ds_name} {date_str}: ERROR - {e}")

        current += timedelta(days=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--days", type=int, help="Download last N days")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--datasets", type=str,
        help=f"Comma-separated datasets. Available: {','.join(DATASETS.keys())}",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, ds in DATASETS.items():
            print(f"  {name:30s} -> {ds['output_stream']}")
        return

    # Date range
    if args.days:
        end_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_date = end_date - timedelta(days=args.days)
    elif args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        if args.end:
            end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        else:
            end_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
    else:
        parser.print_help()
        print("\nProvide --days or --start/--end")
        return

    # Datasets
    if args.datasets:
        ds_names = args.datasets.split(",")
        invalid = [d for d in ds_names if d not in DATASETS]
        if invalid:
            print(f"Unknown datasets: {invalid}")
            print(f"Available: {list(DATASETS.keys())}")
            return
    else:
        # Default: the most important ones for our model
        ds_names = [
            "aggTrades_futures",
            "bookTicker",
            "aggTrades_spot",
            "metrics",
        ]

    print(f"Downloading {len(ds_names)} datasets from {start_date.date()} to {end_date.date()}")
    print(f"Datasets: {ds_names}")
    print(f"Output: {OUTPUT_DIR}")

    download_range(ds_names, start_date, end_date)
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Download historical BTCUSDT data from tardis.dev.

Downloads daily CSV.gz files and converts to parquet matching the same
layout as websocket_capture: data/raw/YYYY-MM-DD/<stream>/full_day.parquet

Usage:
  python -m src.data.download_tardis --days 30
  python -m src.data.download_tardis --start 2025-12-01 --end 2026-03-01
  python -m src.data.download_tardis --start 2025-12-01 --end 2026-03-01 --datasets fut_trades,fut_bookticker
  python -m src.data.download_tardis --list
"""

import argparse
import gzip
import io
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("TARDIS_API_KEY", "")
BASE_URL = "https://datasets.tardis.dev/v1"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# Dataset definitions: tardis endpoint -> our stream name + column mapping
# ---------------------------------------------------------------------------

DATASETS = {
    "fut_trades": {
        "exchange": "binance-futures",
        "data_type": "trades",
        "symbol": "BTCUSDT",
        "output_stream": "trades_futures",
        "column_map": {
            "timestamp": "timestamp_ms",
            "id": "agg_trade_id",
            "price": "price",
            "amount": "qty",
            "side": "_side",
        },
        "post_process": "_process_trades",
    },
    "spot_trades": {
        "exchange": "binance",
        "data_type": "trades",
        "symbol": "BTCUSDT",
        "output_stream": "trades_spot",
        "column_map": {
            "timestamp": "timestamp_ms",
            "id": "agg_trade_id",
            "price": "price",
            "amount": "qty",
            "side": "_side",
        },
        "post_process": "_process_trades",
    },
    "fut_bookticker": {
        "exchange": "binance-futures",
        "data_type": "book_ticker",
        "symbol": "BTCUSDT",
        "output_stream": "bookticker_futures",
        "column_map": {
            "timestamp": "timestamp_ms",
            "bid_price": "best_bid_price",
            "bid_amount": "best_bid_qty",
            "ask_price": "best_ask_price",
            "ask_amount": "best_ask_qty",
        },
    },
    "spot_bookticker": {
        "exchange": "binance",
        "data_type": "book_ticker",
        "symbol": "BTCUSDT",
        "output_stream": "bookticker_spot",
        "column_map": {
            "timestamp": "timestamp_ms",
            "bid_price": "best_bid_price",
            "bid_amount": "best_bid_qty",
            "ask_price": "best_ask_price",
            "ask_amount": "best_ask_qty",
        },
    },
    "fut_book25": {
        "exchange": "binance-futures",
        "data_type": "book_snapshot_25",
        "symbol": "BTCUSDT",
        "output_stream": "orderbook_futures",
        "post_process": "_process_book_snapshot",
    },
    "spot_book25": {
        "exchange": "binance",
        "data_type": "book_snapshot_25",
        "symbol": "BTCUSDT",
        "output_stream": "orderbook_spot",
        "post_process": "_process_book_snapshot",
    },
    "fut_deriv": {
        "exchange": "binance-futures",
        "data_type": "derivative_ticker",
        "symbol": "BTCUSDT",
        "output_stream": "mark_price",
        "column_map": {
            "timestamp": "timestamp_ms",
            "mark_price": "mark_price",
            "index_price": "index_price",
            "funding_rate": "funding_rate",
            "funding_timestamp": "next_funding_time_ms",
            "open_interest": "open_interest",
        },
    },
    "fut_liquidations": {
        "exchange": "binance-futures",
        "data_type": "liquidations",
        "symbol": "BTCUSDT",
        "output_stream": "liquidations",
        "column_map": {
            "timestamp": "timestamp_ms",
            "side": "side",
            "price": "price",
            "amount": "qty",
        },
    },
}

# Defaults: the core datasets we need
DEFAULT_DATASETS = [
    "fut_trades",
    "spot_trades",
    "fut_bookticker",
    "spot_bookticker",
    "fut_book25",
    "spot_book25",
    "fut_deriv",
    "fut_liquidations",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_csv(exchange: str, data_type: str, symbol: str, date_str: str) -> pd.DataFrame | None:
    """Download a single day CSV from tardis. Returns DataFrame or None if 404."""
    year, month, day = date_str.split("-")
    url = f"{BASE_URL}/{exchange}/{data_type}/{year}/{month}/{day}/{symbol}.csv.gz"

    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    resp = requests.get(url, headers=headers, timeout=120, stream=True)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    raw = gzip.decompress(resp.content)
    df = pd.read_csv(io.BytesIO(raw))
    return df


# ---------------------------------------------------------------------------
# Post-processors
# ---------------------------------------------------------------------------

def _process_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tardis trades format to our format."""
    # timestamp already converted to ms by _convert_timestamps
    # tardis has 'side' (buy/sell), we use is_buyer_maker
    # In tardis: side=sell means the taker was selling (buyer was maker)
    df["is_buyer_maker"] = df["_side"] == "sell"
    df = df.drop(columns=["_side"], errors="ignore")
    # Cast types
    df["agg_trade_id"] = df["agg_trade_id"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["qty"] = df["qty"].astype("float64")
    return df


def _process_book_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tardis book_snapshot_25 to our flat orderbook format (20 levels)."""
    # tardis columns: asks[0].price, asks[0].amount, bids[0].price, bids[0].amount, ...
    # We use: bid_price_0, bid_qty_0, ask_price_0, ask_qty_0, ...

    result = {}
    result["timestamp_ms"] = df["timestamp"].values // 1000
    result["recv_ts"] = result["timestamp_ms"]

    for i in range(20):  # we only use 20 levels (tardis has 25)
        result[f"bid_price_{i}"] = df[f"bids[{i}].price"].values
        result[f"bid_qty_{i}"] = df[f"bids[{i}].amount"].values
        result[f"ask_price_{i}"] = df[f"asks[{i}].price"].values
        result[f"ask_qty_{i}"] = df[f"asks[{i}].amount"].values

    return pd.DataFrame(result)


def _apply_column_map(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Rename tardis columns to our standard names, drop extras."""
    # Drop exchange/symbol/local_timestamp columns
    df = df.drop(columns=["exchange", "symbol", "local_timestamp"], errors="ignore")

    rename = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Keep only mapped columns
    keep = [v for v in column_map.values()]
    available = [c for c in keep if c in df.columns]
    df = df[available]

    return df


def _convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert microsecond timestamps to milliseconds."""
    for col in ["timestamp_ms", "next_funding_time_ms"]:
        if col in df.columns:
            # Only convert if values look like microseconds (> 1e15)
            if len(df) > 0 and df[col].iloc[0] > 1e15:
                df[col] = df[col] // 1000
    return df


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_dataset(ds_name: str, date_str: str) -> bool:
    """Download a single dataset for a single date. Returns True if successful."""
    ds = DATASETS[ds_name]

    # Check if already downloaded
    out_path = OUTPUT_DIR / date_str / ds["output_stream"] / "full_day.parquet"
    if out_path.exists():
        return True  # skip

    df = download_csv(ds["exchange"], ds["data_type"], ds["symbol"], date_str)
    if df is None:
        return False

    if len(df) == 0:
        return False

    # Apply column mapping
    if "column_map" in ds:
        df = _apply_column_map(df, ds["column_map"])

    # Convert timestamps
    df = _convert_timestamps(df)

    # Apply post-processor
    post = ds.get("post_process")
    if post == "_process_trades":
        df = _process_trades(df)
    elif post == "_process_book_snapshot":
        df = _process_book_snapshot(df)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    print(f"  {ds_name} {date_str}: {len(df):,} rows -> {ds['output_stream']}/full_day.parquet")
    return True


def download_range(datasets: list[str], start_date: datetime, end_date: datetime):
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
    parser = argparse.ArgumentParser(description="Download Tardis historical data")
    parser.add_argument("--days", type=int, help="Download last N days")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--datasets", type=str,
        help=f"Comma-separated. Available: {','.join(DATASETS.keys())}",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, ds in DATASETS.items():
            print(f"  {name:25s} -> {ds['output_stream']:25s} ({ds['exchange']}/{ds['data_type']})")
        return

    if not API_KEY:
        print("ERROR: Set TARDIS_API_KEY environment variable")
        print("  export TARDIS_API_KEY=your_key_here")
        return

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
        print("\nProvide --days or --start/--end")
        return

    # Datasets
    if args.datasets:
        ds_names = args.datasets.split(",")
        invalid = [d for d in ds_names if d not in DATASETS]
        if invalid:
            print(f"Unknown datasets: {invalid}")
            return
    else:
        ds_names = DEFAULT_DATASETS

    print(f"Downloading {len(ds_names)} datasets from {start_date.date()} to {end_date.date()}")
    print(f"Datasets: {ds_names}")
    print(f"Output: {OUTPUT_DIR}")

    download_range(ds_names, start_date, end_date)
    print("\nDone.")


if __name__ == "__main__":
    main()

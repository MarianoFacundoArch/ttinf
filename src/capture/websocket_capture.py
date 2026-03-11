"""
Binance WebSocket + REST capture for BTCUSDT.

Captures 8 websocket streams + 3 REST endpoints and saves to parquet files
partitioned by day: data/raw/YYYY-MM-DD/<stream>/chunk_HHMMSS.parquet

Usage:
  python -m src.capture.websocket_capture              # capture all streams
  python -m src.capture.websocket_capture --dry-run     # print to stdout, don't save
  python -m src.capture.websocket_capture --streams spot_trades,spot_depth  # subset
"""

import asyncio
import json
import time
import signal
import sys
import ssl
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import certifi
import websockets
import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "btcusdt"
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# How often to flush buffers to parquet (seconds)
FLUSH_INTERVAL = 60

# REST polling intervals (seconds)
OI_POLL_INTERVAL = 30
LS_POLL_INTERVAL = 60

# WebSocket URLs
SPOT_WS = "wss://stream.binance.com:9443/ws"
FUTURES_WS = "wss://fstream.binance.com/ws"

# REST URLs
FUTURES_REST = "https://fapi.binance.com"

# Reconnect config
RECONNECT_DELAY_BASE = 1.0
RECONNECT_DELAY_MAX = 30.0

# ---------------------------------------------------------------------------
# Schemas (pyarrow) for each stream
# ---------------------------------------------------------------------------

SCHEMAS = {
    "trades_spot": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("agg_trade_id", pa.int64()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
        ("is_buyer_maker", pa.bool_()),
        ("recv_ts", pa.int64()),
    ]),
    "trades_futures": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("agg_trade_id", pa.int64()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
        ("is_buyer_maker", pa.bool_()),
        ("recv_ts", pa.int64()),
    ]),
    "orderbook_spot": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("recv_ts", pa.int64()),
        # 20 levels bid price/qty + 20 levels ask price/qty = 80 columns
        *[(f"bid_price_{i}", pa.float64()) for i in range(20)],
        *[(f"bid_qty_{i}", pa.float64()) for i in range(20)],
        *[(f"ask_price_{i}", pa.float64()) for i in range(20)],
        *[(f"ask_qty_{i}", pa.float64()) for i in range(20)],
    ]),
    "orderbook_futures": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("recv_ts", pa.int64()),
        *[(f"bid_price_{i}", pa.float64()) for i in range(20)],
        *[(f"bid_qty_{i}", pa.float64()) for i in range(20)],
        *[(f"ask_price_{i}", pa.float64()) for i in range(20)],
        *[(f"ask_qty_{i}", pa.float64()) for i in range(20)],
    ]),
    "bookticker_spot": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("best_bid_price", pa.float64()),
        ("best_bid_qty", pa.float64()),
        ("best_ask_price", pa.float64()),
        ("best_ask_qty", pa.float64()),
        ("recv_ts", pa.int64()),
    ]),
    "bookticker_futures": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("best_bid_price", pa.float64()),
        ("best_bid_qty", pa.float64()),
        ("best_ask_price", pa.float64()),
        ("best_ask_qty", pa.float64()),
        ("recv_ts", pa.int64()),
    ]),
    "mark_price": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("mark_price", pa.float64()),
        ("index_price", pa.float64()),
        ("funding_rate", pa.float64()),
        ("next_funding_time_ms", pa.int64()),
        ("recv_ts", pa.int64()),
    ]),
    "liquidations": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("side", pa.string()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
        ("avg_price", pa.float64()),
        ("filled_qty", pa.float64()),
        ("status", pa.string()),
        ("recv_ts", pa.int64()),
    ]),
    "open_interest": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("open_interest", pa.float64()),
        ("recv_ts", pa.int64()),
    ]),
    "long_short_global": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("long_short_ratio", pa.float64()),
        ("long_account", pa.float64()),
        ("short_account", pa.float64()),
        ("recv_ts", pa.int64()),
    ]),
    "long_short_top": pa.schema([
        ("timestamp_ms", pa.int64()),
        ("long_short_ratio", pa.float64()),
        ("long_account", pa.float64()),
        ("short_account", pa.float64()),
        ("recv_ts", pa.int64()),
    ]),
}

# ---------------------------------------------------------------------------
# Buffer: collects rows in memory, flushes to parquet
# ---------------------------------------------------------------------------

class StreamBuffer:
    """Accumulates rows for a single stream and flushes to parquet."""

    def __init__(self, stream_name: str, schema: pa.Schema, dry_run: bool = False):
        self.stream_name = stream_name
        self.schema = schema
        self.dry_run = dry_run
        self.rows: list[dict] = []
        self.count = 0
        self.last_flush = time.time()

    def add(self, row: dict):
        if self.dry_run:
            self.count += 1
            if self.count <= 3 or self.count % 100 == 0:
                print(f"  [{self.stream_name}] #{self.count}: {row}")
            return
        self.rows.append(row)
        self.count += 1

    def should_flush(self) -> bool:
        return len(self.rows) > 0 and (time.time() - self.last_flush) >= FLUSH_INTERVAL

    def flush(self):
        if not self.rows or self.dry_run:
            return

        now = datetime.now(timezone.utc)
        day_dir = BASE_DIR / now.strftime("%Y-%m-%d") / self.stream_name
        day_dir.mkdir(parents=True, exist_ok=True)

        filename = f"chunk_{now.strftime('%H%M%S')}.parquet"
        filepath = day_dir / filename

        # Build columnar dict from rows
        columns = {field.name: [] for field in self.schema}
        for row in self.rows:
            for col in columns:
                columns[col].append(row.get(col))

        table = pa.table(columns, schema=self.schema)
        pq.write_table(table, filepath, compression="snappy")

        n = len(self.rows)
        self.rows.clear()
        self.last_flush = time.time()
        print(f"  [{self.stream_name}] flushed {n} rows -> {filepath.name}")


# ---------------------------------------------------------------------------
# Parsers: raw JSON -> dict row
# ---------------------------------------------------------------------------

def parse_agg_trade(msg: dict, recv_ts: int) -> dict:
    return {
        "timestamp_ms": msg["T"],
        "agg_trade_id": msg["a"],
        "price": float(msg["p"]),
        "qty": float(msg["q"]),
        "is_buyer_maker": msg["m"],
        "recv_ts": recv_ts,
    }


def parse_depth20(msg: dict, recv_ts: int) -> dict:
    row = {
        "timestamp_ms": msg.get("T", msg.get("E", recv_ts)),
        "recv_ts": recv_ts,
    }
    for i, level in enumerate(msg.get("bids", msg.get("b", []))[:20]):
        row[f"bid_price_{i}"] = float(level[0])
        row[f"bid_qty_{i}"] = float(level[1])
    for i, level in enumerate(msg.get("asks", msg.get("a", []))[:20]):
        row[f"ask_price_{i}"] = float(level[0])
        row[f"ask_qty_{i}"] = float(level[1])
    return row


def parse_book_ticker(msg: dict, recv_ts: int) -> dict:
    return {
        "timestamp_ms": msg.get("T", msg.get("E", recv_ts)),
        "best_bid_price": float(msg["b"]),
        "best_bid_qty": float(msg["B"]),
        "best_ask_price": float(msg["a"]),
        "best_ask_qty": float(msg["A"]),
        "recv_ts": recv_ts,
    }


def parse_mark_price(msg: dict, recv_ts: int) -> dict:
    return {
        "timestamp_ms": msg["E"],
        "mark_price": float(msg["p"]),
        "index_price": float(msg["i"]),
        "funding_rate": float(msg["r"]),
        "next_funding_time_ms": msg["T"],
        "recv_ts": recv_ts,
    }


def parse_force_order(msg: dict, recv_ts: int) -> dict:
    o = msg["o"]
    return {
        "timestamp_ms": o["T"],
        "side": o["S"],
        "price": float(o["p"]),
        "qty": float(o["q"]),
        "avg_price": float(o["ap"]),
        "filled_qty": float(o["z"]),
        "status": o["X"],
        "recv_ts": recv_ts,
    }


# ---------------------------------------------------------------------------
# WebSocket stream handlers
# ---------------------------------------------------------------------------

def _ssl_context():
    ctx = ssl.create_default_context(cafile=certifi.where())
    return ctx


# Stream definitions: (name, ws_url, subscribe_params, parser, buffer_name)
STREAM_DEFS = {
    "spot_trades": {
        "ws_url": SPOT_WS,
        "subscribe": f"{SYMBOL}@aggTrade",
        "parser": parse_agg_trade,
        "buffer": "trades_spot",
    },
    "spot_depth": {
        "ws_url": SPOT_WS,
        "subscribe": f"{SYMBOL}@depth20@100ms",
        "parser": parse_depth20,
        "buffer": "orderbook_spot",
    },
    "spot_bookticker": {
        "ws_url": SPOT_WS,
        "subscribe": f"{SYMBOL}@bookTicker",
        "parser": parse_book_ticker,
        "buffer": "bookticker_spot",
    },
    "fut_trades": {
        "ws_url": FUTURES_WS,
        "subscribe": f"{SYMBOL}@aggTrade",
        "parser": parse_agg_trade,
        "buffer": "trades_futures",
    },
    "fut_depth": {
        "ws_url": FUTURES_WS,
        "subscribe": f"{SYMBOL}@depth20@100ms",
        "parser": parse_depth20,
        "buffer": "orderbook_futures",
    },
    "fut_bookticker": {
        "ws_url": FUTURES_WS,
        "subscribe": f"{SYMBOL}@bookTicker",
        "parser": parse_book_ticker,
        "buffer": "bookticker_futures",
    },
    "fut_markprice": {
        "ws_url": FUTURES_WS,
        "subscribe": f"{SYMBOL}@markPrice@1s",
        "parser": parse_mark_price,
        "buffer": "mark_price",
    },
    "fut_liquidations": {
        "ws_url": FUTURES_WS,
        "subscribe": f"{SYMBOL}@forceOrder",
        "parser": parse_force_order,
        "buffer": "liquidations",
    },
}


async def ws_stream(stream_name: str, buffers: dict[str, StreamBuffer]):
    """Connect to a single websocket stream with auto-reconnect."""
    sdef = STREAM_DEFS[stream_name]
    buf = buffers[sdef["buffer"]]
    parser = sdef["parser"]
    ws_url = f"{sdef['ws_url']}/{sdef['subscribe']}"

    delay = RECONNECT_DELAY_BASE

    while True:
        try:
            async with websockets.connect(
                ws_url,
                ssl=_ssl_context(),
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                print(f"  [{stream_name}] connected to {sdef['subscribe']}")
                delay = RECONNECT_DELAY_BASE  # reset on success

                async for raw in ws:
                    recv_ts = int(time.time() * 1000)
                    msg = json.loads(raw)
                    # Some streams wrap in {"stream": ..., "data": ...}
                    if "data" in msg and "stream" in msg:
                        msg = msg["data"]
                    row = parser(msg, recv_ts)
                    buf.add(row)

        except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
            print(f"  [{stream_name}] disconnected: {e}. Reconnecting in {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_DELAY_MAX)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [{stream_name}] unexpected error: {e}. Reconnecting in {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_DELAY_MAX)


# ---------------------------------------------------------------------------
# REST polling handlers
# ---------------------------------------------------------------------------

async def poll_open_interest(
    session: aiohttp.ClientSession, buf: StreamBuffer
):
    """Poll open interest every OI_POLL_INTERVAL seconds."""
    url = f"{FUTURES_REST}/fapi/v1/openInterest"
    params = {"symbol": "BTCUSDT"}

    while True:
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    recv_ts = int(time.time() * 1000)
                    buf.add({
                        "timestamp_ms": data.get("time", recv_ts),
                        "open_interest": float(data["openInterest"]),
                        "recv_ts": recv_ts,
                    })
                else:
                    print(f"  [open_interest] HTTP {resp.status}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [open_interest] error: {e}")

        await asyncio.sleep(OI_POLL_INTERVAL)


async def poll_long_short(
    session: aiohttp.ClientSession,
    buf_global: StreamBuffer,
    buf_top: StreamBuffer,
):
    """Poll global and top trader long/short ratios."""
    urls = {
        "global": f"{FUTURES_REST}/futures/data/globalLongShortAccountRatio",
        "top": f"{FUTURES_REST}/futures/data/topLongShortAccountRatio",
    }
    params = {"symbol": "BTCUSDT", "period": "5m", "limit": 1}

    while True:
        for kind, url in urls.items():
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data_list = await resp.json()
                        if data_list:
                            d = data_list[0]
                            recv_ts = int(time.time() * 1000)
                            row = {
                                "timestamp_ms": d.get("timestamp", recv_ts),
                                "long_short_ratio": float(d["longShortRatio"]),
                                "long_account": float(d["longAccount"]),
                                "short_account": float(d["shortAccount"]),
                                "recv_ts": recv_ts,
                            }
                            (buf_global if kind == "global" else buf_top).add(row)
                    else:
                        print(f"  [long_short_{kind}] HTTP {resp.status}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"  [long_short_{kind}] error: {e}")

        await asyncio.sleep(LS_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Flusher: periodically writes buffers to disk
# ---------------------------------------------------------------------------

async def flusher(buffers: dict[str, StreamBuffer]):
    """Periodically flush all buffers to parquet."""
    while True:
        await asyncio.sleep(FLUSH_INTERVAL)
        for buf in buffers.values():
            try:
                if buf.should_flush():
                    buf.flush()
            except Exception as e:
                print(f"  [{buf.stream_name}] flush error: {e}")


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------

async def stats_printer(buffers: dict[str, StreamBuffer]):
    """Print stream stats every 10 seconds."""
    while True:
        await asyncio.sleep(10)
        parts = []
        for name, buf in sorted(buffers.items()):
            parts.append(f"{name}={buf.count}")
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{now} UTC] counts: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(dry_run: bool = False, stream_filter: Optional[set[str]] = None):
    """Launch all capture tasks."""

    print("=" * 60)
    print(f"  BTCUSDT Capture {'(DRY RUN)' if dry_run else ''}")
    print(f"  Flush interval: {FLUSH_INTERVAL}s")
    print(f"  Output: {BASE_DIR}")
    print("=" * 60)

    # Create buffers
    buffers: dict[str, StreamBuffer] = {}
    for name, schema in SCHEMAS.items():
        buffers[name] = StreamBuffer(name, schema, dry_run=dry_run)

    tasks = []

    # WebSocket streams
    for stream_name in STREAM_DEFS:
        if stream_filter and stream_name not in stream_filter:
            continue
        tasks.append(asyncio.create_task(
            ws_stream(stream_name, buffers),
            name=stream_name,
        ))

    # REST polling
    ssl_ctx = _ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    session = aiohttp.ClientSession(connector=connector)

    if not stream_filter or "open_interest" in stream_filter:
        tasks.append(asyncio.create_task(
            poll_open_interest(session, buffers["open_interest"]),
            name="open_interest",
        ))
    if not stream_filter or "long_short" in stream_filter:
        tasks.append(asyncio.create_task(
            poll_long_short(
                session, buffers["long_short_global"], buffers["long_short_top"]
            ),
            name="long_short",
        ))

    # Flusher + stats
    tasks.append(asyncio.create_task(flusher(buffers), name="flusher"))
    tasks.append(asyncio.create_task(stats_printer(buffers), name="stats"))

    print(f"\n  Started {len(tasks)} tasks. Press Ctrl+C to stop.\n")

    # Wait for cancellation
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        # Final flush
        print("\n  Shutting down, final flush...")
        for buf in buffers.values():
            try:
                buf.flush()
            except Exception as e:
                print(f"  [{buf.stream_name}] final flush error: {e}")
        await session.close()
        print("  Done.")


def run():
    dry_run = "--dry-run" in sys.argv
    stream_filter = None

    for arg in sys.argv[1:]:
        if arg.startswith("--streams="):
            stream_filter = set(arg.split("=", 1)[1].split(","))

    loop = asyncio.new_event_loop()

    def _shutdown():
        for task in asyncio.all_tasks(loop):
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, _shutdown)
    loop.add_signal_handler(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(main(dry_run=dry_run, stream_filter=stream_filter))
    except KeyboardInterrupt:
        _shutdown()
        loop.run_until_complete(asyncio.sleep(0.5))
    finally:
        loop.close()


if __name__ == "__main__":
    run()

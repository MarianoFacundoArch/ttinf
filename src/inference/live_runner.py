"""
Live BTC 5-minute block predictor.

Connects to Binance websockets, computes features every second,
runs LightGBM model, prints P(Up) to console.

Usage:
    python -m src.inference.live_runner
    python -m src.inference.live_runner --interval 5   # predict every 5s
"""

import asyncio
import json
import time
import ssl
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import certifi
import websockets
import aiohttp
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.inference.live_buffer import LiveBuffer
from src.inference.live_predictor import LivePredictor


# ---------------------------------------------------------------------------
# Polymarket integration
# ---------------------------------------------------------------------------

POLYMARKET_CRYPTO_API = "https://polymarket.com/api/crypto/crypto-price"
POLYMARKET_LIVE_WS = "wss://ws-live-data.polymarket.com/"


class ChainlinkPriceFeed:
    """Real-time BTC/USD price from Polymarket RTDS websocket."""

    def __init__(self):
        self.price = 0.0
        self.received_at = 0.0  # time.time() when last price arrived

    @property
    def age(self):
        if self.received_at == 0:
            return float('inf')
        return time.time() - self.received_at


class PolymarketTracker:
    """Track Polymarket 5-min markets and poll for price_to_beat."""

    def __init__(self, asset="BTC", timeframe="5m"):
        self.asset = asset
        self.timeframe = timeframe
        self.interval_sec = 900 if timeframe == "15m" else 300
        self.variant = "fifteenminute" if timeframe == "15m" else "fiveminute"

        self.current_open_ts = 0      # Unix seconds of current block open
        self.price_to_beat = None     # Strike price from Polymarket
        self._polling = False         # True while polling for price
        self.enabled = False          # Set to True to activate
        self.chainlink = ChainlinkPriceFeed()  # Real-time Chainlink price

        # Previous block close tracking
        self._pending_close_ts = None   # block_open_ts of block awaiting close
        self._pending_close_strike = None  # strike of that block
        self.last_close_result = None   # dict with close info once resolved

    def get_block_open_ts(self, now_sec=None):
        """Get the current block's open timestamp in seconds."""
        if now_sec is None:
            now_sec = int(time.time())
        return (now_sec // self.interval_sec) * self.interval_sec

    def check_new_block(self):
        """Check if we entered a new block. Returns True if block changed."""
        block_ts = self.get_block_open_ts()
        if block_ts != self.current_open_ts:
            # Queue previous block for close result polling
            if self.current_open_ts > 0 and self.price_to_beat is not None:
                self._pending_close_ts = self.current_open_ts
                self._pending_close_strike = self.price_to_beat
                self.last_close_result = None
            self.current_open_ts = block_ts
            self.price_to_beat = None
            self._polling = True
            return True
        return False

    def get_api_url(self, block_open_ts=None):
        """Build the Polymarket crypto-price API URL for a block."""
        ts = block_open_ts if block_open_ts else self.current_open_ts
        open_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        close_dt = datetime.fromtimestamp(ts + self.interval_sec, tz=timezone.utc)
        open_iso = open_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        close_iso = close_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return (
            f"{POLYMARKET_CRYPTO_API}?symbol={self.asset}"
            f"&eventStartTime={open_iso}"
            f"&variant={self.variant}"
            f"&endDate={close_iso}"
        )


async def polymarket_poller(tracker, session, stop_event):
    """Poll Polymarket API for price_to_beat and previous block close result."""
    if not tracker.enabled:
        return

    while not stop_event.is_set():
        try:
            # Check for new block
            tracker.check_new_block()

            # 1) Poll for current block's price_to_beat
            if tracker._polling and tracker.price_to_beat is None:
                url = tracker.get_api_url()
                try:
                    async with session.get(url, ssl=SSL_CONTEXT, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            data = await r.json()
                            open_price = data.get("openPrice")
                            if open_price and float(open_price) > 0:
                                tracker.price_to_beat = float(open_price)
                                tracker._polling = False
                                block_time = datetime.fromtimestamp(
                                    tracker.current_open_ts, tz=timezone.utc
                                ).strftime("%H:%M")
                                print(f"\n  [polymarket] Block {block_time}: "
                                      f"price_to_beat = ${tracker.price_to_beat:,.2f}")
                except Exception:
                    pass  # Silent retry
                await asyncio.sleep(0.5)

            # 2) Poll for previous block's close result
            elif tracker._pending_close_ts is not None:
                url = tracker.get_api_url(tracker._pending_close_ts)
                try:
                    async with session.get(url, ssl=SSL_CONTEXT, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            data = await r.json()
                            if data.get("completed") and data.get("closePrice"):
                                cp = float(data["closePrice"])
                                op = float(data.get("openPrice", tracker._pending_close_strike))
                                ret_bps = (cp - op) / op * 10_000
                                outcome = "UP" if cp >= op else "DOWN"
                                tracker.last_close_result = {
                                    "close_price": cp,
                                    "open_price": op,
                                    "return_bps": ret_bps,
                                    "outcome": outcome,
                                }
                                print(f"\n  >> CLOSED (Polymarket): {cp:,.2f}  |  "
                                      f"strike: {op:,.2f}  |  "
                                      f"{ret_bps:+.2f} bps  |  {outcome}")
                                tracker._pending_close_ts = None
                except Exception:
                    pass  # Silent retry
                await asyncio.sleep(1)

            else:
                # Nothing to poll — check every 1s for new blocks
                await asyncio.sleep(1)

        except Exception as e:
            print(f"\n  [polymarket] error: {e}")
            await asyncio.sleep(2)


async def ws_polymarket_price_stream(tracker, stop_event):
    """Connect to Polymarket RTDS for real-time Chainlink BTC/USD price."""
    if not tracker.enabled:
        return

    sub_msg = json.dumps({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "filters": json.dumps({"symbol": "btc/usd"}, separators=(",", ":")),
        }],
    })

    while not stop_event.is_set():
        try:
            async with websockets.connect(POLYMARKET_LIVE_WS, ssl=SSL_CONTEXT,
                                          ping_interval=5,
                                          ping_timeout=20) as ws:
                await ws.send(sub_msg)
                print("  [polymarket-price] Connected to RTDS (chainlink btc/usd)")
                async for msg in ws:
                    if stop_event.is_set():
                        break
                    try:
                        raw = msg if isinstance(msg, str) else msg.decode()
                        if not raw.strip():
                            continue
                        d = json.loads(raw)
                        payload = d.get("payload")
                        if not payload:
                            continue

                        # Snapshot: {"payload": {"data": [{timestamp, value}, ...]}}
                        if isinstance(payload, dict) and "data" in payload:
                            data = payload["data"]
                            if isinstance(data, list) and len(data) > 0:
                                last = data[-1]
                                tracker.chainlink.price = float(last["value"])
                                tracker.chainlink.received_at = time.time()
                                print(f"  [polymarket-price] ${tracker.chainlink.price:,.2f}")

                        # Update: {topic, type: "update", payload: {value, ...}}
                        elif d.get("topic") == "crypto_prices_chainlink":
                            tracker.chainlink.price = float(payload["value"])
                            tracker.chainlink.received_at = time.time()

                    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                        pass
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n  [polymarket-price] reconnecting: {e}")
                await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "btcusdt"
SPOT_WS = "wss://stream.binance.com:9443/ws"
FUTURES_WS = "wss://fstream.binance.com/ws"
FUTURES_REST = "https://fapi.binance.com"
SPOT_REST = "https://api.binance.com"

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Cross-exchange websockets
COINBASE_WS = "wss://advanced-trade-ws.coinbase.com"
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"


# ---------------------------------------------------------------------------
# Rate-limited error logging (avoids spamming console)
# ---------------------------------------------------------------------------

_ws_error_counts = {}   # key -> total count
_ws_error_last_log = {} # key -> time.time() of last log


def _log_ws_error(source, stream, error, raw_keys=None):
    """Log a websocket parse error, rate-limited to 1 per 60s per source/stream."""
    key = f"{source}/{stream}"
    _ws_error_counts[key] = _ws_error_counts.get(key, 0) + 1
    now = time.time()
    last = _ws_error_last_log.get(key, 0)
    if now - last >= 60:
        _ws_error_last_log[key] = now
        count = _ws_error_counts[key]
        extra = f" | keys={raw_keys}" if raw_keys else ""
        print(f"\n[WS ERROR] {key}: {error} (count={count}){extra}")

def _load_bucket_stats(model_dir="models"):
    """Load ECE per 5s bucket from metrics_v3.json (generated by training).

    Returns dict: (lo, hi) → (accuracy, ece)
    Falls back to conservative defaults if file not found.
    """
    stats = {}
    for lo in range(0, 300, 5):
        stats[(lo, lo + 5)] = (0.50, 0.02)  # conservative default

    metrics_path = Path(model_dir) / "metrics_v3.json"
    if not metrics_path.exists():
        print(f"[WARNING] {metrics_path} not found — using default ECE=0.02")
        return stats

    try:
        with open(metrics_path) as f:
            m = json.load(f)
        folds = m.get("walk_forward", [])
        if not folds:
            return stats

        for lo in range(0, 300, 5):
            label = f"{lo}-{lo+5}s"
            eces = []
            accs = []
            for fold in folds:
                fine = fold.get("phase_results", {}).get("_fine", {})
                if label in fine:
                    eces.append(fine[label]["ece"])
                    accs.append(fine[label]["accuracy"])
            if eces:
                stats[(lo, lo + 5)] = (float(np.mean(accs)), float(np.mean(eces)))

        print(f"  Loaded ECE stats from {metrics_path} ({len(folds)} folds)")
    except Exception as e:
        print(f"[WARNING] Failed to load ECE from {metrics_path}: {e}")

    return stats


BUCKET_STATS = _load_bucket_stats()


def get_bucket_stats(seconds_to_expiry):
    """Get (accuracy, ece) for the current seconds_to_expiry."""
    s = max(0, min(299, seconds_to_expiry))
    lo = int(s // 5) * 5
    hi = lo + 5
    return BUCKET_STATS.get((lo, hi), (0.50, 0.02))


# ---------------------------------------------------------------------------
# WebSocket Health Tracking
# ---------------------------------------------------------------------------

class WSHealth:
    """Track last message time and reconnection gaps for each websocket."""

    # Longest feature window is 120s (VPIN). After reconnection, data may
    # be incomplete until this many seconds of fresh data have accumulated.
    DATA_COMPLETE_WINDOW_S = 120

    def __init__(self):
        self.last_msg = {
            "binance_futures": 0.0,
            "binance_spot": 0.0,
            "coinbase": 0.0,
            "bybit": 0.0,
        }
        # Track when each source last reconnected (had a gap > 5s)
        self._last_reconnect = {
            "binance_futures": 0.0,
            "binance_spot": 0.0,
            "coinbase": 0.0,
            "bybit": 0.0,
        }

    def update(self, source):
        now = time.time()
        prev = self.last_msg[source]
        # Detect reconnection: gap > 5s since last message
        if prev > 0 and (now - prev) > 5.0:
            self._last_reconnect[source] = now
        # First message ever also counts as reconnect
        elif prev == 0.0:
            self._last_reconnect[source] = now
        self.last_msg[source] = now

    def age(self, source):
        t = self.last_msg[source]
        return time.time() - t if t > 0 else float('inf')

    def is_fresh(self, source, max_age_s=5.0):
        return self.age(source) < max_age_s

    def binance_ok(self):
        """Binance futures must be fresh to predict."""
        return self.is_fresh("binance_futures", 5.0)

    def seconds_since_reconnect(self, source):
        """Seconds since last reconnection for a source."""
        t = self._last_reconnect[source]
        return time.time() - t if t > 0 else float('inf')

    def data_complete(self):
        """True if all critical sources have enough data since last reconnect."""
        for src in ["binance_futures", "binance_spot"]:
            if self.seconds_since_reconnect(src) < self.DATA_COMPLETE_WINDOW_S:
                return False
        return True

    def time_until_complete(self):
        """Seconds until data is considered complete (0 if already complete)."""
        worst = 0.0
        for src in ["binance_futures", "binance_spot"]:
            remaining = self.DATA_COMPLETE_WINDOW_S - self.seconds_since_reconnect(src)
            if remaining > worst:
                worst = remaining
        return max(0.0, worst)

    def status_line(self, poly_tracker=None):
        parts = []
        all_ok = True
        for src in ["binance_futures", "binance_spot", "coinbase", "bybit"]:
            age = self.age(src)
            if age == float('inf'):
                parts.append(f"{src}:OFF")
                all_ok = False
            elif age > 10:
                parts.append(f"{src}:STALE({age:.0f}s)")
                all_ok = False
            else:
                parts.append(f"{src}:OK")
        if poly_tracker and poly_tracker.enabled:
            cl_age = poly_tracker.chainlink.age
            if poly_tracker.chainlink.price <= 0 or cl_age > 10:
                parts.append(f"polymarket_price:OFF")
                all_ok = False
            else:
                parts.append(f"polymarket_price:OK")
            if poly_tracker.price_to_beat is None:
                parts.append(f"polymarket_strike:WAITING")
                all_ok = False
            else:
                parts.append(f"polymarket_strike:OK")
        return " | ".join(parts), all_ok


# ---------------------------------------------------------------------------
# REST Warmup
# ---------------------------------------------------------------------------

async def warmup(buffer, session):
    """Fetch recent historical data to fill buffers before going live."""
    now_ms = int(time.time() * 1000)
    print("Warming up...")

    # 1. Index price klines (last 60 min for ref_price)
    # Uses indexPriceKlines for accurate index_price (not futures close which
    # has basis/premium vs index). ref_price is built from index_price.
    try:
        url = f"{FUTURES_REST}/fapi/v1/indexPriceKlines?pair=BTCUSDT&interval=1m&limit=60"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for k in data:
                ts = int(k[0])
                index_close = float(k[4])
                # mark_price not available from index klines, use index as approximation
                buffer.add_mark_price(ts, index_close, index_close, 0.0, 0)
            print(f"  indexPriceKlines 1m: {len(data)} bars")
    except Exception as e:
        print(f"  indexPriceKlines 1m: ERROR {e}")

    # 2. Premium index (current mark, index, funding)
    try:
        url = f"{FUTURES_REST}/fapi/v1/premiumIndex?symbol=BTCUSDT"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            buffer.add_mark_price(
                int(data["time"]),
                float(data["markPrice"]),
                float(data["indexPrice"]),
                float(data["lastFundingRate"]),
                int(data["nextFundingTime"]),
            )
            print(f"  premiumIndex: mark={data['markPrice']}, index={data['indexPrice']}")
    except Exception as e:
        print(f"  premiumIndex: ERROR {e}")

    # 3. Metrics (long/short, taker, top traders)
    # First pass: global long/short creates the rows
    try:
        url = f"{FUTURES_REST}/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=10"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for item in data:
                ts = int(item["timestamp"])
                ls = float(item["longShortRatio"])
                buffer.add_metrics(ts, ls, 0.0, 0.0, np.nan)
            print(f"  global_ls: {len(data)} bars")
    except Exception as e:
        print(f"  global_ls: ERROR {e}")

    # Second pass: top trader long/short updates existing rows
    try:
        url = f"{FUTURES_REST}/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=5m&limit=10"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for item in data:
                ts = int(item["timestamp"])
                val = float(item["longShortRatio"])
                for i, t in enumerate(buffer.mt_ts):
                    if t == ts:
                        buffer.mt_top_ls[i] = val
                        break
            print(f"  top_ls: {len(data)} bars")
    except Exception as e:
        print(f"  top_ls: ERROR {e}")

    # Third pass: taker buy/sell ratio
    try:
        url = f"{FUTURES_REST}/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=10"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for item in data:
                ts = int(item["timestamp"])
                val = float(item["buySellRatio"])
                for i, t in enumerate(buffer.mt_ts):
                    if t == ts:
                        buffer.mt_taker_ls[i] = val
                        break
            print(f"  taker_ls: {len(data)} bars")
    except Exception as e:
        print(f"  taker_ls: ERROR {e}")

    # 4. Open Interest
    try:
        url = f"{FUTURES_REST}/fapi/v1/openInterest?symbol=BTCUSDT"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            oi = float(data["openInterest"])
            if buffer.mt_ts:
                buffer.mt_oi[-1] = oi
            print(f"  openInterest: {oi:.2f}")
    except Exception as e:
        print(f"  openInterest: ERROR {e}")

    # 5. Recent trades futures (last 120s — needed for VPIN_120s)
    try:
        start_ms = now_ms - 120_000
        url = (f"{FUTURES_REST}/fapi/v1/aggTrades?symbol=BTCUSDT"
               f"&startTime={start_ms}&endTime={now_ms}&limit=1000")
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for t in data:
                buffer.add_trade_futures(
                    int(t["T"]), float(t["p"]), float(t["q"]), bool(t["m"])
                )
            print(f"  trades_futures: {len(data)} trades (last 120s)")
    except Exception as e:
        print(f"  trades_futures: ERROR {e}")

    # 6. Recent trades spot (last 120s)
    try:
        start_ms = now_ms - 120_000
        url = (f"{SPOT_REST}/api/v3/aggTrades?symbol=BTCUSDT"
               f"&startTime={start_ms}&endTime={now_ms}&limit=1000")
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for t in data:
                buffer.add_trade_spot(
                    int(t["T"]), float(t["p"]), float(t["q"]), bool(t["m"])
                )
            print(f"  trades_spot: {len(data)} trades (last 120s)")
    except Exception as e:
        print(f"  trades_spot: ERROR {e}")

    # 7. Current orderbook futures
    try:
        url = f"{FUTURES_REST}/fapi/v1/depth?symbol=BTCUSDT&limit=20"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            ts = int(data.get("T", time.time() * 1000))
            bids = data["bids"]
            asks = data["asks"]
            bp = [float(b[0]) for b in bids[:20]]
            bq = [float(b[1]) for b in bids[:20]]
            ap = [float(a[0]) for a in asks[:20]]
            aq = [float(a[1]) for a in asks[:20]]
            while len(bp) < 20: bp.append(0.0); bq.append(0.0)
            while len(ap) < 20: ap.append(0.0); aq.append(0.0)
            buffer.add_orderbook_futures(ts, bp, bq, ap, aq)
            print(f"  depth_futures: 20 levels")
    except Exception as e:
        print(f"  depth_futures: ERROR {e}")

    # 8. Current orderbook spot
    try:
        url = f"{SPOT_REST}/api/v3/depth?symbol=BTCUSDT&limit=20"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            ts = int(time.time() * 1000)
            bids = data["bids"]
            asks = data["asks"]
            bp = [float(b[0]) for b in bids[:20]]
            bq = [float(b[1]) for b in bids[:20]]
            ap = [float(a[0]) for a in asks[:20]]
            aq = [float(a[1]) for a in asks[:20]]
            while len(bp) < 20: bp.append(0.0); bq.append(0.0)
            while len(ap) < 20: ap.append(0.0); aq.append(0.0)
            buffer.add_orderbook_spot(ts, bp, bq, ap, aq)
            print(f"  depth_spot: 20 levels")
    except Exception as e:
        print(f"  depth_spot: ERROR {e}")

    # 9. Current booktickers
    try:
        url = f"{FUTURES_REST}/fapi/v1/ticker/bookTicker?symbol=BTCUSDT"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            buffer.add_bookticker_futures(
                int(data["time"]), float(data["bidPrice"]), float(data["askPrice"])
            )
            print(f"  bookticker_futures: bid={data['bidPrice']}, ask={data['askPrice']}")
    except Exception as e:
        print(f"  bookticker_futures: ERROR {e}")

    try:
        url = f"{SPOT_REST}/api/v3/ticker/bookTicker?symbol=BTCUSDT"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            buffer.add_bookticker_spot(
                int(time.time() * 1000), float(data["bidPrice"]), float(data["askPrice"])
            )
            print(f"  bookticker_spot: bid={data['bidPrice']}, ask={data['askPrice']}")
    except Exception as e:
        print(f"  bookticker_spot: ERROR {e}")

    stats = buffer.stats()
    print(f"\nWarmup done. Buffer: {stats}")


# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------

async def ws_futures_stream(buffer, health, stop_event):
    """Connect to futures combined stream."""
    streams = [
        f"{SYMBOL}@aggTrade",
        f"{SYMBOL}@bookTicker",
        f"{SYMBOL}@depth20@100ms",
        f"{SYMBOL}@markPrice@1s",
        f"{SYMBOL}@forceOrder",
    ]
    url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ssl=SSL_CONTEXT,
                                          ping_interval=None,
                                          ping_timeout=None,
                                          max_size=2**22) as ws:
                async for msg in ws:
                    if stop_event.is_set():
                        break
                    health.update("binance_futures")
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        d = data.get("data", data)

                        if "aggTrade" in stream:
                            buffer.add_trade_futures(
                                int(d["T"]), float(d["p"]), float(d["q"]), d["m"]
                            )
                        elif "bookTicker" in stream:
                            buffer.add_bookticker_futures(
                                int(d["T"]), float(d["b"]), float(d["a"])
                            )
                        elif "depth20" in stream:
                            ts = int(d.get("T", d.get("E", time.time() * 1000)))
                            bids = d["b"]
                            asks = d["a"]
                            bp = [float(b[0]) for b in bids[:20]]
                            bq = [float(b[1]) for b in bids[:20]]
                            ap = [float(a[0]) for a in asks[:20]]
                            aq = [float(a[1]) for a in asks[:20]]
                            while len(bp) < 20: bp.append(0.0); bq.append(0.0)
                            while len(ap) < 20: ap.append(0.0); aq.append(0.0)
                            buffer.add_orderbook_futures(ts, bp, bq, ap, aq)
                        elif "markPrice" in stream:
                            buffer.add_mark_price(
                                int(d["E"]),
                                float(d["p"]),
                                float(d["i"]),
                                float(d["r"]),
                                int(d["T"]),
                            )
                        elif "forceOrder" in stream:
                            o = d["o"]
                            buffer.add_liquidation(
                                int(o["T"]),
                                o["S"] == "BUY",
                                float(o["q"]),
                            )
                    except (KeyError, ValueError) as e:
                        _log_ws_error("futures", stream, e, list(d.keys()) if isinstance(d, dict) else None)
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[futures ws] reconnecting: {e}")
                await asyncio.sleep(2)


async def ws_spot_stream(buffer, health, stop_event):
    """Connect to spot combined stream."""
    streams = [
        f"{SYMBOL}@aggTrade",
        f"{SYMBOL}@bookTicker",
        f"{SYMBOL}@depth20@100ms",
    ]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ssl=SSL_CONTEXT,
                                          ping_interval=30,
                                          ping_timeout=20) as ws:
                async for msg in ws:
                    if stop_event.is_set():
                        break
                    health.update("binance_spot")
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        d = data.get("data", data)

                        if "aggTrade" in stream:
                            buffer.add_trade_spot(
                                int(d["T"]), float(d["p"]), float(d["q"]), d["m"]
                            )
                        elif "bookTicker" in stream:
                            # Spot bookTicker has no T/E timestamp field (only u,s,b,B,a,A)
                            # Fallback to local arrival time — consistent with training data
                            ts = int(d.get("T", d.get("E", time.time() * 1000)))
                            buffer.add_bookticker_spot(
                                ts, float(d["b"]), float(d["a"])
                            )
                        elif "depth20" in stream:
                            ts = int(d.get("T", d.get("E", time.time() * 1000)))
                            bids = d.get("bids", d.get("b", []))
                            asks = d.get("asks", d.get("a", []))
                            bp = [float(b[0]) for b in bids[:20]]
                            bq = [float(b[1]) for b in bids[:20]]
                            ap = [float(a[0]) for a in asks[:20]]
                            aq = [float(a[1]) for a in asks[:20]]
                            while len(bp) < 20: bp.append(0.0); bq.append(0.0)
                            while len(ap) < 20: ap.append(0.0); aq.append(0.0)
                            buffer.add_orderbook_spot(ts, bp, bq, ap, aq)
                    except (KeyError, ValueError) as e:
                        _log_ws_error("spot", stream, e, list(d.keys()) if isinstance(d, dict) else None)
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[spot ws] reconnecting: {e}")
                await asyncio.sleep(2)


def _coinbase_jwt():
    """Build JWT for Coinbase Advanced Trade WS (CDP API keys).

    Reads from environment variables:
        COINBASE_API_KEY    — e.g. organizations/{org}/apiKeys/{key_id}
        COINBASE_API_SECRET — EC private key in PEM format

    Returns JWT string, or empty string if not configured.
    """
    import os, hashlib, secrets as _secrets
    import jwt as pyjwt
    from cryptography.hazmat.primitives import serialization

    key_name = os.environ.get("COINBASE_API_KEY", "")
    key_secret = os.environ.get("COINBASE_API_SECRET", "")
    if not (key_name and key_secret):
        return ""

    # Handle escaped newlines from .env
    key_secret = key_secret.replace("\\n", "\n")

    try:
        private_key = serialization.load_pem_private_key(
            key_secret.encode("utf-8"), password=None
        )
        payload = {
            "sub": key_name,
            "iss": "coinbase-cloud",
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
        }
        headers = {
            "kid": key_name,
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest(),
        }
        return pyjwt.encode(payload, private_key, algorithm="ES256", headers=headers)
    except Exception as e:
        print(f"[coinbase] JWT generation failed: {e}")
        return ""


async def ws_coinbase_stream(buffer, health, stop_event):
    """Connect to Coinbase Advanced Trade WS for BTC-USD ticker, trades, and L2.

    Uses JWT auth with EC private key (CDP API keys).
    Each subscribe message needs a fresh JWT (expires in 2 min).
    Without keys, subscribes to ticker and market_trades only (no L2).
    """
    has_auth = bool(_coinbase_jwt())
    channels = ["ticker", "market_trades"]
    if has_auth:
        channels.append("level2")
        print("[coinbase] JWT auth OK — subscribing to level2")
    else:
        print("[coinbase] no API keys — skipping level2 (set COINBASE_API_KEY/SECRET)")

    def _make_sub_msg(channel):
        """Build subscribe message with fresh JWT for one channel."""
        msg = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channel": channel,
        }
        token = _coinbase_jwt()
        if token:
            msg["jwt"] = token
        return json.dumps(msg)

    while not stop_event.is_set():
        try:
            async with websockets.connect(COINBASE_WS, ssl=SSL_CONTEXT,
                                          ping_interval=30,
                                          ping_timeout=20,
                                          max_size=2**24) as ws:
                for ch in channels:
                    await ws.send(_make_sub_msg(ch))

                async for msg in ws:
                    if stop_event.is_set():
                        break
                    try:
                        d = json.loads(msg)
                        channel = d.get("channel", "")
                        events = d.get("events", [])

                        for event in events:
                            evt_type = event.get("type", "")

                            if channel == "ticker" and "tickers" in event:
                                for t in event["tickers"]:
                                    ts = int(time.time() * 1000)
                                    bid = float(t.get("best_bid", 0))
                                    ask = float(t.get("best_ask", 0))
                                    if bid > 0 and ask > 0:
                                        buffer.add_coinbase_quote(ts, bid, ask)
                                    price = float(t.get("price", 0))
                                    vol = float(t.get("volume_24_h", 0))
                                    if price > 0:
                                        buffer.add_coinbase_trade(ts, price, 0.0, False)
                                    health.update("coinbase")

                            elif channel == "market_trades" and "trades" in event:
                                for t in event["trades"]:
                                    ts = int(time.time() * 1000)
                                    price = float(t["price"])
                                    qty = float(t["size"])
                                    ibm = t.get("side", "BUY") == "SELL"
                                    buffer.add_coinbase_trade(ts, price, qty, ibm)
                                    health.update("coinbase")

                            elif channel == "l2_data" and evt_type == "snapshot":
                                ts = int(time.time() * 1000)
                                updates = event.get("updates", [])
                                bids = [(u["price_level"], u["new_quantity"])
                                        for u in updates if u.get("side") == "bid"]
                                asks = [(u["price_level"], u["new_quantity"])
                                        for u in updates if u.get("side") == "offer"]
                                buffer.update_coinbase_book(True, bids, asks, ts)
                                health.update("coinbase")

                            elif channel == "l2_data" and evt_type == "update":
                                ts = int(time.time() * 1000)
                                updates = event.get("updates", [])
                                bids = [(u["price_level"], u["new_quantity"])
                                        for u in updates if u.get("side") == "bid"]
                                asks = [(u["price_level"], u["new_quantity"])
                                        for u in updates if u.get("side") == "offer"]
                                buffer.update_coinbase_book(False, bids, asks, ts)
                                health.update("coinbase")

                        if d.get("type") == "error":
                            print(f"\n[coinbase] error: {d.get('message', '?')} — {d.get('reason', '?')}")

                    except (KeyError, ValueError) as e:
                        _log_ws_error("coinbase", channel, e)
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[coinbase ws] reconnecting: {e}")
                await asyncio.sleep(5)


async def ws_bybit_stream(buffer, health, stop_event):
    """Connect to Bybit websocket for BTCUSDT quotes, trades, and orderbook."""
    topics = ["tickers.BTCUSDT", "publicTrade.BTCUSDT", "orderbook.50.BTCUSDT"]

    while not stop_event.is_set():
        try:
            async with websockets.connect(BYBIT_WS, ssl=SSL_CONTEXT,
                                          ping_interval=20,
                                          ping_timeout=10,
                                          max_size=2**22) as ws:
                # Subscribe each topic separately (batch sub fails if any topic is invalid)
                for topic in topics:
                    await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
                last_ping = time.time()
                async for msg in ws:
                    if stop_event.is_set():
                        break
                    # Bybit app-level ping every 20s
                    if time.time() - last_ping > 20:
                        await ws.send('{"op":"ping"}')
                        last_ping = time.time()
                    try:
                        d = json.loads(msg)
                        topic = d.get("topic", "")

                        if topic.startswith("tickers."):
                            data = d.get("data", {})
                            ts = int(d.get("ts", time.time() * 1000))
                            bid = float(data.get("bid1Price", 0))
                            ask = float(data.get("ask1Price", 0))
                            if bid > 0 and ask > 0:
                                buffer.add_bybit_quote(ts, bid, ask)
                                health.update("bybit")

                        elif topic.startswith("publicTrade."):
                            for t in d.get("data", []):
                                ts = int(t.get("T", time.time() * 1000))
                                price = float(t["p"])
                                qty = float(t["v"])
                                # S="Sell" → seller was taker → buyer was maker
                                ibm = t.get("S", "Buy") == "Sell"
                                buffer.add_bybit_trade(ts, price, qty, ibm)
                                health.update("bybit")

                        elif topic.startswith("orderbook."):
                            # Bybit L2: snapshot or delta
                            data = d.get("data", {})
                            is_snap = d.get("type") == "snapshot"
                            bids = data.get("b", [])
                            asks = data.get("a", [])
                            ts = int(d.get("ts", time.time() * 1000))
                            buffer.update_bybit_book(is_snap, bids, asks, ts)
                            health.update("bybit")

                    except (KeyError, ValueError) as e:
                        _log_ws_error("bybit", topic, e)
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[bybit ws] reconnecting: {e}")
                await asyncio.sleep(5)


async def metrics_poller(buffer, session, stop_event):
    """Poll 5-min metrics every 60 seconds."""
    while not stop_event.is_set():
        try:
            # Global long/short
            url = f"{FUTURES_REST}/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=2"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                if data:
                    latest = data[-1]
                    ts = int(latest["timestamp"])
                    ls = float(latest["longShortRatio"])

            # Top trader
            url = f"{FUTURES_REST}/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=5m&limit=2"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                top_ls = float(data[-1]["longShortRatio"]) if data else 0.0

            # Taker
            url = f"{FUTURES_REST}/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=2"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                taker_ls = float(data[-1]["buySellRatio"]) if data else 0.0

            # OI
            url = f"{FUTURES_REST}/fapi/v1/openInterest?symbol=BTCUSDT"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                oi = float(data["openInterest"])

            # Add if new timestamp
            if ts not in buffer.mt_ts:
                buffer.add_metrics(ts, ls, top_ls, taker_ls, oi)

        except Exception as e:
            _log_ws_error("rest", "metrics_poller", e)

        await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# Prediction loop
# ---------------------------------------------------------------------------

async def prediction_loop(buffer, predictor, interval, stop_event,
                          health=None, ws_clients=None, poly_tracker=None):
    """Run prediction every `interval` seconds."""
    await asyncio.sleep(3)

    last_block = 0
    block_count = 0
    last_health_log = 0
    last_no_market_log = 0

    while not stop_event.is_set():
        try:
            now_ms = int(time.time() * 1000)

            # Health check: don't predict if Binance futures is down
            if health and not health.binance_ok():
                age = health.age("binance_futures")
                status, _ = health.status_line(poly_tracker)
                print(f"\r  BINANCE DOWN ({age:.0f}s) — skipping prediction | "
                      f"{status}  ", end="", flush=True)
                await asyncio.sleep(interval)
                continue

            # Periodic health log (every 60s, only if something is down)
            if health and time.time() - last_health_log > 60:
                status, all_ok = health.status_line(poly_tracker)
                if not all_ok:
                    print(f"\n  [health] {status}")
                last_health_log = time.time()

            # Polymarket mode: check block transitions and wait for price_to_beat
            if poly_tracker and poly_tracker.enabled:
                # Also check for new block here to prevent race with poller
                poly_tracker.check_new_block()
                if poly_tracker.price_to_beat is None:
                    if time.time() - last_no_market_log > 5:
                        secs_into = int(time.time()) - poly_tracker.current_open_ts
                        print(f"\r  [polymarket] Waiting for price_to_beat... "
                              f"({secs_into}s into block)  ", end="", flush=True)
                        last_no_market_log = time.time()
                    await asyncio.sleep(interval)
                    continue

            # Polymarket mode: check Chainlink price BEFORE predicting
            poly_mode = poly_tracker and poly_tracker.enabled
            if poly_mode:
                cl_age = poly_tracker.chainlink.age
                if poly_tracker.chainlink.price <= 0 or cl_age > 10:
                    print(f"\r  [polymarket] No price (age={cl_age:.0f}s) — "
                          f"waiting for Chainlink feed  ", end="", flush=True)
                    await asyncio.sleep(interval)
                    continue

            # Trim old data
            buffer.trim(now_ms)

            # Predict (in thread to not block event loop / ws pings)
            open_ref_override = (poly_tracker.price_to_beat
                                 if poly_mode else None)
            current_price_override = (poly_tracker.chainlink.price
                                      if poly_mode else None)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, predictor.predict, buffer, now_ms,
                open_ref_override, current_price_override
            )

            if result is None:
                print(f"\r  Waiting for data...  ", end="", flush=True)
                await asyncio.sleep(interval)
                continue

            # New block?
            if result["block_start_ms"] != last_block:
                # Show close summary of previous block
                if last_block > 0 and not poly_mode:
                    # Non-Polymarket: use Binance index_price
                    if len(predictor.block_results) > 0:
                        prev = predictor.block_results[0]
                        outcome = "UP" if prev['result'] == 1 else "DOWN"
                        print(f"\n  >> CLOSED: {prev['close_ref']:,.2f}  |  "
                              f"open: {prev['open_ref']:,.2f}  |  "
                              f"{prev['return_bps']:+.2f} bps  |  "
                              f"{outcome}")
                # Polymarket close is printed by polymarket_poller when API confirms

                last_block = result["block_start_ms"]
                block_count += 1
                block_time = datetime.fromtimestamp(
                    last_block / 1000, tz=timezone.utc
                ).strftime("%H:%M")
                block_end = datetime.fromtimestamp(
                    (last_block + 300_000) / 1000, tz=timezone.utc
                ).strftime("%H:%M")
                open_label = "Strike (Polymarket)" if poly_mode else "Open"
                print(f"\n{'='*80}")
                print(f"  NEW BLOCK: {block_time}-{block_end} UTC  |  "
                      f"{open_label}: {result['open_ref']:,.2f}  |  "
                      f"Blocks seen: {block_count}")
                print(f"{'='*80}")

            # Format output
            now_str = datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).strftime("%H:%M:%S")

            secs = result["seconds_to_expiry"]
            p_cal = result["p_calibrated"]
            p_bm = result["brownian_prob"]

            # Price and distance for display
            if poly_mode:
                price = poly_tracker.chainlink.price
                strike = poly_tracker.price_to_beat
                dist = (price - strike) / strike * 10_000
            else:
                price = result["price_now"]
                dist = result["dist_to_open_bps"]

            warming = len(predictor.block_results) == 0
            data_incomplete = health and not health.data_complete()
            incomplete_secs = health.time_until_complete() if health else 0

            _, ece_bucket = get_bucket_stats(secs)
            max_buy_up = round(p_cal - ece_bucket, 3)
            max_buy_down = round((1.0 - p_cal) - ece_bucket, 3)

            if p_cal >= 0.5:
                buy_side = f"BUY UP < ${max_buy_up:.3f}"
            else:
                buy_side = f"BUY DN < ${max_buy_down:.3f}"

            incomplete_tag = f" [INCOMPLETE {incomplete_secs:.0f}s]" if data_incomplete else ""

            open_label_short = "vs strike" if poly_mode else "vs open"
            print(f"  {now_str} | {secs:5.0f}s | "
                  f"BTC {price:>10,.2f} | "
                  f"{open_label_short}: {dist:>+7.2f} bps | "
                  f"P(Up): {p_cal:.3f} | "
                  f"{buy_side}{incomplete_tag}")
            if warming:
                print(" *** WARMING UP ***")

            # Broadcast to websocket clients
            if ws_clients:
                ws_data = {
                    "p_up": round(float(p_cal), 4),
                    "max_buy_up": max_buy_up,
                    "max_buy_down": max_buy_down,
                    "seconds_to_expiry": round(secs, 1),
                    "warming_up": warming,
                    "data_incomplete": data_incomplete,
                    "data_complete_in_s": round(incomplete_secs, 0) if data_incomplete else 0,
                    "block_start_ms": result["block_start_ms"],
                    "now_ms": now_ms,
                }
                payload = json.dumps(ws_data)
                dead = set()
                for client in ws_clients:
                    try:
                        await client.send(payload)
                    except Exception:
                        dead.add(client)
                ws_clients -= dead

        except Exception as e:
            print(f"\n  [predict] error: {e}")

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def ws_server_handler(websocket, ws_clients):
    """Handle incoming websocket client connections."""
    ws_clients.add(websocket)
    addr = websocket.remote_address
    print(f"\n  [ws-server] client connected: {addr}")
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)
        print(f"\n  [ws-server] client disconnected: {addr}")


async def main(interval=1, ws_port=8765, use_polymarket=False):
    buffer = LiveBuffer(max_seconds=600)
    predictor = LivePredictor()
    health = WSHealth()
    poly_tracker = PolymarketTracker()
    poly_tracker.enabled = use_polymarket

    # Websocket clients set (shared with prediction_loop)
    ws_clients = set()

    print("=" * 80)
    print("  BTC 5-Min Block Predictor (Live)")
    print("=" * 80)
    print(f"  Model: models/lightgbm_v3.txt")
    print(f"  Features: {len(predictor.feature_cols)}")
    print(f"  Interval: {interval}s")
    print(f"  WS server: ws://localhost:{ws_port}")
    print(f"  Feeds: Binance(futures+spot) + Coinbase + Bybit")
    if use_polymarket:
        print(f"  Polymarket: ENABLED (open_ref from Polymarket strike)")
    print()

    async with aiohttp.ClientSession() as session:
        await warmup(buffer, session)

        stop_event = asyncio.Event()

        # Start websocket server for external clients
        server = await websockets.serve(
            lambda ws: ws_server_handler(ws, ws_clients),
            "0.0.0.0", ws_port,
        )
        print(f"  WS server listening on port {ws_port}\n")

        try:
            await asyncio.gather(
                ws_futures_stream(buffer, health, stop_event),
                ws_spot_stream(buffer, health, stop_event),
                ws_coinbase_stream(buffer, health, stop_event),
                ws_bybit_stream(buffer, health, stop_event),
                metrics_poller(buffer, session, stop_event),
                polymarket_poller(poly_tracker, session, stop_event),
                ws_polymarket_price_stream(poly_tracker, stop_event),
                prediction_loop(buffer, predictor, interval, stop_event,
                                health, ws_clients, poly_tracker),
            )
        except KeyboardInterrupt:
            print("\n\nStopping...")
            stop_event.set()
        finally:
            server.close()
            await server.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live BTC block predictor")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Prediction interval in seconds (default: 1)")
    parser.add_argument("--ws-port", type=int, default=8765,
                        help="WebSocket server port (default: 8765)")
    parser.add_argument("--polymarket", action="store_true",
                        help="Use Polymarket strike as open_ref (skip prediction "
                             "until price_to_beat is available)")
    args = parser.parse_args()

    try:
        asyncio.run(main(interval=args.interval, ws_port=args.ws_port,
                         use_polymarket=args.polymarket))
    except KeyboardInterrupt:
        print("\nDone.")

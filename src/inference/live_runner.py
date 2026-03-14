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

import certifi
import websockets
import aiohttp
import numpy as np

from src.inference.live_buffer import LiveBuffer
from src.inference.live_predictor import LivePredictor


# ---------------------------------------------------------------------------
# Polymarket integration
# ---------------------------------------------------------------------------

POLYMARKET_CRYPTO_API = "https://polymarket.com/api/crypto/crypto-price"
POLYMARKET_LIVE_WS = "wss://ws-live-data.polymarket.com/"


class ChainlinkPriceFeed:
    """Real-time Chainlink BTC/USD price from Polymarket websocket."""

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

    def get_block_open_ts(self, now_sec=None):
        """Get the current block's open timestamp in seconds."""
        if now_sec is None:
            now_sec = int(time.time())
        return (now_sec // self.interval_sec) * self.interval_sec

    def check_new_block(self):
        """Check if we entered a new block. Returns True if block changed."""
        block_ts = self.get_block_open_ts()
        if block_ts != self.current_open_ts:
            self.current_open_ts = block_ts
            self.price_to_beat = None
            self._polling = True
            return True
        return False

    def get_api_url(self):
        """Build the Polymarket crypto-price API URL for current block."""
        open_dt = datetime.fromtimestamp(self.current_open_ts, tz=timezone.utc)
        close_dt = datetime.fromtimestamp(
            self.current_open_ts + self.interval_sec, tz=timezone.utc
        )
        open_iso = open_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        close_iso = close_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return (
            f"{POLYMARKET_CRYPTO_API}?symbol={self.asset}"
            f"&eventStartTime={open_iso}"
            f"&variant={self.variant}"
            f"&endDate={close_iso}"
        )


async def polymarket_poller(tracker, session, stop_event):
    """Poll Polymarket API for price_to_beat when a new block starts."""
    if not tracker.enabled:
        return

    while not stop_event.is_set():
        try:
            # Check for new block
            tracker.check_new_block()

            # Poll if we need the price
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

                # Poll every 500ms until found
                await asyncio.sleep(0.5)
            else:
                # No need to poll — check every 1s for new blocks
                await asyncio.sleep(1)

        except Exception as e:
            print(f"\n  [polymarket] error: {e}")
            await asyncio.sleep(2)


async def ws_chainlink_stream(tracker, stop_event):
    """Connect to Polymarket live-data WS for Chainlink BTC/USD price."""
    if not tracker.enabled:
        return

    sub_msg = json.dumps({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "filters": json.dumps({"symbol": "btc/usd"}),
        }],
    })

    while not stop_event.is_set():
        try:
            async with websockets.connect(POLYMARKET_LIVE_WS, ssl=SSL_CONTEXT,
                                          ping_interval=30,
                                          ping_timeout=20) as ws:
                await ws.send(sub_msg)
                print("  [chainlink] Connected to Polymarket price feed")
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

                        # Format 1: initial snapshot with data array
                        if isinstance(payload, dict) and "data" in payload:
                            data = payload["data"]
                            if isinstance(data, list) and len(data) > 0:
                                last = data[-1]
                                tracker.chainlink.price = float(last["value"])
                                tracker.chainlink.received_at = time.time()
                                print(f"  [chainlink] Initial: "
                                      f"${tracker.chainlink.price:,.2f}")

                        # Format 2: individual update with value
                        elif isinstance(payload, dict) and "value" in payload:
                            tracker.chainlink.price = float(payload["value"])
                            tracker.chainlink.received_at = time.time()

                    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                        pass
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n  [chainlink] reconnecting: {e}")
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
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"

# Accuracy and ECE per 5-second bucket (from walk-forward 8 folds)
# Key: (lo, hi) in seconds_to_expiry → (accuracy, ece)
BUCKET_STATS = {}
for _lo in range(0, 300, 5):
    BUCKET_STATS[(_lo, _lo + 5)] = (0.50, 0.02)  # default
_WF_DATA = [
    (5,10,0.9749,0.0025),(10,15,0.9605,0.0025),(15,20,0.9479,0.0030),
    (20,25,0.9370,0.0035),(25,30,0.9291,0.0032),(30,35,0.9188,0.0039),
    (35,40,0.9114,0.0042),(40,45,0.9044,0.0037),(45,50,0.8976,0.0039),
    (50,55,0.8898,0.0040),(55,60,0.8849,0.0038),(60,65,0.8735,0.0038),
    (65,70,0.8692,0.0044),(70,75,0.8639,0.0041),(75,80,0.8561,0.0043),
    (80,85,0.8516,0.0046),(85,90,0.8457,0.0043),(90,95,0.8359,0.0043),
    (95,100,0.8316,0.0048),(100,105,0.8255,0.0047),(105,110,0.8195,0.0045),
    (110,115,0.8117,0.0047),(115,120,0.8066,0.0052),(120,125,0.7992,0.0073),
    (125,130,0.7927,0.0080),(130,135,0.7890,0.0082),(135,140,0.7827,0.0087),
    (140,145,0.7748,0.0079),(145,150,0.7691,0.0081),(150,155,0.7629,0.0081),
    (155,160,0.7599,0.0082),(160,165,0.7545,0.0083),(165,170,0.7497,0.0079),
    (170,175,0.7425,0.0079),(175,180,0.7367,0.0083),(180,185,0.7278,0.0076),
    (185,190,0.7268,0.0078),(190,195,0.7213,0.0082),(195,200,0.7143,0.0088),
    (200,205,0.7110,0.0086),(205,210,0.7029,0.0092),(210,215,0.6952,0.0096),
    (215,220,0.6899,0.0095),(220,225,0.6809,0.0094),(225,230,0.6730,0.0092),
    (230,235,0.6633,0.0099),(235,240,0.6570,0.0100),(240,245,0.6512,0.0125),
    (245,250,0.6463,0.0133),(250,255,0.6385,0.0139),(255,260,0.6341,0.0144),
    (260,265,0.6273,0.0146),(265,270,0.6244,0.0142),(270,275,0.6171,0.0133),
    (275,280,0.6071,0.0129),(280,285,0.5969,0.0125),(285,290,0.5846,0.0120),
    (290,295,0.5801,0.0121),(295,300,0.5630,0.0110),
]
for _lo, _hi, _acc, _ece in _WF_DATA:
    BUCKET_STATS[(_lo, _hi)] = (_acc, _ece)


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

    def status_line(self):
        parts = []
        for src in ["binance_futures", "binance_spot", "coinbase", "bybit"]:
            age = self.age(src)
            if age == float('inf'):
                parts.append(f"{src}:OFF")
            elif age > 10:
                parts.append(f"{src}:STALE({age:.0f}s)")
            else:
                parts.append(f"{src}:OK")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# REST Warmup
# ---------------------------------------------------------------------------

async def warmup(buffer, session):
    """Fetch recent historical data to fill buffers before going live."""
    now_ms = int(time.time() * 1000)
    print("Warming up...")

    # 1. Klines 1m futures (last 60 min for ref_price)
    try:
        url = f"{FUTURES_REST}/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=60"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for k in data:
                ts = int(k[0])
                close = float(k[4])
                buffer.add_mark_price(ts, close, close, 0.0, 0)
            print(f"  klines 1m: {len(data)} bars")
    except Exception as e:
        print(f"  klines 1m: ERROR {e}")

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
                    except (KeyError, ValueError):
                        pass
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
                            ts = int(d.get("T", d.get("E", time.time() * 1000)))
                            buffer.add_bookticker_spot(
                                ts, float(d["b"]), float(d["a"])
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
                            buffer.add_orderbook_spot(ts, bp, bq, ap, aq)
                    except (KeyError, ValueError):
                        pass
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[spot ws] reconnecting: {e}")
                await asyncio.sleep(2)


async def ws_coinbase_stream(buffer, health, stop_event):
    """Connect to Coinbase websocket for BTC-USD quotes, trades, and L2 book."""
    sub_msg = json.dumps({
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["ticker", "matches", "level2"],
    })

    while not stop_event.is_set():
        try:
            async with websockets.connect(COINBASE_WS, ssl=SSL_CONTEXT,
                                          ping_interval=30,
                                          ping_timeout=20,
                                          max_size=2**22) as ws:
                await ws.send(sub_msg)
                async for msg in ws:
                    if stop_event.is_set():
                        break
                    try:
                        d = json.loads(msg)
                        msg_type = d.get("type", "")

                        if msg_type == "ticker":
                            ts = int(time.time() * 1000)
                            bid = float(d["best_bid"])
                            ask = float(d["best_ask"])
                            buffer.add_coinbase_quote(ts, bid, ask)
                            # Also record as trade from ticker price/size
                            price = float(d.get("price", 0))
                            size = float(d.get("last_size", 0))
                            if price > 0 and size > 0:
                                ibm = d.get("side", "buy") == "sell"
                                buffer.add_coinbase_trade(ts, price, size, ibm)
                            health.update("coinbase")

                        elif msg_type in ("match", "last_match"):
                            ts = int(time.time() * 1000)
                            price = float(d["price"])
                            qty = float(d["size"])
                            # side = taker side: "sell" → buyer was maker
                            ibm = d.get("side", "buy") == "sell"
                            buffer.add_coinbase_trade(ts, price, qty, ibm)
                            health.update("coinbase")

                        elif msg_type == "snapshot":
                            # L2 book snapshot (initial)
                            ts = int(time.time() * 1000)
                            bids = d.get("bids", [])
                            asks = d.get("asks", [])
                            buffer.update_coinbase_book(True, bids, asks, ts)
                            health.update("coinbase")

                        elif msg_type == "l2update":
                            # L2 incremental update
                            ts = int(time.time() * 1000)
                            changes = d.get("changes", [])
                            bids = [(c[1], c[2]) for c in changes if c[0] == "buy"]
                            asks = [(c[1], c[2]) for c in changes if c[0] == "sell"]
                            buffer.update_coinbase_book(False, bids, asks, ts)
                            health.update("coinbase")

                    except (KeyError, ValueError):
                        pass
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

                    except (KeyError, ValueError):
                        pass
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

        except Exception:
            pass

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
                print(f"\r  BINANCE DOWN ({age:.0f}s) — skipping prediction | "
                      f"{health.status_line()}  ", end="", flush=True)
                await asyncio.sleep(interval)
                continue

            # Periodic health log (every 60s)
            if health and time.time() - last_health_log > 60:
                print(f"\n  [health] {health.status_line()}")
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

            # Trim old data
            buffer.trim(now_ms)

            # Predict (in thread to not block event loop / ws pings)
            open_ref_override = (poly_tracker.price_to_beat
                                 if poly_tracker and poly_tracker.enabled
                                 else None)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, predictor.predict, buffer, now_ms, open_ref_override
            )

            if result is None:
                print(f"\r  Waiting for data...  ", end="", flush=True)
                await asyncio.sleep(interval)
                continue

            # New block?
            if result["block_start_ms"] != last_block:
                last_block = result["block_start_ms"]
                block_count += 1
                block_time = datetime.fromtimestamp(
                    last_block / 1000, tz=timezone.utc
                ).strftime("%H:%M")
                block_end = datetime.fromtimestamp(
                    (last_block + 300_000) / 1000, tz=timezone.utc
                ).strftime("%H:%M")
                open_label = "Open"
                if poly_tracker and poly_tracker.enabled:
                    open_label = "Strike (Polymarket)"
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

            # In polymarket mode, show Chainlink price and dist vs strike
            poly_mode = poly_tracker and poly_tracker.enabled
            if poly_mode and poly_tracker.chainlink.price > 0 and poly_tracker.chainlink.age < 30:
                price = poly_tracker.chainlink.price
                strike = poly_tracker.price_to_beat
                dist = (price - strike) / strike * 10_000
                price_src = "CL"
            else:
                price = result["price_now"]
                dist = result["dist_to_open_bps"]
                price_src = "BN"

            if p_cal >= 0.65:
                signal = "UP  ^^^"
            elif p_cal >= 0.55:
                signal = "UP  ^  "
            elif p_cal <= 0.35:
                signal = "DOWN vvv"
            elif p_cal <= 0.45:
                signal = "DOWN v  "
            else:
                signal = "FLAT ~  "

            warming = len(predictor.block_results) == 0
            data_incomplete = health and not health.data_complete()
            incomplete_secs = health.time_until_complete() if health else 0

            acc_bucket, ece_bucket = get_bucket_stats(secs)
            max_buy_up = round(p_cal - ece_bucket, 3)
            max_buy_down = round((1.0 - p_cal) - ece_bucket, 3)

            if p_cal >= 0.5:
                buy_side = f"BUY UP < ${max_buy_up:.3f}"
            else:
                buy_side = f"BUY DN < ${max_buy_down:.3f}"

            incomplete_tag = f" [INCOMPLETE {incomplete_secs:.0f}s]" if data_incomplete else ""

            price_label = f"BTC({price_src})" if poly_mode else "BTC"
            open_label_short = "vs strike" if poly_mode else "vs open"
            print(f"  {now_str} | {secs:5.0f}s | "
                  f"{price_label} {price:>10,.2f} | "
                  f"{open_label_short}: {dist:>+7.2f} bps | "
                  f"P(Up): {p_cal:.3f} | "
                  f"acc: {acc_bucket:.1%} | "
                  f"{buy_side} | "
                  f"{signal}{incomplete_tag}")
            if warming:
                print(" *** WARMING UP ***")

            # Broadcast to websocket clients
            if ws_clients:
                ws_data = {
                    "block_start_ms": result["block_start_ms"],
                    "now_ms": now_ms,
                    "seconds_to_expiry": round(secs, 1),
                    "open_ref": float(result["open_ref"]),
                    "price_now": float(price),
                    "dist_to_open_bps": round(float(dist), 2),
                    "p_up": round(float(p_cal), 4),
                    "p_down": round(1.0 - float(p_cal), 4),
                    "brownian_prob": round(float(p_bm), 4),
                    "edge_vs_brownian": round(float(p_cal - p_bm), 4),
                    "accuracy_at_bucket": round(acc_bucket, 4),
                    "ece_at_bucket": round(ece_bucket, 4),
                    "max_buy_up": max_buy_up,
                    "max_buy_down": max_buy_down,
                    "direction": result["direction"],
                    "warming_up": warming,
                    "data_incomplete": data_incomplete,
                    "data_complete_in_s": round(incomplete_secs, 0) if data_incomplete else 0,
                }
                if poly_tracker and poly_tracker.enabled:
                    ws_data["polymarket_mode"] = True
                    ws_data["price_to_beat"] = poly_tracker.price_to_beat
                    if poly_tracker.chainlink.price > 0:
                        ws_data["chainlink_price"] = poly_tracker.chainlink.price
                        ws_data["chainlink_age_ms"] = round(poly_tracker.chainlink.age * 1000)
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
                ws_chainlink_stream(poly_tracker, stop_event),
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

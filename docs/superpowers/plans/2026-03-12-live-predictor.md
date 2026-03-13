# Live Predictor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a script that connects to Binance websockets, computes features every second, runs the LightGBM model, and prints P(Up) live to console.

**Architecture:** Single async Python script with three layers: (1) LiveBuffer — accumulates websocket data into DayData-like numpy arrays, (2) LivePredictor — wraps model loading, feature computation and prediction, (3) Main loop — connects websockets, warmup via REST, then predict every second. Reuses `compute_features_v3()` directly — no rewrite of feature engine.

**Tech Stack:** asyncio, websockets, aiohttp, numpy, lightgbm, scipy, existing feature_engine_v3

---

## File Structure

```
src/inference/
├── __init__.py
├── live_buffer.py      # LiveBuffer class: accumulates WS data into arrays
├── live_predictor.py   # LivePredictor: model + calibrators + predict()
└── live_runner.py      # Main script: WS connections, REST warmup, predict loop
```

**Why 3 files:**
- `live_buffer.py` — data accumulation logic, independent of model
- `live_predictor.py` — model inference logic, independent of data source
- `live_runner.py` — glues everything together, handles async I/O

---

## Chunk 1: LiveBuffer (data accumulation)

### Task 1: Create LiveBuffer class

**Files:**
- Create: `src/inference/__init__.py`
- Create: `src/inference/live_buffer.py`

The LiveBuffer accumulates websocket messages into numpy arrays that mimic the DayData structure from feature_engine_v3. It keeps a rolling window of data (last 10 minutes) and exposes a `to_day_data()` method that returns a DayData-compatible object.

- [ ] **Step 1: Create empty init file**

Create `src/inference/__init__.py` as empty file.

- [ ] **Step 2: Write LiveBuffer**

Create `src/inference/live_buffer.py` with:

```python
"""
LiveBuffer: accumulates Binance websocket data into numpy arrays
compatible with feature_engine_v3's DayData structure.

Keeps a rolling window of data. Exposes to_day_data() for feature computation.
"""

import numpy as np
import time
from src.features.feature_engine_v3 import DayData, _precompute_book


class LiveBuffer:
    """Accumulates live market data into arrays for feature computation."""

    def __init__(self, max_seconds=600):
        """
        Args:
            max_seconds: how many seconds of data to keep (default 10 min)
        """
        self.max_seconds = max_seconds
        self._max_rows = max_seconds * 100  # generous upper bound per stream

        # --- Trades futures ---
        self.tf_ts = []
        self.tf_price = []
        self.tf_qty = []
        self.tf_ibm = []

        # --- Trades spot ---
        self.ts_ts = []
        self.ts_price = []
        self.ts_qty = []
        self.ts_ibm = []

        # --- Bookticker futures ---
        self.bf_ts = []
        self.bf_bid = []
        self.bf_ask = []

        # --- Bookticker spot ---
        self.bs_ts = []
        self.bs_bid = []
        self.bs_ask = []

        # --- Orderbook futures (latest snapshot) ---
        self.ob_fut_ts = []
        self.ob_fut_bid_prices = []  # list of 20-element arrays
        self.ob_fut_bid_qtys = []
        self.ob_fut_ask_prices = []
        self.ob_fut_ask_qtys = []

        # --- Orderbook spot (latest snapshot) ---
        self.ob_spot_ts = []
        self.ob_spot_bid_prices = []
        self.ob_spot_bid_qtys = []
        self.ob_spot_ask_prices = []
        self.ob_spot_ask_qtys = []

        # --- Mark price ---
        self.mp_ts = []
        self.mp_mark = []
        self.mp_index = []
        self.mp_funding = []
        self.mp_next_ms = []

        # --- Liquidations ---
        self.lq_ts = []
        self.lq_is_buy = []
        self.lq_qty = []

        # --- Metrics (5-min bars) ---
        self.mt_ts = []
        self.mt_ls_ratio = []
        self.mt_top_ls = []
        self.mt_taker_ls = []
        self.mt_oi = []

    # --- Add methods for each stream ---

    def add_trade_futures(self, timestamp_ms, price, qty, is_buyer_maker):
        self.tf_ts.append(timestamp_ms)
        self.tf_price.append(price)
        self.tf_qty.append(qty)
        self.tf_ibm.append(is_buyer_maker)

    def add_trade_spot(self, timestamp_ms, price, qty, is_buyer_maker):
        self.ts_ts.append(timestamp_ms)
        self.ts_price.append(price)
        self.ts_qty.append(qty)
        self.ts_ibm.append(is_buyer_maker)

    def add_bookticker_futures(self, timestamp_ms, bid, ask):
        self.bf_ts.append(timestamp_ms)
        self.bf_bid.append(bid)
        self.bf_ask.append(ask)

    def add_bookticker_spot(self, timestamp_ms, bid, ask):
        self.bs_ts.append(timestamp_ms)
        self.bs_bid.append(bid)
        self.bs_ask.append(ask)

    def add_orderbook_futures(self, timestamp_ms, bid_prices, bid_qtys,
                               ask_prices, ask_qtys):
        self.ob_fut_ts.append(timestamp_ms)
        self.ob_fut_bid_prices.append(bid_prices)
        self.ob_fut_bid_qtys.append(bid_qtys)
        self.ob_fut_ask_prices.append(ask_prices)
        self.ob_fut_ask_qtys.append(ask_qtys)

    def add_orderbook_spot(self, timestamp_ms, bid_prices, bid_qtys,
                            ask_prices, ask_qtys):
        self.ob_spot_ts.append(timestamp_ms)
        self.ob_spot_bid_prices.append(bid_prices)
        self.ob_spot_bid_qtys.append(bid_qtys)
        self.ob_spot_ask_prices.append(ask_prices)
        self.ob_spot_ask_qtys.append(ask_qtys)

    def add_mark_price(self, timestamp_ms, mark, index, funding, next_funding_ms):
        self.mp_ts.append(timestamp_ms)
        self.mp_mark.append(mark)
        self.mp_index.append(index)
        self.mp_funding.append(funding)
        self.mp_next_ms.append(next_funding_ms)

    def add_liquidation(self, timestamp_ms, is_buy, qty):
        self.lq_ts.append(timestamp_ms)
        self.lq_is_buy.append(is_buy)
        self.lq_qty.append(qty)

    def add_metrics(self, timestamp_ms, ls_ratio, top_ls, taker_ls, oi):
        self.mt_ts.append(timestamp_ms)
        self.mt_ls_ratio.append(ls_ratio)
        self.mt_top_ls.append(top_ls)
        self.mt_taker_ls.append(taker_ls)
        self.mt_oi.append(oi)

    # --- Trim old data ---

    def trim(self, now_ms=None):
        """Remove data older than max_seconds."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - self.max_seconds * 1000

        def _trim_list(ts_list, *other_lists):
            if not ts_list:
                return
            # Find first index >= cutoff
            idx = 0
            for i, t in enumerate(ts_list):
                if t >= cutoff:
                    idx = i
                    break
            else:
                idx = len(ts_list)  # all old
            if idx > 0:
                del ts_list[:idx]
                for lst in other_lists:
                    del lst[:idx]

        _trim_list(self.tf_ts, self.tf_price, self.tf_qty, self.tf_ibm)
        _trim_list(self.ts_ts, self.ts_price, self.ts_qty, self.ts_ibm)
        _trim_list(self.bf_ts, self.bf_bid, self.bf_ask)
        _trim_list(self.bs_ts, self.bs_bid, self.bs_ask)
        _trim_list(self.ob_fut_ts, self.ob_fut_bid_prices, self.ob_fut_bid_qtys,
                   self.ob_fut_ask_prices, self.ob_fut_ask_qtys)
        _trim_list(self.ob_spot_ts, self.ob_spot_bid_prices, self.ob_spot_bid_qtys,
                   self.ob_spot_ask_prices, self.ob_spot_ask_qtys)
        _trim_list(self.mp_ts, self.mp_mark, self.mp_index, self.mp_funding,
                   self.mp_next_ms)
        _trim_list(self.lq_ts, self.lq_is_buy, self.lq_qty)
        # Don't trim metrics — we need lagged bars

    # --- Convert to DayData ---

    def to_day_data(self):
        """Convert current buffer to DayData-compatible object."""
        day = DayData()

        # Trades futures
        day.tf_ts = np.array(self.tf_ts, dtype=np.int64)
        day.tf_price = np.array(self.tf_price, dtype=np.float64)
        day.tf_qty = np.array(self.tf_qty, dtype=np.float64)
        day.tf_ibm = np.array(self.tf_ibm, dtype=bool)

        # Trades spot
        day.ts_ts = np.array(self.ts_ts, dtype=np.int64)
        day.ts_price = np.array(self.ts_price, dtype=np.float64)
        day.ts_qty = np.array(self.ts_qty, dtype=np.float64)
        day.ts_ibm = np.array(self.ts_ibm, dtype=bool)

        # Bookticker futures
        day.bf_ts = np.array(self.bf_ts, dtype=np.int64)
        day.bf_bid = np.array(self.bf_bid, dtype=np.float64)
        day.bf_ask = np.array(self.bf_ask, dtype=np.float64)
        day.bf_mid = (day.bf_bid + day.bf_ask) / 2.0

        # Bookticker spot
        day.bs_ts = np.array(self.bs_ts, dtype=np.int64)
        day.bs_bid = np.array(self.bs_bid, dtype=np.float64)
        day.bs_ask = np.array(self.bs_ask, dtype=np.float64)
        day.bs_mid = (day.bs_bid + day.bs_ask) / 2.0

        # Orderbook futures (pre-computed)
        if self.ob_fut_ts:
            ts = np.array(self.ob_fut_ts, dtype=np.int64)
            bp = np.array(self.ob_fut_bid_prices)
            bq = np.array(self.ob_fut_bid_qtys)
            ap = np.array(self.ob_fut_ask_prices)
            aq = np.array(self.ob_fut_ask_qtys)
            day.ob_fut = _precompute_book(ts, bp, bq, ap, aq)
        else:
            day.ob_fut = {'ts': np.array([], dtype=np.int64),
                          'mid': np.array([]), 'spread_bps': np.array([]),
                          'imb_L1': np.array([]), 'imb_L5': np.array([])}

        # Orderbook spot (pre-computed)
        if self.ob_spot_ts:
            ts = np.array(self.ob_spot_ts, dtype=np.int64)
            bp = np.array(self.ob_spot_bid_prices)
            bq = np.array(self.ob_spot_bid_qtys)
            ap = np.array(self.ob_spot_ask_prices)
            aq = np.array(self.ob_spot_ask_qtys)
            day.ob_spot = _precompute_book(ts, bp, bq, ap, aq)
        else:
            day.ob_spot = {'ts': np.array([], dtype=np.int64),
                           'mid': np.array([]), 'spread_bps': np.array([]),
                           'imb_L1': np.array([]), 'imb_L5': np.array([])}

        # Mark price
        day.mp_ts = np.array(self.mp_ts, dtype=np.int64)
        day.mp_mark = np.array(self.mp_mark, dtype=np.float64)
        day.mp_index = np.array(self.mp_index, dtype=np.float64)
        day.mp_funding = np.array(self.mp_funding, dtype=np.float64)
        day.mp_next_ms = np.array(self.mp_next_ms, dtype=np.float64)

        # Liquidations
        day.lq_ts = np.array(self.lq_ts, dtype=np.int64)
        day.lq_is_buy = np.array(self.lq_is_buy, dtype=bool)
        day.lq_qty = np.array(self.lq_qty, dtype=np.float64)

        # Metrics
        day.mt_ts = np.array(self.mt_ts, dtype=np.int64)
        day.mt_ls_ratio = np.array(self.mt_ls_ratio, dtype=np.float64)
        day.mt_top_ls = np.array(self.mt_top_ls, dtype=np.float64)
        day.mt_taker_ls = np.array(self.mt_taker_ls, dtype=np.float64)
        day.mt_oi = np.array(self.mt_oi, dtype=np.float64)

        return day

    def stats(self):
        """Return dict of buffer sizes for debugging."""
        return {
            "trades_fut": len(self.tf_ts),
            "trades_spot": len(self.ts_ts),
            "bookticker_fut": len(self.bf_ts),
            "bookticker_spot": len(self.bs_ts),
            "depth_fut": len(self.ob_fut_ts),
            "depth_spot": len(self.ob_spot_ts),
            "mark_price": len(self.mp_ts),
            "liquidations": len(self.lq_ts),
            "metrics": len(self.mt_ts),
        }
```

- [ ] **Step 3: Commit**

```bash
git add src/inference/__init__.py src/inference/live_buffer.py
git commit -m "feat: add LiveBuffer for real-time data accumulation"
```

---

## Chunk 2: LivePredictor (model inference)

### Task 2: Create LivePredictor class

**Files:**
- Create: `src/inference/live_predictor.py`

Wraps model loading, ref_price building, feature computation, prediction, and calibration into a single `predict()` call.

- [ ] **Step 1: Write LivePredictor**

Create `src/inference/live_predictor.py` with:

```python
"""
LivePredictor: loads model + calibrators, computes features from LiveBuffer,
returns calibrated P(Up) prediction.
"""

import pickle
import numpy as np
import lightgbm as lgb
from pathlib import Path

from src.features.feature_engine_v3 import (
    FEATURE_COLUMNS_V3,
    build_ref_price,
    compute_features_v3,
)


# Calibration time buckets (must match train_model_v3.py)
CALIB_BUCKETS = [
    (240, 300, "240-300"),
    (180, 240, "180-240"),
    (120, 180, "120-180"),
    (60, 120, "60-120"),
    (30, 60, "30-60"),
    (0, 30, "0-30"),
]


class LivePredictor:
    """Loads model and produces calibrated predictions."""

    def __init__(self, model_dir="models"):
        model_dir = Path(model_dir)
        self.model = lgb.Booster(model_file=str(model_dir / "lightgbm_v3.txt"))

        with open(model_dir / "calibrators_v3.pkl", "rb") as f:
            self.calibrators = pickle.load(f)

        self.feature_cols = FEATURE_COLUMNS_V3

        # Block tracking
        self.current_block_start_ms = 0
        self.open_ref = 0.0
        self.open_ref_age_ms = 0
        self.block_results = []  # list of {'return_bps': float, 'result': int}

    def _get_block_start(self, now_ms):
        """Get the deterministic block start for a given timestamp."""
        return (now_ms // 300_000) * 300_000

    def _get_calib_bucket(self, seconds_to_expiry):
        """Get calibrator key for given seconds_to_expiry."""
        for lo, hi, key in CALIB_BUCKETS:
            if lo <= seconds_to_expiry < hi:
                return key
        return "0-30"

    def update_block(self, now_ms, ref):
        """
        Check if we're in a new block. If so, close the old one and
        set open_ref for the new one.

        Returns True if block changed.
        """
        block_start = self._get_block_start(now_ms)

        if block_start == self.current_block_start_ms:
            return False

        # Close old block (if we had one)
        if self.current_block_start_ms > 0 and len(ref['ts']) > 0:
            old_end = self.current_block_start_ms + 300_000
            # Find close_ref (last ref_price <= block_end)
            idx_close = np.searchsorted(ref['ts'], old_end, side='right') - 1
            if idx_close >= 0:
                close_ref = ref['price'][idx_close]
                return_bps = (close_ref - self.open_ref) / self.open_ref * 10_000
                result = 1 if close_ref >= self.open_ref else 0
                self.block_results.insert(0, {
                    'return_bps': float(return_bps),
                    'result': result,
                })
                # Keep only last 6
                self.block_results = self.block_results[:6]

        # Set new block
        self.current_block_start_ms = block_start

        # Find open_ref: last ref_price <= block_start
        if len(ref['ts']) > 0:
            idx = np.searchsorted(ref['ts'], block_start, side='right') - 1
            if idx >= 0:
                self.open_ref = ref['price'][idx]
                self.open_ref_age_ms = block_start - int(ref['ts'][idx])
            else:
                self.open_ref = ref['price'][0]
                self.open_ref_age_ms = 999999
        else:
            self.open_ref = 0.0
            self.open_ref_age_ms = 999999

        return True

    def predict(self, buffer, now_ms=None):
        """
        Compute features and predict P(Up) for the current moment.

        Args:
            buffer: LiveBuffer instance
            now_ms: current timestamp in ms (default: now)

        Returns:
            dict with prediction info, or None if not enough data
        """
        import time as _time
        if now_ms is None:
            now_ms = int(_time.time() * 1000)

        # Build DayData from buffer
        day = buffer.to_day_data()

        # Build ref_price
        ref = build_ref_price(day)

        if len(ref['ts']) == 0:
            return None

        # Update block (sets open_ref, closes previous)
        self.update_block(now_ms, ref)

        if self.open_ref <= 0:
            return None

        block_start = self.current_block_start_ms
        block_end = block_start + 300_000
        seconds_to_expiry = max(0, (block_end - now_ms) / 1000.0)

        # Compute features
        feats = compute_features_v3(
            day, ref, now_ms, block_start, self.open_ref,
            open_ref_age_ms=self.open_ref_age_ms,
            block_results=self.block_results,
        )

        # Build feature vector
        X = np.array([[feats.get(col, 0.0) for col in self.feature_cols]])

        # Predict
        p_raw = float(self.model.predict(X)[0])

        # Calibrate
        bucket_key = self._get_calib_bucket(seconds_to_expiry)
        if bucket_key in self.calibrators:
            p_cal = float(self.calibrators[bucket_key].predict([p_raw])[0])
            p_cal = np.clip(p_cal, 0.01, 0.99)
        else:
            p_cal = p_raw

        # Current ref_price
        idx_now = np.searchsorted(ref['ts'], now_ms, side='right') - 1
        price_now = ref['price'][idx_now] if idx_now >= 0 else self.open_ref

        return {
            "block_start_ms": block_start,
            "now_ms": now_ms,
            "seconds_to_expiry": seconds_to_expiry,
            "open_ref": self.open_ref,
            "price_now": price_now,
            "dist_to_open_bps": feats.get("dist_to_open_bps", 0.0),
            "p_raw": p_raw,
            "p_calibrated": p_cal,
            "confidence": abs(p_cal - 0.5),
            "direction": "UP" if p_cal >= 0.5 else "DOWN",
            "brownian_prob": feats.get("brownian_prob", 0.5),
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/inference/live_predictor.py
git commit -m "feat: add LivePredictor with model loading and calibrated prediction"
```

---

## Chunk 3: LiveRunner (main script)

### Task 3: Create main runner with REST warmup + WebSocket streams + prediction loop

**Files:**
- Create: `src/inference/live_runner.py`

This is the main script the user runs. It:
1. Loads the model
2. Warms up via REST API (trades, klines, metrics)
3. Connects to 8 Binance websocket streams
4. Every second, computes features and prints prediction

- [ ] **Step 1: Write live_runner.py**

Create `src/inference/live_runner.py` with:

```python
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
# Config
# ---------------------------------------------------------------------------

SYMBOL = "btcusdt"
SPOT_WS = "wss://stream.binance.com:9443/ws"
FUTURES_WS = "wss://fstream.binance.com/ws"
FUTURES_REST = "https://fapi.binance.com"
SPOT_REST = "https://api.binance.com"

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


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
                # Use close price as mark/index approximation
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
    for endpoint, name, parser in [
        ("futures/data/globalLongShortAccountRatio", "global_ls",
         lambda d: float(d["longShortRatio"])),
        ("futures/data/topLongShortPositionRatio", "top_ls",
         lambda d: float(d["longShortRatio"])),
        ("futures/data/takerlongshortRatio", "taker_ls",
         lambda d: float(d["buySellRatio"])),
    ]:
        try:
            url = f"{FUTURES_REST}/{endpoint}?symbol=BTCUSDT&period=5m&limit=10"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                for item in data:
                    ts = int(item["timestamp"])
                    val = parser(item)
                    if name == "global_ls":
                        # First call sets all metrics; subsequent calls update
                        existing = [i for i, t in enumerate(buffer.mt_ts) if t == ts]
                        if not existing:
                            buffer.add_metrics(ts, val, 0.0, 0.0, np.nan)
                        else:
                            buffer.mt_ls_ratio[existing[0]] = val
                    elif name == "top_ls":
                        existing = [i for i, t in enumerate(buffer.mt_ts) if t == ts]
                        if existing:
                            buffer.mt_top_ls[existing[0]] = val
                    elif name == "taker_ls":
                        existing = [i for i, t in enumerate(buffer.mt_ts) if t == ts]
                        if existing:
                            buffer.mt_taker_ls[existing[0]] = val
                print(f"  {name}: {len(data)} bars")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    # 4. Open Interest
    try:
        url = f"{FUTURES_REST}/fapi/v1/openInterest?symbol=BTCUSDT"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            oi = float(data["openInterest"])
            # Update latest metrics row with OI
            if buffer.mt_ts:
                buffer.mt_oi[-1] = oi
            print(f"  openInterest: {oi:.2f}")
    except Exception as e:
        print(f"  openInterest: ERROR {e}")

    # 5. Recent trades futures (last 60s, paginated)
    try:
        start_ms = now_ms - 60_000
        url = (f"{FUTURES_REST}/fapi/v1/aggTrades?symbol=BTCUSDT"
               f"&startTime={start_ms}&endTime={now_ms}&limit=1000")
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for t in data:
                buffer.add_trade_futures(
                    int(t["T"]), float(t["p"]), float(t["q"]), bool(t["m"])
                )
            print(f"  trades_futures: {len(data)} trades (last 60s)")
    except Exception as e:
        print(f"  trades_futures: ERROR {e}")

    # 6. Recent trades spot (last 60s, paginated)
    try:
        start_ms = now_ms - 60_000
        url = (f"{SPOT_REST}/api/v3/aggTrades?symbol=BTCUSDT"
               f"&startTime={start_ms}&endTime={now_ms}&limit=1000")
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            for t in data:
                buffer.add_trade_spot(
                    int(t["T"]), float(t["p"]), float(t["q"]), bool(t["m"])
                )
            print(f"  trades_spot: {len(data)} trades (last 60s)")
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
            # Pad to 20 levels if needed
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
            ts = int(time.time() * 1000)  # spot depth has no timestamp
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

    # 9. Current bookticker
    try:
        url = f"{FUTURES_REST}/fapi/v1/ticker/bookTicker?symbol=BTCUSDT"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            buffer.add_bookticker_futures(
                int(data["time"]), float(data["bidPrice"]), float(data["askPrice"])
            )
    except Exception:
        pass

    try:
        url = f"{SPOT_REST}/api/v3/ticker/bookTicker?symbol=BTCUSDT"
        async with session.get(url, ssl=SSL_CONTEXT) as r:
            data = await r.json()
            buffer.add_bookticker_spot(
                int(time.time() * 1000), float(data["bidPrice"]), float(data["askPrice"])
            )
    except Exception:
        pass

    stats = buffer.stats()
    print(f"\nWarmup done. Buffer: {stats}")


# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------

async def ws_futures_stream(buffer, stop_event):
    """Connect to futures combined stream."""
    streams = [
        f"{SYMBOL}@aggTrade",
        f"{SYMBOL}@bookTicker",
        f"{SYMBOL}@depth20@100ms",
        f"{SYMBOL}@markPrice@1s",
        f"{SYMBOL}@forceOrder",
    ]
    url = f"{FUTURES_WS}/{'/'.join(streams)}"

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ssl=SSL_CONTEXT,
                                          ping_interval=20,
                                          ping_timeout=10) as ws:
                async for msg in ws:
                    if stop_event.is_set():
                        break
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
                                float(d["p"]),  # mark price
                                float(d["i"]),  # index price
                                float(d["r"]),  # funding rate
                                int(d["T"]),     # next funding time
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


async def ws_spot_stream(buffer, stop_event):
    """Connect to spot combined stream."""
    streams = [
        f"{SYMBOL}@aggTrade",
        f"{SYMBOL}@bookTicker",
        f"{SYMBOL}@depth20@100ms",
    ]
    url = f"{SPOT_WS}/{'/'.join(streams)}"

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ssl=SSL_CONTEXT,
                                          ping_interval=20,
                                          ping_timeout=10) as ws:
                async for msg in ws:
                    if stop_event.is_set():
                        break
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

            # Check if this timestamp already exists
            if ts not in buffer.mt_ts:
                buffer.add_metrics(ts, ls, top_ls, taker_ls, oi)

        except Exception:
            pass

        await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# Prediction loop
# ---------------------------------------------------------------------------

async def prediction_loop(buffer, predictor, interval, stop_event):
    """Run prediction every `interval` seconds."""
    # Wait a bit for websockets to connect
    await asyncio.sleep(3)

    last_block = 0
    block_count = 0

    while not stop_event.is_set():
        try:
            now_ms = int(time.time() * 1000)

            # Trim old data
            buffer.trim(now_ms)

            # Predict
            result = predictor.predict(buffer, now_ms)

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
                print(f"\n{'='*75}")
                print(f"  NEW BLOCK: {block_time}-{block_end} UTC  |  "
                      f"Open: {result['open_ref']:,.2f}  |  "
                      f"Blocks seen: {block_count}")
                print(f"{'='*75}")

            # Format output
            now_str = datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).strftime("%H:%M:%S")

            secs = result["seconds_to_expiry"]
            dist = result["dist_to_open_bps"]
            p_cal = result["p_calibrated"]
            p_bm = result["brownian_prob"]
            direction = result["direction"]
            conf = result["confidence"]
            price = result["price_now"]

            # Color-like indicators
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

            print(f"  {now_str} | {secs:5.0f}s left | "
                  f"BTC {price:>10,.2f} | "
                  f"vs open: {dist:>+7.2f} bps | "
                  f"P(Up): {p_cal:.3f} | "
                  f"BM: {p_bm:.3f} | "
                  f"edge: {p_cal - p_bm:>+.3f} | "
                  f"{signal}")

        except Exception as e:
            print(f"\n  [predict] error: {e}")

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(interval=1):
    buffer = LiveBuffer(max_seconds=600)
    predictor = LivePredictor()

    print("=" * 75)
    print("  BTC 5-Min Block Predictor (Live)")
    print("=" * 75)
    print(f"  Model: models/lightgbm_v3.txt")
    print(f"  Features: {len(predictor.feature_cols)}")
    print(f"  Interval: {interval}s")
    print()

    # REST warmup
    async with aiohttp.ClientSession() as session:
        await warmup(buffer, session)

        # Start everything
        stop_event = asyncio.Event()

        try:
            await asyncio.gather(
                ws_futures_stream(buffer, stop_event),
                ws_spot_stream(buffer, stop_event),
                metrics_poller(buffer, session, stop_event),
                prediction_loop(buffer, predictor, interval, stop_event),
            )
        except KeyboardInterrupt:
            print("\n\nStopping...")
            stop_event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live BTC block predictor")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Prediction interval in seconds (default: 1)")
    args = parser.parse_args()

    try:
        asyncio.run(main(interval=args.interval))
    except KeyboardInterrupt:
        print("\nDone.")
```

- [ ] **Step 2: Commit**

```bash
git add src/inference/live_runner.py
git commit -m "feat: add live runner with REST warmup, websockets, and prediction loop"
```

---

## Chunk 4: Test locally

### Task 4: Run and verify

- [ ] **Step 1: Run the script**

```bash
python -m src.inference.live_runner
```

Expected output:
```
=========================================================================
  BTC 5-Min Block Predictor (Live)
=========================================================================
  Model: models/lightgbm_v3.txt
  Features: 91
  Interval: 1s

Warming up...
  klines 1m: 60 bars
  premiumIndex: mark=97234.5, index=97230.1
  global_ls: 10 bars
  ...

=========================================================================
  NEW BLOCK: 14:25-14:30 UTC  |  Open: 97,230.50  |  Blocks seen: 1
=========================================================================
  14:25:01 |   299s left | BTC  97,231.20 | vs open:   +0.72 bps | P(Up): 0.512 | BM: 0.508 | edge: +0.004 | FLAT ~
  14:25:02 |   298s left | BTC  97,232.40 | vs open:   +1.95 bps | P(Up): 0.523 | BM: 0.515 | edge: +0.008 | FLAT ~
  ...
```

- [ ] **Step 2: Verify predictions change over time**

Watch for ~60 seconds. Predictions should:
- Update every second
- Show new block header every 5 minutes
- P(Up) should increase when price goes above open
- P(Up) should approach 0 or 1 near block end

- [ ] **Step 3: Ctrl+C to stop, verify clean exit**

- [ ] **Step 4: Final commit**

```bash
git add -A src/inference/
git commit -m "feat: complete live predictor v3 with REST warmup and websocket streams"
```

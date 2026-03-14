#!/usr/bin/env python3
"""
chainlink_drift.py — Binance vs Polymarket/Chainlink real-time drift analysis

Measures whether Chainlink price MOVEMENTS lag behind Binance.
Not comparing price levels (those differ by ~7 bps basis).
Instead: cross-correlation of 1s returns to find timing offset.

Usage:
    python experiments/chainlink_drift.py
"""

import asyncio
import json
import ssl
import time
from collections import deque
from datetime import datetime

import numpy as np
import websockets

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BINANCE_WS = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
POLYMARKET_WS = "wss://ws-live-data.polymarket.com/"
SSL_CONTEXT = ssl.create_default_context()

# ---------------------------------------------------------------------------
# Shared state — store (server_timestamp_s, price) for each tick
# ---------------------------------------------------------------------------
binance_ticks = deque(maxlen=500_000)
chainlink_ticks = deque(maxlen=500_000)

last_bn = {"price": 0.0, "ts": 0.0}
last_cl = {"price": 0.0, "ts": 0.0}

# ---------------------------------------------------------------------------
# Binance WS
# ---------------------------------------------------------------------------
async def binance_stream(stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            async with websockets.connect(BINANCE_WS, ssl=SSL_CONTEXT,
                                          ping_interval=10, ping_timeout=20) as ws:
                print("[binance] Connected")
                async for msg in ws:
                    if stop_event.is_set():
                        break
                    d = json.loads(msg)
                    price = float(d["p"])
                    ts = d["T"] / 1000.0
                    binance_ticks.append((ts, price))
                    last_bn["price"] = price
                    last_bn["ts"] = ts
        except Exception as e:
            if not stop_event.is_set():
                print(f"[binance] reconnecting: {e}")
                await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Polymarket / Chainlink WS
# ---------------------------------------------------------------------------
async def polymarket_stream(stop_event: asyncio.Event):
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
            async with websockets.connect(POLYMARKET_WS, ssl=SSL_CONTEXT,
                                          ping_interval=5, ping_timeout=20) as ws:
                await ws.send(sub_msg)
                print("[chainlink] Connected to Polymarket RTDS")
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

                        price = None
                        cl_ts = None

                        if isinstance(payload, dict) and "data" in payload:
                            data = payload["data"]
                            if isinstance(data, list) and data:
                                last = data[-1]
                                price = float(last["value"])
                                cl_ts = float(last.get("timestamp", 0))
                        elif d.get("topic") == "crypto_prices_chainlink":
                            price = float(payload.get("value", 0))
                            cl_ts = float(payload.get("timestamp", 0))

                        if price and price > 0:
                            if cl_ts and cl_ts > 1e12:
                                cl_ts = cl_ts / 1000.0
                            if not cl_ts or cl_ts < 1e6:
                                cl_ts = time.time()
                            chainlink_ticks.append((cl_ts, price))
                            last_cl["price"] = price
                            last_cl["ts"] = cl_ts

                    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                        pass
        except Exception as e:
            if not stop_event.is_set():
                print(f"[chainlink] reconnecting: {e}")
                await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Resample to 1s grid and compute returns
# ---------------------------------------------------------------------------
def build_1s_returns(ticks, start_ts, end_ts):
    """Forward-fill tick data onto a 1s grid, return (grid, prices, returns)."""
    arr = np.array(list(ticks))
    if len(arr) < 2:
        return None, None, None

    # Filter to range
    mask = (arr[:, 0] >= start_ts) & (arr[:, 0] <= end_ts)
    arr = arr[mask]
    if len(arr) < 2:
        return None, None, None

    grid = np.arange(start_ts, end_ts, 1.0)
    prices = np.zeros(len(grid))

    j = 0
    for i, t in enumerate(grid):
        while j < len(arr) - 1 and arr[j + 1, 0] <= t:
            j += 1
        prices[i] = arr[j, 1]

    returns = np.diff(prices)
    return grid[1:], prices[1:], returns


def compute_xcorr(bn_ret, cl_ret, max_lag_s=30):
    """
    Cross-correlation of returns at integer-second lags.
    Positive lag = Chainlink lags Binance by N seconds.
    """
    n = len(bn_ret)
    if n < max_lag_s * 3:
        return None, None, None

    lags = list(range(-max_lag_s, max_lag_s + 1))
    corrs = []

    bn_norm = (bn_ret - bn_ret.mean())
    cl_norm = (cl_ret - cl_ret.mean())
    bn_std = bn_norm.std()
    cl_std = cl_norm.std()

    if bn_std == 0 or cl_std == 0:
        return None, None, None

    bn_norm = bn_norm / bn_std
    cl_norm = cl_norm / cl_std

    for lag in lags:
        if lag >= 0:
            # Chainlink lags: compare bn[lag:] with cl[:-lag] (or cl[:n-lag])
            length = n - abs(lag)
            if length < 10:
                corrs.append(0)
                continue
            c = np.dot(bn_norm[:length], cl_norm[lag:lag + length]) / length
        else:
            # Chainlink leads: compare bn[:-|lag|] with cl[|lag|:]
            alag = abs(lag)
            length = n - alag
            if length < 10:
                corrs.append(0)
                continue
            c = np.dot(bn_norm[alag:alag + length], cl_norm[:length]) / length
        corrs.append(c)

    corrs = np.array(corrs)
    best_idx = np.argmax(corrs)
    return lags, corrs, lags[best_idx]


# ---------------------------------------------------------------------------
# Display loop — shows prices + xcorr every 30s
# ---------------------------------------------------------------------------
async def display_loop(stop_event: asyncio.Event):
    await asyncio.sleep(4)

    print()
    print(f"{'Time':>10} | {'Binance':>12} | {'Chainlink':>12} | {'Diff $':>8} | {'bps':>6} | {'BN Δ1s':>8} | {'CL Δ1s':>8} | xcorr lag")
    print("-" * 105)

    xcorr_interval = 30
    last_xcorr_time = 0
    last_xcorr_result = ""

    prev_bn = 0.0
    prev_cl = 0.0

    while not stop_event.is_set():
        await asyncio.sleep(1.0)

        bn_p = last_bn["price"]
        cl_p = last_cl["price"]
        if bn_p == 0 or cl_p == 0:
            continue

        diff = cl_p - bn_p
        diff_bps = (diff / bn_p) * 10000

        # 1-second price changes
        bn_delta = bn_p - prev_bn if prev_bn > 0 else 0
        cl_delta = cl_p - prev_cl if prev_cl > 0 else 0
        prev_bn = bn_p
        prev_cl = cl_p

        now = time.time()
        now_str = datetime.now().strftime("%H:%M:%S")

        # Compute cross-correlation every N seconds
        xcorr_str = ""
        if now - last_xcorr_time >= xcorr_interval:
            last_xcorr_time = now

            if len(binance_ticks) > 100 and len(chainlink_ticks) > 30:
                # Use last 5 minutes of data
                end_ts = min(last_bn["ts"], last_cl["ts"])
                start_ts = end_ts - 300

                _, _, bn_ret = build_1s_returns(binance_ticks, start_ts, end_ts)
                _, _, cl_ret = build_1s_returns(chainlink_ticks, start_ts, end_ts)

                if bn_ret is not None and cl_ret is not None:
                    min_len = min(len(bn_ret), len(cl_ret))
                    bn_ret = bn_ret[:min_len]
                    cl_ret = cl_ret[:min_len]

                    lags, corrs, best_lag = compute_xcorr(bn_ret, cl_ret, max_lag_s=15)

                    if lags is not None:
                        best_corr = corrs[np.argmax(corrs)]
                        lag0_corr = corrs[len(corrs) // 2]  # correlation at lag=0

                        # Show top 3 lags
                        top3_idx = np.argsort(corrs)[-3:][::-1]
                        top3 = [(lags[i], corrs[i]) for i in top3_idx]

                        last_xcorr_result = (
                            f"\n  XCORR ({min_len}s data) | "
                            f"best_lag={best_lag:+d}s (r={best_corr:.3f}) | "
                            f"lag0 r={lag0_corr:.3f} | "
                            f"top3: {', '.join(f'{l:+d}s={c:.3f}' for l,c in top3)}"
                        )
                        xcorr_str = last_xcorr_result

        print(
            f"{now_str:>10} | ${bn_p:>11,.2f} | ${cl_p:>11,.2f} | "
            f"{diff:>+8.2f} | {diff_bps:>+5.1f} | "
            f"{bn_delta:>+8.2f} | {cl_delta:>+8.2f} | "
        )

        if xcorr_str:
            print(xcorr_str)
            print("-" * 105)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if not binance_ticks or not chainlink_ticks:
        print("Not enough data")
        return

    bn_arr = np.array(list(binance_ticks))
    cl_arr = np.array(list(chainlink_ticks))

    duration = min(bn_arr[-1, 0], cl_arr[-1, 0]) - max(bn_arr[0, 0], cl_arr[0, 0])
    print(f"Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"Binance ticks: {len(bn_arr)}")
    print(f"Chainlink ticks: {len(cl_arr)}")

    # CL update frequency
    cl_intervals = np.diff(cl_arr[:, 0])
    cl_intervals = cl_intervals[cl_intervals > 0]
    if len(cl_intervals) > 0:
        print(f"Chainlink update interval: mean={np.mean(cl_intervals):.2f}s, "
              f"median={np.median(cl_intervals):.2f}s")

    # Price level difference
    # Sample CL prices and find closest BN price
    diffs_bps = []
    for cl_ts, cl_price in cl_arr[::max(1, len(cl_arr)//200)]:
        idx = np.searchsorted(bn_arr[:, 0], cl_ts)
        if 0 < idx < len(bn_arr):
            bn_price = bn_arr[idx, 1]
            diffs_bps.append((cl_price - bn_price) / bn_price * 10000)
    if diffs_bps:
        d = np.array(diffs_bps)
        print(f"\nPrice level (CL - BN): mean={np.mean(d):+.1f} bps, "
              f"std={np.std(d):.1f} bps, min={np.min(d):+.1f}, max={np.max(d):+.1f}")

    # Full cross-correlation
    end_ts = min(bn_arr[-1, 0], cl_arr[-1, 0])
    start_ts = max(bn_arr[0, 0], cl_arr[0, 0])

    _, _, bn_ret = build_1s_returns(binance_ticks, start_ts, end_ts)
    _, _, cl_ret = build_1s_returns(chainlink_ticks, start_ts, end_ts)

    if bn_ret is not None and cl_ret is not None:
        min_len = min(len(bn_ret), len(cl_ret))
        bn_ret = bn_ret[:min_len]
        cl_ret = cl_ret[:min_len]

        lags, corrs, best_lag = compute_xcorr(bn_ret, cl_ret, max_lag_s=30)

        if lags is not None:
            best_corr = corrs[np.argmax(corrs)]
            lag0_corr = corrs[len(corrs) // 2]

            print(f"\nCROSS-CORRELATION OF 1s RETURNS ({min_len}s of data):")
            print(f"  Best lag: {best_lag:+d}s (correlation: {best_corr:.4f})")
            print(f"  Lag=0 correlation: {lag0_corr:.4f}")
            print(f"  (positive = Chainlink lags Binance by N seconds)")

            # Print correlation profile around the peak
            print(f"\n  Lag profile:")
            for i, lag in enumerate(lags):
                if abs(lag) <= 10:
                    bar = "#" * int(max(0, corrs[i]) * 50)
                    marker = " <<<" if lag == best_lag else ""
                    print(f"    {lag:+3d}s: {corrs[i]:+.4f} {bar}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print("=" * 70)
    print("  BINANCE vs CHAINLINK — Movement Timing Analysis")
    print("=" * 70)
    print()
    print("Measures if Chainlink price MOVEMENTS lag Binance.")
    print("Cross-correlation of 1s returns computed every 30s.")
    print("Let it run 2-3 min minimum. Ctrl+C for full summary.")
    print()

    stop_event = asyncio.Event()
    tasks = [
        asyncio.create_task(binance_stream(stop_event)),
        asyncio.create_task(polymarket_stream(stop_event)),
        asyncio.create_task(display_loop(stop_event)),
    ]

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.sleep(0.5)
        print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

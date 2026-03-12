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

    # 5. Recent trades futures (last 60s)
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

    # 6. Recent trades spot (last 60s)
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

async def ws_futures_stream(buffer, stop_event):
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
                                          ping_interval=30,
                                          ping_timeout=20) as ws:
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


async def ws_spot_stream(buffer, stop_event):
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
                          ws_clients=None):
    """Run prediction every `interval` seconds."""
    await asyncio.sleep(3)

    last_block = 0
    block_count = 0

    while not stop_event.is_set():
        try:
            now_ms = int(time.time() * 1000)

            # Trim old data
            buffer.trim(now_ms)

            # Predict (in thread to not block event loop / ws pings)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, predictor.predict, buffer, now_ms
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
                print(f"\n{'='*80}")
                print(f"  NEW BLOCK: {block_time}-{block_end} UTC  |  "
                      f"Open: {result['open_ref']:,.2f}  |  "
                      f"Blocks seen: {block_count}")
                print(f"{'='*80}")

            # Format output
            now_str = datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).strftime("%H:%M:%S")

            secs = result["seconds_to_expiry"]
            dist = result["dist_to_open_bps"]
            p_cal = result["p_calibrated"]
            p_bm = result["brownian_prob"]
            price = result["price_now"]

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

            warming = ""
            if len(predictor.block_results) == 0:
                warming = " [WARMING UP]"

            print(f"  {now_str} | {secs:5.0f}s | "
                  f"BTC {price:>10,.2f} | "
                  f"vs open: {dist:>+7.2f} bps | "
                  f"P(Up): {p_cal:.3f} | "
                  f"BM: {p_bm:.3f} | "
                  f"edge: {p_cal - p_bm:>+.3f} | "
                  f"{signal}{warming}")

            # Broadcast to websocket clients
            if ws_clients:
                payload = json.dumps({
                    "block_start_ms": result["block_start_ms"],
                    "now_ms": now_ms,
                    "seconds_to_expiry": round(secs, 1),
                    "open_ref": float(result["open_ref"]),
                    "price_now": float(price),
                    "dist_to_open_bps": round(float(dist), 2),
                    "p_up": round(float(p_cal), 4),
                    "p_raw": round(float(result["p_raw"]), 4),
                    "brownian_prob": round(float(p_bm), 4),
                    "edge": round(float(p_cal - p_bm), 4),
                    "direction": result["direction"],
                    "signal": signal.strip(),
                    "warming_up": len(predictor.block_results) == 0,
                })
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


async def main(interval=1, ws_port=8765):
    buffer = LiveBuffer(max_seconds=600)
    predictor = LivePredictor()

    # Websocket clients set (shared with prediction_loop)
    ws_clients = set()

    print("=" * 80)
    print("  BTC 5-Min Block Predictor (Live)")
    print("=" * 80)
    print(f"  Model: models/lightgbm_v3.txt")
    print(f"  Features: {len(predictor.feature_cols)}")
    print(f"  Interval: {interval}s")
    print(f"  WS server: ws://localhost:{ws_port}")
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
                ws_futures_stream(buffer, stop_event),
                ws_spot_stream(buffer, stop_event),
                metrics_poller(buffer, session, stop_event),
                prediction_loop(buffer, predictor, interval, stop_event,
                                ws_clients),
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
    args = parser.parse_args()

    try:
        asyncio.run(main(interval=args.interval, ws_port=args.ws_port))
    except KeyboardInterrupt:
        print("\nDone.")

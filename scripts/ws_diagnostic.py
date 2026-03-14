"""
WebSocket diagnostic: connect to all sources, print raw vs parsed data.
Verifies that what arrives matches what LiveBuffer expects.

Usage:
    python scripts/ws_diagnostic.py
    python scripts/ws_diagnostic.py --seconds 30
"""

import asyncio
import json
import time
import ssl
import argparse
from datetime import datetime, timezone

import certifi
import websockets
import aiohttp

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

FUTURES_WS = "wss://fstream.binance.com/ws"
FUTURES_REST = "https://fapi.binance.com"
SPOT_REST = "https://api.binance.com"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"

SYMBOL = "btcusdt"

# Counters per source
counts = {}
errors = {}
samples = {}


def ts_to_str(ms):
    """Convert ms timestamp to readable string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def now_ms():
    return int(time.time() * 1000)


def log_sample(source, stream_type, parsed):
    """Store first sample of each stream type for final report."""
    key = f"{source}/{stream_type}"
    counts[key] = counts.get(key, 0) + 1
    if key not in samples:
        samples[key] = parsed


def log_error(source, stream_type, error_msg):
    key = f"{source}/{stream_type}"
    errors[key] = errors.get(key, 0) + 1
    if errors[key] <= 3:
        print(f"  ERROR [{key}]: {error_msg}")


async def test_futures(stop_event):
    """Test Binance futures combined stream."""
    streams = [
        f"{SYMBOL}@aggTrade",
        f"{SYMBOL}@bookTicker",
        f"{SYMBOL}@depth20@100ms",
        f"{SYMBOL}@markPrice@1s",
        f"{SYMBOL}@forceOrder",
    ]
    url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"

    try:
        async with websockets.connect(url, ssl=SSL_CONTEXT,
                                      ping_interval=None, max_size=2**22) as ws:
            print("[futures] connected")
            async for msg in ws:
                if stop_event.is_set():
                    break
                d = json.loads(msg)
                stream = d.get("stream", "")
                data = d.get("data", d)

                if "aggTrade" in stream:
                    try:
                        parsed = {
                            "ts_ms": int(data["T"]),
                            "ts_str": ts_to_str(int(data["T"])),
                            "price": float(data["p"]),
                            "qty": float(data["q"]),
                            "is_buyer_maker": data["m"],
                            "latency_ms": now_ms() - int(data["T"]),
                        }
                        log_sample("futures", "aggTrade", parsed)
                    except Exception as e:
                        log_error("futures", "aggTrade", f"{e} | raw keys: {list(data.keys())}")

                elif "bookTicker" in stream:
                    try:
                        parsed = {
                            "ts_ms": int(data["T"]),
                            "ts_str": ts_to_str(int(data["T"])),
                            "bid": float(data["b"]),
                            "ask": float(data["a"]),
                            "spread_bps": (float(data["a"]) - float(data["b"])) / float(data["b"]) * 10000,
                            "latency_ms": now_ms() - int(data["T"]),
                        }
                        log_sample("futures", "bookTicker", parsed)
                    except Exception as e:
                        log_error("futures", "bookTicker", f"{e} | raw keys: {list(data.keys())}")

                elif "depth20" in stream:
                    try:
                        ts = int(data.get("T", data.get("E", now_ms())))
                        bids = data["b"]
                        asks = data["a"]
                        parsed = {
                            "ts_ms": ts,
                            "ts_str": ts_to_str(ts),
                            "n_bids": len(bids),
                            "n_asks": len(asks),
                            "best_bid": float(bids[0][0]) if bids else 0,
                            "best_ask": float(asks[0][0]) if asks else 0,
                            "ts_field_used": "T" if "T" in data else ("E" if "E" in data else "now"),
                            "latency_ms": now_ms() - ts,
                        }
                        log_sample("futures", "depth20", parsed)
                    except Exception as e:
                        log_error("futures", "depth20", f"{e} | raw keys: {list(data.keys())}")

                elif "markPrice" in stream:
                    try:
                        parsed = {
                            "event_ts_ms": int(data["E"]),
                            "ts_str": ts_to_str(int(data["E"])),
                            "mark_price": float(data["p"]),
                            "index_price": float(data["i"]),
                            "funding_rate": float(data["r"]),
                            "next_funding_ms": int(data["T"]),
                            "mark_vs_index_bps": (float(data["p"]) - float(data["i"])) / float(data["i"]) * 10000,
                            "latency_ms": now_ms() - int(data["E"]),
                        }
                        log_sample("futures", "markPrice", parsed)
                    except Exception as e:
                        log_error("futures", "markPrice", f"{e} | raw keys: {list(data.keys())}")

                elif "forceOrder" in stream:
                    try:
                        o = data["o"]
                        parsed = {
                            "ts_ms": int(o["T"]),
                            "ts_str": ts_to_str(int(o["T"])),
                            "side": o["S"],
                            "is_buy": o["S"] == "BUY",
                            "qty": float(o["q"]),
                            "price": float(o["p"]),
                        }
                        log_sample("futures", "forceOrder", parsed)
                    except Exception as e:
                        log_error("futures", "forceOrder", f"{e} | raw keys: {list(data.keys())}")

    except Exception as e:
        print(f"[futures] error: {e}")


async def test_spot(stop_event):
    """Test Binance spot combined stream."""
    streams = [
        f"{SYMBOL}@aggTrade",
        f"{SYMBOL}@bookTicker",
        f"{SYMBOL}@depth20@100ms",
    ]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    try:
        async with websockets.connect(url, ssl=SSL_CONTEXT,
                                      ping_interval=30, ping_timeout=20) as ws:
            print("[spot] connected")
            async for msg in ws:
                if stop_event.is_set():
                    break
                d = json.loads(msg)
                stream = d.get("stream", "")
                data = d.get("data", d)

                if "aggTrade" in stream:
                    try:
                        parsed = {
                            "ts_ms": int(data["T"]),
                            "price": float(data["p"]),
                            "qty": float(data["q"]),
                            "is_buyer_maker": data["m"],
                            "latency_ms": now_ms() - int(data["T"]),
                        }
                        log_sample("spot", "aggTrade", parsed)
                    except Exception as e:
                        log_error("spot", "aggTrade", f"{e}")

                elif "bookTicker" in stream:
                    try:
                        # Spot bookTicker may not have "T" field
                        ts = int(data.get("T", data.get("E", now_ms())))
                        parsed = {
                            "ts_ms": ts,
                            "bid": float(data["b"]),
                            "ask": float(data["a"]),
                            "has_T_field": "T" in data,
                            "has_E_field": "E" in data,
                            "raw_keys": list(data.keys()),
                            "latency_ms": now_ms() - ts,
                        }
                        log_sample("spot", "bookTicker", parsed)
                    except Exception as e:
                        log_error("spot", "bookTicker", f"{e} | raw keys: {list(data.keys())}")

                elif "depth20" in stream:
                    try:
                        ts = int(data.get("T", data.get("E", now_ms())))
                        parsed = {
                            "ts_ms": ts,
                            "n_bids": len(data["b"]),
                            "n_asks": len(data["a"]),
                            "ts_field_used": "T" if "T" in data else ("E" if "E" in data else "now"),
                            "latency_ms": now_ms() - ts,
                        }
                        log_sample("spot", "depth20", parsed)
                    except Exception as e:
                        log_error("spot", "depth20", f"{e}")

    except Exception as e:
        print(f"[spot] error: {e}")


async def test_coinbase(stop_event):
    """Test Coinbase websocket."""
    sub_msg = json.dumps({
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["ticker", "matches", "level2"],
    })

    try:
        async with websockets.connect(COINBASE_WS, ssl=SSL_CONTEXT,
                                      ping_interval=30, ping_timeout=20,
                                      max_size=2**22) as ws:
            await ws.send(sub_msg)
            print("[coinbase] connected")
            async for msg in ws:
                if stop_event.is_set():
                    break
                d = json.loads(msg)
                msg_type = d.get("type", "")

                if msg_type == "ticker":
                    try:
                        parsed = {
                            "ts_local_ms": now_ms(),
                            "best_bid": float(d["best_bid"]),
                            "best_ask": float(d["best_ask"]),
                            "price": float(d.get("price", 0)),
                            "last_size": float(d.get("last_size", 0)),
                            "side": d.get("side", "?"),
                            "note": "side=sell → is_buyer_maker=True in our code",
                        }
                        log_sample("coinbase", "ticker", parsed)
                    except Exception as e:
                        log_error("coinbase", "ticker", f"{e}")

                elif msg_type in ("match", "last_match"):
                    try:
                        parsed = {
                            "ts_local_ms": now_ms(),
                            "price": float(d["price"]),
                            "size": float(d["size"]),
                            "side": d.get("side", "?"),
                            "note": "side=sell → is_buyer_maker=True",
                        }
                        log_sample("coinbase", "match", parsed)
                    except Exception as e:
                        log_error("coinbase", "match", f"{e}")

                elif msg_type == "snapshot":
                    try:
                        parsed = {
                            "n_bids": len(d.get("bids", [])),
                            "n_asks": len(d.get("asks", [])),
                            "note": "L2 initial snapshot",
                        }
                        log_sample("coinbase", "l2_snapshot", parsed)
                    except Exception as e:
                        log_error("coinbase", "l2_snapshot", f"{e}")

                elif msg_type == "l2update":
                    try:
                        changes = d.get("changes", [])
                        parsed = {
                            "n_changes": len(changes),
                            "sample_change": changes[0] if changes else None,
                            "note": "changes: [side, price, qty]",
                        }
                        log_sample("coinbase", "l2update", parsed)
                    except Exception as e:
                        log_error("coinbase", "l2update", f"{e}")

                elif msg_type in ("subscriptions", "error"):
                    log_sample("coinbase", msg_type, d)

    except Exception as e:
        print(f"[coinbase] error: {e}")


async def test_bybit(stop_event):
    """Test Bybit websocket."""
    topics = ["tickers.BTCUSDT", "publicTrade.BTCUSDT", "orderbook.50.BTCUSDT"]

    try:
        async with websockets.connect(BYBIT_WS, ssl=SSL_CONTEXT,
                                      ping_interval=20, ping_timeout=10,
                                      max_size=2**22) as ws:
            for topic in topics:
                await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
            print("[bybit] connected")
            async for msg in ws:
                if stop_event.is_set():
                    break
                d = json.loads(msg)
                topic = d.get("topic", "")

                if topic.startswith("tickers."):
                    try:
                        data = d.get("data", {})
                        ts = int(d.get("ts", now_ms()))
                        parsed = {
                            "ts_ms": ts,
                            "ts_str": ts_to_str(ts),
                            "bid1Price": data.get("bid1Price"),
                            "ask1Price": data.get("ask1Price"),
                            "bid_is_zero": float(data.get("bid1Price", 0)) == 0,
                            "ask_is_zero": float(data.get("ask1Price", 0)) == 0,
                            "latency_ms": now_ms() - ts,
                            "all_keys": list(data.keys()),
                        }
                        log_sample("bybit", "tickers", parsed)
                    except Exception as e:
                        log_error("bybit", "tickers", f"{e}")

                elif topic.startswith("publicTrade."):
                    try:
                        trades = d.get("data", [])
                        if trades:
                            t = trades[0]
                            parsed = {
                                "n_trades_in_msg": len(trades),
                                "ts_ms": int(t.get("T", now_ms())),
                                "price": float(t["p"]),
                                "qty": float(t["v"]),
                                "side": t.get("S", "?"),
                                "note": "S=Sell → is_buyer_maker=True in our code",
                                "latency_ms": now_ms() - int(t.get("T", now_ms())),
                            }
                            log_sample("bybit", "publicTrade", parsed)
                    except Exception as e:
                        log_error("bybit", "publicTrade", f"{e}")

                elif topic.startswith("orderbook."):
                    try:
                        data = d.get("data", {})
                        msg_type = d.get("type", "?")
                        parsed = {
                            "type": msg_type,
                            "ts_ms": int(d.get("ts", now_ms())),
                            "n_bids": len(data.get("b", [])),
                            "n_asks": len(data.get("a", [])),
                            "note": f"type={msg_type} (snapshot vs delta)",
                        }
                        log_sample("bybit", f"orderbook_{msg_type}", parsed)
                    except Exception as e:
                        log_error("bybit", "orderbook", f"{e}")

                elif d.get("op") == "subscribe":
                    log_sample("bybit", "subscribe_ack", d)

    except Exception as e:
        print(f"[bybit] error: {e}")


async def test_metrics():
    """Test REST API metrics (same as metrics_poller)."""
    print("\n--- REST API Metrics ---")
    async with aiohttp.ClientSession() as session:
        # Global long/short
        try:
            url = f"{FUTURES_REST}/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=2"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                latest = data[-1]
                print(f"  global_ls: ts={latest['timestamp']} ({ts_to_str(int(latest['timestamp']))}), "
                      f"ratio={latest['longShortRatio']}")
                print(f"    note: ts is in ms? {int(latest['timestamp']) > 1e12}")
        except Exception as e:
            print(f"  global_ls: ERROR {e}")

        # Top trader
        try:
            url = f"{FUTURES_REST}/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=5m&limit=2"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                latest = data[-1]
                print(f"  top_ls: ts={latest['timestamp']}, ratio={latest['longShortRatio']}")
        except Exception as e:
            print(f"  top_ls: ERROR {e}")

        # Taker
        try:
            url = f"{FUTURES_REST}/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=2"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                latest = data[-1]
                print(f"  taker_ls: ts={latest['timestamp']}, ratio={latest['buySellRatio']}")
        except Exception as e:
            print(f"  taker_ls: ERROR {e}")

        # Warmup klines vs premiumIndex comparison
        print("\n--- Warmup comparison: klines close vs index_price ---")
        try:
            url = f"{FUTURES_REST}/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=1"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                klines = await r.json()
                k_close = float(klines[0][4])
                print(f"  klines 1m close: {k_close:.2f}")
        except Exception as e:
            print(f"  klines: ERROR {e}")

        try:
            url = f"{FUTURES_REST}/fapi/v1/premiumIndex?symbol=BTCUSDT"
            async with session.get(url, ssl=SSL_CONTEXT) as r:
                data = await r.json()
                mark = float(data["markPrice"])
                index = float(data["indexPrice"])
                diff = (k_close - index) / index * 10000
                print(f"  mark_price:  {mark:.2f}")
                print(f"  index_price: {index:.2f}")
                print(f"  klines_close vs index: {diff:+.2f} bps")
                print(f"  note: warmup uses klines close as both mark AND index")
                print(f"         this introduces ~{abs(diff):.1f} bps error in ref_price until real markPrice@1s arrives")
        except Exception as e:
            print(f"  premiumIndex: ERROR {e}")


async def main(seconds=20):
    stop_event = asyncio.Event()
    print(f"Connecting to all websockets for {seconds}s...\n")

    # Test REST first
    await test_metrics()

    print(f"\n--- WebSocket streams ({seconds}s) ---\n")

    tasks = [
        asyncio.create_task(test_futures(stop_event)),
        asyncio.create_task(test_spot(stop_event)),
        asyncio.create_task(test_coinbase(stop_event)),
        asyncio.create_task(test_bybit(stop_event)),
    ]

    await asyncio.sleep(seconds)
    stop_event.set()

    # Give tasks time to clean up
    await asyncio.sleep(1)
    for t in tasks:
        t.cancel()

    # Print report
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC REPORT ({seconds}s)")
    print(f"{'='*70}\n")

    print("Message counts:")
    for key in sorted(counts.keys()):
        rate = counts[key] / seconds
        print(f"  {key:40s}: {counts[key]:6d} msgs ({rate:.1f}/s)")

    if errors:
        print(f"\nErrors:")
        for key in sorted(errors.keys()):
            print(f"  {key:40s}: {errors[key]} errors")

    print(f"\nFirst sample per stream:")
    for key in sorted(samples.keys()):
        print(f"\n  {key}:")
        s = samples[key]
        if isinstance(s, dict):
            for k, v in s.items():
                print(f"    {k}: {v}")
        else:
            print(f"    {s}")

    # Consistency checks
    print(f"\n{'='*70}")
    print(f"  CONSISTENCY CHECKS")
    print(f"{'='*70}\n")

    # Check: spot bookTicker has T field?
    s = samples.get("spot/bookTicker", {})
    if s:
        has_T = s.get("has_T_field", False)
        print(f"  spot bookTicker has 'T' timestamp field: {has_T}")
        if not has_T:
            print(f"    WARNING: live_runner uses d.get('T', d.get('E', time.time()*1000))")
            print(f"    Available keys: {s.get('raw_keys', '?')}")

    # Check: bybit tickers bid/ask zero?
    s = samples.get("bybit/tickers", {})
    if s:
        bid_zero = s.get("bid_is_zero", False)
        ask_zero = s.get("ask_is_zero", False)
        if bid_zero or ask_zero:
            print(f"  WARNING: Bybit tickers has zero bid/ask — may get filtered out")
            print(f"    Keys in data: {s.get('all_keys', '?')}")
        else:
            print(f"  Bybit tickers bid/ask: OK (non-zero)")

    # Check: latencies
    for key in sorted(samples.keys()):
        s = samples[key]
        if isinstance(s, dict) and "latency_ms" in s:
            lat = s["latency_ms"]
            status = "OK" if lat < 1000 else "HIGH"
            print(f"  {key:40s} latency: {lat:6d}ms [{status}]")

    # Check: markPrice mark vs index
    s = samples.get("futures/markPrice", {})
    if s:
        diff = s.get("mark_vs_index_bps", 0)
        print(f"\n  futures/markPrice mark vs index: {diff:+.2f} bps (basis)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.seconds))

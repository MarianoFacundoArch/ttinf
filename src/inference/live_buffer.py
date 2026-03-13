"""
LiveBuffer: accumulates Binance websocket data into numpy arrays
compatible with feature_engine_v3's DayData structure.

Keeps a rolling window of data. Exposes to_day_data() for feature computation.
"""

import numpy as np
import time
from src.features.feature_engine_v3 import DayData, _precompute_book


class LocalBook:
    """Maintains a local L2 orderbook from incremental WS updates."""

    def __init__(self, max_levels=20):
        self.max_levels = max_levels
        self.bids = {}  # price -> qty
        self.asks = {}  # price -> qty

    def apply_snapshot(self, bids, asks):
        """Replace entire book with snapshot data."""
        self.bids = {}
        for price, qty in bids:
            p, q = float(price), float(qty)
            if q > 0:
                self.bids[p] = q
        self.asks = {}
        for price, qty in asks:
            p, q = float(price), float(qty)
            if q > 0:
                self.asks[p] = q

    def apply_delta(self, bids, asks):
        """Apply incremental updates to book."""
        for price, qty in bids:
            p, q = float(price), float(qty)
            if q == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q
        for price, qty in asks:
            p, q = float(price), float(qty)
            if q == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

    def top_levels(self):
        """Return top N bid/ask levels as padded lists of length max_levels."""
        n = self.max_levels
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]

        bp = [p for p, q in sorted_bids]
        bq = [q for p, q in sorted_bids]
        ap = [p for p, q in sorted_asks]
        aq = [q for p, q in sorted_asks]

        while len(bp) < n:
            bp.append(0.0); bq.append(0.0)
        while len(ap) < n:
            ap.append(0.0); aq.append(0.0)

        return bp, bq, ap, aq

    def is_valid(self):
        """Check if book has at least 1 bid and 1 ask."""
        return len(self.bids) > 0 and len(self.asks) > 0


class LiveBuffer:
    """Accumulates live market data into arrays for feature computation."""

    def __init__(self, max_seconds=600):
        self.max_seconds = max_seconds

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

        # --- Orderbook futures (snapshots) ---
        self.ob_fut_ts = []
        self.ob_fut_bid_prices = []
        self.ob_fut_bid_qtys = []
        self.ob_fut_ask_prices = []
        self.ob_fut_ask_qtys = []

        # --- Orderbook spot (snapshots) ---
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

        # --- Cross-exchange: Coinbase quotes ---
        self.cb_ts = []
        self.cb_bid = []
        self.cb_ask = []

        # --- Cross-exchange: Bybit quotes ---
        self.bb_ts = []
        self.bb_bid = []
        self.bb_ask = []

        # --- Cross-exchange: Coinbase trades ---
        self.ct_ts = []
        self.ct_price = []
        self.ct_qty = []
        self.ct_ibm = []

        # --- Cross-exchange: Bybit trades ---
        self.bt_ts = []
        self.bt_price = []
        self.bt_qty = []
        self.bt_ibm = []

        # --- Cross-exchange: Coinbase L2 orderbook ---
        self.book_cb = LocalBook()
        self.ob_cb_ts = []
        self.ob_cb_bid_prices = []
        self.ob_cb_bid_qtys = []
        self.ob_cb_ask_prices = []
        self.ob_cb_ask_qtys = []

        # --- Cross-exchange: Bybit orderbook ---
        self.book_bb = LocalBook()
        self.ob_bb_ts = []
        self.ob_bb_bid_prices = []
        self.ob_bb_bid_qtys = []
        self.ob_bb_ask_prices = []
        self.ob_bb_ask_qtys = []

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

    def add_coinbase_quote(self, timestamp_ms, bid, ask):
        self.cb_ts.append(timestamp_ms)
        self.cb_bid.append(bid)
        self.cb_ask.append(ask)

    def add_bybit_quote(self, timestamp_ms, bid, ask):
        self.bb_ts.append(timestamp_ms)
        self.bb_bid.append(bid)
        self.bb_ask.append(ask)

    def add_coinbase_trade(self, timestamp_ms, price, qty, is_buyer_maker):
        self.ct_ts.append(timestamp_ms)
        self.ct_price.append(price)
        self.ct_qty.append(qty)
        self.ct_ibm.append(is_buyer_maker)

    def add_bybit_trade(self, timestamp_ms, price, qty, is_buyer_maker):
        self.bt_ts.append(timestamp_ms)
        self.bt_price.append(price)
        self.bt_qty.append(qty)
        self.bt_ibm.append(is_buyer_maker)

    def update_coinbase_book(self, is_snapshot, bids, asks, timestamp_ms):
        """Update Coinbase local book and store snapshot (throttled to ~200ms)."""
        if is_snapshot:
            self.book_cb.apply_snapshot(bids, asks)
        else:
            self.book_cb.apply_delta(bids, asks)
        # Throttle: only store snapshot every 200ms
        if self.ob_cb_ts and timestamp_ms - self.ob_cb_ts[-1] < 200:
            return
        if not self.book_cb.is_valid():
            return
        bp, bq, ap, aq = self.book_cb.top_levels()
        self.ob_cb_ts.append(timestamp_ms)
        self.ob_cb_bid_prices.append(bp)
        self.ob_cb_bid_qtys.append(bq)
        self.ob_cb_ask_prices.append(ap)
        self.ob_cb_ask_qtys.append(aq)

    def update_bybit_book(self, is_snapshot, bids, asks, timestamp_ms):
        """Update Bybit local book and store snapshot (throttled to ~200ms)."""
        if is_snapshot:
            self.book_bb.apply_snapshot(bids, asks)
        else:
            self.book_bb.apply_delta(bids, asks)
        if self.ob_bb_ts and timestamp_ms - self.ob_bb_ts[-1] < 200:
            return
        if not self.book_bb.is_valid():
            return
        bp, bq, ap, aq = self.book_bb.top_levels()
        self.ob_bb_ts.append(timestamp_ms)
        self.ob_bb_bid_prices.append(bp)
        self.ob_bb_bid_qtys.append(bq)
        self.ob_bb_ask_prices.append(ap)
        self.ob_bb_ask_qtys.append(aq)

    # --- Trim old data ---

    def trim(self, now_ms=None):
        """Remove data older than max_seconds."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - self.max_seconds * 1000

        def _trim_list(ts_list, *other_lists):
            if not ts_list:
                return
            idx = 0
            for i, t in enumerate(ts_list):
                if t >= cutoff:
                    idx = i
                    break
            else:
                idx = len(ts_list)
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

        # Cross-exchange
        _trim_list(self.cb_ts, self.cb_bid, self.cb_ask)
        _trim_list(self.bb_ts, self.bb_bid, self.bb_ask)
        _trim_list(self.ct_ts, self.ct_price, self.ct_qty, self.ct_ibm)
        _trim_list(self.bt_ts, self.bt_price, self.bt_qty, self.bt_ibm)
        _trim_list(self.ob_cb_ts, self.ob_cb_bid_prices, self.ob_cb_bid_qtys,
                   self.ob_cb_ask_prices, self.ob_cb_ask_qtys)
        _trim_list(self.ob_bb_ts, self.ob_bb_bid_prices, self.ob_bb_bid_qtys,
                   self.ob_bb_ask_prices, self.ob_bb_ask_qtys)

    # --- Convert to DayData ---

    def to_day_data(self):
        """Convert current buffer to DayData-compatible object.

        Snapshots list lengths first to avoid race conditions with
        concurrent websocket appends (prediction runs in a thread).
        """
        day = DayData()

        # Trades futures
        n = len(self.tf_ts)
        day.tf_ts = np.array(self.tf_ts[:n], dtype=np.int64)
        day.tf_price = np.array(self.tf_price[:n], dtype=np.float64)
        day.tf_qty = np.array(self.tf_qty[:n], dtype=np.float64)
        day.tf_ibm = np.array(self.tf_ibm[:n], dtype=bool)

        # Trades spot
        n = len(self.ts_ts)
        day.ts_ts = np.array(self.ts_ts[:n], dtype=np.int64)
        day.ts_price = np.array(self.ts_price[:n], dtype=np.float64)
        day.ts_qty = np.array(self.ts_qty[:n], dtype=np.float64)
        day.ts_ibm = np.array(self.ts_ibm[:n], dtype=bool)

        # Bookticker futures
        n = len(self.bf_ts)
        day.bf_ts = np.array(self.bf_ts[:n], dtype=np.int64)
        day.bf_bid = np.array(self.bf_bid[:n], dtype=np.float64)
        day.bf_ask = np.array(self.bf_ask[:n], dtype=np.float64)
        day.bf_mid = (day.bf_bid + day.bf_ask) / 2.0 if n > 0 else np.array([])

        # Bookticker spot
        n = len(self.bs_ts)
        day.bs_ts = np.array(self.bs_ts[:n], dtype=np.int64)
        day.bs_bid = np.array(self.bs_bid[:n], dtype=np.float64)
        day.bs_ask = np.array(self.bs_ask[:n], dtype=np.float64)
        day.bs_mid = (day.bs_bid + day.bs_ask) / 2.0 if n > 0 else np.array([])

        # Orderbook futures
        n = len(self.ob_fut_ts)
        if n > 0:
            ts = np.array(self.ob_fut_ts[:n], dtype=np.int64)
            bp = np.array(self.ob_fut_bid_prices[:n])
            bq = np.array(self.ob_fut_bid_qtys[:n])
            ap = np.array(self.ob_fut_ask_prices[:n])
            aq = np.array(self.ob_fut_ask_qtys[:n])
            day.ob_fut = _precompute_book(ts, bp, bq, ap, aq)
        else:
            day.ob_fut = {'ts': np.array([], dtype=np.int64),
                          'mid': np.array([]), 'spread_bps': np.array([]),
                          'imb_L1': np.array([]), 'imb_L5': np.array([]),
                          'bid_prices': np.empty((0, 20)), 'bid_qtys': np.empty((0, 20)),
                          'ask_prices': np.empty((0, 20)), 'ask_qtys': np.empty((0, 20))}

        # Orderbook spot
        n = len(self.ob_spot_ts)
        if n > 0:
            ts = np.array(self.ob_spot_ts[:n], dtype=np.int64)
            bp = np.array(self.ob_spot_bid_prices[:n])
            bq = np.array(self.ob_spot_bid_qtys[:n])
            ap = np.array(self.ob_spot_ask_prices[:n])
            aq = np.array(self.ob_spot_ask_qtys[:n])
            day.ob_spot = _precompute_book(ts, bp, bq, ap, aq)
        else:
            day.ob_spot = {'ts': np.array([], dtype=np.int64),
                           'mid': np.array([]), 'spread_bps': np.array([]),
                           'imb_L1': np.array([]), 'imb_L5': np.array([]),
                           'bid_prices': np.empty((0, 20)), 'bid_qtys': np.empty((0, 20)),
                           'ask_prices': np.empty((0, 20)), 'ask_qtys': np.empty((0, 20))}

        # Mark price
        n = len(self.mp_ts)
        day.mp_ts = np.array(self.mp_ts[:n], dtype=np.int64)
        day.mp_mark = np.array(self.mp_mark[:n], dtype=np.float64)
        day.mp_index = np.array(self.mp_index[:n], dtype=np.float64)
        day.mp_funding = np.array(self.mp_funding[:n], dtype=np.float64)
        day.mp_next_ms = np.array(self.mp_next_ms[:n], dtype=np.float64)

        # Liquidations
        n = len(self.lq_ts)
        day.lq_ts = np.array(self.lq_ts[:n], dtype=np.int64)
        day.lq_is_buy = np.array(self.lq_is_buy[:n], dtype=bool)
        day.lq_qty = np.array(self.lq_qty[:n], dtype=np.float64)

        # Metrics
        n = len(self.mt_ts)
        day.mt_ts = np.array(self.mt_ts[:n], dtype=np.int64)
        day.mt_ls_ratio = np.array(self.mt_ls_ratio[:n], dtype=np.float64)
        day.mt_top_ls = np.array(self.mt_top_ls[:n], dtype=np.float64)
        day.mt_taker_ls = np.array(self.mt_taker_ls[:n], dtype=np.float64)
        day.mt_oi = np.array(self.mt_oi[:n], dtype=np.float64)

        # Coinbase quotes
        n = len(self.cb_ts)
        day.cb_ts  = np.array(self.cb_ts[:n], dtype=np.int64)
        day.cb_bid = np.array(self.cb_bid[:n], dtype=np.float64)
        day.cb_ask = np.array(self.cb_ask[:n], dtype=np.float64)
        day.cb_mid = (day.cb_bid + day.cb_ask) / 2.0 if n > 0 else np.array([])

        # Bybit quotes
        n = len(self.bb_ts)
        day.bb_ts  = np.array(self.bb_ts[:n], dtype=np.int64)
        day.bb_bid = np.array(self.bb_bid[:n], dtype=np.float64)
        day.bb_ask = np.array(self.bb_ask[:n], dtype=np.float64)
        day.bb_mid = (day.bb_bid + day.bb_ask) / 2.0 if n > 0 else np.array([])

        # Coinbase trades
        n = len(self.ct_ts)
        day.ct_ts    = np.array(self.ct_ts[:n], dtype=np.int64)
        day.ct_price = np.array(self.ct_price[:n], dtype=np.float64)
        day.ct_qty   = np.array(self.ct_qty[:n], dtype=np.float64)
        day.ct_ibm   = np.array(self.ct_ibm[:n], dtype=bool)

        # Bybit trades
        n = len(self.bt_ts)
        day.bt_ts    = np.array(self.bt_ts[:n], dtype=np.int64)
        day.bt_price = np.array(self.bt_price[:n], dtype=np.float64)
        day.bt_qty   = np.array(self.bt_qty[:n], dtype=np.float64)
        day.bt_ibm   = np.array(self.bt_ibm[:n], dtype=bool)

        # Coinbase orderbook
        n = len(self.ob_cb_ts)
        if n > 0:
            ts = np.array(self.ob_cb_ts[:n], dtype=np.int64)
            bp = np.array(self.ob_cb_bid_prices[:n])
            bq = np.array(self.ob_cb_bid_qtys[:n])
            ap = np.array(self.ob_cb_ask_prices[:n])
            aq = np.array(self.ob_cb_ask_qtys[:n])
            day.ob_cb = _precompute_book(ts, bp, bq, ap, aq)
        else:
            day.ob_cb = {'ts': np.array([], dtype=np.int64),
                         'mid': np.array([]), 'spread_bps': np.array([]),
                         'imb_L1': np.array([]), 'imb_L5': np.array([]),
                         'bid_prices': np.empty((0, 20)), 'bid_qtys': np.empty((0, 20)),
                         'ask_prices': np.empty((0, 20)), 'ask_qtys': np.empty((0, 20))}

        # Bybit orderbook
        n = len(self.ob_bb_ts)
        if n > 0:
            ts = np.array(self.ob_bb_ts[:n], dtype=np.int64)
            bp = np.array(self.ob_bb_bid_prices[:n])
            bq = np.array(self.ob_bb_bid_qtys[:n])
            ap = np.array(self.ob_bb_ask_prices[:n])
            aq = np.array(self.ob_bb_ask_qtys[:n])
            day.ob_bb = _precompute_book(ts, bp, bq, ap, aq)
        else:
            day.ob_bb = {'ts': np.array([], dtype=np.int64),
                         'mid': np.array([]), 'spread_bps': np.array([]),
                         'imb_L1': np.array([]), 'imb_L5': np.array([]),
                         'bid_prices': np.empty((0, 20)), 'bid_qtys': np.empty((0, 20)),
                         'ask_prices': np.empty((0, 20)), 'ask_qtys': np.empty((0, 20))}

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
            "coinbase_quotes": len(self.cb_ts),
            "bybit_quotes": len(self.bb_ts),
            "coinbase_trades": len(self.ct_ts),
            "bybit_trades": len(self.bt_ts),
            "coinbase_depth": len(self.ob_cb_ts),
            "bybit_depth": len(self.ob_bb_ts),
        }

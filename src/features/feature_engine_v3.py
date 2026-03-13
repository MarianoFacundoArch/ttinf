"""
Feature engine V3 for BTC 5-minute block prediction (Polymarket).

Computes ~87 features for a given timestamp T within a 5-minute block.
All features look strictly backward from T. No future leakage.

Key differences vs V2:
  - Binary target (Up/Down per 5-min block) instead of 5-class
  - ref_price based on index_price (not futures mid) for block features
  - Block-state features (dist_to_open_z, crosses, etc.)
  - 5m metrics lagged 1 full bar to prevent leakage
  - ~87 features instead of ~212

Usage:
    from src.features.feature_engine_v3 import (
        load_day_data, build_ref_price, compute_features_v3, FEATURE_COLUMNS_V3
    )

    day = load_day_data("2026-03-01")
    ref = build_ref_price(day)
    features = compute_features_v3(day, ref, T_ms, block_start_ms, open_ref)
"""

import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# np.corrcoef warns when std=0 (constant prices). We handle this with
# np.isfinite checks, so silence the expected warning.
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning, module="numpy")
from pathlib import Path
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# DayData container
# ---------------------------------------------------------------------------

class DayData:
    """Container for one day's market data as numpy arrays."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_before(ts_arr, T_ms):
    """Index of last element with timestamp <= T_ms. Returns -1 if none."""
    if len(ts_arr) == 0:
        return -1
    idx = np.searchsorted(ts_arr, T_ms, side='right') - 1
    return idx if idx >= 0 else -1


def _slice_window(ts_arr, start_ms, end_ms):
    """Return (i_start, i_end) for elements in [start_ms, end_ms)."""
    return (
        np.searchsorted(ts_arr, start_ms, side='left'),
        np.searchsorted(ts_arr, end_ms, side='right'),
    )


def _safe_div(a, b, default=0.0):
    """a / b, returning default if b is zero."""
    return a / b if abs(b) > 1e-15 else default


# ---------------------------------------------------------------------------
# Pre-compute orderbook features (vectorized)
# ---------------------------------------------------------------------------

def _precompute_book(ts, bid_prices, bid_qtys, ask_prices, ask_qtys):
    """Pre-compute orderbook features for every snapshot."""
    mid = (bid_prices[:, 0] + ask_prices[:, 0]) / 2.0
    spread = ask_prices[:, 0] - bid_prices[:, 0]
    spread_bps = np.where(mid > 0, spread / mid * 10_000, 0.0)

    def _imbalance(bq, aq, levels):
        b = bq[:, :levels].sum(axis=1)
        a = aq[:, :levels].sum(axis=1)
        t = b + a
        return np.where(t > 0, (b - a) / t, 0.0)

    imb_L1 = _imbalance(bid_qtys, ask_qtys, 1)
    imb_L5 = _imbalance(bid_qtys, ask_qtys, 5)

    return {
        'ts': ts,
        'mid': mid,
        'spread_bps': spread_bps,
        'imb_L1': imb_L1,
        'imb_L5': imb_L5,
        'bid_prices': bid_prices,
        'bid_qtys': bid_qtys,
        'ask_prices': ask_prices,
        'ask_qtys': ask_qtys,
    }


# ---------------------------------------------------------------------------
# Load day data
# ---------------------------------------------------------------------------

def load_day_data(date_str, data_dir=None):
    """Load all parquet streams for a day into a DayData object."""
    base = Path(data_dir) if data_dir else DATA_DIR
    d = base / date_str
    day = DayData()

    # --- Trades futures ---
    df = pq.read_table(d / "trades_futures" / "full_day.parquet").to_pandas()
    day.tf_ts    = df["timestamp_ms"].values.astype(np.int64)
    day.tf_price = df["price"].values.astype(np.float64)
    day.tf_qty   = df["qty"].values.astype(np.float64)
    day.tf_ibm   = df["is_buyer_maker"].values
    del df

    # --- Trades spot ---
    df = pq.read_table(d / "trades_spot" / "full_day.parquet").to_pandas()
    day.ts_ts    = df["timestamp_ms"].values.astype(np.int64)
    day.ts_price = df["price"].values.astype(np.float64)
    day.ts_qty   = df["qty"].values.astype(np.float64)
    day.ts_ibm   = df["is_buyer_maker"].values
    del df

    # --- Bookticker futures ---
    df = pq.read_table(d / "bookticker_futures" / "full_day.parquet").to_pandas()
    day.bf_ts  = df["timestamp_ms"].values.astype(np.int64)
    day.bf_bid = df["best_bid_price"].values.astype(np.float64)
    day.bf_ask = df["best_ask_price"].values.astype(np.float64)
    day.bf_mid = (day.bf_bid + day.bf_ask) / 2.0
    del df

    # --- Bookticker spot ---
    df = pq.read_table(d / "bookticker_spot" / "full_day.parquet").to_pandas()
    day.bs_ts  = df["timestamp_ms"].values.astype(np.int64)
    day.bs_bid = df["best_bid_price"].values.astype(np.float64)
    day.bs_ask = df["best_ask_price"].values.astype(np.float64)
    day.bs_mid = (day.bs_bid + day.bs_ask) / 2.0
    del df

    # --- Orderbook futures (pre-computed) ---
    df = pq.read_table(d / "orderbook_futures" / "full_day.parquet").to_pandas()
    ts = df["timestamp_ms"].values.astype(np.int64)
    bp = np.column_stack([df[f"bid_price_{i}"].values for i in range(20)])
    bq = np.column_stack([df[f"bid_qty_{i}"].values for i in range(20)])
    ap = np.column_stack([df[f"ask_price_{i}"].values for i in range(20)])
    aq = np.column_stack([df[f"ask_qty_{i}"].values for i in range(20)])
    day.ob_fut = _precompute_book(ts, bp, bq, ap, aq)
    del df, ts, bp, bq, ap, aq

    # --- Orderbook spot (pre-computed) ---
    df = pq.read_table(d / "orderbook_spot" / "full_day.parquet").to_pandas()
    ts = df["timestamp_ms"].values.astype(np.int64)
    bp = np.column_stack([df[f"bid_price_{i}"].values for i in range(20)])
    bq = np.column_stack([df[f"bid_qty_{i}"].values for i in range(20)])
    ap = np.column_stack([df[f"ask_price_{i}"].values for i in range(20)])
    aq = np.column_stack([df[f"ask_qty_{i}"].values for i in range(20)])
    day.ob_spot = _precompute_book(ts, bp, bq, ap, aq)
    del df, ts, bp, bq, ap, aq

    # --- Mark price ---
    df = pq.read_table(d / "mark_price" / "full_day.parquet").to_pandas()
    day.mp_ts      = df["timestamp_ms"].values.astype(np.int64)
    day.mp_mark    = df["mark_price"].values.astype(np.float64)
    day.mp_index   = df["index_price"].values.astype(np.float64)
    day.mp_funding = df["funding_rate"].values.astype(np.float64)
    day.mp_next_ms = df["next_funding_time_ms"].values
    del df

    # --- Liquidations ---
    path = d / "liquidations" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        day.lq_ts     = df["timestamp_ms"].values.astype(np.int64)
        day.lq_is_buy = (df["side"].values.astype(str) == "buy")
        day.lq_qty    = df["qty"].values.astype(np.float64)
        del df
    else:
        day.lq_ts     = np.array([], dtype=np.int64)
        day.lq_is_buy = np.array([], dtype=bool)
        day.lq_qty    = np.array([], dtype=np.float64)

    # --- Metrics (5-min intervals) ---
    path = d / "metrics" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        day.mt_ts       = (pd.to_datetime(df["create_time"])
                           .astype(np.int64) // 1_000_000).values
        day.mt_ls_ratio = df["count_long_short_ratio"].values.astype(np.float64)
        day.mt_top_ls   = df["count_toptrader_long_short_ratio"].values.astype(np.float64)
        day.mt_taker_ls = df["sum_taker_long_short_vol_ratio"].values.astype(np.float64)
        # OI from metrics (sum_open_interest)
        if "sum_open_interest" in df.columns:
            day.mt_oi = df["sum_open_interest"].values.astype(np.float64)
        else:
            day.mt_oi = np.full(len(day.mt_ts), np.nan)
        del df
    else:
        day.mt_ts       = np.array([], dtype=np.int64)
        day.mt_ls_ratio = np.array([], dtype=np.float64)
        day.mt_top_ls   = np.array([], dtype=np.float64)
        day.mt_taker_ls = np.array([], dtype=np.float64)
        day.mt_oi       = np.array([], dtype=np.float64)

    # --- Cross-exchange: Coinbase quotes ---
    path = d / "coinbase_quotes" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        day.cb_ts  = df["timestamp_ms"].values.astype(np.int64)
        day.cb_bid = df["best_bid_price"].values.astype(np.float64)
        day.cb_ask = df["best_ask_price"].values.astype(np.float64)
        day.cb_mid = (day.cb_bid + day.cb_ask) / 2.0
        del df
    else:
        day.cb_ts  = np.array([], dtype=np.int64)
        day.cb_bid = np.array([], dtype=np.float64)
        day.cb_ask = np.array([], dtype=np.float64)
        day.cb_mid = np.array([], dtype=np.float64)

    # --- Cross-exchange: Bybit quotes ---
    path = d / "bybit_quotes" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        day.bb_ts  = df["timestamp_ms"].values.astype(np.int64)
        day.bb_bid = df["best_bid_price"].values.astype(np.float64)
        day.bb_ask = df["best_ask_price"].values.astype(np.float64)
        day.bb_mid = (day.bb_bid + day.bb_ask) / 2.0
        del df
    else:
        day.bb_ts  = np.array([], dtype=np.int64)
        day.bb_bid = np.array([], dtype=np.float64)
        day.bb_ask = np.array([], dtype=np.float64)
        day.bb_mid = np.array([], dtype=np.float64)

    # --- Cross-exchange: Coinbase trades ---
    path = d / "coinbase_trades" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        day.ct_ts    = df["timestamp_ms"].values.astype(np.int64)
        day.ct_price = df["price"].values.astype(np.float64)
        day.ct_qty   = df["qty"].values.astype(np.float64)
        day.ct_ibm   = df["is_buyer_maker"].values
        del df
    else:
        day.ct_ts    = np.array([], dtype=np.int64)
        day.ct_price = np.array([], dtype=np.float64)
        day.ct_qty   = np.array([], dtype=np.float64)
        day.ct_ibm   = np.array([], dtype=bool)

    # --- Cross-exchange: Bybit trades ---
    path = d / "bybit_trades" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        day.bt_ts    = df["timestamp_ms"].values.astype(np.int64)
        day.bt_price = df["price"].values.astype(np.float64)
        day.bt_qty   = df["qty"].values.astype(np.float64)
        day.bt_ibm   = df["is_buyer_maker"].values
        del df
    else:
        day.bt_ts    = np.array([], dtype=np.int64)
        day.bt_price = np.array([], dtype=np.float64)
        day.bt_qty   = np.array([], dtype=np.float64)
        day.bt_ibm   = np.array([], dtype=bool)

    # --- Cross-exchange: Coinbase orderbook (from incremental L2) ---
    path = d / "coinbase_book_l2" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        ts = df["timestamp_ms"].values.astype(np.int64)
        bp = np.column_stack([df[f"bid_price_{i}"].values for i in range(20)])
        bq = np.column_stack([df[f"bid_qty_{i}"].values for i in range(20)])
        ap = np.column_stack([df[f"ask_price_{i}"].values for i in range(20)])
        aq = np.column_stack([df[f"ask_qty_{i}"].values for i in range(20)])
        day.ob_cb = _precompute_book(ts, bp, bq, ap, aq)
        del df, ts, bp, bq, ap, aq
    else:
        day.ob_cb = {'ts': np.array([], dtype=np.int64),
                     'mid': np.array([]), 'spread_bps': np.array([]),
                     'imb_L1': np.array([]), 'imb_L5': np.array([]),
                     'bid_prices': np.empty((0, 20)), 'bid_qtys': np.empty((0, 20)),
                     'ask_prices': np.empty((0, 20)), 'ask_qtys': np.empty((0, 20))}

    # --- Cross-exchange: Bybit orderbook ---
    path = d / "bybit_orderbook" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        ts = df["timestamp_ms"].values.astype(np.int64)
        bp = np.column_stack([df[f"bid_price_{i}"].values for i in range(20)])
        bq = np.column_stack([df[f"bid_qty_{i}"].values for i in range(20)])
        ap = np.column_stack([df[f"ask_price_{i}"].values for i in range(20)])
        aq = np.column_stack([df[f"ask_qty_{i}"].values for i in range(20)])
        day.ob_bb = _precompute_book(ts, bp, bq, ap, aq)
        del df, ts, bp, bq, ap, aq
    else:
        day.ob_bb = {'ts': np.array([], dtype=np.int64),
                     'mid': np.array([]), 'spread_bps': np.array([]),
                     'imb_L1': np.array([]), 'imb_L5': np.array([]),
                     'bid_prices': np.empty((0, 20)), 'bid_qtys': np.empty((0, 20)),
                     'ask_prices': np.empty((0, 20)), 'ask_qtys': np.empty((0, 20))}

    return day


# ---------------------------------------------------------------------------
# Build ref_price_1s (canonical reference price series)
# ---------------------------------------------------------------------------

def build_ref_price(day):
    """
    Build ref_price_1s series from index_price, forward-filled at 1s.

    Returns dict with:
      'ts': int64 array of 1-second timestamps
      'price': float64 array of ref prices (forward-filled index_price)
    """
    if len(day.mp_ts) == 0:
        return {'ts': np.array([], dtype=np.int64),
                'price': np.array([], dtype=np.float64)}

    # Use index_price as primary, fallback to mark_price
    prices = day.mp_index.copy()
    bad = np.isnan(prices) | (prices <= 0)
    if bad.any():
        prices[bad] = day.mp_mark[bad]

    # Build 1-second grid from first to last mark_price timestamp
    first_ms = int(day.mp_ts[0])
    last_ms  = int(day.mp_ts[-1])
    # Align to 1-second boundaries
    first_s = (first_ms // 1000) * 1000
    last_s  = (last_ms // 1000) * 1000

    grid_ts = np.arange(first_s, last_s + 1000, 1000, dtype=np.int64)
    # Vectorized forward-fill: for each grid point, use last index_price <= that time
    indices = np.searchsorted(day.mp_ts, grid_ts, side='right') - 1
    valid = indices >= 0
    grid_price = np.full(len(grid_ts), np.nan, dtype=np.float64)
    grid_price[valid] = prices[indices[valid]]

    return {'ts': grid_ts, 'price': grid_price}


# ---------------------------------------------------------------------------
# GROUP A: Block state features (15 features)
# ---------------------------------------------------------------------------

def compute_block_state(ref, T_ms, block_start_ms, open_ref):
    """
    Block state features using ref_price series.

    Args:
        ref: dict from build_ref_price() with 'ts' and 'price'
        T_ms: current timestamp in ms
        block_start_ms: block start in ms
        open_ref: reference open price for this block
    """
    f = {}
    block_end_ms = block_start_ms + 300_000

    # seconds_to_expiry
    seconds_to_expiry = max(0, (block_end_ms - T_ms) / 1000.0)
    f["seconds_to_expiry"] = seconds_to_expiry

    # Current ref_price
    idx_now = _last_before(ref['ts'], T_ms)
    ref_now = ref['price'][idx_now] if idx_now >= 0 else open_ref

    # dist_to_open_bps
    f["dist_to_open_bps"] = _safe_div(ref_now - open_ref, open_ref) * 10_000

    # Get all ref_price values from block_start to T
    i_start = np.searchsorted(ref['ts'], block_start_ms, side='left')
    i_end   = np.searchsorted(ref['ts'], T_ms, side='right')
    block_prices = ref['price'][i_start:i_end]

    if len(block_prices) < 2:
        f["max_runup_since_open_bps"]     = 0.0
        f["max_drawdown_since_open_bps"]  = 0.0
        f["range_since_open_bps"]         = 0.0
        f["pct_time_above_open"]          = 0.5
        f["pct_time_above_open_last_30s"] = 0.5
        f["num_crosses_open"]             = 0.0
        f["last_cross_age_s"]             = seconds_to_expiry
        f["current_position_in_range"]    = 0.5
        f["realized_vol_since_open"]      = 0.0
        f["realized_vol_10s"]             = 0.0
        f["realized_vol_30s"]             = 0.0
        f["realized_vol_60s"]             = 0.0
        f["dist_to_open_z"]              = 0.0
        return f

    # Deviations from open in bps
    devs_bps = (block_prices - open_ref) / open_ref * 10_000

    f["max_runup_since_open_bps"]    = float(np.nanmax(devs_bps))
    f["max_drawdown_since_open_bps"] = float(np.nanmin(devs_bps))
    f["range_since_open_bps"]        = f["max_runup_since_open_bps"] - f["max_drawdown_since_open_bps"]

    # pct_time_above_open (full block so far)
    above = block_prices >= open_ref
    f["pct_time_above_open"] = float(above.mean())

    # pct_time_above_open_last_30s
    i_30s = np.searchsorted(ref['ts'], T_ms - 30_000, side='left')
    prices_30s = ref['price'][max(i_start, i_30s):i_end]
    if len(prices_30s) > 0:
        f["pct_time_above_open_last_30s"] = float((prices_30s >= open_ref).mean())
    else:
        f["pct_time_above_open_last_30s"] = f["pct_time_above_open"]

    # num_crosses_open
    above_arr = block_prices >= open_ref
    crosses = np.diff(above_arr.astype(np.int8))
    f["num_crosses_open"] = float(np.abs(crosses).sum())

    # last_cross_age_s
    cross_indices = np.where(np.abs(crosses) > 0)[0]
    if len(cross_indices) > 0:
        last_cross_idx = i_start + cross_indices[-1]
        last_cross_ms = ref['ts'][last_cross_idx]
        f["last_cross_age_s"] = (T_ms - last_cross_ms) / 1000.0
    else:
        f["last_cross_age_s"] = (T_ms - block_start_ms) / 1000.0

    # current_position_in_range
    pmin = np.nanmin(block_prices)
    pmax = np.nanmax(block_prices)
    f["current_position_in_range"] = _safe_div(ref_now - pmin, pmax - pmin, 0.5)

    # Realized volatility (std of 1s returns in bps)
    returns_1s = np.diff(block_prices) / block_prices[:-1] * 10_000
    returns_1s = returns_1s[np.isfinite(returns_1s)]
    f["realized_vol_since_open"] = float(np.std(returns_1s)) if len(returns_1s) > 1 else 0.0

    # Realized vol for shorter windows
    for window_s, name in [(10, "realized_vol_10s"), (30, "realized_vol_30s"), (60, "realized_vol_60s")]:
        i_w = np.searchsorted(ref['ts'], T_ms - window_s * 1000, side='left')
        w_prices = ref['price'][max(i_start, i_w):i_end]
        if len(w_prices) > 2:
            w_ret = np.diff(w_prices) / w_prices[:-1] * 10_000
            w_ret = w_ret[np.isfinite(w_ret)]
            f[name] = float(np.std(w_ret)) if len(w_ret) > 1 else 0.0
        else:
            f[name] = 0.0

    # dist_to_open_z = dist_to_open_bps / (sigma * sqrt(seconds_to_expiry))
    sigma = f["realized_vol_60s"] if f["realized_vol_60s"] > 0 else f["realized_vol_since_open"]
    if sigma > 0 and seconds_to_expiry > 0:
        f["dist_to_open_z"] = f["dist_to_open_bps"] / (sigma * np.sqrt(seconds_to_expiry))
    else:
        f["dist_to_open_z"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP B: Microstructure features (37 features)
# ---------------------------------------------------------------------------

def compute_microstructure(day, ref, T_ms):
    """Microstructure features from bookticker, orderbook, trades."""
    f = {}

    # --- Book snapshot (6 features) ---
    for pfx, ob in [("spot", day.ob_spot), ("fut", day.ob_fut)]:
        idx = _last_before(ob['ts'], T_ms)
        if idx >= 0:
            f[f"{pfx}_spread_bps"]   = float(ob['spread_bps'][idx])
            f[f"{pfx}_imbalance_L1"] = float(ob['imb_L1'][idx])
            f[f"{pfx}_imbalance_L5"] = float(ob['imb_L5'][idx])
        else:
            f[f"{pfx}_spread_bps"]   = np.nan
            f[f"{pfx}_imbalance_L1"] = np.nan
            f[f"{pfx}_imbalance_L5"] = np.nan

    # --- Basis (4 features) ---
    si = _last_before(day.bs_ts, T_ms)
    fi = _last_before(day.bf_ts, T_ms)
    if si >= 0 and fi >= 0:
        sm = day.bs_mid[si]
        fm = day.bf_mid[fi]
        f["basis_bps"] = _safe_div(fm - sm, sm) * 10_000

        for delta_s, name in [(5, "basis_delta_5s"), (10, "basis_delta_10s"), (30, "basis_delta_30s")]:
            si_p = _last_before(day.bs_ts, T_ms - delta_s * 1000)
            fi_p = _last_before(day.bf_ts, T_ms - delta_s * 1000)
            if si_p >= 0 and fi_p >= 0:
                sm_p = day.bs_mid[si_p]
                fm_p = day.bf_mid[fi_p]
                basis_past = _safe_div(fm_p - sm_p, sm_p) * 10_000
                f[name] = f["basis_bps"] - basis_past
            else:
                f[name] = 0.0
    else:
        f["basis_bps"]      = np.nan
        f["basis_delta_5s"] = np.nan
        f["basis_delta_10s"] = np.nan
        f["basis_delta_30s"] = np.nan

    # --- Trade flow: buy_pct (8 features) ---
    for pfx, ts_arr, qty_arr, ibm_arr in [
        ("spot", day.ts_ts, day.ts_qty, day.ts_ibm),
        ("fut",  day.tf_ts, day.tf_qty, day.tf_ibm),
    ]:
        for window_s in [5, 10, 30, 60]:
            i0, i1 = _slice_window(ts_arr, T_ms - window_s * 1000, T_ms)
            if i1 > i0:
                q = qty_arr[i0:i1]
                ibm = ibm_arr[i0:i1]
                total = q.sum()
                buy_vol = q[~ibm].sum()  # is_buyer_maker=False means taker buy
                f[f"{pfx}_buy_pct_{window_s}s"] = _safe_div(buy_vol, total, 0.5)
            else:
                f[f"{pfx}_buy_pct_{window_s}s"] = 0.5

    # --- Signed volume z-score (4 features) ---
    for pfx, ts_arr, qty_arr, ibm_arr in [
        ("spot", day.ts_ts, day.ts_qty, day.ts_ibm),
        ("fut",  day.tf_ts, day.tf_qty, day.tf_ibm),
    ]:
        # Compute signed vol for 60s (for std normalization)
        i0_60, i1_60 = _slice_window(ts_arr, T_ms - 60_000, T_ms)
        if i1_60 > i0_60:
            q_60 = qty_arr[i0_60:i1_60]
            ibm_60 = ibm_arr[i0_60:i1_60]
            signed_60 = np.where(ibm_60, -q_60, q_60)  # buyer_maker=True → sell pressure

            # Compute in 1-second buckets for std (vectorized)
            ts_60 = ts_arr[i0_60:i1_60]
            sec_buckets = ((ts_60 - ts_60[0]) // 1000).astype(np.intp)
            n_secs = int(sec_buckets[-1]) + 1 if len(sec_buckets) > 0 else 0
            if n_secs > 2:
                sv_per_sec = np.bincount(sec_buckets, weights=signed_60, minlength=n_secs)
                # Only count non-empty buckets
                nonempty = np.bincount(sec_buckets, minlength=n_secs) > 0
                sv_std = float(np.std(sv_per_sec[nonempty])) if nonempty.sum() > 2 else 0.0
            else:
                sv_std = 0.0

            for window_s in [10, 30]:
                i0_w, i1_w = _slice_window(ts_arr, T_ms - window_s * 1000, T_ms)
                if i1_w > i0_w:
                    q_w = qty_arr[i0_w:i1_w]
                    ibm_w = ibm_arr[i0_w:i1_w]
                    sv = float(np.where(ibm_w, -q_w, q_w).sum())
                    f[f"{pfx}_signed_vol_z_{window_s}s"] = _safe_div(sv, sv_std) if sv_std > 0 else 0.0
                else:
                    f[f"{pfx}_signed_vol_z_{window_s}s"] = 0.0
        else:
            f[f"{pfx}_signed_vol_z_10s"] = 0.0
            f[f"{pfx}_signed_vol_z_30s"] = 0.0

    # --- Trade intensity (6 features) ---
    for pfx, ts_arr in [("spot", day.ts_ts), ("fut", day.tf_ts)]:
        for window_s in [5, 10, 30]:
            i0, i1 = _slice_window(ts_arr, T_ms - window_s * 1000, T_ms)
            f[f"{pfx}_trade_intensity_{window_s}s"] = (i1 - i0) / window_s

    # --- Price returns from ref_price (4 features) ---
    ref_idx = _last_before(ref['ts'], T_ms)
    ref_now = ref['price'][ref_idx] if ref_idx >= 0 else np.nan
    for window_s in [5, 10, 30, 60]:
        ref_past_idx = _last_before(ref['ts'], T_ms - window_s * 1000)
        if ref_past_idx >= 0 and ref_idx >= 0:
            ref_past = ref['price'][ref_past_idx]
            f[f"return_{window_s}s"] = _safe_div(ref_now - ref_past, ref_past) * 10_000
        else:
            f[f"return_{window_s}s"] = 0.0

    # --- VWAP features (3 features) ---
    # block_vwap from futures trades since block_start
    # (We get block_start from the caller context, but here we approximate
    #  using T_ms aligned to 300s boundaries)
    block_start_approx = (T_ms // 300_000) * 300_000
    i0_blk, i1_blk = _slice_window(day.tf_ts, block_start_approx, T_ms)
    if i1_blk > i0_blk:
        prices_blk = day.tf_price[i0_blk:i1_blk]
        qty_blk    = day.tf_qty[i0_blk:i1_blk]
        total_qty  = qty_blk.sum()
        if total_qty > 0:
            block_vwap = (prices_blk * qty_blk).sum() / total_qty
            f["price_vs_block_vwap_bps"] = _safe_div(ref_now - block_vwap, block_vwap) * 10_000
            # We need open_ref for block_vwap_vs_open — caller passes it
            # For now store vwap, compute vs open in wrapper
            f["_block_vwap"] = block_vwap
        else:
            f["price_vs_block_vwap_bps"] = 0.0
            f["_block_vwap"] = np.nan
    else:
        f["price_vs_block_vwap_bps"] = 0.0
        f["_block_vwap"] = np.nan

    # --- Drift comparison (1 feature) ---
    # recent_drift_vs_block_drift: slope 30s vs slope since open
    if ref_idx >= 0 and len(ref['ts']) > 0:
        # Block drift (bps/s since open)
        elapsed_s = max(1, (T_ms - block_start_approx) / 1000.0)
        bs_idx = min(np.searchsorted(ref['ts'], block_start_approx, side='left'), len(ref['ts']) - 1)
        block_drift = f.get("_dist_to_open_bps_temp", _safe_div(ref_now - ref['price'][max(0, bs_idx)], ref_now) * 10_000) / elapsed_s

        # Recent drift (bps/s over 30s)
        ref_30_idx = _last_before(ref['ts'], T_ms - 30_000)
        if ref_30_idx >= 0:
            ref_30 = ref['price'][ref_30_idx]
            recent_drift = _safe_div(ref_now - ref_30, ref_30) * 10_000 / 30.0
        else:
            recent_drift = block_drift

        f["recent_drift_vs_block_drift"] = _safe_div(recent_drift, abs(block_drift), 0.0) if abs(block_drift) > 1e-10 else 0.0
    else:
        f["recent_drift_vs_block_drift"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP C: Regime context (12 features)
# ---------------------------------------------------------------------------

def compute_regime(day, T_ms):
    """Regime context features. 5m metrics use CLOSED bar -1 (lagged)."""
    f = {}

    # --- From mark_price stream (real-time) ---
    idx = _last_before(day.mp_ts, T_ms)
    if idx >= 0:
        f["funding_rate"]     = float(day.mp_funding[idx])
        f["mark_vs_index_bps"] = _safe_div(
            day.mp_mark[idx] - day.mp_index[idx],
            day.mp_index[idx]
        ) * 10_000

        # minutes_to_funding
        next_fund = day.mp_next_ms[idx]
        if not np.isnan(next_fund) and next_fund > 0:
            f["minutes_to_funding"] = max(0, (int(next_fund) - T_ms) / 60_000)
        else:
            # Funding every 8h: compute from time of day
            hour_frac = (T_ms % 86_400_000) / 3_600_000
            next_funding_hour = np.ceil(hour_frac / 8) * 8
            if next_funding_hour <= hour_frac:
                next_funding_hour += 8
            f["minutes_to_funding"] = (next_funding_hour - hour_frac) * 60
    else:
        f["funding_rate"]       = 0.0
        f["mark_vs_index_bps"]  = 0.0
        f["minutes_to_funding"] = 240.0

    # --- From metrics (5-min bars, LAGGED 1 full bar) ---
    # Find the last CLOSED bar before T (its timestamp must be strictly < T - 300_000
    # to ensure the bar is fully closed)
    # Actually: metrics timestamps are at bar close. We want the last one
    # that is at least 1 full bar behind T.
    if len(day.mt_ts) > 0:
        # Last metrics bar with timestamp <= T
        mt_idx = _last_before(day.mt_ts, T_ms)
        # But we want the bar BEFORE that (lag 1), because the current bar
        # may not be fully closed yet
        mt_idx_lagged = mt_idx - 1 if mt_idx >= 1 else -1

        if mt_idx_lagged >= 0:
            f["long_short_ratio"]          = float(day.mt_ls_ratio[mt_idx_lagged])
            f["top_trader_long_short"]     = float(day.mt_top_ls[mt_idx_lagged])
            f["taker_long_short_vol_ratio"] = float(day.mt_taker_ls[mt_idx_lagged])

            # long_short_change (vs bar before that)
            if mt_idx_lagged >= 1:
                f["long_short_change"] = float(
                    day.mt_ls_ratio[mt_idx_lagged] - day.mt_ls_ratio[mt_idx_lagged - 1]
                )
            else:
                f["long_short_change"] = 0.0

            # OI change pct (lagged)
            if mt_idx_lagged >= 1 and not np.isnan(day.mt_oi[mt_idx_lagged]):
                oi_now = day.mt_oi[mt_idx_lagged]
                oi_prev = day.mt_oi[mt_idx_lagged - 1]
                f["oi_change_pct_5m"] = _safe_div(oi_now - oi_prev, oi_prev) * 100
            else:
                f["oi_change_pct_5m"] = 0.0
        else:
            f["long_short_ratio"]          = np.nan
            f["top_trader_long_short"]     = np.nan
            f["taker_long_short_vol_ratio"] = np.nan
            f["long_short_change"]         = 0.0
            f["oi_change_pct_5m"]          = 0.0
    else:
        f["long_short_ratio"]          = np.nan
        f["top_trader_long_short"]     = np.nan
        f["taker_long_short_vol_ratio"] = np.nan
        f["long_short_change"]         = 0.0
        f["oi_change_pct_5m"]          = 0.0

    # --- Liquidation flags ---
    # liq_any_5m
    i0_5m, i1_5m = _slice_window(day.lq_ts, T_ms - 300_000, T_ms)
    f["liq_any_5m"] = 1.0 if (i1_5m - i0_5m) > 0 else 0.0

    # liq_any_30s
    i0_30, i1_30 = _slice_window(day.lq_ts, T_ms - 30_000, T_ms)
    f["liq_any_30s"] = 1.0 if (i1_30 - i0_30) > 0 else 0.0

    # liq_burst_flag (multiple liquidations in 10s)
    i0_10, i1_10 = _slice_window(day.lq_ts, T_ms - 10_000, T_ms)
    f["liq_burst_flag"] = 1.0 if (i1_10 - i0_10) >= 3 else 0.0

    # liq_last_side
    liq_idx = _last_before(day.lq_ts, T_ms)
    if liq_idx >= 0 and (T_ms - day.lq_ts[liq_idx]) < 300_000:
        f["liq_last_side"] = 1.0 if day.lq_is_buy[liq_idx] else -1.0
    else:
        f["liq_last_side"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP D: Block history (10 features)
# ---------------------------------------------------------------------------

def compute_block_history(block_results):
    """
    Features from previous closed blocks.

    Args:
        block_results: list of dicts, each with 'return_bps' and 'result' (1/0),
                       ordered most recent first. At least 6 entries expected.
    """
    f = {}

    # Pad if not enough history
    while len(block_results) < 6:
        block_results.append({'return_bps': 0.0, 'result': 0})

    # Explicit (last 3)
    f["prev_1_return_bps"] = block_results[0]['return_bps']
    f["prev_2_return_bps"] = block_results[1]['return_bps']
    f["prev_3_return_bps"] = block_results[2]['return_bps']
    f["prev_1_result"]     = float(block_results[0]['result'])

    # Aggregates (last 6 = 30 min)
    returns_6 = np.array([b['return_bps'] for b in block_results[:6]])
    results_6 = np.array([b['result'] for b in block_results[:6]])

    f["up_pct_last_6"]          = float(results_6.mean())
    f["mean_return_last_6"]     = float(returns_6.mean())
    f["vol_last_6"]             = float(returns_6.std()) if len(returns_6) > 1 else 0.0
    f["max_abs_return_last_6"]  = float(np.abs(returns_6).max())

    # Realized vol last 15 min (3 blocks) and 30 min (6 blocks)
    # These use the returns of previous blocks
    returns_3 = returns_6[:3]
    f["prev_15m_realized_vol"] = float(returns_3.std()) if len(returns_3) > 1 else 0.0
    f["prev_30m_realized_vol"] = float(returns_6.std()) if len(returns_6) > 1 else 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP E: Temporal features (6 features)
# ---------------------------------------------------------------------------

def compute_temporal(T_ms):
    """Temporal features from timestamp."""
    f = {}

    ms_in_day = T_ms % 86_400_000
    hour = ms_in_day / 3_600_000

    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Day of week (0=Mon ... 6=Sun)
    # Approximate: 1970-01-01 was Thursday (3)
    day_num = (T_ms // 86_400_000 + 3) % 7
    f["dow_sin"] = np.sin(2 * np.pi * day_num / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * day_num / 7.0)

    # is_us_market_hours: 13:30-20:00 UTC
    f["is_us_market_hours"] = 1.0 if 13.5 <= hour < 20.0 else 0.0

    # Direct integer temporals for LightGBM (trees split better on integers)
    f["minute_of_day"] = float(int(ms_in_day // 60_000))  # 0-1439
    f["day_of_week"] = float(day_num)  # 0=Mon, 6=Sun

    # minutes_to_funding is computed in regime group, not duplicated here

    return f


# ---------------------------------------------------------------------------
# GROUP F: Data quality features (7 features)
# ---------------------------------------------------------------------------

def compute_data_quality(day, T_ms, open_ref_age_ms):
    """Data quality/freshness features."""
    f = {}

    # Age of each stream
    def _age(ts_arr):
        idx = _last_before(ts_arr, T_ms)
        if idx >= 0:
            return float(T_ms - ts_arr[idx])
        return 999_999.0

    f["age_bookticker_ms"]  = _age(day.bf_ts)
    f["age_markprice_ms"]   = _age(day.mp_ts)
    f["age_depth_ms"]       = _age(day.ob_fut['ts'])
    f["age_trades_ms"]      = _age(day.tf_ts)

    # Missing streams (no data in last 5s)
    threshold_ms = 5000
    missing = 0
    if f["age_bookticker_ms"] > threshold_ms:
        missing += 1
    if f["age_markprice_ms"] > threshold_ms:
        missing += 1
    if f["age_depth_ms"] > threshold_ms:
        missing += 1
    if f["age_trades_ms"] > threshold_ms:
        missing += 1
    f["missing_stream_count"] = float(missing)

    # open_ref quality
    f["open_ref_quality_flag"] = 1.0 if open_ref_age_ms < 3000 else 0.0

    # Core streams fresh (ref_price aka markprice + bookticker + trades all < 3s)
    f["core_streams_fresh_flag"] = 1.0 if (
        f["age_bookticker_ms"] < 3000 and
        f["age_markprice_ms"] < 3000 and
        f["age_trades_ms"] < 3000
    ) else 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP G: Derived / pre-computed features (5 features)
# Mathematical combinations that trees can't learn efficiently via splits.
# ---------------------------------------------------------------------------

def compute_derived(feats, ref, T_ms, block_start_ms):
    """
    Derived features computed from other features.
    These are mathematical transformations that LightGBM trees
    would need hundreds of splits to approximate.
    """
    f = {}
    T_total = 300.0  # block duration in seconds
    seconds_to_expiry = feats.get("seconds_to_expiry", 0.0)
    t_elapsed = T_total - seconds_to_expiry
    dist_bps = feats.get("dist_to_open_bps", 0.0)
    z = feats.get("dist_to_open_z", 0.0)
    sigma = feats.get("realized_vol_60s", 0.0)
    if sigma <= 0:
        sigma = feats.get("realized_vol_since_open", 0.0)

    # 1. brownian_prob: Φ(z) — the Brownian baseline probability
    f["brownian_prob"] = float(norm.cdf(np.clip(z, -10, 10)))

    # 2. brownian_prob_drift: Φ(z_adjusted) — Brownian with 30s drift
    mu_hat = feats.get("return_30s", 0.0) / 30.0  # bps per second
    if sigma > 0 and seconds_to_expiry > 0:
        z_drift = (dist_bps + mu_hat * seconds_to_expiry) / (sigma * np.sqrt(seconds_to_expiry))
        f["brownian_prob_drift"] = float(norm.cdf(np.clip(z_drift, -10, 10)))
    else:
        f["brownian_prob_drift"] = f["brownian_prob"]

    # 3. z_velocity: rate of change of z over last 5 seconds
    return_5s = feats.get("return_5s", 0.0)
    if sigma > 0 and seconds_to_expiry > 0:
        z_prev = (dist_bps - return_5s) / (sigma * np.sqrt(seconds_to_expiry + 5))
        f["z_velocity"] = (z - z_prev) / 5.0
    else:
        f["z_velocity"] = 0.0

    # 4. bridge_variance: σ² × t_elapsed × t_remaining / T_total
    if sigma > 0:
        f["bridge_variance"] = sigma**2 * t_elapsed * seconds_to_expiry / T_total
    else:
        f["bridge_variance"] = 0.0

    # 5. vol_ratio: realized_vol_since_open / realized_vol_60s
    vol_open = feats.get("realized_vol_since_open", 0.0)
    vol_60 = feats.get("realized_vol_60s", 0.0)
    if vol_60 > 0:
        f["vol_ratio"] = vol_open / vol_60
    else:
        f["vol_ratio"] = 1.0

    return f


# ---------------------------------------------------------------------------
# GROUP H: Flow dynamics (9 features)
# Trade size distribution, flow acceleration, orderbook dynamics.
# All relative/normalized — no absolute dollar values.
# ---------------------------------------------------------------------------

def compute_flow_dynamics(day, T_ms, block_start_ms):
    """
    Flow dynamics features capturing informed trading, flow persistence,
    and orderbook changes over time.
    """
    f = {}

    # --- 1. Trade size distribution (3 features) ---
    # Use futures trades (higher volume, more informative)
    i0_30, i1_30 = _slice_window(day.tf_ts, T_ms - 30_000, T_ms)
    if i1_30 - i0_30 > 10:
        qty_30 = day.tf_qty[i0_30:i1_30]
        ibm_30 = day.tf_ibm[i0_30:i1_30]

        # Threshold for "large" = 90th percentile of recent trades
        p90 = np.percentile(qty_30, 90)
        large_mask = qty_30 >= p90

        # large_trade_pct_30s: fraction of VOLUME from large trades
        total_vol = qty_30.sum()
        large_vol = qty_30[large_mask].sum()
        f["large_trade_pct_30s"] = _safe_div(large_vol, total_vol, 0.0)

        # large_trade_buy_pct_30s: among large trades, what % are buys
        if large_mask.sum() > 0:
            large_qty = qty_30[large_mask]
            large_ibm = ibm_30[large_mask]
            large_buy_vol = large_qty[~large_ibm].sum()  # ~ibm = taker buy
            f["large_trade_buy_pct_30s"] = _safe_div(large_buy_vol, large_qty.sum(), 0.5)
        else:
            f["large_trade_buy_pct_30s"] = 0.5

        # trade_size_cv_30s: coefficient of variation of trade sizes
        mean_qty = qty_30.mean()
        if mean_qty > 0:
            f["trade_size_cv_30s"] = float(qty_30.std() / mean_qty)
        else:
            f["trade_size_cv_30s"] = 0.0
    else:
        f["large_trade_pct_30s"] = 0.0
        f["large_trade_buy_pct_30s"] = 0.5
        f["trade_size_cv_30s"] = 0.0

    # --- 2. Flow acceleration (3 features) ---

    # buy_pressure_acceleration: buy_pct_5s - buy_pct_30s
    # (already computed in microstructure, but we compute here to be self-contained)
    for window_s, ibm_arr, ts_arr, qty_arr in [
        (5, day.tf_ibm, day.tf_ts, day.tf_qty),
    ]:
        i0_5, i1_5 = _slice_window(ts_arr, T_ms - 5_000, T_ms)
        if i1_5 > i0_5:
            q5 = qty_arr[i0_5:i1_5]
            ibm5 = ibm_arr[i0_5:i1_5]
            bp_5s = _safe_div(q5[~ibm5].sum(), q5.sum(), 0.5)
        else:
            bp_5s = 0.5

        if i1_30 > i0_30:
            q30 = day.tf_qty[i0_30:i1_30]
            ibm30 = day.tf_ibm[i0_30:i1_30]
            bp_30s = _safe_div(q30[~ibm30].sum(), q30.sum(), 0.5)
        else:
            bp_30s = 0.5

    f["buy_pressure_acceleration"] = bp_5s - bp_30s

    # volume_acceleration: trades/sec in 10s / trades/sec in 60s
    i0_10, i1_10 = _slice_window(day.tf_ts, T_ms - 10_000, T_ms)
    i0_60, i1_60 = _slice_window(day.tf_ts, T_ms - 60_000, T_ms)
    tps_10 = (i1_10 - i0_10) / 10.0
    tps_60 = (i1_60 - i0_60) / 60.0
    f["volume_acceleration"] = _safe_div(tps_10, tps_60, 1.0)

    # cumulative_volume_delta_norm: net buy volume / total volume since block start
    i0_blk, i1_blk = _slice_window(day.tf_ts, block_start_ms, T_ms)
    if i1_blk > i0_blk:
        q_blk = day.tf_qty[i0_blk:i1_blk]
        ibm_blk = day.tf_ibm[i0_blk:i1_blk]
        buy_vol = q_blk[~ibm_blk].sum()
        sell_vol = q_blk[ibm_blk].sum()
        total_vol = buy_vol + sell_vol
        f["cumulative_volume_delta_norm"] = _safe_div(buy_vol - sell_vol, total_vol, 0.0)
    else:
        f["cumulative_volume_delta_norm"] = 0.0

    # --- 3. Orderbook dynamics (3 features) ---
    ob = day.ob_fut

    # Current orderbook snapshot
    idx_now = _last_before(ob['ts'], T_ms)
    idx_5s = _last_before(ob['ts'], T_ms - 5_000)
    idx_10s = _last_before(ob['ts'], T_ms - 10_000)

    if idx_now >= 0 and idx_5s >= 0:
        # bid_depth_change_pct_5s and ask_depth_change_pct_5s
        # We use imbalance as a proxy for relative depth change
        # imb = (bid - ask) / (bid + ask), so delta_imb captures depth shift
        imb_now = float(ob['imb_L5'][idx_now])
        imb_5s = float(ob['imb_L5'][idx_5s])
        f["bid_depth_change_pct_5s"] = imb_now - imb_5s
        # Positive = bids grew relative to asks (bullish)
        # Negative = asks grew relative to bids (bearish)

        # We split this into bid and ask sides using spread as additional signal
        # If imbalance went up AND spread tightened → bids filling in
        # If imbalance went down AND spread widened → bids pulling out
        spread_now = float(ob['spread_bps'][idx_now])
        spread_5s = float(ob['spread_bps'][idx_5s])
        f["ask_depth_change_pct_5s"] = -(imb_now - imb_5s)
        # Mirror of bid change (if bids grew, asks shrank relatively)
    else:
        f["bid_depth_change_pct_5s"] = 0.0
        f["ask_depth_change_pct_5s"] = 0.0

    if idx_now >= 0 and idx_10s >= 0:
        spread_now = float(ob['spread_bps'][idx_now])
        spread_10s = float(ob['spread_bps'][idx_10s])
        f["spread_change_ratio_10s"] = _safe_div(spread_now, spread_10s, 1.0)
    else:
        f["spread_change_ratio_10s"] = 1.0

    return f


# ---------------------------------------------------------------------------
# Group J: VPIN — Volume-Synchronized Probability of Informed Trading (4 features)
# ---------------------------------------------------------------------------

def compute_vpin(day, T_ms):
    """VPIN features using volume-synchronized buckets.

    Unlike time-based buy_pct, VPIN normalizes for varying trade intensity
    and detects informed flow even when trade counts are balanced but
    volumes are not.
    """
    f = {}

    def _vpin_from_trades(ts, qty, ibm, start_ms, end_ms, n_buckets=10):
        """Compute VPIN and signed VPIN from trades in a time window.

        Proper implementation with trade splitting via np.interp:
        buy/sell volumes are interpolated at exact bucket boundaries,
        ensuring all volume is used and buckets are perfectly equal-volume.
        Returns (vpin, signed_vpin).
        """
        i0 = np.searchsorted(ts, start_ms, side='left')
        i1 = np.searchsorted(ts, end_ms, side='right')
        if i1 - i0 < 10:
            return np.nan, np.nan

        q = qty[i0:i1]
        is_bm = ibm[i0:i1]
        total_vol = q.sum()
        if total_vol <= 0:
            return np.nan, np.nan

        bucket_vol = total_vol / n_buckets
        if bucket_vol <= 0:
            return np.nan, np.nan

        # Cumulative volumes at trade boundaries (the "x axis")
        cum_total = np.cumsum(q)
        cum_buy = np.cumsum(np.where(is_bm, 0.0, q))
        # Prepend 0 for the origin point
        vol_axis = np.empty(len(q) + 1)
        vol_axis[0] = 0.0
        vol_axis[1:] = cum_total
        buy_axis = np.empty(len(q) + 1)
        buy_axis[0] = 0.0
        buy_axis[1:] = cum_buy

        # Bucket boundaries at exact volume thresholds
        # [bucket_vol, 2*bucket_vol, ..., n*bucket_vol]
        thresholds = np.arange(1, n_buckets + 1) * bucket_vol

        # Interpolate buy volume at each boundary
        # np.interp does piecewise-linear interpolation = trade splitting
        buy_at = np.interp(thresholds, vol_axis, buy_axis)

        # Buy/sell per bucket (prepend 0 for diff)
        buy_per = np.diff(np.concatenate(([0.0], buy_at)))
        sell_per = bucket_vol - buy_per  # total per bucket = bucket_vol

        # Imbalances
        abs_imb = np.abs(buy_per - sell_per) / bucket_vol
        signed_imb = (buy_per - sell_per) / bucket_vol

        return float(abs_imb.mean()), float(signed_imb.mean())

    # VPIN over 30s (short-term)
    vpin_30, signed_30 = _vpin_from_trades(
        day.tf_ts, day.tf_qty, day.tf_ibm, T_ms - 30_000, T_ms, n_buckets=10)

    # VPIN over 120s (baseline)
    vpin_120, _ = _vpin_from_trades(
        day.tf_ts, day.tf_qty, day.tf_ibm, T_ms - 120_000, T_ms, n_buckets=20)

    f['vpin_30s'] = vpin_30
    f['vpin_signed_30s'] = signed_30
    f['vpin_120s'] = vpin_120
    f['vpin_spike'] = _safe_div(vpin_30, vpin_120, 1.0)

    return f


# ---------------------------------------------------------------------------
# Group I: Orderbook at open price (4 features)
# ---------------------------------------------------------------------------

def compute_orderbook_at_open(day, T_ms, open_ref):
    """Orderbook features relative to the block open price.

    Measures support/resistance at the critical level that determines
    whether the block closes UP or DOWN.
    """
    f = {}
    THRESHOLD_BPS = 10
    ob = day.ob_fut

    has_raw = ('bid_prices' in ob and len(ob['bid_prices']) > 0)

    if not has_raw or open_ref <= 0 or len(ob['ts']) == 0:
        f['ob_bid_vol_near_open'] = 0.0
        f['ob_ask_vol_near_open'] = 0.0
        f['ob_imbalance_at_open'] = 0.0
        f['ob_volume_to_cross_open_pct'] = 0.0
        return f

    idx = _last_before(ob['ts'], T_ms)
    if idx < 0:
        f['ob_bid_vol_near_open'] = 0.0
        f['ob_ask_vol_near_open'] = 0.0
        f['ob_imbalance_at_open'] = 0.0
        f['ob_volume_to_cross_open_pct'] = 0.0
        return f

    bp = ob['bid_prices'][idx]  # (20,)
    bq = ob['bid_qtys'][idx]
    ap = ob['ask_prices'][idx]
    aq = ob['ask_qtys'][idx]

    threshold_price = open_ref * THRESHOLD_BPS / 10_000

    # Bid volume near open: levels within [open - threshold, open]
    bid_valid = bp > 0
    bid_near = bid_valid & (bp >= open_ref - threshold_price) & (bp <= open_ref)
    bid_near_vol = bq[bid_near].sum()
    total_bid_vol = bq[bid_valid].sum()

    # Ask volume near open: levels within [open, open + threshold]
    ask_valid = ap > 0
    ask_near = ask_valid & (ap >= open_ref) & (ap <= open_ref + threshold_price)
    ask_near_vol = aq[ask_near].sum()
    total_ask_vol = aq[ask_valid].sum()

    # Feature 1: Bid volume concentration near open (support)
    f['ob_bid_vol_near_open'] = float(
        bid_near_vol / total_bid_vol if total_bid_vol > 0 else 0.0)

    # Feature 2: Ask volume concentration near open (resistance)
    f['ob_ask_vol_near_open'] = float(
        ask_near_vol / total_ask_vol if total_ask_vol > 0 else 0.0)

    # Feature 3: Imbalance at open level
    total_near = bid_near_vol + ask_near_vol
    f['ob_imbalance_at_open'] = float(
        (bid_near_vol - ask_near_vol) / total_near if total_near > 0 else 0.0)

    # Feature 4: Volume between current price and open / total on that side
    # Measures how much needs to be traded through to cross the open
    mid = (bp[0] + ap[0]) / 2.0 if bp[0] > 0 and ap[0] > 0 else 0.0
    if mid > open_ref and total_bid_vol > 0:
        # Price above open: bid volume at levels >= open = support
        cross_vol = bq[bid_valid & (bp >= open_ref)].sum()
        f['ob_volume_to_cross_open_pct'] = float(cross_vol / total_bid_vol)
    elif mid < open_ref and total_ask_vol > 0:
        # Price below open: ask volume at levels <= open = resistance
        cross_vol = aq[ask_valid & (ap <= open_ref)].sum()
        f['ob_volume_to_cross_open_pct'] = float(cross_vol / total_ask_vol)
    else:
        f['ob_volume_to_cross_open_pct'] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP L: Pre-block features (9 features) — Ronda 4
#
# What happened in the 30-60 seconds BEFORE the block started?
# Fills the "blind spot" between coarse prev_block returns and block start.
# ---------------------------------------------------------------------------

PREBLOCK_COLUMNS = [
    "pb_momentum_30s",      # Price return in 30s before block open
    "pb_momentum_10s",      # Price return in 10s before block open
    "pb_acceleration",      # momentum_10s - momentum_30s
    "pb_net_flow_30s",      # (buy_vol - sell_vol) / total_vol last 30s
    "pb_net_flow_10s",      # Same but last 10s
    "pb_trade_intensity",   # trades/sec last 30s ÷ trades/sec last 5min
    "pb_vol_30s",           # Realized vol last 30s (std of 1s returns)
    "pb_vol_ratio",         # vol_30s ÷ vol_5min
    "pb_liq_pressure_60s",  # Liq volume / total trade volume last 60s
]


def compute_preblock(day, block_start_ms):
    """
    Pre-block features: what was happening 30-60s before the block opened.

    Uses futures trades (highest liquidity) for momentum and flow.
    These features are STATIC throughout the block — computed once at block start.
    """
    f = {}

    # Time windows (all relative to block_start_ms, looking BACKWARDS)
    t0 = block_start_ms
    t_10 = t0 - 10_000    # 10s before
    t_30 = t0 - 30_000    # 30s before
    t_60 = t0 - 60_000    # 60s before
    t_5m = t0 - 300_000   # 5min before

    # --- Momentum: price return before block open ---
    # Use futures trades for best liquidity
    ts = day.tf_ts
    prices = day.tf_price

    # Price at block open (last trade before t0)
    idx_0 = _last_before(ts, t0)
    # Price 10s before
    idx_10 = _last_before(ts, t_10)
    # Price 30s before
    idx_30 = _last_before(ts, t_30)

    if idx_0 >= 0 and idx_30 >= 0 and prices[idx_0] > 0 and prices[idx_30] > 0:
        p0 = prices[idx_0]
        p30 = prices[idx_30]
        f["pb_momentum_30s"] = (p0 - p30) / p30 * 10_000  # bps
    else:
        f["pb_momentum_30s"] = 0.0

    if idx_0 >= 0 and idx_10 >= 0 and prices[idx_0] > 0 and prices[idx_10] > 0:
        p0 = prices[idx_0]
        p10 = prices[idx_10]
        f["pb_momentum_10s"] = (p0 - p10) / p10 * 10_000  # bps
    else:
        f["pb_momentum_10s"] = 0.0

    # Acceleration: is the move speeding up or slowing down?
    f["pb_acceleration"] = f["pb_momentum_10s"] - f["pb_momentum_30s"]

    # --- Net flow: who's the aggressor? ---
    i_start_30, i_end_30 = _slice_window(ts, t_30, t0)
    if i_end_30 > i_start_30:
        qty_slice = day.tf_qty[i_start_30:i_end_30]
        ibm_slice = day.tf_ibm[i_start_30:i_end_30]
        buy_vol = qty_slice[~ibm_slice].sum()   # is_buyer_maker=False → buyer is taker
        sell_vol = qty_slice[ibm_slice].sum()    # is_buyer_maker=True → seller is taker
        total = buy_vol + sell_vol
        f["pb_net_flow_30s"] = float((buy_vol - sell_vol) / total) if total > 0 else 0.0
    else:
        f["pb_net_flow_30s"] = 0.0

    i_start_10, i_end_10 = _slice_window(ts, t_10, t0)
    if i_end_10 > i_start_10:
        qty_slice = day.tf_qty[i_start_10:i_end_10]
        ibm_slice = day.tf_ibm[i_start_10:i_end_10]
        buy_vol = qty_slice[~ibm_slice].sum()
        sell_vol = qty_slice[ibm_slice].sum()
        total = buy_vol + sell_vol
        f["pb_net_flow_10s"] = float((buy_vol - sell_vol) / total) if total > 0 else 0.0
    else:
        f["pb_net_flow_10s"] = 0.0

    # --- Trade intensity: is the market unusually active? ---
    n_trades_30s = i_end_30 - i_start_30
    i_start_5m, i_end_5m = _slice_window(ts, t_5m, t0)
    n_trades_5m = i_end_5m - i_start_5m

    rate_30s = n_trades_30s / 30.0
    rate_5m = n_trades_5m / 300.0 if n_trades_5m > 0 else 1.0
    f["pb_trade_intensity"] = float(rate_30s / rate_5m) if rate_5m > 0 else 1.0

    # --- Realized vol last 30s (std of 1-second returns) ---
    # Build 1-second price series in [t_30, t0] (vectorized)
    if i_end_30 - i_start_30 > 10:
        ts_slice = ts[i_start_30:i_end_30]
        px_slice = prices[i_start_30:i_end_30]
        sec_grid = np.arange(t_30, t0, 1000)
        sec_idx = np.searchsorted(ts_slice, sec_grid, side='right') - 1
        valid = sec_idx >= 0
        if valid.sum() > 2:
            sec_prices = px_slice[sec_idx[valid]]
            returns = np.diff(sec_prices) / sec_prices[:-1]
            f["pb_vol_30s"] = float(np.std(returns)) * 10_000
        else:
            f["pb_vol_30s"] = 0.0
    else:
        f["pb_vol_30s"] = 0.0

    # --- Vol ratio: vol_30s / vol_5min ---
    # Compute 5min vol the same way (vectorized)
    if i_end_5m - i_start_5m > 10:
        ts_5m_slice = ts[i_start_5m:i_end_5m]
        px_5m_slice = prices[i_start_5m:i_end_5m]
        sec_grid_5m = np.arange(t_5m, t0, 5000)
        sec_idx_5m = np.searchsorted(ts_5m_slice, sec_grid_5m, side='right') - 1
        valid_5m = sec_idx_5m >= 0
        if valid_5m.sum() > 2:
            sec_prices_5m = px_5m_slice[sec_idx_5m[valid_5m]]
            returns_5m = np.diff(sec_prices_5m) / sec_prices_5m[:-1]
            vol_5m = float(np.std(returns_5m)) * 10_000
            f["pb_vol_ratio"] = f["pb_vol_30s"] / vol_5m if vol_5m > 0 else 1.0
        else:
            f["pb_vol_ratio"] = 1.0
    else:
        f["pb_vol_ratio"] = 1.0

    # --- Liquidation pressure: forced vs organic ---
    if hasattr(day, 'lq_ts') and len(day.lq_ts) > 0:
        li_s, li_e = _slice_window(day.lq_ts, t_60, t0)
        liq_vol = day.lq_qty[li_s:li_e].sum() if li_e > li_s else 0.0
        # Total trade volume in last 60s
        ti_s, ti_e = _slice_window(ts, t_60, t0)
        trade_vol = day.tf_qty[ti_s:ti_e].sum() if ti_e > ti_s else 0.0
        f["pb_liq_pressure_60s"] = float(liq_vol / trade_vol) if trade_vol > 0 else 0.0
    else:
        f["pb_liq_pressure_60s"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP M: Previous block structure (2 features) — Ronda 5
#
# HOW the previous block behaved, not just its return.
# Static per block — computed once from trades in [block_start - 300s, block_start].
# ---------------------------------------------------------------------------

PREV_BLOCK_COLUMNS = [
    "prev_trend_linearity",   # R² of price vs time — clean trend vs choppy
    "prev_volume_profile",    # vol last 60s / vol first 60s of prev block
]


def compute_prev_block_structure(day, block_start_ms):
    """Structure of the previous block (300s window before block_start).

    These capture HOW the previous block behaved — its shape, not just
    its return. Static throughout the current block.
    """
    f = {}
    BLOCK_DUR = 300_000  # 5 minutes

    prev_start = block_start_ms - BLOCK_DUR
    prev_end = block_start_ms

    ts = day.tf_ts
    prices = day.tf_price

    i0, i1 = _slice_window(ts, prev_start, prev_end)

    if i1 - i0 < 20:
        f["prev_trend_linearity"] = 0.0
        f["prev_volume_profile"] = 1.0
        return f

    # --- prev_trend_linearity: R² of price vs time ---
    t_slice = ts[i0:i1].astype(np.float64)
    p_slice = prices[i0:i1].astype(np.float64)
    # Normalize time to [0, 1] for numerical stability
    t_norm = (t_slice - t_slice[0]) / max(1.0, float(t_slice[-1] - t_slice[0]))
    # R² = correlation² (faster than full regression)
    corr_matrix = np.corrcoef(t_norm, p_slice)
    r = corr_matrix[0, 1]
    f["prev_trend_linearity"] = float(r * r) if np.isfinite(r) else 0.0

    # --- prev_volume_profile: vol last 60s / vol first 60s ---
    qty = day.tf_qty[i0:i1]
    ts_local = ts[i0:i1]
    first_60_end = prev_start + 60_000
    last_60_start = prev_end - 60_000

    first_mask = ts_local < first_60_end
    last_mask = ts_local >= last_60_start
    vol_first = qty[first_mask].sum()
    vol_last = qty[last_mask].sum()
    f["prev_volume_profile"] = float(vol_last / vol_first) if vol_first > 0 else 1.0

    return f


# ---------------------------------------------------------------------------
# GROUP N: Market microstructure dynamics (4 features) — Ronda 5
#
# Features from data we already collect but don't exploit.
# ---------------------------------------------------------------------------

MICRO_DYNAMICS_COLUMNS = [
    "return_autocorr_30s",       # Autocorrelation of 1s returns — trending vs mean-reverting
    "price_impact_30s",          # Price change per unit signed volume — market fragility
    "ob_seconds_to_cross_open",  # Depth to open / flow per second — difficulty to cross target
    "ob_depth_concentration",    # Vol L1-5 / Vol L1-20 — orderbook shape
]


def compute_micro_dynamics(day, ref, T_ms, open_ref):
    """Advanced microstructure dynamics features.

    These measure properties of the market that no existing feature captures:
    trending vs mean-reverting, market fragility, and orderbook shape.
    """
    f = {}

    # --- return_autocorr_30s: autocorrelation of 1-second returns ---
    ts = day.tf_ts
    prices = day.tf_price
    i0_30, i1_30 = _slice_window(ts, T_ms - 30_000, T_ms)

    if i1_30 - i0_30 > 30:
        ts_slice = ts[i0_30:i1_30]
        px_slice = prices[i0_30:i1_30]
        # Sample at 1-second intervals
        sec_grid = np.arange(T_ms - 30_000, T_ms, 1000)
        sec_idx = np.searchsorted(ts_slice, sec_grid, side='right') - 1
        valid = sec_idx >= 0
        if valid.sum() > 5:
            sec_px = px_slice[sec_idx[valid]]
            rets = np.diff(sec_px) / sec_px[:-1]
            if len(rets) > 3 and np.std(rets) > 0:
                # Lag-1 autocorrelation
                f["return_autocorr_30s"] = float(np.corrcoef(rets[:-1], rets[1:])[0, 1])
                if not np.isfinite(f["return_autocorr_30s"]):
                    f["return_autocorr_30s"] = 0.0
            else:
                f["return_autocorr_30s"] = 0.0
        else:
            f["return_autocorr_30s"] = 0.0
    else:
        f["return_autocorr_30s"] = 0.0

    # --- price_impact_30s: price change per unit of signed volume ---
    # Kyle's lambda: how fragile is the market?
    if i1_30 - i0_30 > 20:
        qty_slice = day.tf_qty[i0_30:i1_30]
        ibm_slice = day.tf_ibm[i0_30:i1_30]
        signed_vol = np.where(ibm_slice, -qty_slice, qty_slice)
        cum_signed = np.cumsum(signed_vol)
        px_slice = prices[i0_30:i1_30]
        px_returns = (px_slice - px_slice[0]) / px_slice[0] * 10_000  # bps from start

        # Simple: total price change / total signed volume
        total_signed = cum_signed[-1]
        total_px_change = px_returns[-1]
        # Require meaningful signed volume (at least 0.1 BTC net)
        if abs(total_signed) > 0.1:
            impact = float(total_px_change / total_signed)
            # Clip to [-50, 50] bps per BTC to avoid outliers
            f["price_impact_30s"] = max(-50.0, min(50.0, impact))
        else:
            f["price_impact_30s"] = 0.0
    else:
        f["price_impact_30s"] = 0.0

    # --- ob_seconds_to_cross_open: depth to open / flow per second ---
    ob = day.ob_fut
    has_raw = ('bid_prices' in ob and len(ob['bid_prices']) > 0)

    if has_raw and open_ref > 0 and len(ob['ts']) > 0:
        idx = _last_before(ob['ts'], T_ms)
        if idx >= 0:
            bp = ob['bid_prices'][idx]
            bq = ob['bid_qtys'][idx]
            ap = ob['ask_prices'][idx]
            aq = ob['ask_qtys'][idx]

            mid = (bp[0] + ap[0]) / 2.0 if bp[0] > 0 and ap[0] > 0 else 0.0

            # Volume between current price and open
            cross_vol = 0.0
            if mid > open_ref:
                # Price above open — need to eat through bids to go below
                valid = (bp > 0) & (bp >= open_ref)
                cross_vol = bq[valid].sum()
            elif mid < open_ref:
                # Price below open — need to eat through asks to go above
                valid = (ap > 0) & (ap <= open_ref)
                cross_vol = aq[valid].sum()

            # Flow rate: average volume per second in last 30s
            if i1_30 > i0_30:
                total_trade_vol = day.tf_qty[i0_30:i1_30].sum()
                flow_per_sec = total_trade_vol / 30.0
                if flow_per_sec > 0:
                    # Cap at 300s (one block) — beyond that it's "very hard"
                    secs = float(cross_vol / flow_per_sec)
                    f["ob_seconds_to_cross_open"] = min(300.0, secs)
                else:
                    f["ob_seconds_to_cross_open"] = 0.0
            else:
                f["ob_seconds_to_cross_open"] = 0.0
        else:
            f["ob_seconds_to_cross_open"] = 0.0
    else:
        f["ob_seconds_to_cross_open"] = 0.0

    # --- ob_depth_concentration: vol L1-5 / vol L1-20 ---
    if has_raw and len(ob['ts']) > 0:
        idx = _last_before(ob['ts'], T_ms)
        if idx >= 0:
            bp = ob['bid_prices'][idx]
            bq = ob['bid_qtys'][idx]
            ap = ob['ask_prices'][idx]
            aq = ob['ask_qtys'][idx]

            bid_valid = bp > 0
            ask_valid = ap > 0
            total_vol = bq[bid_valid].sum() + aq[ask_valid].sum()

            if total_vol > 0:
                # Top 5 levels each side
                bid_5 = bq[bid_valid][:5].sum()
                ask_5 = aq[ask_valid][:5].sum()
                near_vol = bid_5 + ask_5
                f["ob_depth_concentration"] = float(near_vol / total_vol)
            else:
                f["ob_depth_concentration"] = 0.5
        else:
            f["ob_depth_concentration"] = 0.5
    else:
        f["ob_depth_concentration"] = 0.5

    return f


# ---------------------------------------------------------------------------
# GROUP O: Intra-block flow (3 features) — pure flow since block_start
#
# Unlike fut_buy_pct_30s which looks back 30s (crossing block boundary at
# start), these measure flow ONLY within the current block.
# ---------------------------------------------------------------------------

INTRA_BLOCK_COLUMNS = [
    "ib_buy_pct",           # buy_vol / total_vol since block_start
    "ib_signed_vol_z",      # signed vol since block_start, normalized by 60s std
    "ib_volume_rate_ratio", # volume rate in-block vs pre-block 60s
]


def compute_intra_block_flow(day, T_ms, block_start_ms):
    """Flow features computed purely within the current block."""
    f = {}

    ts = day.tf_ts
    qty = day.tf_qty
    ibm = day.tf_ibm

    # Trades since block_start
    i0, i1 = _slice_window(ts, block_start_ms, T_ms)
    elapsed_s = max(1.0, (T_ms - block_start_ms) / 1000.0)

    if i1 - i0 > 0:
        q = qty[i0:i1]
        ib = ibm[i0:i1]
        total = q.sum()
        buy_vol = q[~ib].sum()

        # Feature 1: buy percentage since block start
        f["ib_buy_pct"] = float(_safe_div(buy_vol, total, 0.5))

        # Feature 2: signed volume z-score (normalized by 60s std)
        signed_vol = np.where(ib, -q, q)
        total_signed = float(signed_vol.sum())

        # Get 60s std for normalization
        i0_60, i1_60 = _slice_window(ts, T_ms - 60_000, T_ms)
        if i1_60 - i0_60 > 10:
            ts_60 = ts[i0_60:i1_60]
            q_60 = qty[i0_60:i1_60]
            ibm_60 = ibm[i0_60:i1_60]
            sv_60 = np.where(ibm_60, -q_60, q_60)
            sec_buckets = ((ts_60 - ts_60[0]) // 1000).astype(np.intp)
            n_secs = int(sec_buckets[-1]) + 1 if len(sec_buckets) > 0 else 0
            if n_secs > 2:
                sv_per_sec = np.bincount(sec_buckets, weights=sv_60, minlength=n_secs)
                nonempty = np.bincount(sec_buckets, minlength=n_secs) > 0
                sv_std = float(np.std(sv_per_sec[nonempty])) if nonempty.sum() > 2 else 0.0
            else:
                sv_std = 0.0
            z = float(_safe_div(total_signed, sv_std)) if sv_std > 0 else 0.0
            f["ib_signed_vol_z"] = max(-10.0, min(10.0, z))
        else:
            f["ib_signed_vol_z"] = 0.0

        # Feature 3: volume rate ratio (in-block rate vs pre-block 60s rate)
        ib_rate = float(total) / elapsed_s
        i0_pre, i1_pre = _slice_window(ts, block_start_ms - 60_000, block_start_ms)
        if i1_pre > i0_pre:
            pre_rate = float(qty[i0_pre:i1_pre].sum()) / 60.0
            ratio = float(_safe_div(ib_rate, pre_rate, 1.0))
            f["ib_volume_rate_ratio"] = min(10.0, ratio)
        else:
            f["ib_volume_rate_ratio"] = 1.0
    else:
        f["ib_buy_pct"] = 0.5
        f["ib_signed_vol_z"] = 0.0
        f["ib_volume_rate_ratio"] = 1.0

    return f


# ---------------------------------------------------------------------------
# GROUP P: Orderbook delta since block start (3 features)
#
# Captures how the orderbook CHANGED since block open — not the static
# snapshot but the direction of book pressure evolution.
# ---------------------------------------------------------------------------

OB_DELTA_COLUMNS = [
    "ob_imbalance_delta",    # imbalance L5 now vs at block_start
    "ob_bid_depth_delta",    # bid volume now / bid volume at block_start
    "ob_spread_delta_bps",   # spread now vs at block_start
]


def compute_ob_delta(day, T_ms, block_start_ms, block_cache=None):
    """Orderbook change features since block start.

    Caches the block_start OB snapshot for reuse within the block.
    """
    f = {}
    ob = day.ob_fut

    has_raw = ('bid_prices' in ob and len(ob['bid_prices']) > 0 and len(ob['ts']) > 0)

    if not has_raw:
        f["ob_imbalance_delta"] = 0.0
        f["ob_bid_depth_delta"] = 1.0
        f["ob_spread_delta_bps"] = 0.0
        return f

    # Get or cache block_start OB snapshot
    if block_cache is not None and 'ob_start' in block_cache:
        start_imb = block_cache['ob_start']['imb_L5']
        start_bid_vol = block_cache['ob_start']['bid_vol']
        start_spread = block_cache['ob_start']['spread_bps']
    else:
        idx_start = _last_before(ob['ts'], block_start_ms)
        if idx_start >= 0:
            start_imb = float(ob['imb_L5'][idx_start])
            bp_s = ob['bid_qtys'][idx_start]
            start_bid_vol = float(bp_s[ob['bid_prices'][idx_start] > 0].sum())
            start_spread = float(ob['spread_bps'][idx_start])
        else:
            start_imb = 0.0
            start_bid_vol = 0.0
            start_spread = 0.0

        if block_cache is not None:
            block_cache['ob_start'] = {
                'imb_L5': start_imb,
                'bid_vol': start_bid_vol,
                'spread_bps': start_spread,
            }

    # Current OB snapshot
    idx_now = _last_before(ob['ts'], T_ms)
    if idx_now >= 0:
        now_imb = float(ob['imb_L5'][idx_now])
        bp_n = ob['bid_qtys'][idx_now]
        now_bid_vol = float(bp_n[ob['bid_prices'][idx_now] > 0].sum())
        now_spread = float(ob['spread_bps'][idx_now])

        f["ob_imbalance_delta"] = now_imb - start_imb
        depth_ratio = float(_safe_div(now_bid_vol, start_bid_vol, 1.0))
        f["ob_bid_depth_delta"] = min(10.0, max(0.1, depth_ratio))
        f["ob_spread_delta_bps"] = now_spread - start_spread
    else:
        f["ob_imbalance_delta"] = 0.0
        f["ob_bid_depth_delta"] = 1.0
        f["ob_spread_delta_bps"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP Q: Micro-tick features (8 features) — 1-second resolution signals
#
# Designed for 1s sampling. Capture immediate price velocity, momentum
# consistency, trade bursts, and orderbook micro-changes.
# ---------------------------------------------------------------------------

MICRO_TICK_COLUMNS = [
    "return_1s",                # Price return in last 1 second (bps)
    "return_consistency_5s",    # Fraction of last 5 1s-returns in same direction [0,1]
    "micro_acceleration",       # return_1s vs avg 1s return over 5s
    "realized_vol_3s",          # Std of 1s returns over last 3s (bps)
    "basis_delta_1s",           # Basis change in last 1 second (bps)
    "trade_burst_1s",           # Trades in last 1s / avg per second in 30s
    "ob_imbalance_change_1s",   # Imbalance L5 now vs 1s ago
    "ob_imbalance_change_5s",   # Imbalance L5 now vs 5s ago
]


def compute_micro_tick(day, ref, T_ms):
    """Ultra-short-term features at 1-second resolution."""
    f = {}

    # --- return_1s: price return in last 1 second ---
    ref_idx = _last_before(ref['ts'], T_ms)
    ref_idx_1 = _last_before(ref['ts'], T_ms - 1_000)
    if ref_idx >= 0 and ref_idx_1 >= 0 and ref['price'][ref_idx_1] > 0:
        ref_now = ref['price'][ref_idx]
        ref_1s = ref['price'][ref_idx_1]
        f["return_1s"] = float(_safe_div(ref_now - ref_1s, ref_1s) * 10_000)
    else:
        f["return_1s"] = 0.0

    # --- return_consistency_5s + micro_acceleration ---
    # Compute 1s returns for each of the last 5 seconds
    returns_1s = []
    for lag in range(5):
        t_end = T_ms - lag * 1000
        t_start = t_end - 1000
        i_end = _last_before(ref['ts'], t_end)
        i_start = _last_before(ref['ts'], t_start)
        if i_end >= 0 and i_start >= 0 and ref['price'][i_start] > 0:
            r = _safe_div(ref['price'][i_end] - ref['price'][i_start],
                          ref['price'][i_start]) * 10_000
            returns_1s.append(r)
        else:
            returns_1s.append(0.0)

    if len(returns_1s) >= 5:
        # Consistency: fraction of returns in same direction as most recent
        latest_dir = 1 if returns_1s[0] >= 0 else -1
        same_dir = sum(1 for r in returns_1s if (r >= 0) == (latest_dir >= 0))
        f["return_consistency_5s"] = same_dir / 5.0

        # Acceleration: latest return vs average of all 5
        avg_return = sum(returns_1s) / 5.0
        f["micro_acceleration"] = returns_1s[0] - avg_return
    else:
        f["return_consistency_5s"] = 0.5
        f["micro_acceleration"] = 0.0

    # --- realized_vol_3s: std of 1s returns over last 3 seconds ---
    if len(returns_1s) >= 3:
        vol_3s = np.std(returns_1s[:3])
        f["realized_vol_3s"] = float(vol_3s)
    else:
        f["realized_vol_3s"] = 0.0

    # --- basis_delta_1s: basis change in last 1 second ---
    si = _last_before(day.bs_ts, T_ms)
    fi = _last_before(day.bf_ts, T_ms)
    si_1 = _last_before(day.bs_ts, T_ms - 1_000)
    fi_1 = _last_before(day.bf_ts, T_ms - 1_000)
    if si >= 0 and fi >= 0 and si_1 >= 0 and fi_1 >= 0:
        basis_now = _safe_div(day.bf_mid[fi] - day.bs_mid[si], day.bs_mid[si]) * 10_000
        basis_1s = _safe_div(day.bf_mid[fi_1] - day.bs_mid[si_1], day.bs_mid[si_1]) * 10_000
        f["basis_delta_1s"] = basis_now - basis_1s
    else:
        f["basis_delta_1s"] = 0.0

    # --- trade_burst_1s: trades in last 1s / avg per second in 30s ---
    ts = day.tf_ts
    i0_1, i1_1 = _slice_window(ts, T_ms - 1_000, T_ms)
    i0_30, i1_30 = _slice_window(ts, T_ms - 30_000, T_ms)
    n_trades_1s = i1_1 - i0_1
    n_trades_30s = i1_30 - i0_30
    avg_per_sec = n_trades_30s / 30.0 if n_trades_30s > 0 else 1.0
    burst = float(n_trades_1s / avg_per_sec) if avg_per_sec > 0 else 1.0
    f["trade_burst_1s"] = min(20.0, burst)

    # --- ob_imbalance_change_1s and _5s ---
    ob = day.ob_fut
    if len(ob['ts']) > 0:
        idx_now = _last_before(ob['ts'], T_ms)
        idx_1s = _last_before(ob['ts'], T_ms - 1_000)
        idx_5s = _last_before(ob['ts'], T_ms - 5_000)

        imb_now = float(ob['imb_L5'][idx_now]) if idx_now >= 0 else 0.0

        if idx_1s >= 0:
            f["ob_imbalance_change_1s"] = imb_now - float(ob['imb_L5'][idx_1s])
        else:
            f["ob_imbalance_change_1s"] = 0.0

        if idx_5s >= 0:
            f["ob_imbalance_change_5s"] = imb_now - float(ob['imb_L5'][idx_5s])
        else:
            f["ob_imbalance_change_5s"] = 0.0
    else:
        f["ob_imbalance_change_1s"] = 0.0
        f["ob_imbalance_change_5s"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP K: Cross-exchange features (Ronda 1: 6 quotes + Ronda 2: 4 trades)
# ---------------------------------------------------------------------------

def _has_cross_exchange(day):
    """Check if DayData has cross-exchange fields loaded."""
    return hasattr(day, 'cb_ts') and hasattr(day, 'bb_ts')


def compute_cross_exchange(day, T_ms, open_ref=0.0):
    """Cross-exchange features from Coinbase and Bybit quotes + trades.

    Ronda 1 (quotes - 6 features):
      cb_spread_bps      - Coinbase bid-ask spread
      bb_spread_bps      - Bybit bid-ask spread
      cb_vs_binance_bps  - Coinbase mid vs Binance spot mid
      bb_vs_binance_bps  - Bybit mid vs Binance spot mid
      cross_mean_vs_binance - average of cb and bb vs binance
      cross_price_std_bps   - std dev of mids across 3 exchanges

    Ronda 2 (trades - 4 features):
      cb_buy_pct_30s     - Coinbase buy volume % in last 30s
      bb_buy_pct_30s     - Bybit buy volume % in last 30s
      cb_leads_binance_5s - Coinbase price change 5s ago vs Binance now
      volume_share_binance - Binance fraction of total volume
    """
    f = {}

    if not _has_cross_exchange(day):
        # Return NaN for all cross-exchange features so model handles gracefully
        for col in CROSS_EXCHANGE_COLUMNS:
            f[col] = np.nan
        return f

    # --- Binance spot mid (reference) ---
    bi = _last_before(day.bs_ts, T_ms)
    binance_mid = day.bs_mid[bi] if bi >= 0 else 0.0

    # === RONDA 1: Quotes ===

    # Coinbase spread
    ci = _last_before(day.cb_ts, T_ms)
    if ci >= 0:
        cb_mid = day.cb_mid[ci]
        cb_spread = day.cb_ask[ci] - day.cb_bid[ci]
        f['cb_spread_bps'] = _safe_div(cb_spread, cb_mid) * 10_000
    else:
        cb_mid = 0.0
        f['cb_spread_bps'] = np.nan

    # Bybit spread
    bbi = _last_before(day.bb_ts, T_ms)
    if bbi >= 0:
        bb_mid = day.bb_mid[bbi]
        bb_spread = day.bb_ask[bbi] - day.bb_bid[bbi]
        f['bb_spread_bps'] = _safe_div(bb_spread, bb_mid) * 10_000
    else:
        bb_mid = 0.0
        f['bb_spread_bps'] = np.nan

    # Cross-exchange price divergence
    if binance_mid > 0 and cb_mid > 0:
        f['cb_vs_binance_bps'] = _safe_div(cb_mid - binance_mid, binance_mid) * 10_000
    else:
        f['cb_vs_binance_bps'] = np.nan

    if binance_mid > 0 and bb_mid > 0:
        f['bb_vs_binance_bps'] = _safe_div(bb_mid - binance_mid, binance_mid) * 10_000
    else:
        f['bb_vs_binance_bps'] = np.nan

    # Cross-exchange consensus
    cb_div = f['cb_vs_binance_bps']
    bb_div = f['bb_vs_binance_bps']
    if not np.isnan(cb_div) and not np.isnan(bb_div):
        f['cross_mean_vs_binance'] = (cb_div + bb_div) / 2.0
    elif not np.isnan(cb_div):
        f['cross_mean_vs_binance'] = cb_div
    elif not np.isnan(bb_div):
        f['cross_mean_vs_binance'] = bb_div
    else:
        f['cross_mean_vs_binance'] = np.nan

    # Price dispersion across 3 exchanges
    mids = [m for m in [binance_mid, cb_mid, bb_mid] if m > 0]
    if len(mids) >= 2:
        mean_mid = np.mean(mids)
        f['cross_price_std_bps'] = float(np.std(mids) / mean_mid * 10_000)
    else:
        f['cross_price_std_bps'] = np.nan

    # === RONDA 2: Trades ===

    # Coinbase buy_pct 30s
    if hasattr(day, 'ct_ts') and len(day.ct_ts) > 0:
        i0, i1 = _slice_window(day.ct_ts, T_ms - 30_000, T_ms)
        if i1 > i0:
            q = day.ct_qty[i0:i1]
            ibm = day.ct_ibm[i0:i1]
            buy_vol = q[~ibm].sum()
            f['cb_buy_pct_30s'] = _safe_div(buy_vol, q.sum(), 0.5)
        else:
            f['cb_buy_pct_30s'] = 0.5
    else:
        f['cb_buy_pct_30s'] = np.nan

    # Bybit buy_pct 30s
    if hasattr(day, 'bt_ts') and len(day.bt_ts) > 0:
        i0, i1 = _slice_window(day.bt_ts, T_ms - 30_000, T_ms)
        if i1 > i0:
            q = day.bt_qty[i0:i1]
            ibm = day.bt_ibm[i0:i1]
            buy_vol = q[~ibm].sum()
            f['bb_buy_pct_30s'] = _safe_div(buy_vol, q.sum(), 0.5)
        else:
            f['bb_buy_pct_30s'] = 0.5
    else:
        f['bb_buy_pct_30s'] = np.nan

    # Coinbase leads Binance 5s (price change in CB 5s ago vs Binance now)
    if hasattr(day, 'ct_ts') and len(day.ct_ts) > 0 and bi >= 0:
        # CB price 5s ago vs CB price 10s ago → CB return in that window
        ci_5 = _last_before(day.cb_ts, T_ms - 5_000)
        ci_10 = _last_before(day.cb_ts, T_ms - 10_000)
        if ci_5 >= 0 and ci_10 >= 0 and day.cb_mid[ci_10] > 0:
            cb_return_5s_ago = _safe_div(
                day.cb_mid[ci_5] - day.cb_mid[ci_10], day.cb_mid[ci_10]) * 10_000
        else:
            cb_return_5s_ago = 0.0

        # Binance return over last 5s
        bi_5 = _last_before(day.bs_ts, T_ms - 5_000)
        if bi_5 >= 0 and day.bs_mid[bi_5] > 0:
            bn_return_now = _safe_div(
                binance_mid - day.bs_mid[bi_5], day.bs_mid[bi_5]) * 10_000
        else:
            bn_return_now = 0.0

        # Lead = CB moved 5s ago but Binance hasn't caught up yet
        f['cb_leads_binance_5s'] = cb_return_5s_ago - bn_return_now
    else:
        f['cb_leads_binance_5s'] = np.nan

    # Volume share: Binance / total across 3 exchanges (30s window)
    bn_vol = 0.0
    cb_vol = 0.0
    bb_vol = 0.0

    i0, i1 = _slice_window(day.ts_ts, T_ms - 30_000, T_ms)
    if i1 > i0:
        bn_vol = float(day.ts_qty[i0:i1].sum())

    if hasattr(day, 'ct_ts') and len(day.ct_ts) > 0:
        i0, i1 = _slice_window(day.ct_ts, T_ms - 30_000, T_ms)
        if i1 > i0:
            cb_vol = float(day.ct_qty[i0:i1].sum())

    if hasattr(day, 'bt_ts') and len(day.bt_ts) > 0:
        i0, i1 = _slice_window(day.bt_ts, T_ms - 30_000, T_ms)
        if i1 > i0:
            bb_vol = float(day.bt_qty[i0:i1].sum())

    total_vol = bn_vol + cb_vol + bb_vol
    if total_vol > 0:
        f['volume_share_binance'] = bn_vol / total_vol
    else:
        f['volume_share_binance'] = np.nan

    # === RONDA 3: Depth (orderbook) ===

    # Imbalance L5 from cross-exchange orderbooks
    for pfx, ob_key in [("cb", "ob_cb"), ("bb", "ob_bb")]:
        ob = getattr(day, ob_key, None)
        if ob is not None and len(ob['ts']) > 0:
            idx = _last_before(ob['ts'], T_ms)
            if idx >= 0:
                f[f'{pfx}_imbalance_L5'] = float(ob['imb_L5'][idx])
            else:
                f[f'{pfx}_imbalance_L5'] = np.nan
        else:
            f[f'{pfx}_imbalance_L5'] = np.nan

    # Volume near open price from cross-exchange orderbooks
    THRESHOLD_BPS = 10
    for pfx, ob_key in [("cb", "ob_cb"), ("bb", "ob_bb")]:
        ob = getattr(day, ob_key, None)
        if ob is not None and len(ob['ts']) > 0 and open_ref > 0:
            idx = _last_before(ob['ts'], T_ms)
            if idx >= 0 and 'bid_prices' in ob and len(ob['bid_prices']) > 0:
                bp = ob['bid_prices'][idx]
                bq = ob['bid_qtys'][idx]
                ap = ob['ask_prices'][idx]
                aq = ob['ask_qtys'][idx]

                # Translate open_ref to this exchange's price space
                # using the cross-exchange spread
                threshold = open_ref * THRESHOLD_BPS / 10_000

                bid_valid = bp > 0
                bid_near = bid_valid & (bp >= open_ref - threshold) & (bp <= open_ref)
                ask_valid = ap > 0
                ask_near = ask_valid & (ap >= open_ref) & (ap <= open_ref + threshold)

                bid_near_vol = bq[bid_near].sum()
                ask_near_vol = aq[ask_near].sum()
                total_near = bid_near_vol + ask_near_vol

                f[f'{pfx}_ob_near_open'] = float(
                    (bid_near_vol - ask_near_vol) / total_near if total_near > 0 else 0.0)
            else:
                f[f'{pfx}_ob_near_open'] = np.nan
        else:
            f[f'{pfx}_ob_near_open'] = np.nan

    # Cross-exchange imbalance consensus (average L5 across all 3 exchanges)
    imbalances = []
    # Binance futures imbalance
    if len(day.ob_fut['ts']) > 0:
        idx = _last_before(day.ob_fut['ts'], T_ms)
        if idx >= 0:
            imbalances.append(float(day.ob_fut['imb_L5'][idx]))
    # Coinbase
    if not np.isnan(f.get('cb_imbalance_L5', np.nan)):
        imbalances.append(f['cb_imbalance_L5'])
    # Bybit
    if not np.isnan(f.get('bb_imbalance_L5', np.nan)):
        imbalances.append(f['bb_imbalance_L5'])

    if len(imbalances) >= 2:
        f['cross_imbalance_consensus'] = float(np.mean(imbalances))
    else:
        f['cross_imbalance_consensus'] = np.nan

    # === RONDA 4: Divergence velocity ===
    # Is the cross-exchange divergence growing or closing?

    # Coinbase divergence velocity
    ci_5 = _last_before(day.cb_ts, T_ms - 5_000)
    bi_5 = _last_before(day.bs_ts, T_ms - 5_000)
    if ci >= 0 and ci_5 >= 0 and bi >= 0 and bi_5 >= 0:
        cb_div_now = _safe_div(day.cb_mid[ci] - day.bs_mid[bi],
                               day.bs_mid[bi]) * 10_000
        cb_div_5s = _safe_div(day.cb_mid[ci_5] - day.bs_mid[bi_5],
                              day.bs_mid[bi_5]) * 10_000
        f['cb_divergence_velocity'] = cb_div_now - cb_div_5s
    else:
        f['cb_divergence_velocity'] = np.nan

    # Bybit divergence velocity
    bbi_5 = _last_before(day.bb_ts, T_ms - 5_000)
    if bbi >= 0 and bbi_5 >= 0 and bi >= 0 and bi_5 >= 0:
        bb_div_now = _safe_div(day.bb_mid[bbi] - day.bs_mid[bi],
                               day.bs_mid[bi]) * 10_000
        bb_div_5s = _safe_div(day.bb_mid[bbi_5] - day.bs_mid[bi_5],
                              day.bs_mid[bi_5]) * 10_000
        f['bb_divergence_velocity'] = bb_div_now - bb_div_5s
    else:
        f['bb_divergence_velocity'] = np.nan

    return f


# Cross-exchange column names (used for NaN defaults when data missing)
CROSS_EXCHANGE_COLUMNS = [
    # Ronda 1: quotes
    "cb_spread_bps", "bb_spread_bps",
    "cb_vs_binance_bps", "bb_vs_binance_bps",
    "cross_mean_vs_binance", "cross_price_std_bps",
    # Ronda 2: trades
    "cb_buy_pct_30s", "bb_buy_pct_30s",
    "cb_leads_binance_5s", "volume_share_binance",
    # Ronda 3: depth
    "cb_imbalance_L5", "bb_imbalance_L5",
    "cb_ob_near_open", "bb_ob_near_open",
    "cross_imbalance_consensus",
    # Ronda 4: divergence velocity
    "cb_divergence_velocity", "bb_divergence_velocity",
]


# ---------------------------------------------------------------------------
# Group R: Theoretical & statistical features (14 features)
# ---------------------------------------------------------------------------

THEORETICAL_COLUMNS = [
    "hurst_exponent_30s",
    "ou_reversion_speed",
    "ou_half_life",
    "variance_ratio_5s_1s",
    "shannon_entropy_30s",
    "skew_ret_30s",
    "kurt_ret_30s",
    "dist_to_round_number_bps",
    "microprice_vs_mid_bps",
    "amihud_illiquidity_30s",
    "vol_of_vol_60s",
    "absret_acf_lag1",
    "obv_norm",
    "bounce_count_open",
]


def _get_1s_returns(ref, T_ms, n_seconds):
    """Get array of 1-second returns (in bps) for the last n_seconds.

    Optimized: direct indexing into 1s grid instead of binary search per second.
    """
    ref_ts0 = ref['ts'][0]
    ref_len = len(ref['ts'])
    prices = ref['price']

    idx_now = int((T_ms - ref_ts0) // 1000)
    idx_start = idx_now - n_seconds

    idx_start = max(0, idx_start)
    idx_now = min(ref_len - 1, idx_now)

    if idx_now - idx_start < 2:
        return np.zeros(n_seconds)

    p = prices[idx_start:idx_now + 1]
    denom = p[:-1]
    safe = denom > 0
    rets_forward = np.zeros(len(p) - 1)
    rets_forward[safe] = (p[1:][safe] - denom[safe]) / denom[safe] * 10_000

    rets = rets_forward[::-1]

    if len(rets) < n_seconds:
        rets = np.concatenate([rets, np.zeros(n_seconds - len(rets))])
    else:
        rets = rets[:n_seconds]

    return rets


def compute_theoretical(day, ref, T_ms, block_start_ms, open_ref):
    """Compute theoretical & statistical features (Group R)."""
    f = {}

    ref_ts0 = ref['ts'][0]
    ref_len = len(ref['ts'])
    prices = ref['price']

    def _ref_idx(t_ms):
        return max(0, min(ref_len - 1, int((t_ms - ref_ts0) // 1000)))

    # Get 1-second returns for various windows
    rets_30 = _get_1s_returns(ref, T_ms, 30)
    rets_60 = _get_1s_returns(ref, T_ms, 60)

    # ---------------------------------------------------------------
    # #1 Hurst exponent (30s) via variance ratio proxy
    # H = 0.5 * log2(VR) + 0.5, where VR = Var(5s) / (5 * Var(1s))
    # ---------------------------------------------------------------
    var_1s = np.var(rets_30) if len(rets_30) >= 5 else 0.0
    if var_1s > 1e-12 and len(rets_30) >= 10:
        # Build 5-second returns from non-overlapping blocks
        rets_5s = []
        for i in range(0, len(rets_30) - 4, 5):
            rets_5s.append(np.sum(rets_30[i:i+5]))
        var_5s = np.var(rets_5s) if len(rets_5s) >= 2 else 0.0
        vr = var_5s / (5.0 * var_1s) if var_1s > 1e-12 else 1.0
        vr = max(0.01, min(100.0, vr))  # clip
        hurst = 0.5 * np.log2(vr) + 0.5
        f["hurst_exponent_30s"] = float(np.clip(hurst, 0.0, 1.0))
    else:
        f["hurst_exponent_30s"] = 0.5  # default = random walk

    # ---------------------------------------------------------------
    # #2 OU mean reversion speed (60s window)
    # Regress dx(t) on x(t) where x = price - open
    # theta = -slope. Positive = mean-reverting.
    # ---------------------------------------------------------------
    if len(rets_60) >= 10:
        # Build price path relative to open (direct slice)
        idx_end = _ref_idx(T_ms)
        idx_start = _ref_idx(T_ms - 60_000)
        price_path = prices[idx_start:idx_end + 1] - open_ref
        if len(price_path) >= 3:
            dx = np.diff(price_path)
            x = price_path[:-1]
            # Simple regression: theta = -cov(dx, x) / var(x)
            var_x = np.var(x)
            if var_x > 1e-15:
                cov_dx_x = np.mean((dx - dx.mean()) * (x - x.mean()))
                theta = -cov_dx_x / var_x
                f["ou_reversion_speed"] = float(np.clip(theta, -5.0, 5.0))
            else:
                f["ou_reversion_speed"] = 0.0
        else:
            f["ou_reversion_speed"] = 0.0
    else:
        f["ou_reversion_speed"] = 0.0

    # ---------------------------------------------------------------
    # #3 OU half-life (derived from #2)
    # half_life = ln(2) / theta. If theta <= 0, no reversion.
    # ---------------------------------------------------------------
    theta = f["ou_reversion_speed"]
    if theta > 0.001:
        f["ou_half_life"] = float(np.clip(np.log(2) / theta, 0.1, 999.0))
    else:
        f["ou_half_life"] = 999.0  # no reversion

    # ---------------------------------------------------------------
    # #4 Variance ratio (5s vs 1s)
    # VR = Var(5s_returns) / (5 * Var(1s_returns))
    # VR > 1 = momentum, VR < 1 = mean reversion, VR = 1 = random walk
    # ---------------------------------------------------------------
    if var_1s > 1e-12 and len(rets_30) >= 10:
        # Reuse rets_5s from hurst calculation
        rets_5s_vr = []
        for i in range(0, len(rets_30) - 4, 5):
            rets_5s_vr.append(np.sum(rets_30[i:i+5]))
        var_5s_vr = np.var(rets_5s_vr) if len(rets_5s_vr) >= 2 else var_1s * 5
        vr_val = var_5s_vr / (5.0 * var_1s)
        f["variance_ratio_5s_1s"] = float(np.clip(vr_val, 0.1, 5.0))
    else:
        f["variance_ratio_5s_1s"] = 1.0

    # ---------------------------------------------------------------
    # #5 Shannon entropy of returns (30s)
    # Discretize into 5 bins, compute normalized entropy.
    # 0 = perfectly predictable, 1 = maximum disorder
    # ---------------------------------------------------------------
    if len(rets_30) >= 5 and np.std(rets_30) > 1e-10:
        # Use fixed bins: very_neg, neg, neutral, pos, very_pos
        std_r = np.std(rets_30)
        bins = [-np.inf, -std_r, -std_r * 0.2, std_r * 0.2, std_r, np.inf]
        counts = np.histogram(rets_30, bins=bins)[0]
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(5)
        f["shannon_entropy_30s"] = float(entropy / max_entropy) if max_entropy > 0 else 1.0
    else:
        f["shannon_entropy_30s"] = 1.0  # maximum uncertainty

    # ---------------------------------------------------------------
    # #6 Skewness of returns (30s)
    # Positive = more extreme up moves. Negative = more extreme down moves.
    # ---------------------------------------------------------------
    if len(rets_30) >= 5 and np.std(rets_30) > 1e-10:
        m3 = np.mean((rets_30 - np.mean(rets_30)) ** 3)
        s3 = np.std(rets_30) ** 3
        skew = m3 / s3 if s3 > 1e-15 else 0.0
        f["skew_ret_30s"] = float(np.clip(skew, -5.0, 5.0))
    else:
        f["skew_ret_30s"] = 0.0

    # ---------------------------------------------------------------
    # #7 Kurtosis of returns (30s) — excess kurtosis
    # High = fat tails = extreme moves likely. 0 = normal distribution.
    # ---------------------------------------------------------------
    if len(rets_30) >= 5 and np.std(rets_30) > 1e-10:
        m4 = np.mean((rets_30 - np.mean(rets_30)) ** 4)
        s4 = np.std(rets_30) ** 4
        kurt = (m4 / s4 - 3.0) if s4 > 1e-15 else 0.0
        f["kurt_ret_30s"] = float(np.clip(kurt, 0.0, 20.0))
    else:
        f["kurt_ret_30s"] = 0.0

    # ---------------------------------------------------------------
    # #8 Distance to nearest round number (bps)
    # Round numbers: multiples of 1000 and 5000 in USD
    # ---------------------------------------------------------------
    ref_idx = _ref_idx(T_ms)
    price_now = prices[ref_idx]
    if price_now > 0:
        # Find nearest multiples of 1000 and 5000
        r1000 = round(price_now / 1000) * 1000
        r5000 = round(price_now / 5000) * 5000
        dist_1000 = abs(price_now - r1000) / price_now * 10_000
        dist_5000 = abs(price_now - r5000) / price_now * 10_000
        f["dist_to_round_number_bps"] = float(min(dist_1000, dist_5000))
    else:
        f["dist_to_round_number_bps"] = 999.0

    # ---------------------------------------------------------------
    # #9 Microprice minus midprice (bps)
    # microprice = (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
    # Positive = bid-side pressure (bullish)
    # ---------------------------------------------------------------
    ob = day.ob_fut
    if len(ob['ts']) > 0 and 'bid_prices' in ob and len(ob['bid_prices']) > 0:
        oi = _last_before(ob['ts'], T_ms)
        if oi >= 0:
            bid = float(ob['bid_prices'][oi][0])
            ask = float(ob['ask_prices'][oi][0])
            bid_qty = float(ob['bid_qtys'][oi][0])
            ask_qty = float(ob['ask_qtys'][oi][0])
            mid = (bid + ask) / 2.0
            total_qty = bid_qty + ask_qty
            if total_qty > 0 and mid > 0:
                microprice = (bid * ask_qty + ask * bid_qty) / total_qty
                f["microprice_vs_mid_bps"] = float((microprice - mid) / mid * 10_000)
            else:
                f["microprice_vs_mid_bps"] = 0.0
        else:
            f["microprice_vs_mid_bps"] = 0.0
    else:
        f["microprice_vs_mid_bps"] = 0.0

    # ---------------------------------------------------------------
    # #10 Amihud illiquidity (30s)
    # |return_30s| / total_volume_30s. High = fragile market.
    # ---------------------------------------------------------------
    ret_30 = abs(float(np.sum(rets_30))) if len(rets_30) > 0 else 0.0
    i0, i1 = _slice_window(day.tf_ts, T_ms - 30_000, T_ms)
    total_vol = float(np.sum(day.tf_qty[i0:i1])) if i1 > i0 else 0.0
    if total_vol > 1e-6:
        amihud = ret_30 / total_vol
        f["amihud_illiquidity_30s"] = float(np.clip(np.log1p(amihud * 1e6), 0, 20))
    else:
        f["amihud_illiquidity_30s"] = 0.0

    # ---------------------------------------------------------------
    # #11 Vol of vol (60s) — std of realized_vol_10s sampled every 10s
    # High = regime instability
    # ---------------------------------------------------------------
    if len(rets_60) >= 30:
        vols = []
        for offset in range(0, 60, 10):
            chunk = rets_60[offset:offset + 10]
            if len(chunk) >= 5:
                vols.append(float(np.std(chunk)))
        if len(vols) >= 3:
            f["vol_of_vol_60s"] = float(np.clip(np.std(vols), 0, 50))
        else:
            f["vol_of_vol_60s"] = 0.0
    else:
        f["vol_of_vol_60s"] = 0.0

    # ---------------------------------------------------------------
    # #12 Autocorrelation of |returns| lag 1 (30s)
    # Measures volatility clustering (ARCH effect)
    # ---------------------------------------------------------------
    if len(rets_30) >= 5:
        abs_rets = np.abs(rets_30)
        if np.std(abs_rets) > 1e-10:
            corr = np.corrcoef(abs_rets[:-1], abs_rets[1:])[0, 1]
            f["absret_acf_lag1"] = float(corr) if np.isfinite(corr) else 0.0
        else:
            f["absret_acf_lag1"] = 0.0
    else:
        f["absret_acf_lag1"] = 0.0

    # ---------------------------------------------------------------
    # #13 On-balance volume normalized (since block start)
    # OBV: +vol when price up, -vol when price down. Normalized [-1,1].
    # Simplified: use 1-second price bars and aggregate volume per bar.
    # ---------------------------------------------------------------
    i0_block = _ref_idx(block_start_ms)
    i1_block = _ref_idx(T_ms) + 1
    i1_block = min(i1_block, ref_len)

    if i1_block - i0_block >= 2:
        block_prices = prices[i0_block:i1_block]
        n_bars = i1_block - i0_block
        block_ts_start = int(ref['ts'][i0_block])
        block_ts_end = int(ref['ts'][i1_block - 1])

        # Pre-bin all trade volume into 1-second buckets (vectorized)
        ti0, ti1 = _slice_window(day.tf_ts, block_ts_start, block_ts_end + 1000)
        if ti1 > ti0:
            trade_ts = day.tf_ts[ti0:ti1]
            trade_qty = day.tf_qty[ti0:ti1]
            bucket_idx = ((trade_ts - block_ts_start) // 1000).astype(np.intp)
            bucket_idx = np.clip(bucket_idx, 0, n_bars - 1)
            vol_per_bar = np.bincount(bucket_idx, weights=trade_qty, minlength=n_bars)
        else:
            vol_per_bar = np.zeros(n_bars)

        direction = np.sign(np.diff(block_prices))
        bar_vols = vol_per_bar[:n_bars - 1]
        obv_val = float(np.sum(direction * bar_vols))
        total_v = float(np.sum(bar_vols))
        f["obv_norm"] = float(obv_val / total_v) if total_v > 0 else 0.0
    else:
        f["obv_norm"] = 0.0

    # ---------------------------------------------------------------
    # #14 Bounce count at open (since block start)
    # Count times price approached open (within 1 bps) but reversed
    # without crossing. Many bounces = open is strong support/resistance.
    # ---------------------------------------------------------------
    if i1_block - i0_block >= 3 and open_ref > 0:
        block_prices = prices[i0_block:i1_block]
        threshold_abs = open_ref * 1.0 / 10_000
        bounces = 0
        in_zone = False
        entered_side = 0  # 1 = from above, -1 = from below
        for i in range(1, len(block_prices)):
            dist = block_prices[i] - open_ref
            near = abs(dist) <= threshold_abs
            if near and not in_zone:
                in_zone = True
                entered_side = 1 if block_prices[i - 1] > open_ref else -1
            elif not near and in_zone:
                # Exited zone — check if same side as entry (= bounce)
                exit_side = 1 if block_prices[i] > open_ref else -1
                if exit_side == entered_side:
                    bounces += 1
                in_zone = False
        f["bounce_count_open"] = float(min(bounces, 20))
    else:
        f["bounce_count_open"] = 0.0

    return f


# ---------------------------------------------------------------------------
# Feature column list (programmatically generated)
# ---------------------------------------------------------------------------

def _build_feature_columns():
    """Build ordered list of all feature column names."""
    cols = []

    # Group A: Block state (15)
    cols += [
        "seconds_to_expiry",
        "dist_to_open_bps",
        "max_runup_since_open_bps",
        "max_drawdown_since_open_bps",
        "range_since_open_bps",
        "pct_time_above_open",
        "pct_time_above_open_last_30s",
        "num_crosses_open",
        "last_cross_age_s",
        "current_position_in_range",
        "realized_vol_since_open",
        "realized_vol_10s",
        "realized_vol_30s",
        "realized_vol_60s",
        "dist_to_open_z",
    ]

    # Group B: Microstructure (37)
    cols += [
        "spot_spread_bps", "fut_spread_bps",
        "spot_imbalance_L1", "spot_imbalance_L5",
        "fut_imbalance_L1", "fut_imbalance_L5",
        "basis_bps", "basis_delta_5s", "basis_delta_10s", "basis_delta_30s",
        "spot_buy_pct_5s", "spot_buy_pct_10s", "spot_buy_pct_30s", "spot_buy_pct_60s",
        "fut_buy_pct_5s", "fut_buy_pct_10s", "fut_buy_pct_30s", "fut_buy_pct_60s",
        "spot_signed_vol_z_10s", "spot_signed_vol_z_30s",
        "fut_signed_vol_z_10s", "fut_signed_vol_z_30s",
        "spot_trade_intensity_5s", "spot_trade_intensity_10s", "spot_trade_intensity_30s",
        "fut_trade_intensity_5s", "fut_trade_intensity_10s", "fut_trade_intensity_30s",
        "return_5s", "return_10s", "return_30s", "return_60s",
        "price_vs_block_vwap_bps", "block_vwap_vs_open_bps",
        "recent_drift_vs_block_drift",
    ]

    # Group C: Regime (12)
    cols += [
        "funding_rate", "minutes_to_funding", "mark_vs_index_bps",
        "oi_change_pct_5m",
        "long_short_ratio", "long_short_change",
        "top_trader_long_short", "taker_long_short_vol_ratio",
        "liq_any_5m", "liq_any_30s", "liq_burst_flag", "liq_last_side",
    ]

    # Group D: Block history (10)
    cols += [
        "prev_1_return_bps", "prev_2_return_bps", "prev_3_return_bps",
        "prev_1_result",
        "up_pct_last_6", "mean_return_last_6", "vol_last_6", "max_abs_return_last_6",
        "prev_15m_realized_vol", "prev_30m_realized_vol",
    ]

    # Group E: Temporal (8)
    cols += [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "minutes_to_funding",  # NOTE: computed in regime, listed here for grouping
        "is_us_market_hours",
        "minute_of_day", "day_of_week",
    ]

    # Group F: Data quality (7)
    cols += [
        "age_bookticker_ms", "age_markprice_ms", "age_depth_ms", "age_trades_ms",
        "missing_stream_count", "open_ref_quality_flag", "core_streams_fresh_flag",
    ]

    # Group G: Derived / pre-computed (5)
    cols += [
        "brownian_prob", "brownian_prob_drift", "z_velocity",
        "bridge_variance", "vol_ratio",
    ]

    # Group H: Flow dynamics (9)
    cols += [
        "large_trade_pct_30s", "large_trade_buy_pct_30s", "trade_size_cv_30s",
        "buy_pressure_acceleration", "volume_acceleration", "cumulative_volume_delta_norm",
        "bid_depth_change_pct_5s", "ask_depth_change_pct_5s", "spread_change_ratio_10s",
    ]

    # Group I: Orderbook at open (4)
    cols += [
        "ob_bid_vol_near_open", "ob_ask_vol_near_open",
        "ob_imbalance_at_open", "ob_volume_to_cross_open_pct",
    ]

    # Group J: VPIN (4)
    cols += [
        "vpin_30s", "vpin_signed_30s", "vpin_120s", "vpin_spike",
    ]

    # Group K: Cross-exchange (15)
    cols += CROSS_EXCHANGE_COLUMNS

    # Group L: Pre-block (9)
    cols += PREBLOCK_COLUMNS

    # Group M: Previous block structure (2)
    cols += PREV_BLOCK_COLUMNS

    # Group N: Market microstructure dynamics (4)
    cols += MICRO_DYNAMICS_COLUMNS

    # Group O: Intra-block flow (3)
    cols += INTRA_BLOCK_COLUMNS

    # Group P: Orderbook delta since block start (3)
    cols += OB_DELTA_COLUMNS

    # Group Q: Micro-tick features (8)
    cols += MICRO_TICK_COLUMNS

    # Group R: Theoretical & statistical (14)
    cols += THEORETICAL_COLUMNS

    # Deduplicate (minutes_to_funding appears in C and E)
    seen = set()
    deduped = []
    for c in cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


FEATURE_COLUMNS_V3 = _build_feature_columns()


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_features_v3(day, ref, T_ms, block_start_ms, open_ref,
                        open_ref_age_ms=0, block_results=None,
                        block_cache=None):
    """
    Compute all ~132 features for timestamp T within a block.

    Args:
        day: DayData object
        ref: dict from build_ref_price()
        T_ms: current timestamp (ms)
        block_start_ms: block start (ms)
        open_ref: reference open price
        open_ref_age_ms: age of the open_ref data point in ms
        block_results: list of previous block results (most recent first),
                       each dict with 'return_bps' and 'result'
        block_cache: optional dict, reused across rows of the same block
                     to avoid recomputing static features (preblock, history)

    Returns:
        dict mapping feature name → float value
    """
    if block_results is None:
        block_results = []

    feats = {}

    # Group A: Block state (dynamic — depends on T_ms)
    feats.update(compute_block_state(ref, T_ms, block_start_ms, open_ref))

    # Group B: Microstructure (dynamic)
    micro = compute_microstructure(day, ref, T_ms)
    # Compute block_vwap_vs_open_bps using open_ref
    vwap = micro.pop("_block_vwap", np.nan)
    if not np.isnan(vwap):
        micro["block_vwap_vs_open_bps"] = _safe_div(vwap - open_ref, open_ref) * 10_000
    else:
        micro["block_vwap_vs_open_bps"] = 0.0
    feats.update(micro)

    # Group C: Regime (dynamic — liq features depend on T_ms)
    regime = compute_regime(day, T_ms)
    feats.update(regime)

    # Group D: Block history (STATIC per block — cache)
    if block_cache is not None and 'block_history' in block_cache:
        feats.update(block_cache['block_history'])
    else:
        bh = compute_block_history(block_results)
        feats.update(bh)
        if block_cache is not None:
            block_cache['block_history'] = bh

    # Group E: Temporal (dynamic)
    temporal = compute_temporal(T_ms)
    feats.update(temporal)
    # minutes_to_funding already set by regime, temporal doesn't overwrite

    # Group F: Data quality (dynamic)
    feats.update(compute_data_quality(day, T_ms, open_ref_age_ms))

    # Group G: Derived / pre-computed (dynamic)
    feats.update(compute_derived(feats, ref, T_ms, block_start_ms))

    # Group H: Flow dynamics (dynamic)
    feats.update(compute_flow_dynamics(day, T_ms, block_start_ms))

    # Group I: Orderbook at open (dynamic — OB changes over time)
    feats.update(compute_orderbook_at_open(day, T_ms, open_ref))

    # Group J: VPIN (dynamic)
    feats.update(compute_vpin(day, T_ms))

    # Group K: Cross-exchange (dynamic)
    feats.update(compute_cross_exchange(day, T_ms, open_ref=open_ref))

    # Group L: Pre-block (STATIC per block — cache)
    if block_cache is not None and 'preblock' in block_cache:
        feats.update(block_cache['preblock'])
    else:
        pb = compute_preblock(day, block_start_ms)
        feats.update(pb)
        if block_cache is not None:
            block_cache['preblock'] = pb

    # Group M: Previous block structure (STATIC per block — cache)
    if block_cache is not None and 'prev_block' in block_cache:
        feats.update(block_cache['prev_block'])
    else:
        pvb = compute_prev_block_structure(day, block_start_ms)
        feats.update(pvb)
        if block_cache is not None:
            block_cache['prev_block'] = pvb

    # Group N: Market microstructure dynamics (dynamic)
    feats.update(compute_micro_dynamics(day, ref, T_ms, open_ref))

    # Group O: Intra-block flow (dynamic)
    feats.update(compute_intra_block_flow(day, T_ms, block_start_ms))

    # Group P: Orderbook delta since block start (dynamic, caches start snapshot)
    feats.update(compute_ob_delta(day, T_ms, block_start_ms, block_cache=block_cache))

    # Group Q: Micro-tick features (dynamic, 1s resolution)
    feats.update(compute_micro_tick(day, ref, T_ms))

    # Group R: Theoretical & statistical (dynamic)
    feats.update(compute_theoretical(day, ref, T_ms, block_start_ms, open_ref))

    return feats

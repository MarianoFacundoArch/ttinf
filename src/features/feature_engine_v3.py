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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
    grid_price = np.full(len(grid_ts), np.nan, dtype=np.float64)

    # Forward-fill: for each grid point, use last index_price <= that time
    for i, t in enumerate(grid_ts):
        idx = _last_before(day.mp_ts, t)
        if idx >= 0:
            grid_price[i] = prices[idx]

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

            # Compute in 1-second buckets for std
            ts_60 = ts_arr[i0_60:i1_60]
            sec_buckets = (ts_60 - ts_60[0]) // 1000
            unique_secs = np.unique(sec_buckets)
            if len(unique_secs) > 2:
                sv_per_sec = np.array([signed_60[sec_buckets == s].sum() for s in unique_secs])
                sv_std = float(np.std(sv_per_sec))
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
        block_drift = f.get("_dist_to_open_bps_temp", _safe_div(ref_now - ref['price'][max(0, np.searchsorted(ref['ts'], block_start_approx, side='left'))], ref_now) * 10_000) / elapsed_s

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

    # 2. brownian_prob_drift: Φ(z_adjusted) — Brownian adjusted by recent drift
    #    Uses return_30s as drift estimate over 30 seconds
    mu_hat = feats.get("return_30s", 0.0) / 30.0  # bps per second
    if sigma > 0 and seconds_to_expiry > 0:
        drift_adjustment = mu_hat * seconds_to_expiry
        z_drift = (dist_bps + drift_adjustment) / (sigma * np.sqrt(seconds_to_expiry))
        f["brownian_prob_drift"] = float(norm.cdf(np.clip(z_drift, -10, 10)))
    else:
        f["brownian_prob_drift"] = f["brownian_prob"]

    # 3. z_velocity: rate of change of z over last 5 seconds
    #    Approximated from return_5s normalized by sigma and sqrt(time)
    return_5s = feats.get("return_5s", 0.0)
    if sigma > 0 and seconds_to_expiry > 0:
        # z at T-5s ≈ (dist_bps - return_5s) / (sigma * sqrt(seconds_to_expiry + 5))
        z_prev = (dist_bps - return_5s) / (sigma * np.sqrt(seconds_to_expiry + 5))
        f["z_velocity"] = (z - z_prev) / 5.0
    else:
        f["z_velocity"] = 0.0

    # 4. bridge_variance: σ² × t_elapsed × t_remaining / T_total
    #    Peaks at mid-block, measures uncertainty independent of direction
    if sigma > 0:
        f["bridge_variance"] = sigma**2 * t_elapsed * seconds_to_expiry / T_total
    else:
        f["bridge_variance"] = 0.0

    # 5. vol_ratio: realized_vol_since_open / realized_vol_60s
    #    Detects volatility regime change within the block
    vol_open = feats.get("realized_vol_since_open", 0.0)
    vol_60 = feats.get("realized_vol_60s", 0.0)
    if vol_60 > 0:
        f["vol_ratio"] = vol_open / vol_60
    else:
        f["vol_ratio"] = 1.0

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

    # Group E: Temporal (6)
    cols += [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "minutes_to_funding",  # NOTE: computed in regime, listed here for grouping
        "is_us_market_hours",
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
                        open_ref_age_ms=0, block_results=None):
    """
    Compute all ~87 features for timestamp T within a block.

    Args:
        day: DayData object
        ref: dict from build_ref_price()
        T_ms: current timestamp (ms)
        block_start_ms: block start (ms)
        open_ref: reference open price
        open_ref_age_ms: age of the open_ref data point in ms
        block_results: list of previous block results (most recent first),
                       each dict with 'return_bps' and 'result'

    Returns:
        dict mapping feature name → float value
    """
    if block_results is None:
        block_results = []

    feats = {}

    # Group A: Block state
    feats.update(compute_block_state(ref, T_ms, block_start_ms, open_ref))

    # Group B: Microstructure
    micro = compute_microstructure(day, ref, T_ms)
    # Compute block_vwap_vs_open_bps using open_ref
    vwap = micro.pop("_block_vwap", np.nan)
    if not np.isnan(vwap):
        micro["block_vwap_vs_open_bps"] = _safe_div(vwap - open_ref, open_ref) * 10_000
    else:
        micro["block_vwap_vs_open_bps"] = 0.0
    feats.update(micro)

    # Group C: Regime
    regime = compute_regime(day, T_ms)
    feats.update(regime)

    # Group D: Block history
    feats.update(compute_block_history(block_results))

    # Group E: Temporal
    temporal = compute_temporal(T_ms)
    feats.update(temporal)
    # minutes_to_funding already set by regime, temporal doesn't overwrite

    # Group F: Data quality
    feats.update(compute_data_quality(day, T_ms, open_ref_age_ms))

    # Group G: Derived / pre-computed
    feats.update(compute_derived(feats, ref, T_ms, block_start_ms))

    return feats

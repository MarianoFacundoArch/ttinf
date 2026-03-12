"""
Feature engine for BTC 15-second price prediction.

Computes ~212 features from raw market data for a given timestamp T.
All features look backward only (max 120s). No future leakage.

Usage:
    from src.features.feature_engine import load_day_data, compute_features, FEATURE_COLUMNS

    day = load_day_data("2026-03-01")
    features = compute_features(day, T_ms=1741046400000)
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

BOOK_LOOKBACKS = [1, 3, 5, 10, 30]          # seconds
TRADE_LOOKBACKS = [1, 3, 5, 10, 30, 60]     # seconds
BOOKTICKER_LOOKBACKS = [1, 5, 10]            # seconds


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
# Pre-compute orderbook derived features (vectorized over all snapshots)
# ---------------------------------------------------------------------------

def _precompute_book(ts, bid_prices, bid_qtys, ask_prices, ask_qtys):
    """
    Pre-compute scalar features for every orderbook snapshot.

    Args:
        ts:         (N,) int64 timestamps
        bid_prices: (N, 20) float64
        bid_qtys:   (N, 20) float64
        ask_prices: (N, 20) float64
        ask_qtys:   (N, 20) float64

    Returns dict of 1-D arrays keyed by feature name.
    """
    mid = (bid_prices[:, 0] + ask_prices[:, 0]) / 2.0
    spread = ask_prices[:, 0] - bid_prices[:, 0]
    spread_bps = np.where(mid > 0, spread / mid * 10_000, 0.0)

    # Microprice
    top_qty = bid_qtys[:, 0] + ask_qtys[:, 0]
    microprice = np.where(
        top_qty > 0,
        (bid_prices[:, 0] * ask_qtys[:, 0] + ask_prices[:, 0] * bid_qtys[:, 0]) / top_qty,
        mid,
    )
    microprice_dist = np.where(mid > 0, (microprice - mid) / mid * 10_000, 0.0)

    # Imbalances at different depth levels
    def _imbalance(bq, aq, levels):
        b = bq[:, :levels].sum(axis=1)
        a = aq[:, :levels].sum(axis=1)
        t = b + a
        return np.where(t > 0, (b - a) / t, 0.0)

    imb_L1  = _imbalance(bid_qtys, ask_qtys, 1)
    imb_L5  = _imbalance(bid_qtys, ask_qtys, 5)
    imb_L10 = _imbalance(bid_qtys, ask_qtys, 10)
    imb_L20 = _imbalance(bid_qtys, ask_qtys, 20)

    # Depth totals
    total_bid = bid_qtys.sum(axis=1)
    total_ask = ask_qtys.sum(axis=1)
    total_depth = total_bid + total_ask
    bid_depth_ratio = np.where(total_depth > 0, total_bid / total_depth, 0.5)

    # Depth concentration (L1 / total)
    depth_conc_bid = np.where(total_bid > 0, bid_qtys[:, 0] / total_bid, 0.0)
    depth_conc_ask = np.where(total_ask > 0, ask_qtys[:, 0] / total_ask, 0.0)

    # Walls: any level qty > mean + 3*std across that snapshot's levels
    bid_mean = bid_qtys.mean(axis=1, keepdims=True)
    bid_std  = bid_qtys.std(axis=1, keepdims=True)
    wall_bid = (bid_qtys > bid_mean + 3 * bid_std).any(axis=1).astype(np.float64)

    ask_mean = ask_qtys.mean(axis=1, keepdims=True)
    ask_std  = ask_qtys.std(axis=1, keepdims=True)
    wall_ask = (ask_qtys > ask_mean + 3 * ask_std).any(axis=1).astype(np.float64)

    return {
        'ts': ts,
        'mid': mid,
        'spread_bps': spread_bps,
        'microprice_dist': microprice_dist,
        'imb_L1': imb_L1,
        'imb_L5': imb_L5,
        'imb_L10': imb_L10,
        'imb_L20': imb_L20,
        'bid_depth_ratio': bid_depth_ratio,
        'depth_conc_bid': depth_conc_bid,
        'depth_conc_ask': depth_conc_ask,
        'wall_bid': wall_bid,
        'wall_ask': wall_ask,
        'total_depth': total_depth,
    }


# ---------------------------------------------------------------------------
# Load day data
# ---------------------------------------------------------------------------

def load_day_data(date_str, data_dir=None):
    """Load all 9 parquet streams for a day into a DayData object."""
    base = Path(data_dir) if data_dir else DATA_DIR
    d = base / date_str
    day = DayData()

    # --- Trades futures ---
    df = pq.read_table(d / "trades_futures" / "full_day.parquet").to_pandas()
    day.tf_ts    = df["timestamp_ms"].values.astype(np.int64)
    day.tf_price = df["price"].values.astype(np.float64)
    day.tf_qty   = df["qty"].values.astype(np.float64)
    day.tf_ibm   = df["is_buyer_maker"].values  # bool
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
    day.bf_bqt = df["best_bid_qty"].values.astype(np.float64)
    day.bf_ask = df["best_ask_price"].values.astype(np.float64)
    day.bf_aqt = df["best_ask_qty"].values.astype(np.float64)
    day.bf_mid = (day.bf_bid + day.bf_ask) / 2.0
    del df

    # --- Bookticker spot ---
    df = pq.read_table(d / "bookticker_spot" / "full_day.parquet").to_pandas()
    day.bs_ts  = df["timestamp_ms"].values.astype(np.int64)
    day.bs_bid = df["best_bid_price"].values.astype(np.float64)
    day.bs_bqt = df["best_bid_qty"].values.astype(np.float64)
    day.bs_ask = df["best_ask_price"].values.astype(np.float64)
    day.bs_aqt = df["best_ask_qty"].values.astype(np.float64)
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
    day.mp_next_ms = df["next_funding_time_ms"].values.astype(np.int64)
    day.mp_oi      = df["open_interest"].values.astype(np.float64)
    del df

    # --- Liquidations ---
    df = pq.read_table(d / "liquidations" / "full_day.parquet").to_pandas()
    day.lq_ts     = df["timestamp_ms"].values.astype(np.int64)
    day.lq_is_buy = (df["side"].values.astype(str) == "buy")
    day.lq_price  = df["price"].values.astype(np.float64)
    day.lq_qty    = df["qty"].values.astype(np.float64)
    del df

    # --- Metrics (5-min intervals) ---
    path = d / "metrics" / "full_day.parquet"
    if path.exists():
        df = pq.read_table(path).to_pandas()
        day.mt_ts        = (pd.to_datetime(df["create_time"])
                            .astype(np.int64) // 1_000_000).values
        day.mt_ls_ratio  = df["count_long_short_ratio"].values.astype(np.float64)
        day.mt_top_ls    = df["count_toptrader_long_short_ratio"].values.astype(np.float64)
        day.mt_taker_ls  = df["sum_taker_long_short_vol_ratio"].values.astype(np.float64)
        del df
    else:
        day.mt_ts       = np.array([], dtype=np.int64)
        day.mt_ls_ratio = np.array([], dtype=np.float64)
        day.mt_top_ls   = np.array([], dtype=np.float64)
        day.mt_taker_ls = np.array([], dtype=np.float64)

    # Day boundaries (from spot bookticker)
    day.day_start_ms = int(day.bs_ts[0])  if len(day.bs_ts) else 0
    day.day_end_ms   = int(day.bs_ts[-1]) if len(day.bs_ts) else 0

    return day


# ---------------------------------------------------------------------------
# GROUP 1: Order book snapshot (25 features)
# ---------------------------------------------------------------------------

def compute_book_snapshot(day, T_ms):
    f = {}

    for pfx, ob in [("spot", day.ob_spot), ("fut", day.ob_fut)]:
        idx = _last_before(ob['ts'], T_ms)
        if idx < 0:
            for n in ['spread_bps', 'microprice_dist',
                       'imbalance_L1', 'imbalance_L5', 'imbalance_L10', 'imbalance_L20',
                       'bid_depth_ratio', 'depth_concentration_bid', 'depth_concentration_ask',
                       'wall_bid', 'wall_ask']:
                f[f"{pfx}_{n}"] = np.nan
            continue

        f[f"{pfx}_spread_bps"]              = float(ob['spread_bps'][idx])
        f[f"{pfx}_microprice_dist"]         = float(ob['microprice_dist'][idx])
        f[f"{pfx}_imbalance_L1"]            = float(ob['imb_L1'][idx])
        f[f"{pfx}_imbalance_L5"]            = float(ob['imb_L5'][idx])
        f[f"{pfx}_imbalance_L10"]           = float(ob['imb_L10'][idx])
        f[f"{pfx}_imbalance_L20"]           = float(ob['imb_L20'][idx])
        f[f"{pfx}_bid_depth_ratio"]         = float(ob['bid_depth_ratio'][idx])
        f[f"{pfx}_depth_concentration_bid"] = float(ob['depth_conc_bid'][idx])
        f[f"{pfx}_depth_concentration_ask"] = float(ob['depth_conc_ask'][idx])
        f[f"{pfx}_wall_bid"]                = float(ob['wall_bid'][idx])
        f[f"{pfx}_wall_ask"]                = float(ob['wall_ask'][idx])

    # Cross features
    si = _last_before(day.ob_spot['ts'], T_ms)
    fi = _last_before(day.ob_fut['ts'], T_ms)
    if si >= 0 and fi >= 0:
        sm = day.ob_spot['mid'][si]
        fm = day.ob_fut['mid'][fi]
        f["basis_bps"]         = _safe_div(fm - sm, sm) * 10_000
        f["spread_diff_bps"]   = f.get("fut_spread_bps", 0.0) - f.get("spot_spread_bps", 0.0)
        f["imbalance_diff_L5"] = f.get("fut_imbalance_L5", 0.0) - f.get("spot_imbalance_L5", 0.0)
    else:
        f["basis_bps"] = f["spread_diff_bps"] = f["imbalance_diff_L5"] = np.nan

    return f


# ---------------------------------------------------------------------------
# GROUP 2: Order book dynamics (45 features)
# ---------------------------------------------------------------------------

def compute_book_dynamics(day, T_ms):
    f = {}

    for pfx, ob in [("spot", day.ob_spot), ("fut", day.ob_fut)]:
        idx_now = _last_before(ob['ts'], T_ms)
        for lb in BOOK_LOOKBACKS:
            if idx_now < 0:
                f[f"{pfx}_imbalance_L5_delta_{lb}s"]     = np.nan
                f[f"{pfx}_bid_depth_ratio_delta_{lb}s"]   = np.nan
                f[f"{pfx}_spread_delta_bps_{lb}s"]        = np.nan
                f[f"{pfx}_microprice_move_{lb}s"]         = np.nan
                continue

            idx_past = _last_before(ob['ts'], T_ms - lb * 1000)
            if idx_past < 0:
                f[f"{pfx}_imbalance_L5_delta_{lb}s"]     = np.nan
                f[f"{pfx}_bid_depth_ratio_delta_{lb}s"]   = np.nan
                f[f"{pfx}_spread_delta_bps_{lb}s"]        = np.nan
                f[f"{pfx}_microprice_move_{lb}s"]         = np.nan
                continue

            f[f"{pfx}_imbalance_L5_delta_{lb}s"]   = float(ob['imb_L5'][idx_now] - ob['imb_L5'][idx_past])
            f[f"{pfx}_bid_depth_ratio_delta_{lb}s"] = float(ob['bid_depth_ratio'][idx_now] - ob['bid_depth_ratio'][idx_past])
            f[f"{pfx}_spread_delta_bps_{lb}s"]      = float(ob['spread_bps'][idx_now] - ob['spread_bps'][idx_past])
            m_now  = ob['mid'][idx_now]
            m_past = ob['mid'][idx_past]
            f[f"{pfx}_microprice_move_{lb}s"] = _safe_div(m_now - m_past, m_past) * 10_000

    # Cross: basis delta
    si_now = _last_before(day.ob_spot['ts'], T_ms)
    fi_now = _last_before(day.ob_fut['ts'], T_ms)
    for lb in BOOK_LOOKBACKS:
        si_past = _last_before(day.ob_spot['ts'], T_ms - lb * 1000)
        fi_past = _last_before(day.ob_fut['ts'], T_ms - lb * 1000)
        if min(si_now, fi_now, si_past, fi_past) >= 0:
            b_now  = _safe_div(day.ob_fut['mid'][fi_now] - day.ob_spot['mid'][si_now],
                               day.ob_spot['mid'][si_now]) * 10_000
            b_past = _safe_div(day.ob_fut['mid'][fi_past] - day.ob_spot['mid'][si_past],
                               day.ob_spot['mid'][si_past]) * 10_000
            f[f"basis_delta_{lb}s"] = b_now - b_past
        else:
            f[f"basis_delta_{lb}s"] = np.nan

    return f


# ---------------------------------------------------------------------------
# GROUP 3: Trade flow (66 features)
# ---------------------------------------------------------------------------

def compute_trade_flow(day, T_ms):
    f = {}

    for pfx, ts_a, qty_a, ibm_a in [
        ("spot", day.ts_ts, day.ts_qty, day.ts_ibm),
        ("fut",  day.tf_ts, day.tf_qty, day.tf_ibm),
    ]:
        # 60s baseline for normalization
        i0_60, i1_60 = _slice_window(ts_a, T_ms - 60_000, T_ms)
        n_60 = i1_60 - i0_60
        if n_60 > 0:
            q60     = qty_a[i0_60:i1_60]
            ibm60   = ibm_a[i0_60:i1_60]
            vol_60  = q60.sum()
            int_60  = n_60 / 60.0
            med_60  = float(np.median(q60))
            p95_60  = float(np.percentile(q60, 95)) if n_60 >= 20 else float(q60.max())
            vol_rate_60 = vol_60 / 60.0
        else:
            vol_60 = 0.0; int_60 = 1.0; med_60 = 1.0; p95_60 = 1.0; vol_rate_60 = 1.0

        for lb in TRADE_LOOKBACKS:
            i0, i1 = _slice_window(ts_a, T_ms - lb * 1000, T_ms)
            n = i1 - i0
            if n > 0:
                q   = qty_a[i0:i1]
                ibm = ibm_a[i0:i1]
                # is_buyer_maker=True → taker was seller → sell trade
                buy_mask  = ~ibm
                sell_mask = ibm
                buy_vol  = q[buy_mask].sum()
                sell_vol = q[sell_mask].sum()
                tot_vol  = q.sum()

                f[f"{pfx}_buy_pct_{lb}s"]           = _safe_div(buy_vol, tot_vol, 0.5)
                f[f"{pfx}_signed_vol_zscore_{lb}s"]  = _safe_div((buy_vol - sell_vol) / lb, vol_rate_60)
                f[f"{pfx}_trade_intensity_{lb}s"]    = _safe_div(n / lb, int_60, 1.0)
                f[f"{pfx}_avg_size_ratio_{lb}s"]     = _safe_div(tot_vol / n, med_60, 1.0)

                # Large trade imbalance
                large = q > p95_60
                if large.any():
                    lb_buy  = q[large & buy_mask].sum()
                    lb_sell = q[large & sell_mask].sum()
                    f[f"{pfx}_large_trade_imb_{lb}s"] = _safe_div(lb_buy - lb_sell, lb_buy + lb_sell)
                else:
                    f[f"{pfx}_large_trade_imb_{lb}s"] = 0.0
            else:
                f[f"{pfx}_buy_pct_{lb}s"]           = 0.5
                f[f"{pfx}_signed_vol_zscore_{lb}s"]  = 0.0
                f[f"{pfx}_trade_intensity_{lb}s"]    = 0.0
                f[f"{pfx}_avg_size_ratio_{lb}s"]     = 1.0
                f[f"{pfx}_large_trade_imb_{lb}s"]    = 0.0

    # Cross: fut/spot volume ratio
    for lb in TRADE_LOOKBACKS:
        ms = lb * 1000
        si0, si1 = _slice_window(day.ts_ts, T_ms - ms, T_ms)
        fi0, fi1 = _slice_window(day.tf_ts, T_ms - ms, T_ms)
        sv = day.ts_qty[si0:si1].sum() if si1 > si0 else 0.0
        fv = day.tf_qty[fi0:fi1].sum() if fi1 > fi0 else 0.0
        f[f"fut_spot_vol_ratio_{lb}s"] = _safe_div(fv, sv, 1.0)

    return f


# ---------------------------------------------------------------------------
# GROUP 4: Price momentum (31 features)
# ---------------------------------------------------------------------------

def compute_price_momentum(day, T_ms):
    f = {}
    NAN_FILL = np.nan

    # Current spot mid
    idx_s = _last_before(day.bs_ts, T_ms)
    if idx_s < 0:
        # No data → fill everything NaN
        for lb in TRADE_LOOKBACKS:
            f[f"return_{lb}s"] = f[f"realized_vol_{lb}s"] = NAN_FILL
            f[f"fut_return_{lb}s"] = f[f"fut_leads_spot_{lb}s"] = NAN_FILL
        for n in ['price_slope_10s', 'price_slope_30s', 'price_acceleration',
                   'return_skew_30s', 'return_skew_60s', 'max_drawdown_60s', 'max_runup_60s']:
            f[n] = NAN_FILL
        return f

    price_now = day.bs_mid[idx_s]

    # -- Spot returns & realized vol --
    for lb in TRADE_LOOKBACKS:
        past = _last_before(day.bs_ts, T_ms - lb * 1000)
        if past >= 0:
            f[f"return_{lb}s"] = _safe_div(price_now - day.bs_mid[past], day.bs_mid[past]) * 10_000
        else:
            f[f"return_{lb}s"] = 0.0

        # Realized vol (std of tick returns, subsampled)
        i0, i1 = _slice_window(day.bs_ts, T_ms - lb * 1000, T_ms)
        n = i1 - i0
        if n > 2:
            mids = day.bs_mid[i0:i1]
            if len(mids) > 500:
                mids = mids[::len(mids) // 500]
            rets = np.diff(mids) / mids[:-1]
            f[f"realized_vol_{lb}s"] = float(rets.std()) * 10_000
        else:
            f[f"realized_vol_{lb}s"] = 0.0

    # -- Futures returns & leads --
    idx_f = _last_before(day.bf_ts, T_ms)
    fprice_now = day.bf_mid[idx_f] if idx_f >= 0 else price_now

    for lb in TRADE_LOOKBACKS:
        fpast = _last_before(day.bf_ts, T_ms - lb * 1000)
        if fpast >= 0 and idx_f >= 0:
            fret = _safe_div(fprice_now - day.bf_mid[fpast], day.bf_mid[fpast]) * 10_000
            f[f"fut_return_{lb}s"]     = fret
            f[f"fut_leads_spot_{lb}s"] = fret - f.get(f"return_{lb}s", 0.0)
        else:
            f[f"fut_return_{lb}s"]     = 0.0
            f[f"fut_leads_spot_{lb}s"] = 0.0

    # -- Price slope (linear regression, bps/sec) --
    for lb, name in [(10, "price_slope_10s"), (30, "price_slope_30s")]:
        i0, i1 = _slice_window(day.bs_ts, T_ms - lb * 1000, T_ms)
        n = i1 - i0
        if n > 5:
            mids  = day.bs_mid[i0:i1]
            times = (day.bs_ts[i0:i1] - day.bs_ts[i0]).astype(np.float64) / 1000.0
            if len(mids) > 200:
                step = len(mids) // 200
                mids = mids[::step]; times = times[::step]
            xm = times.mean(); ym = mids.mean()
            dx = times - xm
            denom = (dx * dx).sum()
            slope = (dx * (mids - ym)).sum() / denom if denom > 0 else 0.0
            f[name] = slope / price_now * 10_000
        else:
            f[name] = 0.0

    f["price_acceleration"] = f["price_slope_10s"] - f["price_slope_30s"]

    # -- Return skewness --
    for lb, name in [(30, "return_skew_30s"), (60, "return_skew_60s")]:
        i0, i1 = _slice_window(day.bs_ts, T_ms - lb * 1000, T_ms)
        if i1 - i0 > 10:
            mids = day.bs_mid[i0:i1]
            if len(mids) > 500:
                mids = mids[::len(mids) // 500]
            rets = np.diff(mids) / mids[:-1]
            if len(rets) > 2:
                m = rets.mean(); s = rets.std()
                if s > 0:
                    f[name] = float(np.mean(((rets - m) / s) ** 3))
                else:
                    f[name] = 0.0
            else:
                f[name] = 0.0
        else:
            f[name] = 0.0

    # -- Max drawdown & runup (60s) --
    i0, i1 = _slice_window(day.bs_ts, T_ms - 60_000, T_ms)
    if i1 - i0 > 2:
        mids = day.bs_mid[i0:i1]
        if len(mids) > 2000:
            mids = mids[::len(mids) // 2000]
        peak = np.maximum.accumulate(mids)
        dd   = (mids - peak) / peak * 10_000
        f["max_drawdown_60s"] = float(dd.min())

        trough = np.minimum.accumulate(mids)
        ru     = np.where(trough > 0, (mids - trough) / trough * 10_000, 0.0)
        f["max_runup_60s"] = float(ru.max())
    else:
        f["max_drawdown_60s"] = 0.0
        f["max_runup_60s"]    = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP 5: Book ticker (14 features)
# ---------------------------------------------------------------------------

def compute_book_ticker(day, T_ms):
    f = {}

    for pfx, bt_ts, bt_bid, bt_ask, bt_mid, ob in [
        ("spot", day.bs_ts, day.bs_bid, day.bs_ask, day.bs_mid, day.ob_spot),
        ("fut",  day.bf_ts, day.bf_bid, day.bf_ask, day.bf_mid, day.ob_fut),
    ]:
        # Mid vs depth mid
        bi = _last_before(bt_ts, T_ms)
        oi = _last_before(ob['ts'], T_ms)
        if bi >= 0 and oi >= 0:
            f[f"{pfx}_mid_vs_depth_mid"] = _safe_div(bt_mid[bi] - ob['mid'][oi], ob['mid'][oi]) * 10_000
        else:
            f[f"{pfx}_mid_vs_depth_mid"] = 0.0

        # Update counts & flip rates
        for lb in BOOKTICKER_LOOKBACKS:
            i0, i1 = _slice_window(bt_ts, T_ms - lb * 1000, T_ms)
            f[f"{pfx}_bookticker_updates_{lb}s"] = float(i1 - i0)

            if i1 - i0 > 1:
                bid_ch = np.count_nonzero(np.diff(bt_bid[i0:i1]))
                ask_ch = np.count_nonzero(np.diff(bt_ask[i0:i1]))
                f[f"{pfx}_bid_ask_flip_rate_{lb}s"] = (bid_ch + ask_ch) / lb
            else:
                f[f"{pfx}_bid_ask_flip_rate_{lb}s"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP 6: Derivatives / context (14 features)
# ---------------------------------------------------------------------------

def compute_derivatives(day, T_ms):
    f = {}

    # Mark price / funding / OI
    mi = _last_before(day.mp_ts, T_ms)
    if mi >= 0:
        f["funding_rate"]            = float(day.mp_funding[mi])
        f["funding_rate_annualized"] = float(day.mp_funding[mi]) * 3 * 365
        f["mark_vs_index_bps"]       = _safe_div(
            day.mp_mark[mi] - day.mp_index[mi], day.mp_index[mi]) * 10_000

        # OI change vs previous reading
        if mi > 0:
            f["oi_change_pct"] = _safe_div(
                day.mp_oi[mi] - day.mp_oi[mi - 1], day.mp_oi[mi - 1]) * 100
        else:
            f["oi_change_pct"] = 0.0

        # OI change 5min
        mi5 = _last_before(day.mp_ts, T_ms - 300_000)
        if mi5 >= 0:
            f["oi_change_pct_5m"] = _safe_div(
                day.mp_oi[mi] - day.mp_oi[mi5], day.mp_oi[mi5]) * 100
        else:
            f["oi_change_pct_5m"] = 0.0
    else:
        f["funding_rate"] = f["funding_rate_annualized"] = f["mark_vs_index_bps"] = 0.0
        f["oi_change_pct"] = f["oi_change_pct_5m"] = 0.0

    # Liquidations (60s)
    li0, li1 = _slice_window(day.lq_ts, T_ms - 60_000, T_ms)
    lq_n = li1 - li0
    if lq_n > 0:
        q = day.lq_qty[li0:li1]
        b = day.lq_is_buy[li0:li1]
        bv = q[b].sum();  sv = q[~b].sum()

        # Normalize by avg futures trade vol per second
        fi0, fi1 = _slice_window(day.tf_ts, T_ms - 60_000, T_ms)
        avg_v = day.tf_qty[fi0:fi1].sum() / 60.0 if fi1 > fi0 else 1.0

        f["liq_buy_vol_ratio"]  = _safe_div(bv, avg_v)
        f["liq_sell_vol_ratio"] = _safe_div(sv, avg_v)
        f["liq_net_ratio"]      = _safe_div(bv - sv, avg_v)
        f["liq_count_60s"]      = float(lq_n)

        li30_0, li30_1 = _slice_window(day.lq_ts, T_ms - 30_000, T_ms)
        f["liq_is_cascading"] = 1.0 if (li30_1 - li30_0) > 3 else 0.0
    else:
        f["liq_buy_vol_ratio"] = f["liq_sell_vol_ratio"] = f["liq_net_ratio"] = 0.0
        f["liq_count_60s"] = 0.0
        f["liq_is_cascading"] = 0.0

    # Metrics (long/short ratios, 5-min freq)
    mti = _last_before(day.mt_ts, T_ms)
    if mti >= 0:
        f["long_short_ratio"]          = float(day.mt_ls_ratio[mti])
        f["top_trader_long_short"]     = float(day.mt_top_ls[mti])
        f["taker_long_short_vol_ratio"] = float(day.mt_taker_ls[mti])
        f["long_short_change"] = float(day.mt_ls_ratio[mti] - day.mt_ls_ratio[mti - 1]) if mti > 0 else 0.0
    else:
        f["long_short_ratio"] = f["top_trader_long_short"] = f["taker_long_short_vol_ratio"] = 1.0
        f["long_short_change"] = 0.0

    return f


# ---------------------------------------------------------------------------
# GROUP 7: Volatility / regime (7 features)
# ---------------------------------------------------------------------------

def compute_volatility_regime(day, T_ms, features):
    f = {}

    # vol ratio
    rv10 = features.get("realized_vol_10s", 0.0)
    rv60 = features.get("realized_vol_60s", 0.0)
    f["vol_ratio_10s_60s"] = _safe_div(rv10, rv60, 1.0)

    # spread vs mean 60s
    i0, i1 = _slice_window(day.ob_spot['ts'], T_ms - 60_000, T_ms)
    if i1 > i0:
        mean_sp = day.ob_spot['spread_bps'][i0:i1].mean()
        f["spread_vs_mean_60s"] = _safe_div(features.get("spot_spread_bps", 0.0), mean_sp, 1.0)
    else:
        f["spread_vs_mean_60s"] = 1.0

    # activity ratio (spot trades 10s vs 60s)
    ti0_10, ti1_10 = _slice_window(day.ts_ts, T_ms - 10_000, T_ms)
    ti0_60, ti1_60 = _slice_window(day.ts_ts, T_ms - 60_000, T_ms)
    tps_10 = (ti1_10 - ti0_10) / 10.0
    tps_60 = (ti1_60 - ti0_60) / 60.0
    f["activity_ratio"] = _safe_div(tps_10, tps_60, 1.0)

    # book depth vs mean 60s
    oi = _last_before(day.ob_spot['ts'], T_ms)
    if oi >= 0 and i1 > i0:
        cur_depth  = float(day.ob_spot['total_depth'][oi])
        mean_depth = float(day.ob_spot['total_depth'][i0:i1].mean())
        f["book_depth_vs_mean_60s"] = _safe_div(cur_depth, mean_depth, 1.0)
    else:
        f["book_depth_vs_mean_60s"] = 1.0

    # price range
    for lb, name in [(60, "price_range_60s_bps"), (10, "price_range_10s_bps")]:
        j0, j1 = _slice_window(day.bs_ts, T_ms - lb * 1000, T_ms)
        if j1 - j0 > 0:
            m = day.bs_mid[j0:j1]
            mid_v = m.mean()
            f[name] = _safe_div(float(m.max() - m.min()), mid_v) * 10_000
        else:
            f[name] = 0.0

    f["range_ratio"] = _safe_div(f["price_range_10s_bps"], f["price_range_60s_bps"], 1.0)

    return f


# ---------------------------------------------------------------------------
# GROUP 8: Temporal (10 features)
# ---------------------------------------------------------------------------

def compute_temporal(T_ms):
    f = {}
    dt = datetime.fromtimestamp(T_ms / 1000.0, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    dow  = dt.weekday()

    f["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    f["dow_sin"]   = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"]   = np.cos(2 * np.pi * dow / 7)
    f["is_weekend"] = 1.0 if dow >= 5 else 0.0

    h = dt.hour
    f["session_asia"]    = 1.0 if 0 <= h < 8 else 0.0
    f["session_europe"]  = 1.0 if 8 <= h < 16 else 0.0
    f["session_us"]      = 1.0 if 13 <= h < 21 else 0.0
    f["session_overlap"] = 1.0 if 13 <= h < 16 else 0.0

    # Minutes to next funding (00:00, 08:00, 16:00 UTC)
    curr_min = h * 60 + dt.minute
    funding_mins = [0, 480, 960, 1440]  # 0h, 8h, 16h, 24h
    min_to = min((fm - curr_min) for fm in funding_mins if fm > curr_min) if curr_min < 1440 else 0
    f["minutes_to_funding"] = float(min_to) if min_to > 0 else float(1440 - curr_min)

    return f


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_features(day, T_ms):
    """Compute all ~212 features for timestamp T_ms. Returns dict."""
    features = {}
    features.update(compute_book_snapshot(day, T_ms))
    features.update(compute_book_dynamics(day, T_ms))
    features.update(compute_trade_flow(day, T_ms))
    features.update(compute_price_momentum(day, T_ms))
    features.update(compute_book_ticker(day, T_ms))
    features.update(compute_derivatives(day, T_ms))
    features.update(compute_volatility_regime(day, T_ms, features))
    features.update(compute_temporal(T_ms))
    return features


# ---------------------------------------------------------------------------
# Feature column list (programmatic, ensures consistency)
# ---------------------------------------------------------------------------

def _build_feature_columns():
    cols = []
    # G1: book snapshot (25)
    for p in ["spot", "fut"]:
        cols += [f"{p}_spread_bps", f"{p}_microprice_dist",
                 f"{p}_imbalance_L1", f"{p}_imbalance_L5", f"{p}_imbalance_L10", f"{p}_imbalance_L20",
                 f"{p}_bid_depth_ratio", f"{p}_depth_concentration_bid", f"{p}_depth_concentration_ask",
                 f"{p}_wall_bid", f"{p}_wall_ask"]
    cols += ["basis_bps", "spread_diff_bps", "imbalance_diff_L5"]

    # G2: book dynamics (45)
    for p in ["spot", "fut"]:
        for lb in BOOK_LOOKBACKS:
            cols += [f"{p}_imbalance_L5_delta_{lb}s", f"{p}_bid_depth_ratio_delta_{lb}s",
                     f"{p}_spread_delta_bps_{lb}s", f"{p}_microprice_move_{lb}s"]
    for lb in BOOK_LOOKBACKS:
        cols.append(f"basis_delta_{lb}s")

    # G3: trade flow (66)
    for p in ["spot", "fut"]:
        for lb in TRADE_LOOKBACKS:
            cols += [f"{p}_buy_pct_{lb}s", f"{p}_signed_vol_zscore_{lb}s",
                     f"{p}_trade_intensity_{lb}s", f"{p}_avg_size_ratio_{lb}s",
                     f"{p}_large_trade_imb_{lb}s"]
    for lb in TRADE_LOOKBACKS:
        cols.append(f"fut_spot_vol_ratio_{lb}s")

    # G4: price momentum (31)
    for lb in TRADE_LOOKBACKS:
        cols.append(f"return_{lb}s")
    for lb in TRADE_LOOKBACKS:
        cols.append(f"realized_vol_{lb}s")
    cols += ["price_slope_10s", "price_slope_30s", "price_acceleration",
             "return_skew_30s", "return_skew_60s", "max_drawdown_60s", "max_runup_60s"]
    for lb in TRADE_LOOKBACKS:
        cols.append(f"fut_return_{lb}s")
    for lb in TRADE_LOOKBACKS:
        cols.append(f"fut_leads_spot_{lb}s")

    # G5: book ticker (14)
    for p in ["spot", "fut"]:
        cols.append(f"{p}_mid_vs_depth_mid")
        for lb in BOOKTICKER_LOOKBACKS:
            cols.append(f"{p}_bookticker_updates_{lb}s")
        for lb in BOOKTICKER_LOOKBACKS:
            cols.append(f"{p}_bid_ask_flip_rate_{lb}s")

    # G6: derivatives (14)
    cols += ["funding_rate", "funding_rate_annualized", "mark_vs_index_bps",
             "oi_change_pct", "oi_change_pct_5m",
             "liq_buy_vol_ratio", "liq_sell_vol_ratio", "liq_net_ratio",
             "liq_count_60s", "liq_is_cascading",
             "long_short_ratio", "top_trader_long_short",
             "long_short_change", "taker_long_short_vol_ratio"]

    # G7: volatility / regime (7)
    cols += ["vol_ratio_10s_60s", "spread_vs_mean_60s", "activity_ratio",
             "book_depth_vs_mean_60s", "price_range_60s_bps", "price_range_10s_bps", "range_ratio"]

    # G8: temporal (10)
    cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
             "session_asia", "session_europe", "session_us", "session_overlap",
             "minutes_to_funding"]

    return cols


FEATURE_COLUMNS = _build_feature_columns()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    date = "2026-03-01"
    print(f"Loading {date}...")
    t0 = time.time()
    day = load_day_data(date)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"  spot bookticker: {len(day.bs_ts):,} rows")
    print(f"  fut  bookticker: {len(day.bf_ts):,} rows")
    print(f"  spot trades:     {len(day.ts_ts):,} rows")
    print(f"  fut  trades:     {len(day.tf_ts):,} rows")
    print(f"  ob   futures:    {len(day.ob_fut['ts']):,} snapshots")
    print(f"  ob   spot:       {len(day.ob_spot['ts']):,} snapshots")
    print(f"  mark price:      {len(day.mp_ts):,} rows")
    print(f"  liquidations:    {len(day.lq_ts):,} rows")
    print(f"  metrics:         {len(day.mt_ts):,} rows")

    # Pick a timestamp: noon UTC + 120s buffer
    T = day.day_start_ms + 120_000 + 43_200_000
    print(f"\nComputing features at T={T} ({datetime.fromtimestamp(T/1000, tz=timezone.utc)})...")
    t0 = time.time()
    feats = compute_features(day, T)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"Computed {len(feats)} features in {elapsed_ms:.1f}ms")

    print(f"\nExpected: {len(FEATURE_COLUMNS)}, Got: {len(feats)}")

    # Show first 20
    for k in FEATURE_COLUMNS[:20]:
        print(f"  {k:45s} = {feats.get(k, 'MISSING')}")
    print(f"  ... ({len(feats)} total)")

    # Check all expected columns are present
    missing = [c for c in FEATURE_COLUMNS if c not in feats]
    extra   = [c for c in feats if c not in FEATURE_COLUMNS]
    if missing:
        print(f"\nMISSING columns: {missing}")
    if extra:
        print(f"\nEXTRA columns: {extra}")
    if not missing and not extra:
        print("\nAll columns match FEATURE_COLUMNS perfectly.")

    # Benchmark
    print("\nBenchmarking 100 consecutive seconds...")
    t0 = time.time()
    for i in range(100):
        compute_features(day, T + i * 1000)
    elapsed = time.time() - t0
    print(f"100 samples in {elapsed:.2f}s ({elapsed / 100 * 1000:.1f}ms per sample)")

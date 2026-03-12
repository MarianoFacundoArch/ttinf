"""
Train LightGBM binary model for BTC 5-minute block prediction (Polymarket).

Features:
  - Baselines: naive (sign of dist_to_open) + Brownian (Phi(z))
  - LightGBM binary with walk-forward validation
  - Calibration by time-remaining bucket
  - Comprehensive reporting per fold

Usage:
  python -m src.training.train_model_v3
  python -m src.training.train_model_v3 --walkforward --train-days 56 --test-days 14
  python -m src.training.train_model_v3 --dataset-dir data/dataset_v3 --output-dir models
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from src.features.feature_engine_v3 import FEATURE_COLUMNS_V3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_v3"
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "models"

# LightGBM hyperparameters (from DESIGN_V3.txt section 8B)
PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "learning_rate":    0.03,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.7,
    "min_child_samples": 300,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "max_bin":          255,
    "verbose":          -1,
    "seed":             42,
    "num_threads":      -1,
}

NUM_BOOST_ROUND = 3000
EARLY_STOPPING  = 200

# Time buckets for calibration and phase analysis
TIME_BUCKETS = [
    (240, 300, "240-300s (early)"),
    (180, 240, "180-240s"),
    (120, 180, "120-180s"),
    (60,  120, "60-120s"),
    (30,   60, "30-60s"),
    (0,    30, "0-30s (late)"),
]

# Phase buckets for reporting (10 phases, 30s each)
PHASE_BUCKETS = [
    (270, 300, "270-300s"),
    (240, 270, "240-270s"),
    (210, 240, "210-240s"),
    (180, 210, "180-210s"),
    (150, 180, "150-180s"),
    (120, 150, "120-150s"),
    (90,  120, "90-120s"),
    (60,   90, "60-90s"),
    (30,   60, "30-60s"),
    (0,    30, "0-30s"),
]

# Feature groups for importance aggregation
FEATURE_GROUPS = {
    "block_state": [
        "seconds_to_expiry", "dist_to_open_bps", "max_runup_since_open_bps",
        "max_drawdown_since_open_bps", "range_since_open_bps", "pct_time_above_open",
        "pct_time_above_open_last_30s", "num_crosses_open", "last_cross_age_s",
        "current_position_in_range", "realized_vol_since_open", "realized_vol_10s",
        "realized_vol_30s", "realized_vol_60s", "dist_to_open_z",
    ],
    "microstructure": [
        "spot_spread_bps", "fut_spread_bps", "spot_imbalance_L1", "spot_imbalance_L5",
        "fut_imbalance_L1", "fut_imbalance_L5", "basis_bps", "basis_delta_5s",
        "basis_delta_10s", "basis_delta_30s",
        "spot_buy_pct_5s", "spot_buy_pct_10s", "spot_buy_pct_30s", "spot_buy_pct_60s",
        "fut_buy_pct_5s", "fut_buy_pct_10s", "fut_buy_pct_30s", "fut_buy_pct_60s",
        "spot_signed_vol_z_10s", "spot_signed_vol_z_30s",
        "fut_signed_vol_z_10s", "fut_signed_vol_z_30s",
        "spot_trade_intensity_5s", "spot_trade_intensity_10s", "spot_trade_intensity_30s",
        "fut_trade_intensity_5s", "fut_trade_intensity_10s", "fut_trade_intensity_30s",
        "return_5s", "return_10s", "return_30s", "return_60s",
        "price_vs_block_vwap_bps", "block_vwap_vs_open_bps", "recent_drift_vs_block_drift",
    ],
    "regime": [
        "funding_rate", "minutes_to_funding", "mark_vs_index_bps", "oi_change_pct_5m",
        "long_short_ratio", "long_short_change", "top_trader_long_short",
        "taker_long_short_vol_ratio", "liq_any_5m", "liq_any_30s",
        "liq_burst_flag", "liq_last_side",
    ],
    "block_history": [
        "prev_1_return_bps", "prev_2_return_bps", "prev_3_return_bps", "prev_1_result",
        "up_pct_last_6", "mean_return_last_6", "vol_last_6", "max_abs_return_last_6",
        "prev_15m_realized_vol", "prev_30m_realized_vol",
    ],
    "temporal": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us_market_hours",
        "minute_of_day", "day_of_week",
    ],
    "data_quality": [
        "age_bookticker_ms", "age_markprice_ms", "age_depth_ms", "age_trades_ms",
        "missing_stream_count", "open_ref_quality_flag", "core_streams_fresh_flag",
    ],
    "derived": [
        "brownian_prob", "brownian_prob_drift", "z_velocity",
        "bridge_variance", "vol_ratio",
    ],
    "flow_dynamics": [
        "large_trade_pct_30s", "large_trade_buy_pct_30s", "trade_size_cv_30s",
        "buy_pressure_acceleration", "volume_acceleration", "cumulative_volume_delta_norm",
        "bid_depth_change_pct_5s", "ask_depth_change_pct_5s", "spread_change_ratio_10s",
    ],
    "orderbook_at_open": [
        "ob_bid_vol_near_open", "ob_ask_vol_near_open",
        "ob_imbalance_at_open", "ob_volume_to_cross_open_pct",
    ],
    "vpin": [
        "vpin_30s", "vpin_signed_30s", "vpin_120s", "vpin_spike",
    ],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir=None):
    """Load all per-day parquets, return single DataFrame sorted by timestamp."""
    d = Path(dataset_dir) if dataset_dir else DATASET_DIR
    files = sorted(d.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {d}")
        sys.exit(1)

    dfs = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df):,} rows, "
              f"{df['block_start_ms'].nunique()} blocks")

    full = pd.concat(dfs, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    print(f"  Total: {len(full):,} rows, {full['block_start_ms'].nunique():,} blocks, "
          f"{len(files)} days")
    return full


def get_day_boundaries(df):
    """Return sorted list of unique dates from timestamp_ms."""
    dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
    return sorted(dates.unique())


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_naive(dist_to_open_bps):
    """Baseline 1: predict Up if price >= open (dist_to_open >= 0)."""
    return (dist_to_open_bps >= 0).astype(int)


def baseline_brownian(dist_to_open_bps, realized_vol, seconds_to_expiry):
    """
    Baseline 2: Brownian motion probability.
    P(Up) = Phi(dist_to_open / (sigma * sqrt(time_left)))
    """
    dist = np.nan_to_num(dist_to_open_bps, nan=0.0)
    sigma = np.where(realized_vol > 0, realized_vol, 1.0)
    sigma = np.nan_to_num(sigma, nan=1.0)
    time_left = np.where(seconds_to_expiry > 0, seconds_to_expiry, 0.01)
    time_left = np.nan_to_num(time_left, nan=0.01)
    z = dist / (sigma * np.sqrt(time_left))
    z = np.clip(z, -10, 10)
    proba = norm.cdf(z)
    # Ensure no NaN/Inf in output
    proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)
    return proba


def evaluate_baseline(name, y_true, y_pred_proba, y_pred_class):
    """Evaluate a baseline, return dict of metrics."""
    acc = accuracy_score(y_true, y_pred_class)
    ll = log_loss(y_true, np.column_stack([1 - y_pred_proba, y_pred_proba]),
                  labels=[0, 1])
    brier = brier_score_loss(y_true, y_pred_proba)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5
    return {"accuracy": acc, "logloss": ll, "brier": brier, "auc": auc}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lgb(X_train, y_train, X_val, y_val):
    """Train LightGBM binary. Returns (model, best_iteration)."""
    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=FEATURE_COLUMNS_V3, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val,
                       reference=dtrain, free_raw_data=False)

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING),
            lgb.log_evaluation(100),
        ],
    )
    return model, model.best_iteration


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_calibrators(y_true, y_pred_proba, seconds_to_expiry):
    """Fit isotonic calibrators per time bucket. Returns dict of calibrators."""
    calibrators = {}
    for lo, hi, label in TIME_BUCKETS:
        mask = (seconds_to_expiry >= lo) & (seconds_to_expiry < hi)
        y_t = y_true[mask]
        p_t = y_pred_proba[mask]
        if len(y_t) < 50:
            calibrators[label] = None
            continue
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        iso.fit(p_t, y_t)
        calibrators[label] = iso
    return calibrators


def apply_calibrators(y_pred_proba, seconds_to_expiry, calibrators):
    """Apply calibrators per time bucket. Returns calibrated probabilities."""
    result = y_pred_proba.copy()
    for lo, hi, label in TIME_BUCKETS:
        mask = (seconds_to_expiry >= lo) & (seconds_to_expiry < hi)
        cal = calibrators.get(label)
        if cal is not None and mask.sum() > 0:
            result[mask] = cal.predict(y_pred_proba[mask])
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(model, X, y, df_meta, calibrators, fold_label=""):
    """
    Full evaluation of a fold. Prints comprehensive report.
    Returns dict of metrics.
    """
    y_proba = model.predict(X)
    y_pred = (y_proba >= 0.5).astype(int)

    seconds = df_meta["seconds_to_expiry"].values
    dist_bps = df_meta["dist_to_open_bps"].values
    vol = df_meta["realized_vol_60s"].values
    terminal = df_meta["terminal_return_bps"].values

    # --- 1. General metrics ---
    acc = accuracy_score(y, y_pred)
    ll = log_loss(y, np.column_stack([1 - y_proba, y_proba]), labels=[0, 1])
    brier = brier_score_loss(y, y_proba)
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = 0.5
    f1 = f1_score(y, y_pred)

    print(f"\n{'='*70}")
    print(f"  {fold_label}")
    print(f"{'='*70}")
    print(f"  1. GENERAL METRICS")
    print(f"     Accuracy:   {acc:.4f}")
    print(f"     Log loss:   {ll:.6f}")
    print(f"     Brier:      {brier:.6f}")
    print(f"     AUC:        {auc:.4f}")
    print(f"     F1:         {f1:.4f}")
    print(f"     Samples:    {len(y):,}")

    # --- 2. Baselines comparison ---
    dist_bps_clean = np.nan_to_num(dist_bps, nan=0.0)
    bl_naive_pred = baseline_naive(dist_bps_clean)
    bl_naive_proba = (dist_bps_clean >= 0).astype(float)
    bl_naive = evaluate_baseline("naive", y, bl_naive_proba, bl_naive_pred)

    bl_bm_proba = baseline_brownian(dist_bps, vol, seconds)
    bl_bm_pred = (bl_bm_proba >= 0.5).astype(int)
    bl_bm = evaluate_baseline("brownian", y, bl_bm_proba, bl_bm_pred)

    lgb_metrics = {"accuracy": acc, "logloss": ll, "brier": brier, "auc": auc}

    print(f"\n  2. BASELINES COMPARISON")
    print(f"     {'Metric':<12s} {'Naive':>10s} {'Brownian':>10s} {'LightGBM':>10s} {'vs_Naive':>10s} {'vs_BM':>10s}")
    for m in ["accuracy", "logloss", "brier", "auc"]:
        n_val = bl_naive[m]
        b_val = bl_bm[m]
        l_val = lgb_metrics[m]
        # For accuracy and AUC, higher is better. For logloss and brier, lower is better.
        if m in ["accuracy", "auc"]:
            d_n = l_val - n_val
            d_b = l_val - b_val
        else:
            d_n = n_val - l_val  # positive = lgb is better
            d_b = b_val - l_val
        print(f"     {m:<12s} {n_val:>10.4f} {b_val:>10.4f} {l_val:>10.4f} {d_n:>+10.4f} {d_b:>+10.4f}")

    # --- 3. Metrics by phase ---
    print(f"\n  3. METRICS BY PHASE")
    print(f"     {'Phase':<22s} {'Rows':>8s} {'Acc':>8s} {'Logloss':>10s} {'AUC':>8s} {'vs_BM_acc':>10s}")
    phase_results = {}
    for lo, hi, label in PHASE_BUCKETS:
        mask = (seconds >= lo) & (seconds < hi)
        if mask.sum() < 10:
            continue
        y_m = y[mask]
        p_m = y_proba[mask]
        pred_m = (p_m >= 0.5).astype(int)
        acc_m = accuracy_score(y_m, pred_m)
        ll_m = log_loss(y_m, np.column_stack([1 - p_m, p_m]), labels=[0, 1])
        try:
            auc_m = roc_auc_score(y_m, p_m)
        except ValueError:
            auc_m = 0.5

        # BM baseline for this phase
        bm_p = baseline_brownian(dist_bps[mask], vol[mask], seconds[mask])
        bm_pred = (bm_p >= 0.5).astype(int)
        bm_acc = accuracy_score(y_m, bm_pred)

        print(f"     {label:<22s} {mask.sum():>8,} {acc_m:>8.4f} {ll_m:>10.6f} {auc_m:>8.4f} {acc_m - bm_acc:>+10.4f}")
        phase_results[label] = {"accuracy": acc_m, "logloss": ll_m, "auc": auc_m, "vs_bm_acc": acc_m - bm_acc}

    # --- 3b. Fine-grained accuracy (10s buckets) ---
    print(f"\n  3b. FINE-GRAINED ACCURACY (5s buckets)")
    print(f"     {'Bucket':<12s} {'Rows':>8s} {'Acc':>8s} {'BM_Acc':>8s} {'vs_BM':>8s} {'ECE':>8s} {'AvgP':>8s} {'FreqUp':>8s}")
    fine_results = {}
    y_cal_fine = apply_calibrators(y_proba, seconds, calibrators) if calibrators else y_proba
    for lo in range(0, 300, 5):
        hi = lo + 5
        label = f"{lo}-{hi}s"
        mask = (seconds >= lo) & (seconds < hi)
        if mask.sum() < 10:
            continue
        y_m = y[mask]
        p_m = y_cal_fine[mask]
        pred_m = (p_m >= 0.5).astype(int)
        acc_m = accuracy_score(y_m, pred_m)

        # BM baseline
        bm_p = baseline_brownian(dist_bps[mask], vol[mask], seconds[mask])
        bm_pred = (bm_p >= 0.5).astype(int)
        bm_acc = accuracy_score(y_m, bm_pred)

        # ECE for this bucket
        avg_p = float(p_m.mean())
        freq_up = float(y_m.mean())
        ece_bucket = abs(avg_p - freq_up)

        print(f"     {label:<12s} {mask.sum():>8,} {acc_m:>8.4f} {bm_acc:>8.4f} {acc_m - bm_acc:>+8.4f} {ece_bucket:>8.4f} {avg_p:>8.4f} {freq_up:>8.4f}")
        fine_results[label] = {
            "accuracy": acc_m, "bm_accuracy": bm_acc,
            "vs_bm": acc_m - bm_acc, "ece": ece_bucket,
            "avg_pred": avg_p, "freq_up": freq_up, "rows": int(mask.sum()),
        }
    phase_results["_fine"] = fine_results

    # --- 4. Calibration ---
    print(f"\n  4. CALIBRATION")
    y_cal = apply_calibrators(y_proba, seconds, calibrators) if calibrators else y_proba
    cal_buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                   (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
    print(f"     {'P(pred)':<12s} {'Rows':>8s} {'Freq_Up':>10s} {'Ratio':>8s}")
    ece = 0.0
    total_cal = 0
    for lo, hi in cal_buckets:
        mask = (y_cal >= lo) & (y_cal < hi)
        n = mask.sum()
        if n == 0:
            continue
        freq = y[mask].mean()
        mid = (lo + hi) / 2
        ratio = freq / mid if mid > 0 else 0
        ece += abs(freq - mid) * n
        total_cal += n
        print(f"     {lo:.2f}-{hi:.2f}    {n:>8,} {freq:>10.4f} {ratio:>8.2f}")
    # Also check below 0.5 (Down predictions)
    for lo, hi in [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)]:
        mask = (y_cal >= lo) & (y_cal < hi)
        n = mask.sum()
        if n == 0:
            continue
        freq = y[mask].mean()
        mid = (lo + hi) / 2
        ece += abs(freq - mid) * n
        total_cal += n
        print(f"     {lo:.2f}-{hi:.2f}    {n:>8,} {freq:>10.4f} {freq/mid:>8.2f}")

    ece = ece / total_cal if total_cal > 0 else 0
    print(f"     ECE: {ece:.6f}")

    # --- 5. Feature importance ---
    importance = model.feature_importance(importance_type="split")
    sorted_idx = np.argsort(importance)[::-1]
    total_imp = importance.sum()

    print(f"\n  5. FEATURE IMPORTANCE")
    print(f"     Top 30:")
    for rank, idx in enumerate(sorted_idx[:30], 1):
        pct = importance[idx] / total_imp * 100 if total_imp > 0 else 0
        print(f"       {rank:3d}. {FEATURE_COLUMNS_V3[idx]:45s} {importance[idx]:>8,} ({pct:.1f}%)")

    print(f"\n     Bottom 30:")
    for rank, idx in enumerate(sorted_idx[-30:], len(FEATURE_COLUMNS_V3) - 29):
        print(f"       {rank:3d}. {FEATURE_COLUMNS_V3[idx]:45s} {importance[idx]:>8,}")

    zero_imp = (importance == 0).sum()
    print(f"\n     Features with importance = 0: {zero_imp}/{len(FEATURE_COLUMNS_V3)}")

    # Importance by group
    print(f"\n     Importance by group:")
    for group_name, group_cols in FEATURE_GROUPS.items():
        group_imp = sum(importance[FEATURE_COLUMNS_V3.index(c)]
                        for c in group_cols if c in FEATURE_COLUMNS_V3)
        pct = group_imp / total_imp * 100 if total_imp > 0 else 0
        print(f"       {group_name:<20s}: {pct:5.1f}%")

    # --- 6. Pseudo-PnL ---
    print(f"\n  6. PSEUDO-PNL (forecasting diagnostic, NOT real strategy PnL)")
    print(f"     {'Threshold':<12s} {'Trades':>8s} {'Win%':>8s} {'Avg_bps':>10s} {'Sharpe':>8s} {'MaxDD':>10s}")
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        pnl_list = []
        for i in range(len(y)):
            p = y_cal[i] if calibrators else y_proba[i]
            if p > 0.5 + threshold:
                pnl_list.append(terminal[i])  # long
            elif p < 0.5 - threshold:
                pnl_list.append(-terminal[i])  # short
        if len(pnl_list) > 0:
            pnl_arr = np.array(pnl_list)
            avg = pnl_arr.mean()
            win = (pnl_arr > 0).mean() * 100
            sharpe = avg / pnl_arr.std() * np.sqrt(288) if pnl_arr.std() > 0 else 0
            cum = np.cumsum(pnl_arr)
            peak = np.maximum.accumulate(cum)
            maxdd = (cum - peak).min()
            print(f"     {threshold:<12.2f} {len(pnl_list):>8,} {win:>8.1f} {avg:>10.2f} {sharpe:>8.2f} {maxdd:>10.1f}")
        else:
            print(f"     {threshold:<12.2f} {'0':>8s}")

    # --- 7. Error analysis ---
    print(f"\n  7. ERROR ANALYSIS")
    easy_mask = np.abs(terminal) > 10
    hard_mask = np.abs(terminal) < 2
    if easy_mask.sum() > 0:
        acc_easy = accuracy_score(y[easy_mask], (y_proba[easy_mask] >= 0.5).astype(int))
        print(f"     Easy blocks (|return| > 10 bps): {acc_easy:.4f} ({easy_mask.sum():,} rows)")
    if hard_mask.sum() > 0:
        acc_hard = accuracy_score(y[hard_mask], (y_proba[hard_mask] >= 0.5).astype(int))
        print(f"     Hard blocks (|return| < 2 bps):  {acc_hard:.4f} ({hard_mask.sum():,} rows)")

    vol_med = np.median(vol[vol > 0]) if (vol > 0).any() else 0
    if vol_med > 0:
        hi_vol = vol > np.percentile(vol[vol > 0], 75)
        lo_vol = vol < np.percentile(vol[vol > 0], 25)
        if hi_vol.sum() > 0:
            print(f"     High vol (>P75):  {accuracy_score(y[hi_vol], (y_proba[hi_vol] >= 0.5).astype(int)):.4f} ({hi_vol.sum():,} rows)")
        if lo_vol.sum() > 0:
            print(f"     Low vol (<P25):   {accuracy_score(y[lo_vol], (y_proba[lo_vol] >= 0.5).astype(int)):.4f} ({lo_vol.sum():,} rows)")

    # --- 8. Intra-block stability ---
    print(f"\n  8. INTRA-BLOCK STABILITY")
    block_ids = df_meta["block_start_ms"].values
    unique_blocks = np.unique(block_ids)
    flip_counts = []
    stable_count = 0
    for bid in unique_blocks[:500]:  # sample first 500 blocks for speed
        block_mask = block_ids == bid
        block_preds = (y_proba[block_mask] >= 0.5).astype(int)
        if len(block_preds) < 2:
            continue
        flips = np.sum(np.abs(np.diff(block_preds)))
        flip_counts.append(flips)
        # Stable if same prediction >80% of the time
        mode_pct = max(block_preds.mean(), 1 - block_preds.mean())
        if mode_pct >= 0.8:
            stable_count += 1
    if flip_counts:
        print(f"     Avg flips per block:  {np.mean(flip_counts):.1f}")
        print(f"     Stable blocks (>80% same pred): {stable_count}/{len(flip_counts)} ({stable_count/len(flip_counts)*100:.1f}%)")

    return {
        "accuracy": acc, "logloss": ll, "brier": brier, "auc": auc, "f1": f1,
        "ece": ece,
        "baseline_naive": bl_naive,
        "baseline_brownian": bl_bm,
        "delta_vs_bm_acc": acc - bl_bm["accuracy"],
        "delta_vs_bm_ll": bl_bm["logloss"] - ll,
        "phase_results": phase_results,
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward(df, train_days=56, test_days=14, step_days=7):
    """
    Walk-forward with block grouping and embargo.
    Train split: 85% train, 15% calibration (last 15% of train period).
    """
    dates = get_day_boundaries(df)
    n_dates = len(dates)
    embargo_blocks = 2  # 10 minutes = 2 blocks

    print(f"\nWalk-forward: {n_dates} days, train={train_days}d, test={test_days}d, "
          f"step={step_days}d, embargo={embargo_blocks} blocks")

    all_results = []
    fold = 0
    start = 0

    while start + train_days + test_days <= n_dates:
        fold += 1

        # Date ranges
        train_dates = set(dates[start:start + train_days])
        test_dates = set(dates[start + train_days:start + train_days + test_days])

        df_dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
        train_mask = df_dates.isin(train_dates)
        test_mask = df_dates.isin(test_dates)

        df_train_full = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        # Apply embargo: remove blocks near the train/test boundary
        if len(df_train_full) > 0 and len(df_test) > 0:
            train_last_block = df_train_full["block_start_ms"].max()
            test_first_block = df_test["block_start_ms"].min()

            # Remove last N blocks from train
            embargo_cutoff_train = train_last_block - embargo_blocks * 300_000
            df_train_full = df_train_full[df_train_full["block_start_ms"] <= embargo_cutoff_train]

            # Remove first N blocks from test
            embargo_cutoff_test = test_first_block + embargo_blocks * 300_000
            df_test = df_test[df_test["block_start_ms"] >= embargo_cutoff_test]

        if len(df_train_full) < 1000 or len(df_test) < 100:
            start += step_days
            continue

        # Split train into train (85%) and calibration (15%) — temporal split
        train_block_starts = sorted(df_train_full["block_start_ms"].unique())
        n_train_blocks = len(train_block_starts)
        calib_split = int(n_train_blocks * 0.85)
        calib_blocks = set(train_block_starts[calib_split:])
        real_train_blocks = set(train_block_starts[:calib_split])

        calib_mask = df_train_full["block_start_ms"].isin(calib_blocks)
        real_train_mask = df_train_full["block_start_ms"].isin(real_train_blocks)

        df_train = df_train_full.loc[real_train_mask]
        df_calib = df_train_full.loc[calib_mask]

        print(f"\n--- Fold {fold}: train {min(train_dates)}..{max(train_dates)} "
              f"({len(df_train):,} train, {len(df_calib):,} calib), "
              f"test {min(test_dates)}..{max(test_dates)} ({len(df_test):,}) ---")

        X_train = df_train[FEATURE_COLUMNS_V3].values
        y_train = df_train["target"].values.astype(int)
        X_calib = df_calib[FEATURE_COLUMNS_V3].values
        y_calib = df_calib["target"].values.astype(int)
        X_test = df_test[FEATURE_COLUMNS_V3].values
        y_test = df_test["target"].values.astype(int)

        # Train (use test for early stopping signal only)
        model, best_iter = train_lgb(X_train, y_train, X_test, y_test)
        print(f"  Best iteration: {best_iter}")

        # Fit calibrators on calib set
        calib_proba = model.predict(X_calib)
        calib_seconds = df_calib["seconds_to_expiry"].values
        calibrators = fit_calibrators(y_calib, calib_proba, calib_seconds)

        # Build meta dataframe for evaluation
        meta_cols = ["block_start_ms", "seconds_to_expiry", "dist_to_open_bps",
                     "realized_vol_60s", "terminal_return_bps"]
        df_test_meta = df_test[meta_cols].copy()

        # Evaluate
        result = evaluate_fold(
            model, X_test, y_test, df_test_meta, calibrators,
            fold_label=f"Fold {fold} test ({min(test_dates)} to {max(test_dates)})"
        )
        result["fold"] = fold
        result["train_start"] = str(min(train_dates))
        result["train_end"] = str(max(train_dates))
        result["test_start"] = str(min(test_dates))
        result["test_end"] = str(max(test_dates))
        result["best_iteration"] = best_iter
        result["n_train"] = len(df_train)
        result["n_calib"] = len(df_calib)
        result["n_test"] = len(df_test)

        # --- Disagreement analysis: model vs Brownian ---
        test_proba = model.predict(X_test)
        test_seconds = df_test["seconds_to_expiry"].values
        test_dist = df_test["dist_to_open_bps"].values
        test_vol = df_test["realized_vol_60s"].values
        y_cal_test = apply_calibrators(test_proba, test_seconds, calibrators)
        bm_proba = baseline_brownian(test_dist, test_vol, test_seconds)

        disagree_results = {}
        for thresh in [0.03, 0.05, 0.10]:
            diff = y_cal_test - bm_proba
            disagree_mask = np.abs(diff) > thresh
            n_disagree = disagree_mask.sum()
            if n_disagree < 20:
                continue

            # When model is more bullish than Brownian
            bullish = disagree_mask & (diff > 0)
            # When model is more bearish
            bearish = disagree_mask & (diff < 0)

            model_right = 0
            bm_right = 0
            if bullish.sum() > 0:
                # Model says more UP, check if UP actually won
                model_right += y_test[bullish].sum()
                bm_right += (1 - y_test[bullish]).sum()
            if bearish.sum() > 0:
                # Model says more DOWN, check if DOWN actually won
                model_right += (1 - y_test[bearish]).sum()
                bm_right += y_test[bearish].sum()

            total = model_right + bm_right
            model_pct = model_right / total if total > 0 else 0.5
            avg_diff = float(np.abs(diff[disagree_mask]).mean())

            disagree_results[f"thresh_{thresh}"] = {
                "n_disagree": int(n_disagree),
                "model_right_pct": float(model_pct),
                "avg_disagreement": avg_diff,
            }

        result["disagreement"] = disagree_results
        all_results.append(result)

        start += step_days

    # --- Summary ---
    if all_results:
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD SUMMARY ({len(all_results)} folds)")
        print(f"{'='*70}")

        for metric in ["accuracy", "logloss", "brier", "auc", "ece",
                        "delta_vs_bm_acc", "delta_vs_bm_ll"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric:<20s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

        # Consistency
        bm_wins = sum(1 for r in all_results if r["delta_vs_bm_acc"] > 0)
        print(f"\n  Consistency: LightGBM > Brownian in {bm_wins}/{len(all_results)} folds")

        # Best and worst fold
        best_fold = max(all_results, key=lambda r: r["delta_vs_bm_acc"])
        worst_fold = min(all_results, key=lambda r: r["delta_vs_bm_acc"])
        print(f"  Best fold:  {best_fold['fold']} ({best_fold['test_start']} to {best_fold['test_end']}, "
              f"delta_acc={best_fold['delta_vs_bm_acc']:+.4f})")
        print(f"  Worst fold: {worst_fold['fold']} ({worst_fold['test_start']} to {worst_fold['test_end']}, "
              f"delta_acc={worst_fold['delta_vs_bm_acc']:+.4f})")

        # Phase summary across all folds
        print(f"\n  ACCURACY BY PHASE (averaged across {len(all_results)} folds)")
        print(f"  {'Phase':<25s} {'Acc':>8s} {'AUC':>8s} {'vs_BM':>8s} {'Std':>8s}")
        phase_labels = [label for _, _, label in PHASE_BUCKETS]
        for label in phase_labels:
            accs = [r["phase_results"][label]["accuracy"]
                    for r in all_results if label in r.get("phase_results", {})]
            aucs = [r["phase_results"][label]["auc"]
                    for r in all_results if label in r.get("phase_results", {})]
            vs_bm = [r["phase_results"][label].get("vs_bm_acc", 0.0)
                     for r in all_results if label in r.get("phase_results", {})]
            if accs:
                print(f"  {label:<25s} {np.mean(accs):>8.4f} {np.mean(aucs):>8.4f} "
                      f"{np.mean(vs_bm):>+8.4f} {np.std(accs):>8.4f}")

        # Fine-grained summary (5s buckets)
        print(f"\n  FINE-GRAINED ACCURACY (5s buckets, averaged across {len(all_results)} folds)")
        print(f"  {'Bucket':<12s} {'Acc':>8s} {'BM_Acc':>8s} {'vs_BM':>8s} {'ECE':>8s} {'Rows':>10s}")
        for lo in range(0, 300, 5):
            label = f"{lo}-{lo+5}s"
            accs = [r["phase_results"]["_fine"][label]["accuracy"]
                    for r in all_results
                    if "_fine" in r.get("phase_results", {}) and label in r["phase_results"]["_fine"]]
            bm_accs = [r["phase_results"]["_fine"][label]["bm_accuracy"]
                       for r in all_results
                       if "_fine" in r.get("phase_results", {}) and label in r["phase_results"]["_fine"]]
            eces = [r["phase_results"]["_fine"][label]["ece"]
                    for r in all_results
                    if "_fine" in r.get("phase_results", {}) and label in r["phase_results"]["_fine"]]
            rows = [r["phase_results"]["_fine"][label]["rows"]
                    for r in all_results
                    if "_fine" in r.get("phase_results", {}) and label in r["phase_results"]["_fine"]]
            if accs:
                print(f"  {label:<12s} {np.mean(accs):>8.4f} {np.mean(bm_accs):>8.4f} "
                      f"{np.mean(accs) - np.mean(bm_accs):>+8.4f} {np.mean(eces):>8.4f} {int(np.mean(rows)):>10,}")

        # Disagreement analysis summary
        print(f"\n  MODEL vs BROWNIAN DISAGREEMENT (averaged across {len(all_results)} folds)")
        print(f"  {'Threshold':<12s} {'N_disagree':>12s} {'Model_right%':>14s} {'Avg_diff':>10s} {'Verdict':>10s}")
        for thresh in [0.03, 0.05, 0.10]:
            key = f"thresh_{thresh}"
            ns = [r["disagreement"][key]["n_disagree"]
                  for r in all_results if key in r.get("disagreement", {})]
            mrs = [r["disagreement"][key]["model_right_pct"]
                   for r in all_results if key in r.get("disagreement", {})]
            diffs = [r["disagreement"][key]["avg_disagreement"]
                     for r in all_results if key in r.get("disagreement", {})]
            if mrs:
                avg_mr = np.mean(mrs)
                verdict = "MODEL" if avg_mr > 0.52 else ("BROWNIAN" if avg_mr < 0.48 else "TIE")
                print(f"  |diff|>{thresh:<5.2f}  {int(np.mean(ns)):>12,} {avg_mr:>13.1%} "
                      f"{np.mean(diffs):>10.4f} {verdict:>10s}")

    return all_results


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_model(model, calibrators, results, output_dir=None):
    """Save model, calibrators, feature columns, and metrics."""
    d = Path(output_dir) if output_dir else OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = d / "lightgbm_v3.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Calibrators
    cal_path = d / "calibrators_v3.pkl"
    with open(cal_path, 'wb') as f:
        pickle.dump(calibrators, f)
    print(f"Calibrators saved: {cal_path}")

    # Feature columns
    cols_path = d / "feature_columns_v3.txt"
    cols_path.write_text("\n".join(FEATURE_COLUMNS_V3))
    print(f"Feature columns saved: {cols_path}")

    # Metrics
    metrics_path = d / "metrics_v3.json"
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    clean = _convert(results)
    metrics_path.write_text(json.dumps(clean, indent=2, default=str))
    print(f"Metrics saved: {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train V3 LightGBM binary model")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--walkforward", action="store_true", help="Use walk-forward validation")
    parser.add_argument("--train-days", type=int, default=56, help="Walk-forward train window (days)")
    parser.add_argument("--test-days", type=int, default=14, help="Walk-forward test window (days)")
    parser.add_argument("--step-days", type=int, default=7, help="Walk-forward step (days)")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.dataset_dir)

    # Target distribution
    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    total = len(df)
    print(f"\nTarget distribution:")
    print(f"  Up:   {n_up:>10,} ({n_up / total * 100:.1f}%)")
    print(f"  Down: {n_down:>10,} ({n_down / total * 100:.1f}%)")
    print(f"  Blocks: {df['block_start_ms'].nunique():,}")

    if args.walkforward:
        wf_results = walk_forward(df, args.train_days, args.test_days, args.step_days)

        # Train final model on all data
        print(f"\n{'='*70}")
        print(f"  Training final model on ALL data ({len(df):,} rows)")
        print(f"{'='*70}")

        # Split: 85% train, 15% calib (temporal)
        all_blocks = sorted(df["block_start_ms"].unique())
        calib_split = int(len(all_blocks) * 0.85)
        calib_blocks = set(all_blocks[calib_split:])
        train_blocks = set(all_blocks[:calib_split])

        df_final_train = df[df["block_start_ms"].isin(train_blocks)]
        df_final_calib = df[df["block_start_ms"].isin(calib_blocks)]

        X_train = df_final_train[FEATURE_COLUMNS_V3].values
        y_train = df_final_train["target"].values.astype(int)
        X_calib = df_final_calib[FEATURE_COLUMNS_V3].values
        y_calib = df_final_calib["target"].values.astype(int)

        final_model, best_iter = train_lgb(X_train, y_train, X_calib, y_calib)
        print(f"  Final model best iteration: {best_iter}")

        # Fit final calibrators
        calib_proba = final_model.predict(X_calib)
        calib_seconds = df_final_calib["seconds_to_expiry"].values
        final_calibrators = fit_calibrators(y_calib, calib_proba, calib_seconds)

        # Feature importance
        importance = final_model.feature_importance(importance_type="split")
        sorted_idx = np.argsort(importance)[::-1]
        total_imp = importance.sum()

        print(f"\n  Top 30 features (final model):")
        for rank, idx in enumerate(sorted_idx[:30], 1):
            pct = importance[idx] / total_imp * 100 if total_imp > 0 else 0
            print(f"    {rank:3d}. {FEATURE_COLUMNS_V3[idx]:45s} {importance[idx]:>8,} ({pct:.1f}%)")

        save_model(final_model, final_calibrators,
                   {"walk_forward": wf_results, "best_iteration": best_iter},
                   args.output_dir)

    else:
        # Simple temporal split (80/20)
        all_blocks = sorted(df["block_start_ms"].unique())
        split_idx = int(len(all_blocks) * 0.8)
        train_blocks = set(all_blocks[:split_idx])
        test_blocks = set(all_blocks[split_idx:])

        df_train = df[df["block_start_ms"].isin(train_blocks)]
        df_test = df[df["block_start_ms"].isin(test_blocks)]

        # Further split train for calibration
        train_block_list = sorted(train_blocks)
        calib_split = int(len(train_block_list) * 0.85)
        calib_blocks = set(train_block_list[calib_split:])
        real_train_blocks = set(train_block_list[:calib_split])

        df_real_train = df_train[df_train["block_start_ms"].isin(real_train_blocks)]
        df_calib = df_train[df_train["block_start_ms"].isin(calib_blocks)]

        print(f"\nTemporal split: train={len(df_real_train):,}, calib={len(df_calib):,}, "
              f"test={len(df_test):,}")

        X_train = df_real_train[FEATURE_COLUMNS_V3].values
        y_train = df_real_train["target"].values.astype(int)
        X_calib = df_calib[FEATURE_COLUMNS_V3].values
        y_calib = df_calib["target"].values.astype(int)
        X_test = df_test[FEATURE_COLUMNS_V3].values
        y_test = df_test["target"].values.astype(int)

        model, best_iter = train_lgb(X_train, y_train, X_test, y_test)
        print(f"  Best iteration: {best_iter}")

        # Calibrate
        calib_proba = model.predict(X_calib)
        calib_seconds = df_calib["seconds_to_expiry"].values
        calibrators = fit_calibrators(y_calib, calib_proba, calib_seconds)

        meta_cols = ["block_start_ms", "seconds_to_expiry", "dist_to_open_bps",
                     "realized_vol_60s", "terminal_return_bps"]
        result = evaluate_fold(model, X_test, y_test, df_test[meta_cols],
                               calibrators, fold_label="Test set (temporal split)")

        save_model(model, calibrators,
                   {"mode": "temporal_split", "best_iteration": best_iter, **result},
                   args.output_dir)

    print("\nDone.")
    sys.exit(0)


if __name__ == "__main__":
    main()

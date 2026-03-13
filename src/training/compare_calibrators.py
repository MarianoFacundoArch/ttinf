"""
Compare calibration methods on existing model predictions.
No re-training — just different post-processing of the same raw probabilities.

Compares:
  1. Isotonic 30s buckets (current)
  2. Isotonic 5s buckets
  3. Beta calibration 30s buckets
  4. Beta calibration 5s buckets

Usage:
  python -m src.training.compare_calibrators
  python -m src.training.compare_calibrators --dataset-dir data/dataset_v3
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import binomtest, norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.features.feature_engine_v3 import FEATURE_COLUMNS_V3

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_v3"

# Time buckets
BUCKETS_30S = [
    (240, 300, "240-300s"),
    (180, 240, "180-240s"),
    (120, 180, "120-180s"),
    (60, 120, "60-120s"),
    (30, 60, "30-60s"),
    (0, 30, "0-30s"),
]

BUCKETS_5S = [(lo, lo + 5, f"{lo}-{lo+5}s") for lo in range(0, 300, 5)]


# ---------------------------------------------------------------------------
# Beta calibration
# ---------------------------------------------------------------------------

class BetaCalibrator:
    """
    Beta calibration: maps predicted probabilities through a parametric transform.
    P_cal = 1 / (1 + 1/exp(a * logit(p) + b))
    which is equivalent to: P_cal = sigmoid(a * logit(p) + b)

    3-parameter version: P_cal = sigmoid(a * log(p) + b * log(1-p) + c)
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.c = 0.0
        self.mode = "2param"  # "2param" or "3param"

    def fit(self, p_pred, y_true, mode="2param"):
        """Fit calibrator. p_pred: raw probabilities, y_true: 0/1 labels."""
        self.mode = mode
        p = np.clip(p_pred, 1e-6, 1 - 1e-6)
        y = y_true.astype(float)

        if mode == "2param":
            # sigmoid(a * logit(p) + b)
            s = logit(p)

            def neg_log_likelihood(params):
                a, b = params
                q = expit(a * s + b)
                q = np.clip(q, 1e-8, 1 - 1e-8)
                return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

            result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead",
                              options={"maxiter": 5000})
            self.a, self.b = result.x

        elif mode == "3param":
            # sigmoid(a * log(p) + b * log(1-p) + c)
            log_p = np.log(p)
            log_1p = np.log(1 - p)

            def neg_log_likelihood(params):
                a, b, c = params
                q = expit(a * log_p + b * log_1p + c)
                q = np.clip(q, 1e-8, 1 - 1e-8)
                return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

            result = minimize(neg_log_likelihood, [1.0, -1.0, 0.0], method="Nelder-Mead",
                              options={"maxiter": 5000})
            self.a, self.b, self.c = result.x

    def predict(self, p_pred):
        """Apply calibration to raw probabilities."""
        p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)

        if self.mode == "2param":
            s = logit(p)
            return expit(self.a * s + self.b)
        else:
            return expit(self.a * np.log(p) + self.b * np.log(1 - p) + self.c)


# ---------------------------------------------------------------------------
# Calibrator fitting
# ---------------------------------------------------------------------------

def fit_isotonic(y_true, y_pred, seconds, buckets):
    """Fit isotonic calibrators per bucket."""
    calibrators = {}
    for lo, hi, label in buckets:
        mask = (seconds >= lo) & (seconds < hi)
        y_t, p_t = y_true[mask], y_pred[mask]
        if len(y_t) < 50:
            calibrators[label] = None
            continue
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        iso.fit(p_t, y_t)
        calibrators[label] = iso
    return calibrators


def fit_beta(y_true, y_pred, seconds, buckets, mode="2param"):
    """Fit beta calibrators per bucket."""
    calibrators = {}
    for lo, hi, label in buckets:
        mask = (seconds >= lo) & (seconds < hi)
        y_t, p_t = y_true[mask], y_pred[mask]
        if len(y_t) < 50:
            calibrators[label] = None
            continue
        cal = BetaCalibrator()
        cal.fit(p_t, y_t, mode=mode)
        calibrators[label] = cal
    return calibrators


def apply_calibrators(y_pred, seconds, calibrators, buckets):
    """Apply calibrators per bucket. Returns calibrated probabilities."""
    result = y_pred.copy()
    for lo, hi, label in buckets:
        mask = (seconds >= lo) & (seconds < hi)
        cal = calibrators.get(label)
        if cal is not None and mask.sum() > 0:
            preds = y_pred[mask]
            if hasattr(cal, 'predict'):
                result[mask] = cal.predict(preds)
            elif hasattr(cal, 'transform'):
                result[mask] = cal.transform(preds)
    return np.clip(result, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_calibration(name, y_true, y_cal, seconds, phase_buckets):
    """Evaluate calibration quality. Returns dict of metrics."""
    y_pred = (y_cal >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_cal)
    ll = log_loss(y_true, np.column_stack([1 - y_cal, y_cal]), labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, y_cal)
    except ValueError:
        auc = 0.5

    # ECE
    ece = 0.0
    total = 0
    for p_lo in np.arange(0.0, 1.0, 0.05):
        p_hi = p_lo + 0.05
        mask = (y_cal >= p_lo) & (y_cal < p_hi)
        n = mask.sum()
        if n == 0:
            continue
        freq = y_true[mask].mean()
        mid = (p_lo + p_hi) / 2
        ece += abs(freq - mid) * n
        total += n
    ece = ece / total if total > 0 else 0

    # Phase metrics
    phase_results = {}
    for lo, hi, label in phase_buckets:
        mask = (seconds >= lo) & (seconds < hi)
        if mask.sum() < 50:
            continue
        y_m = y_true[mask]
        p_m = y_cal[mask]
        pred_m = (p_m >= 0.5).astype(int)

        acc_m = accuracy_score(y_m, pred_m)
        brier_m = brier_score_loss(y_m, p_m)

        avg_p = float(p_m.mean())
        freq_up = float(y_m.mean())
        ece_m = abs(avg_p - freq_up)

        phase_results[label] = {
            "acc": acc_m, "brier": brier_m, "ece": ece_m,
            "avg_pred": avg_p, "freq_up": freq_up, "n": int(mask.sum()),
        }

    return {
        "name": name, "accuracy": acc, "brier": brier, "logloss": ll,
        "auc": auc, "ece": ece, "phase": phase_results,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir=None):
    d = Path(dataset_dir) if dataset_dir else DATASET_DIR
    files = sorted(d.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {d}")
        sys.exit(1)
    dfs = [pq.read_table(f).to_pandas() for f in files]
    full = pd.concat(dfs, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    print(f"Loaded {len(full):,} rows, {full['block_start_ms'].nunique():,} blocks, {len(files)} days")
    return full


def get_day_boundaries(df):
    dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
    return sorted(dates.unique())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare calibration methods")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--train-days", type=int, default=56)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.dataset_dir)

    dates = get_day_boundaries(df)
    n_dates = len(dates)
    embargo_blocks = 2

    # Phase buckets for reporting
    phase_buckets = [
        (270, 300, "270-300s"),
        (240, 270, "240-270s"),
        (210, 240, "210-240s"),
        (180, 210, "180-210s"),
        (150, 180, "150-180s"),
        (120, 150, "120-150s"),
        (90, 120, "90-120s"),
        (60, 90, "60-90s"),
        (30, 60, "30-60s"),
        (0, 30, "0-30s"),
    ]

    # Calibrator configs to compare
    configs = [
        ("Isotonic-30s", "isotonic", BUCKETS_30S),
        ("Isotonic-5s", "isotonic", BUCKETS_5S),
        ("Beta2-30s", "beta2", BUCKETS_30S),
        ("Beta2-5s", "beta2", BUCKETS_5S),
        ("Beta3-30s", "beta3", BUCKETS_30S),
        ("Beta3-5s", "beta3", BUCKETS_5S),
    ]

    all_fold_results = {name: [] for name, _, _ in configs}
    fold = 0
    start = 0

    print(f"\nWalk-forward: {n_dates} days, train={args.train_days}d, "
          f"test={args.test_days}d, step={args.step_days}d")

    while start + args.train_days + args.test_days <= n_dates:
        fold += 1

        train_dates = set(dates[start:start + args.train_days])
        test_dates = set(dates[start + args.train_days:start + args.train_days + args.test_days])

        df_dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
        train_mask = df_dates.isin(train_dates)
        test_mask = df_dates.isin(test_dates)

        df_train_full = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        # Embargo
        if len(df_train_full) > 0 and len(df_test) > 0:
            train_last_block = df_train_full["block_start_ms"].max()
            test_first_block = df_test["block_start_ms"].min()
            embargo_cutoff_train = train_last_block - embargo_blocks * 300_000
            embargo_cutoff_test = test_first_block + embargo_blocks * 300_000
            df_train_full = df_train_full[df_train_full["block_start_ms"] <= embargo_cutoff_train]
            df_test = df_test[df_test["block_start_ms"] >= embargo_cutoff_test]

        if len(df_train_full) < 1000 or len(df_test) < 100:
            start += args.step_days
            continue

        # Split train into train (85%) and calibration (15%)
        train_block_starts = sorted(df_train_full["block_start_ms"].unique())
        n_train_blocks = len(train_block_starts)
        calib_split = int(n_train_blocks * 0.85)
        calib_blocks = set(train_block_starts[calib_split:])
        real_train_blocks = set(train_block_starts[:calib_split])

        df_train = df_train_full.loc[df_train_full["block_start_ms"].isin(real_train_blocks)]
        df_calib = df_train_full.loc[df_train_full["block_start_ms"].isin(calib_blocks)]

        print(f"\n--- Fold {fold}: train {min(train_dates)}..{max(train_dates)}, "
              f"test {min(test_dates)}..{max(test_dates)} ---")
        print(f"  Train: {len(df_train):,}, Calib: {len(df_calib):,}, Test: {len(df_test):,}")

        # Train LightGBM (same as standard model)
        X_train = df_train[FEATURE_COLUMNS_V3].values
        y_train = df_train["target"].values.astype(int)
        X_calib = df_calib[FEATURE_COLUMNS_V3].values
        y_calib = df_calib["target"].values.astype(int)
        X_test = df_test[FEATURE_COLUMNS_V3].values
        y_test = df_test["target"].values.astype(int)

        dtrain = lgb.Dataset(X_train, label=y_train,
                             feature_name=FEATURE_COLUMNS_V3, free_raw_data=False)
        dval = lgb.Dataset(X_test, label=y_test,
                           reference=dtrain, free_raw_data=False)

        params = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.01,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7,
            "min_child_samples": 300, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "max_bin": 255, "verbose": -1, "seed": 42, "num_threads": -1,
        }

        model = lgb.train(
            params, dtrain, num_boost_round=3000,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(500), lgb.log_evaluation(500)],
        )
        print(f"  Best iteration: {model.best_iteration}")

        # Get raw predictions on calib and test
        calib_proba = model.predict(X_calib)
        test_proba = model.predict(X_test)
        calib_seconds = df_calib["seconds_to_expiry"].values
        test_seconds = df_test["seconds_to_expiry"].values

        # Compare each calibrator
        for config_name, cal_type, buckets in configs:
            # Fit
            if cal_type == "isotonic":
                cals = fit_isotonic(y_calib, calib_proba, calib_seconds, buckets)
            elif cal_type == "beta2":
                cals = fit_beta(y_calib, calib_proba, calib_seconds, buckets, mode="2param")
            elif cal_type == "beta3":
                cals = fit_beta(y_calib, calib_proba, calib_seconds, buckets, mode="3param")

            # Apply to test
            test_cal = apply_calibrators(test_proba, test_seconds, cals, buckets)

            # Evaluate
            result = evaluate_calibration(
                config_name, y_test, test_cal, test_seconds, phase_buckets
            )
            result["fold"] = fold
            all_fold_results[config_name].append(result)

        start += args.step_days

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"  CALIBRATION COMPARISON SUMMARY ({fold} folds)")
    print(f"{'='*80}")

    print(f"\n  {'Method':<20s} {'Accuracy':>10s} {'Brier':>10s} {'Logloss':>10s} "
          f"{'AUC':>10s} {'ECE':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    summary = {}
    for config_name, _, _ in configs:
        results = all_fold_results[config_name]
        acc = np.mean([r["accuracy"] for r in results])
        brier = np.mean([r["brier"] for r in results])
        ll = np.mean([r["logloss"] for r in results])
        auc = np.mean([r["auc"] for r in results])
        ece = np.mean([r["ece"] for r in results])
        print(f"  {config_name:<20s} {acc:>10.4f} {brier:>10.6f} {ll:>10.6f} "
              f"{auc:>10.4f} {ece:>10.6f}")
        summary[config_name] = {"acc": acc, "brier": brier, "ll": ll, "auc": auc, "ece": ece}

    # Phase comparison
    print(f"\n  BRIER BY PHASE (lower is better)")
    print(f"  {'Phase':<15s}", end="")
    for config_name, _, _ in configs:
        print(f" {config_name:>14s}", end="")
    print()

    for _, _, phase_label in phase_buckets:
        print(f"  {phase_label:<15s}", end="")
        for config_name, _, _ in configs:
            results = all_fold_results[config_name]
            briers = [r["phase"][phase_label]["brier"]
                      for r in results if phase_label in r.get("phase", {})]
            if briers:
                print(f" {np.mean(briers):>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    # ECE by phase
    print(f"\n  ECE BY PHASE (lower is better)")
    print(f"  {'Phase':<15s}", end="")
    for config_name, _, _ in configs:
        print(f" {config_name:>14s}", end="")
    print()

    for _, _, phase_label in phase_buckets:
        print(f"  {phase_label:<15s}", end="")
        for config_name, _, _ in configs:
            results = all_fold_results[config_name]
            eces = [r["phase"][phase_label]["ece"]
                    for r in results if phase_label in r.get("phase", {})]
            if eces:
                print(f" {np.mean(eces):>14.6f}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    # Best method
    best_brier = min(summary.items(), key=lambda x: x[1]["brier"])
    best_ece = min(summary.items(), key=lambda x: x[1]["ece"])
    print(f"\n  BEST by Brier: {best_brier[0]} ({best_brier[1]['brier']:.6f})")
    print(f"  BEST by ECE:   {best_ece[0]} ({best_ece[1]['ece']:.6f})")

    print("\nDone.")


if __name__ == "__main__":
    main()

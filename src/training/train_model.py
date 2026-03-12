"""
Train LightGBM multiclass model for BTC 15-second price prediction.

Validation strategies:
  1. Simple temporal split (80/20)
  2. Walk-forward validation (train N days, test next K days, advance)

Usage:
  python -m src.training.train_model
  python -m src.training.train_model --walkforward --train-days 60 --test-days 7
  python -m src.training.train_model --dataset-dir data/dataset --output-dir models
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.features.feature_engine import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset"
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "models"

TARGET_NAMES = ["STRONG_DOWN", "DOWN", "FLAT", "UP", "STRONG_UP"]

# LightGBM hyperparameters (from DESIGN.txt section 7)
PARAMS = {
    "objective":        "multiclass",
    "num_class":        5,
    "metric":           "multi_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "learning_rate":    0.03,
    "subsample":        0.8,
    "subsample_freq":   1,
    "colsample_bytree": 0.7,
    "min_child_samples": 100,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "max_bin":          255,
    "verbose":          -1,
    "seed":             42,
    "num_threads":      -1,
}

NUM_BOOST_ROUND = 5000
EARLY_STOPPING  = 200


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
        print(f"  Loaded {f.name}: {len(df):,} rows")

    full = pd.concat(dfs, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    print(f"  Total: {len(full):,} rows, {len(files)} days")
    return full


def split_temporal(df, train_frac=0.8):
    """Simple temporal split: first train_frac% for train, rest for test."""
    n = len(df)
    split_idx = int(n * train_frac)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    return train, test


def get_day_boundaries(df):
    """Return sorted list of unique dates from timestamp_ms."""
    dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
    return sorted(dates.unique())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_sample_weights(y):
    """Compute sample weights inversely proportional to class frequency."""
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    # weight = n_samples / (n_classes * count_of_class)
    class_weights = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weights[yi] for yi in y], dtype=np.float64)


def train_lgb(X_train, y_train, X_val, y_val):
    """Train LightGBM with early stopping and class balancing. Returns (model, best_iteration)."""
    w_train = _compute_sample_weights(y_train)
    w_val   = _compute_sample_weights(y_val)
    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train, feature_name=FEATURE_COLUMNS, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain, free_raw_data=False)

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING),
            lgb.log_evaluation(50),
        ],
    )
    return model, model.best_iteration


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, change_bps, label=""):
    """Full evaluation: accuracy, F1, confusion matrix, STRONG precision, PnL sim."""
    y_proba = model.predict(X)
    y_pred  = y_proba.argmax(axis=1)

    acc   = accuracy_score(y, y_pred)
    f1_m  = f1_score(y, y_pred, average="macro")
    ll    = log_loss(y, y_proba, labels=[0, 1, 2, 3, 4])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  F1 macro:      {f1_m:.4f}")
    print(f"  Log loss:      {ll:.4f}")
    print(f"  Samples:       {len(y):,}")

    # Classification report
    print(f"\n  Classification report:")
    report = classification_report(y, y_pred, target_names=TARGET_NAMES, digits=4)
    for line in report.split("\n"):
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4])
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"    {'':15s}", end="")
    for name in TARGET_NAMES:
        print(f"{name:>12s}", end="")
    print()
    for i, name in enumerate(TARGET_NAMES):
        print(f"    {name:15s}", end="")
        for j in range(5):
            print(f"{cm[i, j]:12,}", end="")
        print()

    # Worst error: STRONG_UP predicted but actually STRONG_DOWN (and vice versa)
    worst_up   = cm[0, 4]  # actual STRONG_DOWN, predicted STRONG_UP
    worst_down = cm[4, 0]  # actual STRONG_UP, predicted STRONG_DOWN
    total_strong_pred = (y_pred == 0).sum() + (y_pred == 4).sum()
    print(f"\n  Catastrophic errors (STRONG predicted as opposite STRONG):")
    print(f"    Pred STRONG_UP  but actual STRONG_DOWN: {worst_up:,}")
    print(f"    Pred STRONG_DOWN but actual STRONG_UP:  {worst_down:,}")
    if total_strong_pred > 0:
        print(f"    Catastrophic rate: {(worst_up + worst_down) / total_strong_pred:.4f}")

    # FLAT coverage: how often the model predicts FLAT (= "don't trade")
    flat_pct = (y_pred == 2).sum() / len(y_pred) * 100
    print(f"\n  FLAT coverage (no-trade): {flat_pct:.1f}%")

    # Simple PnL simulation
    _simulate_pnl(y_proba, y_pred, change_bps, label)

    return {
        "accuracy": acc, "f1_macro": f1_m, "log_loss": ll,
        "confusion_matrix": cm.tolist(),
    }


def _simulate_pnl(y_proba, y_pred, change_bps, label):
    """Simulate PnL: trade only when P(STRONG) > threshold."""
    print(f"\n  PnL simulation (trade on STRONG signals):")

    for threshold in [0.3, 0.4, 0.5]:
        pnl = 0.0
        n_trades = 0
        pnl_list = []

        for i in range(len(y_pred)):
            p_strong_up   = y_proba[i, 4]
            p_strong_down = y_proba[i, 0]

            if p_strong_up > threshold:
                # Go long
                pnl_list.append(change_bps[i])
                pnl += change_bps[i]
                n_trades += 1
            elif p_strong_down > threshold:
                # Go short
                pnl_list.append(-change_bps[i])
                pnl -= change_bps[i]
                n_trades += 1

        if n_trades > 0:
            pnl_arr = np.array(pnl_list)
            avg_pnl = pnl_arr.mean()
            win_rate = (pnl_arr > 0).sum() / n_trades * 100
            sharpe = avg_pnl / pnl_arr.std() * np.sqrt(86400) if pnl_arr.std() > 0 else 0

            # Max drawdown
            cumulative = np.cumsum(pnl_arr)
            peak = np.maximum.accumulate(cumulative)
            max_dd = (cumulative - peak).min()

            print(f"    threshold={threshold}: {n_trades:,} trades, "
                  f"total={pnl:.1f}bps, avg={avg_pnl:.2f}bps, "
                  f"win={win_rate:.1f}%, sharpe={sharpe:.2f}, maxDD={max_dd:.1f}bps")
        else:
            print(f"    threshold={threshold}: 0 trades")


def print_feature_importance(model, top_n=30):
    """Print top features by importance (split count)."""
    importance = model.feature_importance(importance_type="split")
    sorted_idx = np.argsort(importance)[::-1]

    print(f"\n{'='*60}")
    print(f"  Top {top_n} features (by split count)")
    print(f"{'='*60}")
    for rank, idx in enumerate(sorted_idx[:top_n], 1):
        print(f"  {rank:3d}. {FEATURE_COLUMNS[idx]:45s} {importance[idx]:,}")


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward(df, train_days=60, test_days=7, step_days=7):
    """
    Walk-forward validation: train on [0..train_days], test on [train_days..train_days+test_days],
    advance by step_days, repeat.
    """
    dates = get_day_boundaries(df)
    n_dates = len(dates)
    print(f"\nWalk-forward: {n_dates} days available, train={train_days}d, test={test_days}d, step={step_days}d")

    all_results = []
    fold = 0
    start = 0

    while start + train_days + test_days <= n_dates:
        fold += 1
        train_dates = set(dates[start:start + train_days])
        test_dates  = set(dates[start + train_days:start + train_days + test_days])

        df_dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
        train_mask = df_dates.isin(train_dates)
        test_mask  = df_dates.isin(test_dates)

        df_train = df.loc[train_mask]
        df_test  = df.loc[test_mask]

        print(f"\n--- Fold {fold}: train {min(train_dates)}..{max(train_dates)} ({len(df_train):,}), "
              f"test {min(test_dates)}..{max(test_dates)} ({len(df_test):,}) ---")

        X_train = df_train[FEATURE_COLUMNS].values
        y_train = df_train["target"].values.astype(int)
        X_test  = df_test[FEATURE_COLUMNS].values
        y_test  = df_test["target"].values.astype(int)
        change_test = df_test["change_bps"].values

        model, best_iter = train_lgb(X_train, y_train, X_test, y_test)
        print(f"  Best iteration: {best_iter}")

        result = evaluate(model, X_test, y_test, change_test,
                          label=f"Fold {fold} test ({min(test_dates)} to {max(test_dates)})")
        result["fold"] = fold
        result["train_start"] = str(min(train_dates))
        result["train_end"]   = str(max(train_dates))
        result["test_start"]  = str(min(test_dates))
        result["test_end"]    = str(max(test_dates))
        result["best_iteration"] = best_iter
        all_results.append(result)

        start += step_days

    # Summary
    if all_results:
        accs = [r["accuracy"] for r in all_results]
        f1s  = [r["f1_macro"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"  Walk-forward summary ({len(all_results)} folds)")
        print(f"{'='*60}")
        print(f"  Accuracy:  mean={np.mean(accs):.4f}, std={np.std(accs):.4f}, "
              f"min={np.min(accs):.4f}, max={np.max(accs):.4f}")
        print(f"  F1 macro:  mean={np.mean(f1s):.4f}, std={np.std(f1s):.4f}, "
              f"min={np.min(f1s):.4f}, max={np.max(f1s):.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_model(model, results, output_dir=None):
    """Save model, feature columns, and metrics."""
    d = Path(output_dir) if output_dir else OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)

    # Model (native LightGBM format)
    model_path = d / "lightgbm_v2.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Feature columns
    cols_path = d / "feature_columns.txt"
    cols_path.write_text("\n".join(FEATURE_COLUMNS))
    print(f"Feature columns saved: {cols_path}")

    # Metrics
    metrics_path = d / "metrics.json"
    # Convert numpy types for JSON
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(results, default=_convert))
    metrics_path.write_text(json.dumps(clean, indent=2))
    print(f"Metrics saved: {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--walkforward", action="store_true", help="Use walk-forward validation")
    parser.add_argument("--train-days", type=int, default=60, help="Walk-forward train window")
    parser.add_argument("--test-days", type=int, default=7, help="Walk-forward test window")
    parser.add_argument("--step-days", type=int, default=7, help="Walk-forward step size")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Simple split train fraction")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.dataset_dir)

    # Target distribution
    print(f"\nTarget distribution:")
    for cls in range(5):
        n = (df["target"] == cls).sum()
        print(f"  {cls} ({TARGET_NAMES[cls]:12s}): {n:>10,} ({n / len(df) * 100:.1f}%)")

    if args.walkforward:
        # Walk-forward validation
        wf_results = walk_forward(df, args.train_days, args.test_days, args.step_days)

        # Train final model on all data (for deployment)
        print(f"\n{'='*60}")
        print(f"  Training final model on ALL data ({len(df):,} rows)")
        print(f"{'='*60}")
        X_all = df[FEATURE_COLUMNS].values
        y_all = df["target"].values.astype(int)

        # Use last test_days as validation for early stopping only
        dates = get_day_boundaries(df)
        val_dates = set(dates[-args.test_days:])
        df_dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
        val_mask = df_dates.isin(val_dates)
        X_val = df.loc[val_mask, FEATURE_COLUMNS].values
        y_val = df.loc[val_mask, "target"].values.astype(int)

        final_model, best_iter = train_lgb(X_all, y_all, X_val, y_val)
        print(f"  Final model best iteration: {best_iter}")
        print_feature_importance(final_model)
        save_model(final_model, {"walk_forward": wf_results, "best_iteration": best_iter},
                   args.output_dir)

    else:
        # Simple temporal split
        train_df, test_df = split_temporal(df, args.train_frac)
        print(f"\nTemporal split: train={len(train_df):,}, test={len(test_df):,}")

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df["target"].values.astype(int)
        X_test  = test_df[FEATURE_COLUMNS].values
        y_test  = test_df["target"].values.astype(int)
        change_test = test_df["change_bps"].values

        model, best_iter = train_lgb(X_train, y_train, X_test, y_test)
        print(f"  Best iteration: {best_iter}")

        evaluate(model, X_test, y_test, change_test, label="Test set (temporal split)")
        print_feature_importance(model)
        save_model(model, {"mode": "temporal_split", "train_frac": args.train_frac,
                           "best_iteration": best_iter}, args.output_dir)

    print("\nDone.")
    sys.exit(0)


if __name__ == "__main__":
    main()

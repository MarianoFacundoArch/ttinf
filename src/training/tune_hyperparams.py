"""
Optuna hyperparameter tuning for LightGBM walk-forward.

Optimizes LightGBM params using walk-forward AUC as the objective.
Uses MedianPruner to cut bad trials early (after 2 folds).

Usage:
  python -m src.training.tune_hyperparams --n-trials 100
  python -m src.training.tune_hyperparams --n-trials 50 --timeout 7200
  python -m src.training.tune_hyperparams --resume  # continue from previous study
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from src.features.feature_engine_v3 import FEATURE_COLUMNS_V3
from src.training.train_model_v3 import (
    EARLY_STOPPING,
    get_day_boundaries,
    fit_calibrators,
    predict_with_init_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_v3"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "optuna"
STUDY_NAME = "lgb_walkforward_v3"
DB_PATH = RESULTS_DIR / "optuna_study.db"

# Walk-forward config (fixed during tuning)
TRAIN_DAYS = 56
TEST_DAYS = 14
STEP_DAYS = 7
EMBARGO_BLOCKS = 2

# Training
MAX_BOOST_ROUND = 5000
FEATURE_COLS = FEATURE_COLUMNS_V3


# ---------------------------------------------------------------------------
# Data loading (once)
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir=None):
    """Load all parquet files into single DataFrame."""
    d = Path(dataset_dir) if dataset_dir else DATASET_DIR
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {d}")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    df.sort_values("timestamp_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df):,} rows from {len(files)} files")
    return df


# ---------------------------------------------------------------------------
# Single fold train+evaluate (lightweight, no printing)
# ---------------------------------------------------------------------------

def train_evaluate_fold(params, df_train, df_calib, df_test):
    """Train one fold, return AUC on test set. Returns NaN if training fails."""
    try:
        X_train = df_train[FEATURE_COLS].values
        y_train = df_train["target"].values.astype(int)
        X_test = df_test[FEATURE_COLS].values
        y_test = df_test["target"].values.astype(int)

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS,
                             free_raw_data=False)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain,
                           free_raw_data=False)

        model = lgb.train(
            params, dtrain,
            num_boost_round=MAX_BOOST_ROUND,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING),
                lgb.log_evaluation(0),  # silent
            ],
        )

        y_pred = model.predict(X_test)

        # Calibrate
        X_calib = df_calib[FEATURE_COLS].values
        y_calib = df_calib["target"].values.astype(int)
        calib_proba = model.predict(X_calib)
        calib_seconds = df_calib["seconds_to_expiry"].values
        calibrators = fit_calibrators(y_calib, calib_proba, calib_seconds)

        # Apply calibration to test
        test_seconds = df_test["seconds_to_expiry"].values
        from src.training.train_model_v3 import TIME_BUCKETS
        y_cal = np.copy(y_pred)
        for lo, hi, key in TIME_BUCKETS:
            mask = (test_seconds >= lo) & (test_seconds < hi)
            if mask.any() and key in calibrators:
                y_cal[mask] = calibrators[key].predict(y_pred[mask])
        y_cal = np.clip(y_cal, 0.01, 0.99)

        auc = roc_auc_score(y_test, y_cal)
        acc = float(np.mean((y_cal >= 0.5) == y_test))

        return auc, acc, model.best_iteration

    except Exception as e:
        print(f"    Fold failed: {e}")
        return float('nan'), float('nan'), 0


# ---------------------------------------------------------------------------
# Walk-forward for one set of params
# ---------------------------------------------------------------------------

def walk_forward_objective(params, df, trial=None):
    """Run walk-forward, return mean AUC. Supports Optuna pruning."""
    dates = get_day_boundaries(df)
    n_dates = len(dates)

    fold = 0
    aucs = []
    accs = []
    start = 0

    while start + TRAIN_DAYS + TEST_DAYS <= n_dates:
        fold += 1

        train_dates = set(dates[start:start + TRAIN_DAYS])
        test_dates = set(dates[start + TRAIN_DAYS:start + TRAIN_DAYS + TEST_DAYS])

        df_dates = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date
        train_mask = df_dates.isin(train_dates)
        test_mask = df_dates.isin(test_dates)

        df_train_full = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        # Embargo
        if len(df_train_full) > 0 and len(df_test) > 0:
            train_last_block = df_train_full["block_start_ms"].max()
            test_first_block = df_test["block_start_ms"].min()
            embargo_cutoff_train = train_last_block - EMBARGO_BLOCKS * 300_000
            embargo_cutoff_test = test_first_block + EMBARGO_BLOCKS * 300_000
            df_train_full = df_train_full[df_train_full["block_start_ms"] <= embargo_cutoff_train]
            df_test = df_test[df_test["block_start_ms"] >= embargo_cutoff_test]

        if len(df_train_full) < 1000 or len(df_test) < 100:
            start += STEP_DAYS
            continue

        # Split train into train (85%) and calibration (15%)
        train_block_starts = sorted(df_train_full["block_start_ms"].unique())
        n_train_blocks = len(train_block_starts)
        calib_split = int(n_train_blocks * 0.85)
        calib_blocks = set(train_block_starts[calib_split:])
        real_train_blocks = set(train_block_starts[:calib_split])

        df_train = df_train_full.loc[df_train_full["block_start_ms"].isin(real_train_blocks)]
        df_calib = df_train_full.loc[df_train_full["block_start_ms"].isin(calib_blocks)]

        auc, acc, best_iter = train_evaluate_fold(params, df_train, df_calib, df_test)

        if np.isnan(auc):
            start += STEP_DAYS
            continue

        aucs.append(auc)
        accs.append(acc)

        # Optuna pruning: report intermediate result after each fold
        if trial is not None:
            trial.report(np.mean(aucs), fold - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        start += STEP_DAYS

    if len(aucs) == 0:
        return 0.0, 0.0

    return float(np.mean(aucs)), float(np.mean(accs))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, df):
    """Optuna objective: suggest params, run walk-forward, return AUC."""

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
        "num_threads": -1,

        # Tuned parameters
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
        "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
    }

    mean_auc, mean_acc = walk_forward_objective(params, df, trial=trial)

    # Store accuracy as user attribute for analysis
    trial.set_user_attr("mean_accuracy", mean_acc)

    return mean_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials (default: 100)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds (default: no limit)")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume previous study instead of starting fresh")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading dataset...")
    df = load_dataset(args.dataset_dir)
    print(f"  Days: {len(get_day_boundaries(df))}")
    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(FEATURE_COLS)}")

    # Create or load study
    storage = f"sqlite:///{DB_PATH}"

    if args.resume:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
        print(f"\nResuming study '{STUDY_NAME}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            load_if_exists=True,
        )
        print(f"\nStudy '{STUDY_NAME}' ({len(study.trials)} existing trials)")

    n_existing = len(study.trials)
    print(f"Running {args.n_trials} new trials...")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {TEST_DAYS}d test, {STEP_DAYS}d step")
    print()

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, df),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    # Results
    print(f"\n{'='*60}")
    print(f"  OPTUNA RESULTS")
    print(f"{'='*60}")
    print(f"  Total trials: {len(study.trials)} ({len(study.trials) - n_existing} new)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best accuracy: {study.best_trial.user_attrs.get('mean_accuracy', 'N/A')}")
    print(f"\n  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save best params
    best_params_file = RESULTS_DIR / "best_params.json"
    result = {
        "best_auc": study.best_value,
        "best_accuracy": study.best_trial.user_attrs.get("mean_accuracy"),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
    }
    best_params_file.write_text(json.dumps(result, indent=2))
    print(f"\n  Best params saved to: {best_params_file}")

    # Top 5 trials
    print(f"\n  Top 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        acc = t.user_attrs.get("mean_accuracy", "?")
        print(f"    {i+1}. AUC={t.value:.4f} acc={acc} (trial {t.number})")

    # Comparison with current defaults
    print(f"\n  Current defaults for reference:")
    print(f"    learning_rate=0.01, num_leaves=63, min_child_samples=300")
    print(f"    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0")


if __name__ == "__main__":
    main()

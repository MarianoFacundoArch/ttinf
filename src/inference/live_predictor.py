"""
LivePredictor: loads model + calibrators, computes features from LiveBuffer,
returns calibrated P(Up) prediction.
"""

import json
import pickle
import numpy as np
import lightgbm as lgb
from pathlib import Path

from src.features.feature_engine_v3 import (
    build_ref_price,
    compute_features_v3,
)


# Calibration time buckets (must match train_model_v3.py TIME_BUCKETS)
CALIB_BUCKETS = [
    (240, 300, "240-300s (early)"),
    (180, 240, "180-240s"),
    (120, 180, "120-180s"),
    (60, 120, "60-120s"),
    (30, 60, "30-60s"),
    (0, 30, "0-30s (late)"),
]


class LivePredictor:
    """Loads model and produces calibrated predictions."""

    def __init__(self, model_dir="models"):
        model_dir = Path(model_dir)
        self.model = lgb.Booster(model_file=str(model_dir / "lightgbm_v3.txt"))

        with open(model_dir / "calibrators_v3.pkl", "rb") as f:
            self.calibrators = pickle.load(f)

        # Read feature columns from model file (matches the trained model)
        cols_file = model_dir / "feature_columns_v3.txt"
        with open(cols_file) as fh:
            self.feature_cols = [line.strip() for line in fh if line.strip()]

        # Load model config (residual mode flag)
        config_file = model_dir / "model_config_v3.json"
        if config_file.exists():
            with open(config_file) as fh:
                config = json.load(fh)
            self.residual_mode = config.get("residual_mode", False)
        else:
            self.residual_mode = False

        # Block tracking
        self.current_block_start_ms = 0
        self.open_ref = 0.0
        self.open_ref_age_ms = 0
        self.block_results = []  # list of {'return_bps': float, 'result': int}
        self.block_cache = {}  # cache static features within a block

    def _get_block_start(self, now_ms):
        """Get the deterministic block start for a given timestamp."""
        return (now_ms // 300_000) * 300_000

    def _get_calib_bucket(self, seconds_to_expiry):
        """Get calibrator key for given seconds_to_expiry."""
        for lo, hi, key in CALIB_BUCKETS:
            if lo <= seconds_to_expiry < hi:
                return key
        return "0-30s (late)"

    def update_block(self, now_ms, ref):
        """
        Check if we're in a new block. If so, close the old one and
        set open_ref for the new one.

        Returns True if block changed.
        """
        block_start = self._get_block_start(now_ms)

        if block_start == self.current_block_start_ms:
            return False

        # Close old block (if we had one)
        if self.current_block_start_ms > 0 and len(ref['ts']) > 0:
            old_end = self.current_block_start_ms + 300_000
            # Find close_ref (last ref_price <= block_end)
            idx_close = np.searchsorted(ref['ts'], old_end, side='right') - 1
            if idx_close >= 0:
                close_ref = ref['price'][idx_close]
                return_bps = (close_ref - self.open_ref) / self.open_ref * 10_000
                result = 1 if close_ref >= self.open_ref else 0
                self.block_results.insert(0, {
                    'return_bps': float(return_bps),
                    'result': result,
                })
                # Keep only last 6
                self.block_results = self.block_results[:6]

        # Set new block
        self.current_block_start_ms = block_start
        self.block_cache = {}  # reset cache for new block

        # Find open_ref: last ref_price <= block_start
        if len(ref['ts']) > 0:
            idx = np.searchsorted(ref['ts'], block_start, side='right') - 1
            if idx >= 0:
                self.open_ref = ref['price'][idx]
                self.open_ref_age_ms = block_start - int(ref['ts'][idx])
            else:
                self.open_ref = ref['price'][0]
                self.open_ref_age_ms = 999999
        else:
            self.open_ref = 0.0
            self.open_ref_age_ms = 999999

        return True

    def predict(self, buffer, now_ms=None):
        """
        Compute features and predict P(Up) for the current moment.

        Args:
            buffer: LiveBuffer instance
            now_ms: current timestamp in ms (default: now)

        Returns:
            dict with prediction info, or None if not enough data
        """
        import time as _time
        if now_ms is None:
            now_ms = int(_time.time() * 1000)

        # Build DayData from buffer
        day = buffer.to_day_data()

        # Build ref_price
        ref = build_ref_price(day)

        if len(ref['ts']) == 0:
            return None

        # Update block (sets open_ref, closes previous)
        self.update_block(now_ms, ref)

        if self.open_ref <= 0:
            return None

        block_start = self.current_block_start_ms
        block_end = block_start + 300_000
        seconds_to_expiry = max(0, (block_end - now_ms) / 1000.0)

        # Compute features (block_cache avoids recomputing static features)
        feats = compute_features_v3(
            day, ref, now_ms, block_start, self.open_ref,
            open_ref_age_ms=self.open_ref_age_ms,
            block_results=self.block_results,
            block_cache=self.block_cache,
        )

        # Build feature vector
        X = np.array([[feats.get(col, 0.0) for col in self.feature_cols]])

        # Predict (residual mode: add brownian logit back)
        if self.residual_mode:
            brownian_p = feats.get("brownian_prob_drift", feats.get("brownian_prob", 0.5))
            if not np.isfinite(brownian_p):
                brownian_p = 0.5
            brownian_p = np.clip(brownian_p, 1e-4, 1 - 1e-4)
            init_score = np.log(brownian_p / (1 - brownian_p))
            raw_margin = float(self.model.predict(X, raw_score=True)[0])
            p_raw = float(1.0 / (1.0 + np.exp(-(init_score + raw_margin))))
        else:
            p_raw = float(self.model.predict(X)[0])

        # Calibrate
        bucket_key = self._get_calib_bucket(seconds_to_expiry)
        if bucket_key in self.calibrators:
            p_cal = float(self.calibrators[bucket_key].predict([p_raw])[0])
            p_cal = np.clip(p_cal, 0.01, 0.99)
        else:
            p_cal = p_raw

        # Current ref_price
        idx_now = np.searchsorted(ref['ts'], now_ms, side='right') - 1
        price_now = ref['price'][idx_now] if idx_now >= 0 else self.open_ref

        return {
            "block_start_ms": block_start,
            "now_ms": now_ms,
            "seconds_to_expiry": seconds_to_expiry,
            "open_ref": self.open_ref,
            "price_now": price_now,
            "dist_to_open_bps": feats.get("dist_to_open_bps", 0.0),
            "p_raw": p_raw,
            "p_calibrated": p_cal,
            "confidence": abs(p_cal - 0.5),
            "direction": "UP" if p_cal >= 0.5 else "DOWN",
            "brownian_prob": feats.get("brownian_prob", 0.5),
        }

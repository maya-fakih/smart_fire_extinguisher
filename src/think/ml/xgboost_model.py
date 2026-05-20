# src/think/ml/xgboost_model.py

import logging
import xgboost as xgb
import numpy as np
from think.ml.base_model import BaseModel
from exceptions import ModelError

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        xgb_cfg = config.get("think", {}).get("xgboost", {})
        self._n_estimators = xgb_cfg.get("n_estimators", 100)
        self._max_depth = xgb_cfg.get("max_depth", 4)
        self._learning_rate = xgb_cfg.get("learning_rate", 0.1)
        self._subsample = xgb_cfg.get("subsample", 0.8)
        self._colsample_bytree = xgb_cfg.get("colsample_bytree", 0.8)

    def fit(self, X, y):
        try:
            logger.debug(f"XGBoostModel: fitting | X_shape={X.shape} | y_shape={np.array(y).shape}")
            # Shift labels 1-5 → 0-4 to match XGBoost's expected range.
            # predict() adds 1 back, so danger_level output is always 1-5.
            y_shifted = [label - 1 for label in y]
            self._model = xgb.XGBClassifier(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                learning_rate=self._learning_rate,
                subsample=self._subsample,
                colsample_bytree=self._colsample_bytree,
                objective="multi:softmax",
                num_class=5,
                random_state=42,
            )
            self._model.fit(X, y_shifted)
            logger.info(f"XGBoostModel: training completed | n_samples={len(X)}")
        except Exception as e:
            logger.error(
                f"XGBoostModel: training failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise ModelError(f"Training failed: {e}")

    def predict(self, features: dict) -> int:
        if self._model is None:
            logger.error("XGBoostModel: model not loaded")
            raise ModelError("Model not loaded")
        if not features:
            logger.warning("XGBoostModel: empty features dict, returning default danger_level=1")
            return 1
        try:
            logger.debug(f"XGBoostModel: predicting | feature_count={len(features)}")
            # Feature-ordering safety: sort by key so the column order here is
            # identical to the column order used at fit() time. Both paths call
            # build_feature_vector (same keys) then sorted() — guaranteed match.
            # Without this, dict insertion order could silently desync train vs
            # predict and XGBoost would learn garbage.
            vals = [features[k] for k in sorted(features.keys())]
            X = np.array([vals])
            pred = self._model.predict(X)[0]
            danger_level = int(pred) + 1
            logger.debug(f"XGBoostModel: prediction_result | danger_level={danger_level}")
            return danger_level
        except Exception as e:
            logger.error(
                f"XGBoostModel: prediction failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise ModelError(f"Prediction failed: {e}")

    def save(self, path: str):
        if self._model is None:
            logger.error("XGBoostModel: no model to save")
            raise ModelError("No model to save")
        try:
            logger.debug(f"XGBoostModel: saving model | path={path}")
            # Defensive: first-run case — the model_weights folder may not
            # exist yet. xgboost.save_model raises on a missing parent dir.
            import os
            os.makedirs(path, exist_ok=True)
            self._model.save_model(f"{path}/xgboost_model.json")
            logger.info(f"XGBoostModel: model saved | path={path}/xgboost_model.json")
        except Exception as e:
            logger.error(
                f"XGBoostModel: save failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise ModelError(f"Failed to save model: {e}")

    def load(self, path: str):
        try:
            logger.debug(f"XGBoostModel: loading model | path={path}")
            self._model = xgb.XGBClassifier()
            self._model.load_model(f"{path}/xgboost_model.json")
            logger.info(f"XGBoostModel: model loaded | path={path}/xgboost_model.json")
        except Exception as e:
            logger.error(
                f"XGBoostModel: load failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise ModelError(f"Failed to load model: {e}")

    def feature_importance(self) -> dict:
        if self._model is None:
            logger.error("XGBoostModel: model not loaded for feature importance")
            raise ModelError("Model not loaded")
        try:
            logger.debug("XGBoostModel: calculating feature importance")
            importance = self._model.get_booster().get_score(importance_type="weight")
            result = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            logger.debug(f"XGBoostModel: feature_importance | count={len(result)}")
            return result
        except Exception as e:
            logger.error(
                f"XGBoostModel: feature_importance failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise ModelError(f"Failed to get feature importance: {e}")
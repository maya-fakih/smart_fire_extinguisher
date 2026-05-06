# src/think/ml/xgboost_model.py

import xgboost as xgb
import numpy as np
from think.ml.base_model import BaseModel
from exceptions import ModelError


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
            self._model.fit(X, y)
        except Exception as e:
            raise ModelError(f"Training failed: {e}")

    def predict(self, features: dict) -> int:
        if self._model is None:
            raise ModelError("Model not loaded")
        if not features:
            return 1
        try:
            vals = list(features.values())
            X = np.array([vals])
            pred = self._model.predict(X)[0]
            return int(pred) + 1
        except Exception as e:
            raise ModelError(f"Prediction failed: {e}")

    def save(self, path: str):
        if self._model is None:
            raise ModelError("No model to save")
        try:
            self._model.save_model(f"{path}/xgboost_model.json")
        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")

    def load(self, path: str):
        try:
            self._model = xgb.XGBClassifier()
            self._model.load_model(f"{path}/xgboost_model.json")
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")

    def feature_importance(self) -> dict:
        if self._model is None:
            raise ModelError("Model not loaded")
        try:
            importance = self._model.get_booster().get_score(importance_type="weight")
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            raise ModelError(f"Failed to get feature importance: {e}")
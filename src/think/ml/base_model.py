# src/think/ml/base_model.py

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


class BaseModel(ABC):
    def __init__(self, config: dict):
        self._config = config
        self._model = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, features: dict) -> int:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def feature_importance(self) -> dict:
        pass

    def evaluate(self, X, y_true) -> dict:
        if self._model is None:
            raise ValueError("Model not trained or loaded")

        y_pred = self._model.predict(X)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def evaluate_per_class(self, X, y_true) -> dict:
        if self._model is None:
            raise ValueError("Model not trained or loaded")

        y_pred = self._model.predict(X)
        classes = sorted(set(y_true) | set(y_pred))

        return {
            "precision": precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0).tolist(),
            "recall": recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0).tolist(),
            "f1": f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0).tolist(),
            "classes": classes,
        }
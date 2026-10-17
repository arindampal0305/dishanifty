"""Shared interface so Random Forest, XGBoost, and LSTM are interchangeable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

N_CLASSES = 3  # 0 = DOWN, 1 = NEUTRAL, 2 = UP


class DirectionModel(ABC):
    """
    Every model takes the same features and returns a (n_rows, 3) probability
    matrix whose columns are always ordered DOWN, NEUTRAL, UP — regardless of
    which classes happened to appear in its training fold.
    """

    name: str = "base"
    lookback: int = 1  # rows of history each prediction needs; >1 for sequence models

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectionModel":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        """
        context: the `lookback - 1` rows immediately preceding X, used by
        sequence models so the first prediction in a fold is not thrown away.
        Tabular models ignore it.
        """
        ...

    def predict(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba(X, context), axis=1)

    def save(self, stem: Path) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, stem.with_suffix(".joblib"))

    @classmethod
    def load(cls, stem: Path) -> "DirectionModel":
        return joblib.load(stem.with_suffix(".joblib"))


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Expand a classifier's probability matrix to all three classes.

    A fold can legitimately contain only two of the three labels; scikit-learn
    then returns two columns and `classes_` tells us which. Dropping that on the
    floor would silently mislabel UP as NEUTRAL, so pad the missing class with
    zero probability instead.
    """
    if proba.shape[1] == N_CLASSES and list(classes) == list(range(N_CLASSES)):
        return proba

    full = np.zeros((proba.shape[0], N_CLASSES))
    for col, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, col]
    return full


def build_sequences(array: np.ndarray, lookback: int) -> np.ndarray:
    """
    Turn (n_rows, n_features) into (n_rows - lookback + 1, lookback, n_features).
    Sequence i ends on row i + lookback - 1, so it only ever looks backwards.
    """
    if len(array) < lookback:
        raise ValueError(f"Need at least {lookback} rows to build a sequence, got {len(array)}")
    windows = [array[i - lookback + 1: i + 1] for i in range(lookback - 1, len(array))]
    return np.stack(windows)

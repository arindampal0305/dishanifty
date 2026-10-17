"""Model 1: Random Forest — the nonlinear baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.base import DirectionModel, align_proba


class RandomForestDirectionModel(DirectionModel):
    name = "Random Forest"

    def __init__(self, **overrides):
        params = dict(
            n_estimators=500,
            max_depth=8,
            # Daily returns are mostly noise; shallow trees with fat leaves stop
            # the forest from memorising individual days.
            min_samples_leaf=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        params.update(overrides)
        self.model = RandomForestClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestDirectionModel":
        self.model.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        return align_proba(self.model.predict_proba(X), self.model.classes_)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)

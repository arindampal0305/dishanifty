"""Model 2: XGBoost — gradient boosted trees."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.models.base import DirectionModel, align_proba


class XGBoostDirectionModel(DirectionModel):
    name = "XGBoost"

    def __init__(self, **overrides):
        # Tuned on the walk-forward validation folds only. The previous settings
        # (400 trees, depth 4) reached 0.88 train accuracy against 0.35 validation
        # — the booster was memorising individual days. Depth-2 stumps with heavy
        # L2 and fat leaves cut the gap from +0.53 to +0.11 and lifted validation
        # accuracy by ~2 points.
        params = dict(
            objective="multi:softprob",
            num_class=3,
            n_estimators=150,
            learning_rate=0.02,
            max_depth=2,
            subsample=0.6,
            colsample_bytree=0.5,
            min_child_weight=40,
            reg_lambda=20.0,
            reg_alpha=1.0,
            gamma=0.5,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        params.update(overrides)
        self.model = XGBClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostDirectionModel":
        # XGBoost has no class_weight parameter, so balance via sample weights.
        weights = compute_sample_weight("balanced", y)
        self.model.fit(X, y, sample_weight=weights, verbose=False)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        return align_proba(self.model.predict_proba(X), self.model.classes_)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self.feature_names_
        ).sort_values(ascending=False)

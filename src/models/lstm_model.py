"""Model 3: LSTM — the sequence model, fed 30 days of history per prediction."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.models.base import DirectionModel, N_CLASSES, build_sequences


class LSTMDirectionModel(DirectionModel):
    name = "LSTM"

    def __init__(self, lookback: int = 30, units: tuple[int, int] = (64, 32),
                 dropout: float = 0.2, learning_rate: float = 1e-3,
                 epochs: int = 60, batch_size: int = 32, seed: int = 42,
                 clip: float = 5.0):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        # StandardScaler, not MinMaxScaler. Daily returns are heavy-tailed: a
        # handful of COVID-era days set the min and max, and MinMax then squashes
        # the middle 50% of days into ~5% of the [0,1] range. The network sees
        # near-constant input and its loss sits at ln(3) forever.
        self.scaler = StandardScaler()
        self.clip = clip  # z-score cap, so single outliers can't dominate again
        self.model = None

    def _build(self, n_features: int):
        import tensorflow as tf
        from tensorflow import keras

        tf.keras.utils.set_random_seed(self.seed)
        model = keras.Sequential([
            keras.layers.Input(shape=(self.lookback, n_features)),
            keras.layers.LSTM(self.units[0], return_sequences=True),
            keras.layers.Dropout(self.dropout),
            keras.layers.LSTM(self.units[1]),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(N_CLASSES, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMDirectionModel":
        from tensorflow import keras

        self.feature_names_ = list(X.columns)
        scaled = np.clip(self.scaler.fit_transform(X), -self.clip, self.clip)  # fit on train only

        sequences = build_sequences(scaled, self.lookback)
        targets = y.to_numpy()[self.lookback - 1:]

        # Chronological hold-out for early stopping. A random split would leak
        # future days into the stopping decision.
        split = int(len(sequences) * 0.85)
        classes = np.unique(targets[:split])
        weights = compute_class_weight("balanced", classes=classes, y=targets[:split])
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        self.model = self._build(scaled.shape[1])
        self.model.fit(
            sequences[:split], targets[:split],
            validation_data=(sequences[split:], targets[split:]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True)],
            verbose=0,
        )
        return self

    def predict_proba(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")

        needed = self.lookback - 1
        if context is not None and len(context) >= needed:
            history = context.iloc[-needed:][self.feature_names_]
        else:
            # No history available (very start of the series): repeat the first
            # row so the window is full. Affects only the first 29 predictions.
            first = X.iloc[[0]][self.feature_names_]
            available = context[self.feature_names_] if context is not None else first.iloc[0:0]
            pad = pd.concat([first] * (needed - len(available)))
            history = pd.concat([pad, available])

        combined = pd.concat([history, X[self.feature_names_]])
        scaled = np.clip(self.scaler.transform(combined), -self.clip, self.clip)
        sequences = build_sequences(scaled, self.lookback)
        return self.model.predict(sequences, verbose=0)

    # Keras graphs do not survive joblib, so the network is saved separately
    # from the scaler and hyperparameters.
    def save(self, stem: Path) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(stem.with_suffix(".keras"))
        state = {k: v for k, v in self.__dict__.items() if k != "model"}
        joblib.dump(state, stem.with_suffix(".joblib"))

    @classmethod
    def load(cls, stem: Path) -> "LSTMDirectionModel":
        from tensorflow import keras

        state = joblib.load(stem.with_suffix(".joblib"))
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj.model = keras.models.load_model(stem.with_suffix(".keras"))
        return obj

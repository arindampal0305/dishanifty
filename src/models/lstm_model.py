"""
lstm_model.py — DishaNifty Step 12-14 : LSTM model (Khushi's part)

Covers:
  Step 12 : Scale features, build LSTM sequence data (30-day lookback)
  Step 13 : Build and tune LSTM on all walk-forward folds
  Step 14 : Run backtest for LSTM

Design notes / assumptions (documented so they can be defended to the mentor):
  - Uses src/validation.py for fold boundaries. Does not redefine them.
  - Uses src/backtest.py for the backtest engine. Does not redefine it.
  - Feature set = all engineered indicator columns + Close + Volume.
    Open/High/Low are dropped because ATR_14 already encodes the High-Low
    range and Close is the reference price everywhere else in the pipeline.
  - Scaling: MinMaxScaler, fit ONLY on each fold's training window, then
    applied to both that fold's train and validation window. A fresh
    scaler is fit per fold (this matches walk-forward validation: the
    model must never see val/test statistics during fit).
  - Sequence construction: to predict day t's Target (direction of the
    t -> t+1 move) the model consumes the trailing `lookback` days of
    scaled features ending at day t (i.e. days t-lookback+1 ... t).
    For the first (lookback-1) days of a validation window, the required
    history necessarily comes from the preceding training window. This
    is NOT leakage: those are real past days, already known at time t,
    and the scaler used to transform them was fit on train only.
  - Class imbalance: handled via sklearn's balanced class weights during
    training (not by resampling), per the roadmap's guidance in Section 4.
  - Output format matches src/backtest.py's contract exactly: a
    pandas Series named 'Predicted_Target', indexed by date, values in
    {0, 1, 2} (DOWN / NEUTRAL / UP) -- ready to pass straight into
    run_backtest() / run_walk_forward_backtest().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow / Keras is only needed once you actually train — importing it
# lazily inside functions keeps this module importable (e.g. for unit
# testing make_sequences / prepare_fold) even in environments where TF
# is not installed yet.


FEATURE_COLS = [
    "Close", "Volume", "Return", "Volatility_10",
    "SMA_10", "SMA_20", "SMA_50",
    "EMA_10", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width",
    "ATR_14", "Momentum_10",
]

TARGET_COL = "Target"
DEFAULT_LOOKBACK = 30


# --------------------------------------------------------------------------
# Step 12a : Scaling
# --------------------------------------------------------------------------

def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> MinMaxScaler:
    """Fit a MinMaxScaler on the TRAIN window only. Never fit on val/test."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler,
                  feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(df[feature_cols])
    return out


# --------------------------------------------------------------------------
# Step 12b : Sequence construction (30-day lookback)
# --------------------------------------------------------------------------

def make_sequences(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS,
                    target_col: str = TARGET_COL, lookback: int = DEFAULT_LOOKBACK):
    """
    Build (X, y, dates) from a chronologically-sorted, already-scaled df.

    X[i] = features for days [t-lookback+1 .. t]   shape (lookback, n_features)
    y[i] = Target at day t                          (label for the t->t+1 move)
    dates[i] = day t (the date the prediction is "as of")

    Requires len(df) >= lookback.
    """
    if len(df) < lookback:
        return (np.empty((0, lookback, len(feature_cols))),
                np.empty((0,), dtype=int),
                pd.DatetimeIndex([]))

    data = df[feature_cols].to_numpy(dtype="float32")
    targets = df[target_col].to_numpy()
    dates = df.index.to_numpy()

    X, y, seq_dates = [], [], []
    for i in range(lookback - 1, len(df)):
        X.append(data[i - lookback + 1: i + 1])
        y.append(targets[i])
        seq_dates.append(dates[i])

    return np.array(X), np.array(y, dtype=int), pd.DatetimeIndex(seq_dates)


def prepare_fold(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  feature_cols: list[str] = FEATURE_COLS,
                  lookback: int = DEFAULT_LOOKBACK) -> dict:
    """
    Turn a (train_df, val_df) pair -- as produced by a fold from
    src.validation.get_walk_forward_folds() or get_test_split() -- into
    scaled, sequenced train/val tensors ready for the LSTM.

    Scaler is fit on train_df only. train_df and val_df are concatenated
    BEFORE sequencing so that the first (lookback-1) rows of val_df have
    valid history to draw on; the resulting sequences are then split back
    out by target date so no val-window label ever appears in the
    training set.
    """
    scaler = fit_scaler(train_df, feature_cols)

    combined = pd.concat([train_df, val_df]).sort_index()
    combined_scaled = apply_scaler(combined, scaler, feature_cols)

    X_all, y_all, dates_all = make_sequences(combined_scaled, feature_cols, TARGET_COL, lookback)

    val_start = np.datetime64(val_df.index.min())
    is_val = dates_all >= val_start

    return {
        "X_train": X_all[~is_val], "y_train": y_all[~is_val], "dates_train": dates_all[~is_val],
        "X_val": X_all[is_val], "y_val": y_all[is_val], "dates_val": dates_all[is_val],
        "scaler": scaler,
    }


# --------------------------------------------------------------------------
# Step 13a : Model architecture
# --------------------------------------------------------------------------

def build_lstm_model(input_shape: tuple, lstm_units=(64, 32), dropout: float = 0.3,
                      learning_rate: float = 0.001):
    """2 LSTM layers + dropout + dense head, per roadmap Section 5.3."""
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(lstm_units[0], return_sequences=True),
        keras.layers.Dropout(dropout),
        keras.layers.LSTM(lstm_units[1]),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),  # DOWN=0, NEUTRAL=1, UP=2
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_class_weights(y_train: np.ndarray) -> dict:
    """Balanced class weights to counter NEUTRAL-day dominance (roadmap Section 4)."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


# --------------------------------------------------------------------------
# Step 13b : Train + evaluate a single fold
# --------------------------------------------------------------------------

def train_and_evaluate_fold(train_df: pd.DataFrame, val_df: pd.DataFrame,
                             feature_cols: list[str] = FEATURE_COLS,
                             lookback: int = DEFAULT_LOOKBACK,
                             lstm_units=(64, 32), dropout: float = 0.3,
                             learning_rate: float = 0.001,
                             epochs: int = 50, batch_size: int = 32,
                             fold_label: str = "", verbose: int = 0) -> dict:
    from tensorflow import keras

    data = prepare_fold(train_df, val_df, feature_cols, lookback)
    class_weights = get_class_weights(data["y_train"])

    model = build_lstm_model(
        input_shape=(lookback, len(feature_cols)),
        lstm_units=lstm_units, dropout=dropout, learning_rate=learning_rate,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=verbose,
    )

    val_probs = model.predict(data["X_val"], verbose=0)
    val_preds = np.argmax(val_probs, axis=1)

    acc = accuracy_score(data["y_val"], val_preds)
    macro_f1 = f1_score(data["y_val"], val_preds, average="macro")
    cm = confusion_matrix(data["y_val"], val_preds, labels=[0, 1, 2])

    predictions = pd.Series(
        val_preds, index=pd.DatetimeIndex(data["dates_val"]), name="Predicted_Target"
    )

    return {
        "fold": fold_label,
        "model": model,
        "history": history.history,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "predictions": predictions,
        "scaler": data["scaler"],
        "params": {
            "lookback": lookback, "lstm_units": lstm_units,
            "dropout": dropout, "learning_rate": learning_rate,
        },
    }


# --------------------------------------------------------------------------
# Step 13c : Hyperparameter tuning across the 4 walk-forward folds
# --------------------------------------------------------------------------

def tune_hyperparameters(folds: list[dict], feature_cols: list[str] = FEATURE_COLS,
                          param_grid: list[dict] | None = None,
                          epochs: int = 30, verbose: int = 0) -> pd.DataFrame:
    """
    folds: the list returned by src.validation.get_walk_forward_folds(df)
           -- each item is {'fold': n, 'train': df, 'val': df}.
    param_grid: list of dicts, each with any of
        lookback, lstm_units, dropout, learning_rate.
        Defaults to a small, defensible grid if not given.

    Scores each config by its AVERAGE macro-F1 across all 4 folds
    (not accuracy -- see roadmap Section 7.1 on why accuracy alone is
    misleading under class imbalance). Returns a results table sorted
    best-first; does NOT touch the final 2024-2026 test set.
    """
    if param_grid is None:
        param_grid = [
            {"lookback": 30, "lstm_units": (64, 32), "dropout": 0.2, "learning_rate": 0.001},
            {"lookback": 30, "lstm_units": (64, 32), "dropout": 0.3, "learning_rate": 0.001},
            {"lookback": 30, "lstm_units": (128, 64), "dropout": 0.3, "learning_rate": 0.0005},
            {"lookback": 20, "lstm_units": (64, 32), "dropout": 0.3, "learning_rate": 0.001},
            {"lookback": 45, "lstm_units": (64, 32), "dropout": 0.3, "learning_rate": 0.001},
        ]

    rows = []
    for params in param_grid:
        fold_f1s, fold_accs = [], []
        for fold in folds:
            res = train_and_evaluate_fold(
                fold["train"], fold["val"], feature_cols,
                epochs=epochs, fold_label=f"Fold {fold['fold']}", verbose=verbose, **params
            )
            fold_f1s.append(res["macro_f1"])
            fold_accs.append(res["accuracy"])
        row = {**params, "avg_macro_f1": np.mean(fold_f1s), "avg_accuracy": np.mean(fold_accs)}
        rows.append(row)
        print(f"{params} -> avg macro F1: {row['avg_macro_f1']:.4f} | avg accuracy: {row['avg_accuracy']:.4f}")

    return pd.DataFrame(rows).sort_values("avg_macro_f1", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Final retrain (2015-2023) + evaluate once on 2024-2026 test (Step 15 input)
# --------------------------------------------------------------------------

def run_final_test(train_df: pd.DataFrame, test_df: pd.DataFrame, best_params: dict,
                    feature_cols: list[str] = FEATURE_COLS, epochs: int = 50) -> dict:
    """
    Retrain with the frozen best hyperparameters on the full 2015-2023
    window and evaluate ONCE on the 2024-2026 held-out test window.
    Call this only after tune_hyperparameters() has been run and a
    config has been chosen -- do not tune again after seeing this result.
    """
    return train_and_evaluate_fold(
        train_df, test_df, feature_cols,
        epochs=epochs, fold_label="Final Test", verbose=0, **best_params
    )


# --------------------------------------------------------------------------
# Persistence helpers (predictions -> CSV, matching src/backtest.py contract)
# --------------------------------------------------------------------------

def save_predictions(predictions: pd.Series, path: str) -> None:
    """Writes a CSV with a 'Date' column and 'Predicted_Target' column,
    exactly matching src.backtest.load_predictions()'s expected format."""
    out = predictions.rename("Predicted_Target").to_frame()
    out.index.name = "Date"
    out.to_csv(path)
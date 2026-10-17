"""
Ask a trained model what to do about a stock tomorrow.

    python -m src.predict --ticker TCS --model xgb

Uses the "live" model — the one retrained on all available history — because a
forward-looking call should use every day of data that exists. The metrics in
results/ come from the separate final-test model, which never saw 2024 onwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.config import INSTRUMENTS, Instrument
from src.models.base import DirectionModel
from src.recommend import format_recommendation, recommend

MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM"}


def model_stem(instrument: Instrument, model_key: str, variant: str = "live") -> Path:
    return config.SAVED_MODELS_DIR / f"{instrument.slug}_{model_key}_{variant}"


def has_saved_model(instrument: Instrument, model_key: str, variant: str = "live") -> bool:
    return model_stem(instrument, model_key, variant).with_suffix(".joblib").exists()


def available_models(instrument: Instrument) -> list[str]:
    """Model keys that have a live model on disk and can answer right now."""
    return [key for key in MODEL_LABELS if has_saved_model(instrument, key)]


def load_saved_model(instrument: Instrument, model_key: str, variant: str = "live") -> DirectionModel:
    stem = model_stem(instrument, model_key, variant)
    if not stem.with_suffix(".joblib").exists():
        raise FileNotFoundError(
            f"No trained {model_key} model for {instrument.key}. "
            f"Run: python -m src.evaluate --ticker {instrument.key} --model {model_key}"
        )
    if model_key == "lstm":
        from src.models.lstm_model import LSTMDirectionModel
        return LSTMDirectionModel.load(stem)
    return DirectionModel.load(stem)


def load_feature_history(instrument: Instrument) -> pd.DataFrame:
    path = instrument.features_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run: python -m src.features --ticker {instrument.key}"
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def predict_latest(instrument: Instrument, model_key: str = "xgb") -> dict:
    """Recommendation for the next trading day, based on the most recent close."""
    features = load_feature_history(instrument)
    model = load_saved_model(instrument, model_key)

    columns = config.model_features_for(instrument)
    X = features[columns]

    # The sequence model needs the lookback-1 rows sitting just before the row
    # being predicted; tabular models get None.
    context = X.iloc[-model.lookback:-1] if model.lookback > 1 else None
    proba = model.predict_proba(X.iloc[[-1]], context)[0]

    last_row = features.iloc[-1]
    rec = recommend(proba, last_row)
    rec.update({
        "ticker": instrument.key,
        "name": instrument.name,
        "model": MODEL_LABELS.get(model_key, model_key),
        "as_of": features.index[-1],
        "close": float(last_row["Close"]),
    })
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Next-day direction and BUY/SELL/HOLD call.")
    parser.add_argument("--ticker", default="TCS",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}, or ALL")
    parser.add_argument("--model", default="xgb", choices=list(MODEL_LABELS))
    args = parser.parse_args()

    keys = list(INSTRUMENTS) if args.ticker.upper() == "ALL" else [args.ticker]
    for key in keys:
        instrument = config.get_instrument(key)
        try:
            rec = predict_latest(instrument, args.model)
        except FileNotFoundError as exc:
            print(f"{instrument.key}: {exc}\n")
            continue
        print(format_recommendation(
            rec, f"{instrument.name} [{rec['model']}]", rec["as_of"].date(), rec["close"]))
        print()


if __name__ == "__main__":
    main()

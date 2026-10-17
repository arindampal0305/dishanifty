"""
Score each out-of-sample prediction against what the market actually did.

The predictions come from the *final-test* model — trained on data before 2024
and never refitted — so every row here is a genuine out-of-sample call, not a
fit to the day it is predicting.

A row dated t is the call made after t's close about the move from t to t+1.
`Next_Return` on that row is the move that actually happened.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.config import Instrument, LABEL_NAMES

PROB_COLUMNS = ["prob_down", "prob_neutral", "prob_up"]


def load_track_record(instrument: Instrument, model_key: str,
                      window: str = "finaltest") -> pd.DataFrame | None:
    """Predicted vs actual direction per day. None if the model hasn't been run."""
    path = config.PREDICTIONS_DIR / f"{instrument.slug}_{model_key}_{window}.csv"
    if not path.exists() or not instrument.labeled_path().exists():
        return None

    preds = pd.read_csv(path, index_col=0, parse_dates=True)
    actual = pd.read_csv(instrument.labeled_path(), index_col=0, parse_dates=True)

    df = preds.join(actual[["Close", "Next_Return", "Target"]], how="inner")
    df["Predicted"] = df["Predicted_Target"].map(LABEL_NAMES)
    df["Actual"] = df["Target"].map(LABEL_NAMES)
    df["Correct"] = df["Predicted_Target"] == df["Target"]
    df["Confidence"] = df[PROB_COLUMNS].max(axis=1)
    return df


def hit_rates(df: pd.DataFrame) -> dict:
    """Overall accuracy, plus accuracy restricted to days the model took a side."""
    directional = df[df["Predicted"] != "NEUTRAL"]
    return {
        "n": len(df),
        "accuracy": float(df["Correct"].mean()) if len(df) else float("nan"),
        "n_directional": len(directional),
        "directional_accuracy": (float(directional["Correct"].mean())
                                 if len(directional) else float("nan")),
    }

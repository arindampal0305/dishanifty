"""
Walk-forward training and evaluation for every model, on any instrument.

Produces, per model:
  * per-fold Accuracy / Macro F1 and backtest Sharpe, CAGR, Max Drawdown
  * one final-test row (train < 2024, evaluate on 2024 onwards, exactly once)
  * a saved "live" model retrained on all available data, used by the dashboard
    to call tomorrow. The live model is never scored — scoring it would mean
    testing on its own training data.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src import config
from src.backtest import run_backtest
from src.config import INSTRUMENTS, LABEL_NAMES, Instrument
from src.models.base import DirectionModel
from src.models.rf_model import RandomForestDirectionModel
from src.models.xgb_model import XGBoostDirectionModel
from src.validation import get_test_split, get_walk_forward_folds, load_labeled

MODEL_REGISTRY = {
    "rf": RandomForestDirectionModel,
    "xgb": XGBoostDirectionModel,
    "lstm": None,  # imported lazily; TensorFlow takes ~15s to load
}


def build_model(model_key: str) -> DirectionModel:
    if model_key == "lstm":
        from src.models.lstm_model import LSTMDirectionModel
        return LSTMDirectionModel()
    return MODEL_REGISTRY[model_key]()


def _context_for(model: DirectionModel, train: pd.DataFrame, features: list[str]):
    """Tail of the training window a sequence model needs to predict row 1 of val."""
    if model.lookback <= 1:
        return None
    return train[features].iloc[-(model.lookback - 1):]


def _predictions_frame(proba: np.ndarray, index) -> pd.DataFrame:
    df = pd.DataFrame(proba, index=index, columns=["prob_down", "prob_neutral", "prob_up"])
    df["Predicted_Target"] = np.argmax(proba, axis=1)
    df.index.name = "Date"
    return df


def _buy_and_hold(df: pd.DataFrame, index) -> dict:
    """Benchmark: always long. The number every strategy has to beat."""
    always_long = pd.Series(2, index=index)
    result = run_backtest(df, always_long, label="Buy & Hold")
    return {"sharpe": result.sharpe, "cagr": result.cagr, "max_drawdown": result.max_drawdown}


def _majority_baseline(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Accuracy floor: guess the training set's most common class every single day.

    This, not 1/3, is the number a model has to beat. Uniform chance assumes the
    three classes are equally frequent; they are not, and a model can clear 0.33
    while still being worse than this one-line strategy.
    """
    majority = int(train["Target"].mode().iloc[0])
    y_pred = pd.Series(majority, index=test.index)
    bt = run_backtest(df, y_pred, label="Majority class")
    return {
        "model": f"Always {LABEL_NAMES[majority]}",
        "window": "Final Test",
        "n_rows": len(test),
        "accuracy": accuracy_score(test["Target"], y_pred),
        "macro_f1": f1_score(test["Target"], y_pred, average="macro", zero_division=0),
        "sharpe": bt.sharpe,
        "cagr": bt.cagr,
        "max_drawdown": bt.max_drawdown,
        "n_trades": bt.n_trades,
    }


def evaluate_window(df: pd.DataFrame, model: DirectionModel, train: pd.DataFrame,
                    evaluate_on: pd.DataFrame, features: list[str], label: str) -> tuple[dict, pd.DataFrame]:
    model.fit(train[features], train["Target"])
    proba = model.predict_proba(evaluate_on[features], _context_for(model, train, features))
    preds_df = _predictions_frame(proba, evaluate_on.index)

    y_true = evaluate_on["Target"]
    y_pred = preds_df["Predicted_Target"]
    bt = run_backtest(df, y_pred, label=f"{model.name} — {label}")

    metrics = {
        "model": model.name,
        "window": label,
        "n_rows": len(evaluate_on),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "sharpe": bt.sharpe,
        "cagr": bt.cagr,
        "max_drawdown": bt.max_drawdown,
        "n_trades": bt.n_trades,
    }
    return metrics, preds_df


def run_model(instrument: Instrument, model_key: str, df: pd.DataFrame,
              save_live: bool = True) -> pd.DataFrame:
    features = config.model_features_for(instrument)
    model_name = build_model(model_key).name
    print(f"\n--- {instrument.key} / {model_name} ---")

    rows = []
    for fold in get_walk_forward_folds(df, verbose=False):
        metrics, preds = evaluate_window(
            df, build_model(model_key), fold.train, fold.val, features, fold.label)
        rows.append(metrics)
        preds.to_csv(config.PREDICTIONS_DIR / f"{instrument.slug}_{model_key}_{fold.label.replace(' ', '').lower()}.csv")
        print(f"  {fold.label}: acc {metrics['accuracy']:.3f} | macroF1 {metrics['macro_f1']:.3f} "
              f"| Sharpe {metrics['sharpe']:.2f}")

    train, test = get_test_split(df, verbose=False)
    final_model = build_model(model_key)
    metrics, preds = evaluate_window(df, final_model, train, test, features, "Final Test")
    rows.append(metrics)
    preds.to_csv(config.PREDICTIONS_DIR / f"{instrument.slug}_{model_key}_finaltest.csv")
    final_model.save(config.SAVED_MODELS_DIR / f"{instrument.slug}_{model_key}_final")

    cm = confusion_matrix(test["Target"], preds["Predicted_Target"], labels=[0, 1, 2])
    pd.DataFrame(cm,
                 index=[f"true_{LABEL_NAMES[i]}" for i in range(3)],
                 columns=[f"pred_{LABEL_NAMES[i]}" for i in range(3)]
                 ).to_csv(config.RESULTS_DIR / f"{instrument.slug}_{model_key}_confusion.csv")

    print(f"  FINAL TEST: acc {metrics['accuracy']:.3f} | macroF1 {metrics['macro_f1']:.3f} "
          f"| Sharpe {metrics['sharpe']:.2f} | MaxDD {metrics['max_drawdown']:.1%}")

    if save_live:
        # Trained on everything including the test window, so it can only be
        # used for forward-looking predictions — never for reported metrics.
        live = build_model(model_key)
        live.fit(df[features], df["Target"])
        live.save(config.SAVED_MODELS_DIR / f"{instrument.slug}_{model_key}_live")
        print(f"  live model saved (trained through {df.index.max().date()})")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward train and evaluate models.")
    parser.add_argument("--ticker", default="TCS",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}")
    parser.add_argument("--model", default="all",
                        help="rf, xgb, lstm, or all")
    parser.add_argument("--no-live", action="store_true",
                        help="Skip retraining the live model used by the dashboard")
    args = parser.parse_args()

    config.ensure_dirs()
    instrument = config.get_instrument(args.ticker)
    df = load_labeled(instrument.key)

    model_keys = list(MODEL_REGISTRY) if args.model == "all" else [args.model]

    all_metrics = [run_model(instrument, key, df, save_live=not args.no_live)
                   for key in model_keys]
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    train, test = get_test_split(df, verbose=False)
    bh = _buy_and_hold(df, test.index)
    metrics_df = pd.concat([metrics_df, pd.DataFrame([
        _majority_baseline(df, train, test),
        {"model": "Buy & Hold", "window": "Final Test", "n_rows": len(test),
         "accuracy": np.nan, "macro_f1": np.nan, "n_trades": 1, **bh},
    ])], ignore_index=True)

    out = config.RESULTS_DIR / f"{instrument.slug}_metrics.csv"
    metrics_df.to_csv(out, index=False)

    print(f"\n=== Final test comparison — {instrument.name} ===")
    final = metrics_df[metrics_df["window"] == "Final Test"]
    print(final[["model", "accuracy", "macro_f1", "sharpe", "cagr", "max_drawdown"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved -> {out.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

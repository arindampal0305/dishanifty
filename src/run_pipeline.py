"""
End-to-end pipeline for one instrument.

    python -m src.run_pipeline --ticker TCS
    python -m src.run_pipeline --ticker TCS --model xgb --skip-download

Runs: download -> clean -> features -> labels -> walk-forward evaluation, then
prints the current BUY/SELL/HOLD call.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.config import INSTRUMENTS
from src.data_cleaner import clean
from src.data_loader import download_instrument, download_macro
from src.features import build_features
from src.evaluate import MODEL_REGISTRY, run_model, _buy_and_hold, _majority_baseline
from src.predict import predict_latest
from src.recommend import format_recommendation
from src.target import generate_target, class_balance
from src.validation import get_test_split, load_labeled


def prepare_data(ticker: str, download: bool = True, force: bool = False) -> None:
    instrument = config.get_instrument(ticker)
    config.ensure_dirs()

    if download:
        download_instrument(instrument, force=force)
        download_macro(force=force)

    # Individual stocks use NIFTY 50 as a market-context feature, so the index
    # has to be downloaded and cleaned even when it isn't the target.
    needed = [instrument.key] + list(config.MACRO_INSTRUMENTS)
    if instrument.kind == "stock":
        needed.append(config.MARKET_INDEX_KEY)

    for key in dict.fromkeys(needed):
        dep = config.get_instrument(key)
        if download and dep.key != instrument.key:
            download_instrument(dep, force=force)
        clean(dep).to_csv(dep.clean_path())

    features = build_features(instrument)
    features.to_csv(instrument.features_path())

    labeled = generate_target(features, instrument.threshold)
    labeled.to_csv(instrument.labeled_path())
    print(f"{instrument.key} @ {instrument.threshold:.2%} class balance:")
    print(class_balance(labeled).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full DishaNifty pipeline.")
    parser.add_argument("--ticker", default="TCS",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}")
    parser.add_argument("--model", default="all",
                        help="rf, xgb, lstm, all, or a comma list like rf,xgb")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse the CSVs already in data/raw/")
    parser.add_argument("--force", action="store_true", help="Re-download raw data")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Refresh data and features, leave the saved models alone")
    args = parser.parse_args()

    instrument = config.get_instrument(args.ticker)

    print(f"=== {instrument.name} ({instrument.symbol}) ===\n")
    prepare_data(instrument.key, download=not args.skip_download, force=args.force)

    if args.prepare_only:
        print("\nData and features refreshed. Models untouched.")
        return

    df = load_labeled(instrument.key)
    if args.model == "all":
        model_keys = list(MODEL_REGISTRY)
    else:
        model_keys = [k.strip() for k in args.model.split(",") if k.strip()]
        unknown = [k for k in model_keys if k not in MODEL_REGISTRY]
        if unknown:
            parser.error(f"Unknown model(s): {', '.join(unknown)}. "
                         f"Choose from {', '.join(MODEL_REGISTRY)} or 'all'.")
    metrics = pd.concat([run_model(instrument, key, df) for key in model_keys],
                        ignore_index=True)

    train, test = get_test_split(df, verbose=False)
    metrics = pd.concat([metrics, pd.DataFrame([
        _majority_baseline(df, train, test),
        {"model": "Buy & Hold", "window": "Final Test", "n_rows": len(test),
         "n_trades": 1, **_buy_and_hold(df, test.index)},
    ])], ignore_index=True)
    # Training a subset must not delete the rows of models it did not touch,
    # or a 20-second rf+xgb run would wipe an LSTM score that took four minutes.
    metrics_path = config.RESULTS_DIR / f"{instrument.slug}_metrics.csv"
    if metrics_path.exists():
        previous = pd.read_csv(metrics_path)
        fresh = set(metrics["model"])
        kept = previous[~previous["model"].isin(fresh)]
        metrics = pd.concat([metrics, kept], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)

    print(f"\n=== Final test comparison — {instrument.name} ===")
    final = metrics[metrics["window"] == "Final Test"]
    print(final[["model", "accuracy", "macro_f1", "sharpe", "cagr", "max_drawdown"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n=== Current call ===")
    for key in model_keys:
        rec = predict_latest(instrument, key)
        print(format_recommendation(
            rec, f"{instrument.name} [{rec['model']}]", rec["as_of"].date(), rec["close"]))
        print()


if __name__ == "__main__":
    main()

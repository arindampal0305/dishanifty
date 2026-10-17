"""Label each day by the direction of the *next* day's return."""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.config import INSTRUMENTS, LABEL_NAMES, Instrument

THRESHOLD_GRID = [0.001, 0.0025, 0.005]


def generate_target(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["Next_Return"] = df["Close"].pct_change().shift(-1)

    df["Target"] = 1
    df.loc[df["Next_Return"] > threshold, "Target"] = 2    # UP
    df.loc[df["Next_Return"] < -threshold, "Target"] = 0   # DOWN

    # The final row has no next day, so it has no label.
    df = df.dropna(subset=["Next_Return"])
    df["Target"] = df["Target"].astype(int)
    return df


def class_balance(df: pd.DataFrame) -> pd.Series:
    counts = df["Target"].value_counts(normalize=True).sort_index()
    counts.index = [LABEL_NAMES[i] for i in counts.index]
    return counts.round(3)


def threshold_report(df: pd.DataFrame, instrument: Instrument) -> pd.DataFrame:
    """Class proportions at each candidate threshold — roadmap step 6."""
    rows = {}
    for t in THRESHOLD_GRID:
        rows[f"{t:.2%}"] = class_balance(generate_target(df, t))
    report = pd.DataFrame(rows).T.fillna(0.0)
    print(f"\nClass balance by threshold — {instrument.name}")
    print(report.to_string())
    print(f"Selected threshold: {instrument.threshold:.2%}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UP/DOWN/NEUTRAL labels.")
    parser.add_argument("--ticker", default="NIFTY50",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}, or ALL")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the instrument's default threshold (e.g. 0.005)")
    parser.add_argument("--report", action="store_true",
                        help="Print class balance across the 0.1/0.25/0.5%% grid")
    args = parser.parse_args()

    config.ensure_dirs()
    keys = list(INSTRUMENTS) if args.ticker.upper() == "ALL" else [args.ticker]

    for key in keys:
        instrument = config.get_instrument(key)
        df = pd.read_csv(instrument.features_path(), index_col=0, parse_dates=True)

        if args.report:
            threshold_report(df, instrument)

        threshold = args.threshold if args.threshold is not None else instrument.threshold
        labeled = generate_target(df, threshold)
        labeled.to_csv(instrument.labeled_path())

        print(f"\n{instrument.key} @ {threshold:.2%} -> {len(labeled)} rows")
        print(class_balance(labeled).to_string())
        print(f"  saved -> {instrument.labeled_path().relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

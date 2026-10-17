"""Walk-forward fold definitions. Never a random split — time order is the point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from src import config
from src.config import INSTRUMENTS

TRAIN_START = "2015-01-01"
TEST_START = "2024-01-01"

FOLD_SPECS = [
    {"fold": 1, "train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31", "note": "COVID year"},
    {"fold": 2, "train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2021-12-31", "note": "Post-COVID recovery"},
    {"fold": 3, "train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-12-31", "note": "Rate hike cycle"},
    {"fold": 4, "train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-12-31", "note": "Stabilization year"},
]


@dataclass
class Fold:
    label: str
    train: pd.DataFrame
    val: pd.DataFrame
    note: str = ""


def get_walk_forward_folds(df: pd.DataFrame, verbose: bool = True) -> list[Fold]:
    folds = []
    for spec in FOLD_SPECS:
        train = df[df.index <= spec["train_end"]]
        val = df[(df.index >= spec["val_start"]) & (df.index <= spec["val_end"])]
        if val.empty:
            continue
        folds.append(Fold(label=f"Fold {spec['fold']}", train=train, val=val, note=spec["note"]))
        if verbose:
            print(f"Fold {spec['fold']} | train {TRAIN_START}..{spec['train_end']} ({len(train)}) "
                  f"| val {spec['val_start']}..{spec['val_end']} ({len(val)}) | {spec['note']}")
    return folds


def get_test_split(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df.index < TEST_START]
    test = df[df.index >= TEST_START]
    if verbose:
        print(f"Final test | train {TRAIN_START}..2023-12-31 ({len(train)}) "
              f"| test {TEST_START}..present ({len(test)})")
    return train, test


def load_labeled(ticker: str) -> pd.DataFrame:
    instrument = config.get_instrument(ticker)
    path = instrument.labeled_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run: python -m src.target --ticker {instrument.key}"
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show walk-forward fold sizes.")
    parser.add_argument("--ticker", default="NIFTY50",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}")
    args = parser.parse_args()

    df = load_labeled(args.ticker)
    print(f"=== Walk-forward folds — {args.ticker.upper()} ===")
    get_walk_forward_folds(df)
    print()
    get_test_split(df)


if __name__ == "__main__":
    main()

"""Flatten and clean raw yfinance CSVs into a tidy OHLCV frame."""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.config import INSTRUMENTS, Instrument

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def read_raw(path) -> pd.DataFrame:
    """
    yfinance writes a three-row header (Price / Ticker / Date). Read the first
    two as a MultiIndex, flatten to the price level, then drop the leftover
    'Date' spacer row along with anything else that isn't a real timestamp.
    """
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.columns = [col[0] for col in df.columns]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df.index.name = "Date"
    return df.apply(pd.to_numeric, errors="coerce")


def clean(instrument: Instrument) -> pd.DataFrame:
    df = read_raw(instrument.raw_path())

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{instrument.key}: raw file is missing columns {missing}")

    df = df[OHLCV]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.ffill().dropna(subset=["Close"])

    # India VIX and USD/INR carry no traded volume; a zero column is honest there
    # but would be a dead feature, so leave it and let Volume_Ratio handle it.
    print(f"{instrument.key}: cleaned to {df.shape[0]} rows, "
          f"{df.index.min().date()} to {df.index.max().date()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw market data.")
    parser.add_argument("--ticker", default="NIFTY50",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}, or ALL")
    args = parser.parse_args()

    config.ensure_dirs()
    keys = list(INSTRUMENTS) if args.ticker.upper() == "ALL" else [args.ticker]

    for key in keys + list(config.MACRO_INSTRUMENTS):
        instrument = config.get_instrument(key)
        df = clean(instrument)
        df.to_csv(instrument.clean_path())
        print(f"  saved -> {instrument.clean_path().relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

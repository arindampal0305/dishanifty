"""
Technical indicators plus the scale-invariant feature set the models consume.

Two kinds of column come out of here:

  * raw indicators (SMA_20, MACD, BB_Upper, ...) — kept for the dashboard and
    EDA, never handed to a model;
  * model features (Close_vs_SMA20, MACD_Norm, BB_Position, ...) — ratios,
    percentages, and bounded oscillators only.

The split exists because raw moving averages are absolute price levels. A tree
trained on 2015 NIFTY data learns splits around 8,000; by 2024 every row sits on
one side of them and the feature is dead. Dividing by Close fixes that, and it
is what makes a single pipeline work for both an index and an individual stock.

Everything here uses only data available by the close of day t.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pandas_ta as ta

from src import config
from src.config import INSTRUMENTS, Instrument


def _pick(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Select a pandas-ta output column by prefix; its suffixes vary by version."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise KeyError(f"pandas-ta returned no column starting with '{prefix}'. Got: {list(df.columns)}")
    return df[matches[0]]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    df["Return"] = close.pct_change()
    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volatility_20"] = df["Return"].rolling(20).std()

    for n in (10, 20, 50):
        df[f"SMA_{n}"] = ta.sma(close, length=n)
        df[f"EMA_{n}"] = ta.ema(close, length=n)

    df["RSI_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    df["MACD"] = _pick(macd, "MACD_")
    df["MACD_Signal"] = _pick(macd, "MACDs_")
    df["MACD_Hist"] = _pick(macd, "MACDh_")

    bb = ta.bbands(close, length=20, std=2)
    df["BB_Upper"] = _pick(bb, "BBU_")
    df["BB_Middle"] = _pick(bb, "BBM_")
    df["BB_Lower"] = _pick(bb, "BBL_")

    df["ATR_14"] = ta.atr(high, low, close, length=14)
    df["Momentum_10"] = ta.mom(close, length=10)
    return df


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    for lag in (1, 2, 3):
        df[f"Return_Lag{lag}"] = df["Return"].shift(lag)
    df["Return_5d"] = close.pct_change(5)

    for n in (10, 20, 50):
        df[f"Close_vs_SMA{n}"] = close / df[f"SMA_{n}"] - 1
        df[f"Close_vs_EMA{n}"] = close / df[f"EMA_{n}"] - 1
    df["SMA10_vs_SMA20"] = df["SMA_10"] / df["SMA_20"] - 1
    df["SMA20_vs_SMA50"] = df["SMA_20"] / df["SMA_50"] - 1

    df["MACD_Norm"] = df["MACD"] / close
    df["MACD_Hist_Norm"] = df["MACD_Hist"] / close

    bb_range = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Position"] = (close - df["BB_Lower"]) / bb_range
    df["BB_Width"] = bb_range / df["BB_Middle"]

    df["ATR_Pct"] = df["ATR_14"] / close
    df["Momentum_10_Pct"] = close / close.shift(10) - 1

    volume_avg = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = (df["Volume"] / volume_avg).where(volume_avg > 0, 1.0)
    return df


def _daily_close(instrument_key: str) -> pd.Series:
    instrument = config.get_instrument(instrument_key)
    path = instrument.clean_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run: python -m src.data_cleaner --ticker {instrument_key}"
        )
    series = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
    return series[~series.index.duplicated(keep="first")].sort_index()


def add_macro_features(df: pd.DataFrame, instrument: Instrument) -> pd.DataFrame:
    """
    Attach India VIX, USD/INR, and (for individual stocks) NIFTY 50 context.

    Macro series trade on a slightly different calendar, so they are reindexed
    onto this instrument's dates and forward-filled. Forward-fill only ever
    carries a *past* observation onto a later date, so no future data leaks in.
    """
    df = df.copy()

    vix = _daily_close("INDIA_VIX").reindex(df.index).ffill()
    df["VIX_Level"] = vix
    df["VIX_Change"] = vix.pct_change()

    usdinr = _daily_close("USDINR").reindex(df.index).ffill()
    df["USDINR_Return"] = usdinr.pct_change()

    if instrument.kind == "stock":
        market = _daily_close(config.MARKET_INDEX_KEY).reindex(df.index).ffill()
        df["Market_Return"] = market.pct_change()
        df["Market_Return_5d"] = market.pct_change(5)

    return df


def build_features(instrument: Instrument) -> pd.DataFrame:
    path = instrument.clean_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run: python -m src.data_cleaner --ticker {instrument.key}"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    df = add_indicators(df)
    df = add_model_features(df)
    df = add_macro_features(df, instrument)

    required = config.model_features_for(instrument)
    df = df.dropna(subset=required)

    print(f"{instrument.key}: {df.shape[0]} rows x {len(required)} model features "
          f"({df.index.min().date()} to {df.index.max().date()})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build technical and model features.")
    parser.add_argument("--ticker", default="NIFTY50",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}, or ALL")
    args = parser.parse_args()

    config.ensure_dirs()
    keys = list(INSTRUMENTS) if args.ticker.upper() == "ALL" else [args.ticker]

    for key in keys:
        instrument = config.get_instrument(key)
        df = build_features(instrument)
        df.to_csv(instrument.features_path())
        print(f"  saved -> {instrument.features_path().relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

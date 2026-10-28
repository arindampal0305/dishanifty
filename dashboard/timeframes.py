"""Chart timeframes for the price panel.

The research pipeline is daily-only: every model in this project is fitted on
daily bars and predicts the next daily close. Nothing in this module feeds a
model -- it exists so the price chart can be *read* at other resolutions, the
way a trading terminal shows it.

Two sources:

* ``local`` -- resampled from the daily frame already on disk. Free, instant,
  and the same numbers the models were trained on.
* ``yahoo`` -- fetched live, because intraday bars are stored nowhere. Yahoo
  caps intraday history hard (1m: 7 days, 5m: 60 days, 1h: 730 days), so each
  intraday preset asks for a span comfortably inside its own cap.

Each preset fetches more history than it displays, so the moving averages are
already warm at the left edge of the chart instead of starting as a gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta
import streamlit as st
import yfinance as yf

OHLC_AGG = {"Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"}


@dataclass(frozen=True)
class Timeframe:
    """One button on the timeframe bar: a complete view spec."""

    label: str
    source: str       # "local" or "yahoo"
    interval: str     # yfinance interval, "" when source is local
    fetch: str        # yfinance period, "" when source is local
    resample: str     # pandas offset alias, "" to keep the native bar size
    show_days: int    # how much to display; 0 means everything available
    caption: str

    @property
    def intraday(self) -> bool:
        return self.source == "yahoo"

    @property
    def rangebreaks(self) -> list[dict]:
        """Gaps to cut out of the x axis so the candles sit flush.

        Daily and slower bars need nothing — plotly already spaces them well.
        Intraday bars have to have the nights and weekends removed, or twelve
        dead hours sit between every session and the candles bunch into thin
        clusters separated by whitespace.
        """
        if not self.intraday:
            return []
        breaks = [dict(bounds=["sat", "mon"])]
        if not self.resample:
            # NSE trades 09:15-15:30. Resampled 5h bins spill past the close,
            # so only native intraday bars get the session clamp.
            breaks.append(dict(bounds=[15.55, 9.25], pattern="hour"))
        return breaks


# Ordered exactly as the bar renders. Spans differ between presets on purpose,
# so no two buttons produce the same chart.
TIMEFRAMES: dict[str, Timeframe] = {
    tf.label: tf for tf in [
        Timeframe("1m", "yahoo", "1m", "5d", "", 1,
                  "1-minute bars, most recent session. Yahoo keeps 7 days of these."),
        Timeframe("5m", "yahoo", "5m", "1mo", "", 5,
                  "5-minute bars, last 5 sessions. Yahoo keeps 60 days of these."),
        Timeframe("1H", "yahoo", "1h", "3mo", "", 30,
                  "Hourly bars, last month. Yahoo keeps 2 years of these."),
        Timeframe("5H", "yahoo", "1h", "1y", "5h", 90,
                  "5-hour bars resampled from hourly, last quarter. NSE trades "
                  "6h15m a day, so these bins straddle sessions."),
        Timeframe("1D", "local", "", "", "", 90,
                  "Daily bars, last quarter. This is the resolution the models see."),
        Timeframe("1W", "local", "", "", "W-FRI", 365,
                  "Weekly bars, last year, resampled from daily."),
        Timeframe("1M", "local", "", "", "ME", 0,
                  "Monthly bars, full history since 2015, resampled from daily."),
        Timeframe("6M", "local", "", "", "", 182,
                  "Daily bars, last six months."),
        Timeframe("1Y", "local", "", "", "", 365,
                  "Daily bars, last year."),
        Timeframe("3Y", "local", "", "", "W-FRI", 1095,
                  "Weekly bars, last three years."),
        Timeframe("5Y", "local", "", "", "W-FRI", 1825,
                  "Weekly bars, last five years."),
    ]
}

DEFAULT = "6M"


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Live intraday pull. Cached for five minutes so switching tabs is free."""
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance returns (field, ticker) columns even for a single symbol.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Show the exchange's own clock, then drop the zone so the axis reads as
    # plain local time rather than a UTC offset.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Collapse bars into bigger ones, keeping OHLCV semantics intact."""
    agg = {col: how for col, how in OHLC_AGG.items() if col in df.columns}
    # origin="start" bins from the first bar so intraday bins line up with the
    # session rather than with midnight. pandas only honours it for tick-like
    # rules, and warns on calendar ones like W-FRI or ME, so pass it selectively.
    unit = rule.lstrip("0123456789").lower()
    kwargs = {"origin": "start"} if unit in {"h", "min", "s", "ms"} else {}
    out = df.resample(rule, **kwargs).agg(agg)
    return out.dropna(subset=["Close"])


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Overlays computed on the bars being shown, not on daily bars.

    A 20-period average means 20 *of these* bars. Carrying the daily SMA onto a
    weekly chart would draw a line that belongs to a different series.
    """
    df = df.copy()
    close = df["Close"]
    df["SMA_20"] = close.rolling(20).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df["RSI_14"] = ta.rsi(close, length=14) if len(close) > 15 else pd.NA
    return df


def load(symbol: str, daily: pd.DataFrame, key: str) -> tuple[pd.DataFrame, str]:
    """Return (bars, error). An empty frame means the error explains why."""
    tf = TIMEFRAMES[key]

    if tf.intraday:
        frame = _fetch_intraday(symbol, tf.interval, tf.fetch)
        if frame.empty:
            return pd.DataFrame(), (
                f"Yahoo returned no {tf.interval} data for {symbol}. Intraday "
                "history is short-lived and unavailable outside market hours "
                "for some symbols. Showing nothing rather than guessing."
            )
    else:
        frame = daily

    if tf.resample:
        frame = _resample(frame, tf.resample)

    frame = _with_indicators(frame)

    if tf.show_days and not frame.empty:
        cutoff = frame.index.max() - pd.Timedelta(days=tf.show_days)
        frame = frame[frame.index >= cutoff]

    if frame.empty:
        return pd.DataFrame(), f"No bars left for {key} after resampling."
    return frame, ""

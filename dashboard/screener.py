"""Cross-sectional screens over every prepared instrument.

Nothing here involves a model, which is the point: these are arithmetic screens
over indicators, so they work on all 51 instruments the moment their data is on
disk, trained or not.

Bollinger bands are recomputed from Close rather than read from the stored
BB_* columns, because those are fixed at 20 periods / 2 sigma and the whole
point of this tab is letting someone change both.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config

# Enough history for a long Bollinger window plus a readable chart behind it.
TAIL = 400

# Conventional readings. These are textbook technical analysis, NOT findings of
# this project -- see NOTE below, which the tab prints verbatim.
RSI_ZONES: list[tuple[float, str, str]] = [
    (70, "OVERBOUGHT", "Conventionally read as a stretched advance: buyers have "
                       "pushed hard and the move is often due a pause. Note that "
                       "a strong trend can sit above 70 for weeks."),
    (60, "STRONG", "Firm upward momentum, short of the overbought line."),
    (40, "NEUTRAL", "No momentum signal either way. Most days live here."),
    (30, "WEAK", "Downward momentum, short of the oversold line."),
    (0, "OVERSOLD", "Conventionally read as exhausted selling, sometimes ahead of "
                    "a bounce. A falling stock can stay below 30 for a long time."),
]

NOTE = (
    "These are the textbook interpretations of RSI, not predictions and not this "
    "project's own findings. Nothing on this tab comes from the trained models. "
    "This project's measured result is that next-day direction is close to "
    "unpredictable from indicators like these, so treat the readings as "
    "description of where a price has been, never as a recommendation."
)


def rsi_zone(value: float) -> tuple[str, str]:
    """Label and textbook note for one RSI reading."""
    if pd.isna(value):
        return "NO DATA", "Not enough history to compute RSI."
    for floor, label, note in RSI_ZONES:
        if value >= floor:
            return label, note
    return "OVERSOLD", RSI_ZONES[-1][2]


def data_signature() -> tuple[int, float]:
    """Changes whenever any features file is rewritten, so the cache drops."""
    paths = list(config.PROCESSED_DIR.glob("*_features.csv"))
    return len(paths), max((p.stat().st_mtime for p in paths), default=0.0)


@st.cache_data(show_spinner="scanning every instrument")
def load_panel(signature: tuple[int, float], tail: int = TAIL) -> dict[str, pd.DataFrame]:
    """Last `tail` sessions per instrument, only the columns the screens need.

    Reading all 48 feature columns across 51 files would be ~250MB for data this
    tab never looks at, so usecols keeps it to five.
    """
    wanted = ["Date", "Open", "High", "Low", "Close", "RSI_14"]
    panel: dict[str, pd.DataFrame] = {}

    for key, inst in config.all_instruments().items():
        path = inst.features_path()
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True, usecols=wanted)
        except (ValueError, OSError):
            continue
        if not df.empty:
            panel[key] = df.tail(tail)
    return panel


def bollinger(close: pd.Series, period: int, n_std: float) -> pd.DataFrame:
    """Bands from a rolling mean and population sigma.

    ddof=0 matches TA-Lib and the charting platforms; pandas defaults to the
    sample estimate, which would draw slightly wider bands than a trader expects.
    """
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * sd,
                         "lower": mid - n_std * sd})


def rsi_screen(panel: dict[str, pd.DataFrame], threshold: float,
               above: bool) -> pd.DataFrame:
    """Every instrument whose latest RSI sits on the requested side."""
    instruments = config.all_instruments()
    rows = []
    for key, df in panel.items():
        rsi = df["RSI_14"].iloc[-1]
        if pd.isna(rsi) or (rsi >= threshold) != above:
            continue
        close = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2] if len(df) > 1 else close
        rows.append({
            "Company": instruments[key].name,
            "RSI": float(rsi),
            "Close": float(close),
            "1d": float(close / prev - 1) if prev else 0.0,
            "Zone": rsi_zone(rsi)[0],
            "key": key,
        })
    if not rows:
        return pd.DataFrame(columns=["Company", "RSI", "Close", "1d", "Zone", "key"])
    return (pd.DataFrame(rows)
            .sort_values("RSI", ascending=not above)
            .reset_index(drop=True))


def breakout_screen(panel: dict[str, pd.DataFrame], period: int, n_std: float,
                    lookback: int) -> pd.DataFrame:
    """Instruments whose close pierced a band within the last `lookback` sessions.

    Reports the most recent break per instrument, and how far outside it closed.
    """
    instruments = config.all_instruments()
    rows = []

    for key, df in panel.items():
        if len(df) < period + 1:
            continue
        bands = bollinger(df["Close"], period, n_std)
        recent = df.tail(lookback)
        band = bands.tail(lookback)

        over = recent["Close"] > band["upper"]
        under = recent["Close"] < band["lower"]
        hits = recent.index[over | under]
        if len(hits) == 0:
            continue

        when = hits[-1]
        close = float(recent.loc[when, "Close"])
        up = bool(over.loc[when])
        edge = float(band.loc[when, "upper" if up else "lower"])
        rows.append({
            "Company": instruments[key].name,
            "Broke": when,
            "Side": "ABOVE" if up else "BELOW",
            "Close": close,
            "Band": edge,
            "Outside": abs(close - edge) / edge if edge else 0.0,
            "Breaks": int((over | under).sum()),
            "key": key,
        })

    if not rows:
        return pd.DataFrame(columns=["Company", "Broke", "Side", "Close", "Band",
                                     "Outside", "Breaks", "key"])
    return (pd.DataFrame(rows)
            .sort_values(["Broke", "Outside"], ascending=[False, False])
            .reset_index(drop=True))

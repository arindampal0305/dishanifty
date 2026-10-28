"""Central registry of tradeable instruments, paths, and the model feature set."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
CUSTOM_INSTRUMENTS_FILE = PROJECT_ROOT / "data" / "custom_instruments.json"

START_DATE = "2015-01-01"
END_DATE = None  # None = up to today, so the test window keeps extending

LABEL_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass(frozen=True)
class Instrument:
    key: str
    symbol: str
    name: str
    kind: str  # "index" or "stock"
    threshold: float

    @property
    def slug(self) -> str:
        return self.key.lower()

    def raw_path(self) -> Path:
        return RAW_DIR / f"{self.slug}_raw.csv"

    def clean_path(self) -> Path:
        return PROCESSED_DIR / f"{self.slug}_clean.csv"

    def features_path(self) -> Path:
        return PROCESSED_DIR / f"{self.slug}_features.csv"

    def labeled_path(self) -> Path:
        return PROCESSED_DIR / f"{self.slug}_labeled.csv"


# Individual stocks move roughly twice as much per day as the index, so the same
# UP/DOWN threshold would leave them with almost no NEUTRAL days. 0.5% keeps the
# three classes at comparable proportions across both; both values are from the
# 0.1/0.25/0.5% grid the roadmap allows.
_INDEX_THRESHOLD = 0.0025
_STOCK_THRESHOLD = 0.005

def _key_for(symbol: str) -> str:
    """'SBIN.NS' -> 'SBIN', '^NSEBANK' -> 'NSEBANK'. Keys become filenames.

    Punctuation collapses to an underscore, so 'M&M.NS' keys as M_M and
    'BAJAJ-AUTO.NS' as BAJAJ_AUTO. The Yahoo symbol keeps its real spelling.
    """
    stem = symbol.upper().split(".")[0]
    key = re.sub(r"[^A-Z0-9]+", "_", stem).strip("_")
    if not key:
        raise ValueError(f"Cannot derive an instrument key from '{symbol}'.")
    return key


# The eight hand-curated entries the project was built on. They keep their
# hand-written display names even when the index list below also contains them.
_CURATED: list[Instrument] = [
    Instrument("NIFTY50", "^NSEI", "NIFTY 50", "index", _INDEX_THRESHOLD),
    Instrument("TCS", "TCS.NS", "Tata Consultancy Services", "stock", _STOCK_THRESHOLD),
    Instrument("RELIANCE", "RELIANCE.NS", "Reliance Industries", "stock", _STOCK_THRESHOLD),
    Instrument("INFY", "INFY.NS", "Infosys", "stock", _STOCK_THRESHOLD),
    Instrument("HDFCBANK", "HDFCBANK.NS", "HDFC Bank", "stock", _STOCK_THRESHOLD),
    Instrument("ICICIBANK", "ICICIBANK.NS", "ICICI Bank", "stock", _STOCK_THRESHOLD),
    Instrument("SBIN", "SBIN.NS", "State Bank of India", "stock", _STOCK_THRESHOLD),
    Instrument("ITC", "ITC.NS", "ITC Limited", "stock", _STOCK_THRESHOLD),
]

# NIFTY 50 membership is re-constituted a few times a year, so the list lives in
# a data file rather than in this module. src/fetch_universe.py refreshes it.
NIFTY50_FILE = PROJECT_ROOT / "data" / "nifty50_constituents.json"


def _load_universe() -> list[Instrument]:
    """Index constituents from disk. A missing or malformed file is not fatal --
    the app falls back to the curated eight rather than refusing to start."""
    try:
        rows = json.loads(NIFTY50_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    out: list[Instrument] = []
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        name = str(row.get("name", "")).strip()
        if not symbol or not name:
            continue
        out.append(Instrument(_key_for(symbol), f"{symbol}.NS", name,
                              "stock", _STOCK_THRESHOLD))
    return out


INSTRUMENTS: dict[str, Instrument] = {inst.key: inst for inst in _CURATED}
for _inst in _load_universe():
    INSTRUMENTS.setdefault(_inst.key, _inst)   # curated names win

MACRO_INSTRUMENTS: dict[str, Instrument] = {
    inst.key: inst
    for inst in [
        Instrument("INDIA_VIX", "^INDIAVIX", "India VIX", "index", _INDEX_THRESHOLD),
        Instrument("USDINR", "USDINR=X", "USD/INR", "index", _INDEX_THRESHOLD),
    ]
}

MARKET_INDEX_KEY = "NIFTY50"

# Instruments the user adds at runtime from the dashboard. Kept in a JSON file so
# a symbol added in one session is still there after a restart, and separate from
# INSTRUMENTS so the eight curated defaults can never be edited away.
_CUSTOM: dict[str, Instrument] = {}


def load_custom_instruments() -> dict[str, Instrument]:
    """Re-read the custom registry from disk. Safe to call repeatedly."""
    _CUSTOM.clear()
    if not CUSTOM_INSTRUMENTS_FILE.exists():
        return _CUSTOM
    try:
        entries = json.loads(CUSTOM_INSTRUMENTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _CUSTOM
    for entry in entries:
        try:
            inst = Instrument(**entry)
        except TypeError:
            continue
        if inst.key not in INSTRUMENTS:
            _CUSTOM[inst.key] = inst
    return _CUSTOM


def all_instruments() -> dict[str, Instrument]:
    """The eight built-ins first, then anything the user added."""
    return {**INSTRUMENTS, **_CUSTOM}


def is_custom(key: str) -> bool:
    return key.upper() in _CUSTOM


def register_instrument(symbol: str, name: str | None = None,
                        kind: str = "stock") -> Instrument:
    """
    Add a Yahoo Finance symbol to the registry and persist it.

    Returns the existing Instrument if the symbol is already known, so adding
    'TCS.NS' twice is a no-op rather than a duplicate.
    """
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("Symbol is empty.")
    if not re.fullmatch(r"[A-Za-z0-9^.=\-&]+", symbol):
        raise ValueError(f"'{symbol}' is not a valid Yahoo Finance symbol.")

    key = _key_for(symbol)
    existing = all_instruments().get(key)
    if existing is not None:
        return existing

    if kind not in ("stock", "index"):
        raise ValueError("kind must be 'stock' or 'index'.")
    inst = Instrument(
        key=key,
        symbol=symbol.upper(),
        name=(name or "").strip() or key,
        kind=kind,
        threshold=_INDEX_THRESHOLD if kind == "index" else _STOCK_THRESHOLD,
    )

    _CUSTOM[key] = inst
    CUSTOM_INSTRUMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_INSTRUMENTS_FILE.write_text(
        json.dumps([asdict(i) for i in _CUSTOM.values()], indent=2), encoding="utf-8"
    )
    return inst


def get_instrument(key: str) -> Instrument:
    key = key.upper()
    pool = all_instruments()
    if key in pool:
        return pool[key]
    if key in MACRO_INSTRUMENTS:
        return MACRO_INSTRUMENTS[key]
    raise KeyError(f"Unknown instrument '{key}'. Available: {sorted(pool)}")


load_custom_instruments()


# Every entry is a ratio, a percentage, or a bounded oscillator. Raw price levels
# and raw moving averages are deliberately excluded: a split learned at
# "SMA_50 < 9000" on 2015 data is meaningless once the series trades at 24000.
MODEL_FEATURES = [
    "Return",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_5d",
    "Volatility_10",
    "Volatility_20",
    "Close_vs_SMA10",
    "Close_vs_SMA20",
    "Close_vs_SMA50",
    "SMA10_vs_SMA20",
    "SMA20_vs_SMA50",
    "Close_vs_EMA10",
    "Close_vs_EMA20",
    "Close_vs_EMA50",
    "RSI_14",
    "MACD_Norm",
    "MACD_Hist_Norm",
    "BB_Position",
    "BB_Width",
    "ATR_Pct",
    "Momentum_10_Pct",
    "Volume_Ratio",
    "VIX_Level",
    "VIX_Change",
    "USDINR_Return",
]

# Added only for individual stocks — how the broad market moved that day.
MARKET_FEATURES = ["Market_Return", "Market_Return_5d"]

# Kept in the processed CSV for the dashboard and EDA, never fed to a model.
DISPLAY_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_20", "SMA_50",
    "EMA_10", "EMA_20", "EMA_50",
    "MACD", "MACD_Signal",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "ATR_14", "Momentum_10",
]


def model_features_for(instrument: Instrument) -> list[str]:
    if instrument.kind == "stock":
        return MODEL_FEATURES + MARKET_FEATURES
    return list(MODEL_FEATURES)


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, RESULTS_DIR, SAVED_MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

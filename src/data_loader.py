"""Download raw OHLCV data for any registered instrument via yfinance."""

from __future__ import annotations

import argparse

import yfinance as yf

from src import config
from src.config import INSTRUMENTS, MACRO_INSTRUMENTS, Instrument


def download_instrument(instrument: Instrument,
                        start: str = config.START_DATE,
                        end: str = config.END_DATE,
                        force: bool = False) -> None:
    path = instrument.raw_path()
    if path.exists() and not force:
        print(f"[skip] {instrument.key}: {path.name} already exists (use --force to re-download)")
        return

    print(f"Downloading {instrument.name} ({instrument.symbol})...")
    # auto_adjust=True back-adjusts for splits and bonus issues. Without it a
    # stock like TCS shows a fabricated -50% "crash" on its split date.
    df = yf.download(instrument.symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {instrument.symbol}")

    df.to_csv(path)
    print(f"  saved {len(df)} rows -> {path.relative_to(config.PROJECT_ROOT)}")


def download_macro(force: bool = False) -> None:
    for instrument in MACRO_INSTRUMENTS.values():
        download_instrument(instrument, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw market data.")
    parser.add_argument("--ticker", default="NIFTY50",
                        help=f"Instrument key. One of: {', '.join(INSTRUMENTS)}, or ALL")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.ticker.upper() == "ALL":
        targets = list(INSTRUMENTS.values())
    else:
        targets = [config.get_instrument(args.ticker)]

    for instrument in targets:
        download_instrument(instrument, force=args.force)

    download_macro(force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()

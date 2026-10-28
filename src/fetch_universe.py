"""Refresh the NIFTY 50 constituent list.

    python -m src.fetch_universe

Index membership is re-constituted a few times a year, so the list is *data*,
not a constant baked into config.py. This writes data/nifty50_constituents.json
and nothing else reads the network: the dashboard and the pipeline both load
that file, so a blocked network can never stop the app from starting.

Source is the Wikipedia NIFTY 50 article, whose constituents table is the only
machine-readable list that is reliably reachable without an NSE session cookie.
Re-run it after an index review; if it fails, the previous file stays valid.
"""

from __future__ import annotations

import json

import pandas as pd

from src import config

WIKI_URL = "https://en.wikipedia.org/wiki/NIFTY_50"
OUTPUT = config.PROJECT_ROOT / "data" / "nifty50_constituents.json"
WANTED = {"Company name", "Symbol", "Sector"}


def fetch() -> list[dict]:
    """Return [{name, symbol, sector}, ...] for the current constituents."""
    tables = pd.read_html(WIKI_URL, storage_options={"User-Agent": "Mozilla/5.0"})

    for table in tables:
        cols = {str(c).split("[")[0].strip() for c in table.columns}
        if WANTED <= cols:
            table = table.copy()
            table.columns = [str(c).split("[")[0].strip() for c in table.columns]
            rows = (table[["Company name", "Symbol", "Sector"]]
                    .dropna()
                    .rename(columns={"Company name": "name", "Symbol": "symbol",
                                     "Sector": "sector"}))
            return rows.to_dict("records")

    raise RuntimeError(
        f"No constituents table at {WIKI_URL}. The page layout changed; "
        "update WANTED or write the JSON by hand."
    )


def main() -> None:
    rows = fetch()
    if not 40 <= len(rows) <= 60:
        raise RuntimeError(f"Got {len(rows)} constituents, expected ~50. "
                           "Refusing to overwrite the list with something odd.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} constituents to {OUTPUT.relative_to(config.PROJECT_ROOT)}")
    for row in rows:
        print(f"  {row['symbol']:<12} {row['name']}")


if __name__ == "__main__":
    main()

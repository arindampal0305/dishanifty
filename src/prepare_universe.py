"""Download and prepare every instrument, without training anything.

    python -m src.prepare_universe              # everything still missing
    python -m src.prepare_universe --force      # re-download all of it
    python -m src.prepare_universe --only TCS SBIN

Training is the expensive half (minutes per company, most of it the LSTM), so
it stays on demand in the dashboard. This script does the *cheap* half ahead of
time -- fetch, clean, build features, write labels -- so that selecting a
company in the UI never waits on the network.

It also writes data/universe_status.json, recording how much history each
company actually has. That matters: several NIFTY 50 members listed well after
2015 (Jio Financial in 2023, Eternal in 2021), and the walk-forward design
needs data before 2020 for its first fold to mean anything. The dashboard reads
this file to warn before someone waits on a train that cannot succeed.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime

import pandas as pd

from src import config
from src.run_pipeline import prepare_data

STATUS_FILE = config.PROJECT_ROOT / "data" / "universe_status.json"

# Fold 1 of the walk-forward trains through 2019 and validates on 2020. A
# company listed after that cannot fill the early folds, and one listed after
# the test split opens cannot be scored at all.
FULL_HISTORY_BY = pd.Timestamp("2019-01-01")
TEST_START = pd.Timestamp("2024-01-01")


def _classify(first: pd.Timestamp) -> tuple[str, str]:
    if first <= FULL_HISTORY_BY:
        return "full", "All four walk-forward folds are meaningful."
    if first < TEST_START:
        return "partial", (f"Listed {first:%b %Y}, so the early folds have little "
                           "or no training data. The final test still works.")
    return "insufficient", (f"Listed {first:%b %Y}, after the sealed test window "
                            "opens. Not enough history to train honestly.")


def prepare_one(inst, force: bool) -> dict:
    """Prepare one instrument and describe what came back."""
    prepare_data(inst.key, download=True, force=force)

    labeled = pd.read_csv(inst.labeled_path(), index_col=0, parse_dates=True)
    first, last = labeled.index.min(), labeled.index.max()
    status, note = _classify(first)
    return {
        "key": inst.key, "symbol": inst.symbol, "name": inst.name,
        "rows": int(len(labeled)),
        "first": first.strftime("%Y-%m-%d"), "last": last.strftime("%Y-%m-%d"),
        "status": status, "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even where a file already exists")
    ap.add_argument("--only", nargs="*", default=None,
                    help="limit to these instrument keys")
    args = ap.parse_args()

    instruments = config.all_instruments()
    if args.only:
        wanted = {k.upper() for k in args.only}
        instruments = {k: v for k, v in instruments.items() if k in wanted}

    config.ensure_dirs()
    results: list[dict] = []
    failures: list[tuple[str, str]] = []

    for i, inst in enumerate(instruments.values(), 1):
        head = f"[{i:>2}/{len(instruments)}] {inst.key:<12} {inst.symbol:<16}"
        try:
            row = prepare_one(inst, args.force)
        except Exception as exc:                      # one bad symbol must not
            failures.append((inst.key, str(exc)))     # abort the other 49
            print(f"{head} FAILED  {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            continue
        results.append(row)
        print(f"{head} {row['rows']:>5} rows  {row['first']} -> {row['last']}  "
              f"[{row['status']}]", flush=True)

    STATUS_FILE.write_text(json.dumps({
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "instruments": {r["key"]: r for r in results},
        "failed": dict(failures),
    }, indent=2) + "\n", encoding="utf-8")

    counts = pd.Series([r["status"] for r in results]).value_counts().to_dict()
    print(f"\nPrepared {len(results)}/{len(instruments)}  {counts}")
    if failures:
        print(f"Failed: {', '.join(k for k, _ in failures)}")
    print(f"Status written to {STATUS_FILE.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

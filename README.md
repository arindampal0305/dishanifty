# DishaNifty: Indian Market Direction Prediction

Predicts the next trading day's direction (**UP / DOWN / NEUTRAL**) for the NIFTY 50 index
*or any individual NSE stock*, and turns that prediction into a **BUY / SELL / HOLD** call.

Two companion documents go deeper than this README:
[DATA_FLOW.md](DATA_FLOW.md) traces every file and function from download to on-screen call,
and [APPLICATION_GUIDE.md](APPLICATION_GUIDE.md) explains the app and each feature in
non-technical language. The full project report, as submitted, is in
[report/DishaNifty_Project_Report.docx](report/DishaNifty_Project_Report.docx).

## Research Question
Does increasing model complexity (Random Forest → XGBoost → LSTM) improve predictive accuracy
and trading performance on the Indian stock market?

A second question falls out of the multi-instrument design: **is an individual stock easier or
harder to predict than the index?** The pipeline runs identically on both, so the comparison is
apples to apples.

## Supported Instruments

| Key | Symbol | Kind | Threshold |
|-----|--------|------|-----------|
| `NIFTY50` | `^NSEI` | index | 0.25% |
| `TCS` | `TCS.NS` | stock | 0.50% |
| `RELIANCE` | `RELIANCE.NS` | stock | 0.50% |
| `INFY` | `INFY.NS` | stock | 0.50% |
| `HDFCBANK` | `HDFCBANK.NS` | stock | 0.50% |
| `ICICIBANK` | `ICICIBANK.NS` | stock | 0.50% |
| `SBIN` | `SBIN.NS` | stock | 0.50% |
| `ITC` | `ITC.NS` | stock | 0.50% |

Add more in `src/config.py` — nothing else needs to change.

Individual stocks move roughly twice as much per day as the index, so they use a wider
UP/DOWN threshold. At 0.25% a stock like TCS has almost no NEUTRAL days; at 0.50% the three
classes land near 31/36/33. Both values come from the 0.1 / 0.25 / 0.5% grid, and the
threshold is frozen before the test set is touched.

## Setup

### Prerequisites

- **Python 3.12** (3.12.10 was used throughout development)
- Roughly **500 MB** of free disk space once data and models are built
- An internet connection for the first data download

### 1. Clone and create the environment

```bash
git clone -b final_disha https://github.com/arindampal0305/dishanifty.git
cd dishanifty
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS or Linux
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Fetch the market data

**A fresh clone has no price data and no trained models.** Both are deliberately excluded
from version control (see `.gitignore`): they run to several hundred megabytes, they are
fully reproducible from the code, and committing regenerated copies would bloat the history
on every run. The first command you run after installing therefore has to build them.

```bash
python -m src.prepare_universe
```

This downloads, cleans and builds features and labels for the NIFTY 50 index and all fifty
constituent companies, plus the India VIX and USD/INR macro series. It takes a few minutes,
is resumable, and a failure on one symbol does not abort the rest. On completion it writes
`data/universe_status.json`, recording how much history each instrument actually has.

To prepare only a few instruments instead of all fifty one:

```bash
python -m src.prepare_universe --only NIFTY50 TCS SBIN
```

To re-download everything from scratch, add `--force`.

### 3. Train at least one instrument

Preparing data does not train anything. Until a model exists for an instrument, the
dashboard will show a **NOT TRAINED** panel for it rather than a prediction.

```bash
python -m src.run_pipeline --ticker NIFTY50 --model rf,xgb   # about 20 seconds
python -m src.run_pipeline --ticker NIFTY50 --model all      # about 4 minutes, adds the LSTM
```

The LSTM accounts for roughly 93% of training time on its own, which is why the two tree
models are offered as a separate fast path. You can equally do this from the interface:
select any instrument and press **QUICK TRAIN** (Random Forest and XGBoost) or **FULL
TRAIN** (all three).

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open <http://localhost:8501> if it does not open by itself.

### Pulling updates later

```bash
git pull
pip install -r requirements.txt       # only if requirements.txt changed
```

Your local data and models are untouched by a pull, because they are not tracked. To bring
prices up to the latest session, either press **REFRESH MARKET DATA** in the sidebar or run:

```bash
python -m src.prepare_universe --force
```

### Refreshing the NIFTY 50 constituent list (optional)

Index membership is reviewed a few times a year, so the constituent list lives in a data
file rather than in source. To refresh it:

```bash
python -m src.fetch_universe
```

This rewrites `data/nifty50_constituents.json`. Nothing else in the application touches the
network at import time, so a blocked connection can never prevent the app from starting.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Every instrument shows `[---]` and NOT TRAINED | Normal on a fresh clone. Run step 3, or press QUICK TRAIN. |
| `price data: not downloaded` | Step 2 has not been run for that instrument. |
| An instrument is flagged with `[ ! ]` | It listed after 2015 and lacks the history the early folds need. Expected for Jio Financial, Eternal and Max Healthcare. |
| Intraday chart is empty | Yahoo keeps only 7 days of 1-minute, 60 days of 5-minute and 2 years of hourly bars, and serves nothing outside market hours for some symbols. |
| Edits to `dashboard/retro.py` have no effect | Streamlit caches imported modules. Restart the server rather than reloading the browser. |

## Usage

Full pipeline for one instrument — download, clean, features, labels, train, evaluate, recommend:
```bash
python -m src.run_pipeline --ticker TCS
python -m src.run_pipeline --ticker NIFTY50 --model xgb
```

Individual stages:
```bash
python -m src.data_loader   --ticker TCS      # or --ticker ALL
python -m src.data_cleaner  --ticker TCS
python -m src.features      --ticker TCS
python -m src.target        --ticker TCS --report   # class balance per threshold
python -m src.validation    --ticker TCS
python -m src.evaluate      --ticker TCS --model all
```

Current recommendation:
```bash
python -m src.predict --ticker TCS --model xgb
python -m src.predict --ticker ALL --model xgb
```

Dashboard:
```bash
streamlit run dashboard/app.py
```

Four tabs, plus a screener reached from the glowing button beside the logo:

- **Today** — the BUY/SELL/HOLD call for the next session, class probabilities against the
  45% action threshold, indicator context, and a price chart with eleven timeframe presets
  from 1-minute through to 5-year bars.
- **Track record** — every past call plotted on the price line, green where the market agreed
  and red where it did not, plus rolling accuracy for all three models against the
  majority-class baseline. These come from the final-test model, so they are genuinely
  out-of-sample.
- **Models** — accuracy against the majority-class baseline, the full metrics table, equity
  curves versus Buy & Hold, and the confusion matrix.
- **How it works** — the pipeline diagram and the methodology in plain language.
- **Screener** (separate view) — filter all 51 instruments by RSI level, or by Bollinger band
  breach with a configurable period and sigma, and chart the breaching candles. These screens
  read the indicator files directly, so they work for every instrument whether or not a model
  has been fitted.

**Refresh market data** in the sidebar re-downloads from Yahoo Finance, rebuilds the
indicators, and re-runs the saved model, so the page is never stuck on an old session. It
does not retrain — that is `python -m src.evaluate`. The sidebar always shows the date of the
last bar held on disk. Note that Yahoo serves end-of-day bars: refreshing while the NSE is
open gives a partial, still-moving candle for today.

## Features

Models are fed **only scale-invariant features** — ratios, percentages, and bounded oscillators.
Raw price levels and raw moving averages are computed and stored for charting but never
reach a model: a tree trained on 2015 NIFTY data learns splits around 8,000, and by 2024
every row sits on one side of them. This is also what lets one pipeline serve both a
3,000-rupee stock and a 25,000-point index.

| Group | Features |
|-------|----------|
| Returns | `Return`, `Return_Lag1..3`, `Return_5d` |
| Volatility | `Volatility_10`, `Volatility_20`, `ATR_Pct` |
| Trend | `Close_vs_SMA10/20/50`, `Close_vs_EMA10/20/50`, `SMA10_vs_SMA20`, `SMA20_vs_SMA50` |
| Oscillators | `RSI_14`, `MACD_Norm`, `MACD_Hist_Norm`, `BB_Position`, `BB_Width`, `Momentum_10_Pct` |
| Liquidity | `Volume_Ratio` |
| Macro | `VIX_Level`, `VIX_Change`, `USDINR_Return` |
| Market (stocks only) | `Market_Return`, `Market_Return_5d` — how NIFTY 50 moved that day |

## Target and Signal

```
R_t+1 = (Close_t+1 - Close_t) / Close_t

R_t+1 >  threshold  ->  UP      (2)  ->  BUY
R_t+1 < -threshold  ->  DOWN    (0)  ->  SELL
otherwise           ->  NEUTRAL (1)  ->  HOLD
```

The BUY/SELL/HOLD mapping is **not** a bare argmax. A three-class softmax over noisy daily
returns routinely puts its top class at 0.36 — technically a winner, far too weak to act on.
Anything below 45% confidence is downgraded to HOLD, while the predicted direction is still
reported. See `src/recommend.py`.

## Models

| Model | Input | Scaling | Class imbalance |
|-------|-------|---------|-----------------|
| Random Forest | one row of features | not needed | `class_weight="balanced_subsample"` |
| XGBoost | one row of features | not needed | balanced `sample_weight` |
| LSTM | 30-day sequence | StandardScaler + ±5σ clip, fit on train only | Keras `class_weight` |

All three implement the same `DirectionModel` interface (`src/models/base.py`) and return a
`(n, 3)` probability matrix ordered DOWN / NEUTRAL / UP, so they are fully interchangeable.

Two models are saved per instrument:
- `*_final` — trained on data before 2024, scored once on 2024 onwards. **This is what the
  reported metrics come from.**
- `*_live` — retrained on all available history, used only for forward-looking predictions.
  Never scored, because scoring it would mean testing on its own training data.

## Validation

Expanding-window walk-forward; never a random split.

| Fold | Train | Validate |
|------|-------|----------|
| 1 | 2015–2019 | 2020 |
| 2 | 2015–2020 | 2021 |
| 3 | 2015–2021 | 2022 |
| 4 | 2015–2022 | 2023 |
| Final test | 2015–2023 | 2024–present |

## Results

Held-out test set, 2024-01-01 to 2026-10-16 (~690 trading days), never touched during
development. Transaction cost 0.05% per position change.

**The baseline that matters is not 33.3%.** Uniform chance assumes the three classes are
equally frequent. They are not, so the honest floor is *guess the training set's most common
class every day* — reported below as its own row. A model can clear 33.3% and still be worse
than that one-liner. One standard error at n≈690 is **1.8 percentage points**, so any gap
narrower than about 3.6 points is indistinguishable from noise.

**NIFTY 50**

| Model | Accuracy | Macro F1 | Sharpe | CAGR | Max DD | Trades |
|-------|----------|----------|--------|------|--------|--------|
| Random Forest | 0.388 | 0.383 | 0.44 | +4.8% | −21.4% | 233 |
| XGBoost | 0.358 | 0.338 | −1.56 | −17.6% | −42.0% | 224 |
| LSTM | 0.360 | 0.328 | 0.13 | +0.8% | −17.0% | 38 |
| **Always UP** (baseline) | **0.369** | 0.180 | 0.22 | +2.1% | −15.8% | 1 |
| Buy & Hold | — | — | 0.22 | +2.1% | −15.8% | 1 |

**TCS**

| Model | Accuracy | Macro F1 | Sharpe | CAGR | Max DD | Trades |
|-------|----------|----------|--------|------|--------|--------|
| Random Forest | 0.333 | 0.328 | −0.51 | −12.7% | −54.5% | 215 |
| XGBoost | 0.333 | 0.322 | −0.39 | −9.9% | −40.1% | 198 |
| LSTM | 0.319 | 0.304 | −0.55 | −12.4% | −49.9% | 48 |
| **Always NEUTRAL** (baseline) | **0.349** | 0.173 | 0.00 | 0.0% | 0.0% | 0 |
| Buy & Hold | — | — | −0.46 | −13.2% | −53.4% | 1 |

Macro F1 is the one place the models clearly win: the baseline scores 0.17–0.18 because it
never predicts two of the three classes. The models do spread their predictions across all
three — they are just not spreading them correctly often enough to beat the floor on accuracy.

### Answering the research question

**Does more complexity help? No.** Random Forest — the simplest of the three — is the best
model on the index, and the LSTM is the worst on both instruments. The ordering
RF → XGBoost → LSTM is not a ladder of increasing performance. Added capacity mostly bought
more ways to fit noise: the LSTM's validation loss starts rising after the second epoch, so
early stopping keeps a barely-trained network.

**Is a stock easier to predict than the index? No — harder.** Every model loses roughly five
accuracy points moving from NIFTY 50 to TCS, and every TCS strategy has a negative Sharpe.
Single-stock returns carry idiosyncratic news that none of these features observe, while the
index averages that noise away.

**Honest reading of the numbers.** Exactly one cell in these tables beats the majority-class
baseline: Random Forest on NIFTY 50, 0.388 against 0.369. That is 1.9 points, or about one
standard error — not a significant result. On TCS no model beats the floor at all; all three
land at or below "always say NEUTRAL". The correct conclusion is that **none of these models
demonstrates a reliable edge in next-day direction**, which is the expected finding for daily
price-history-only prediction and is consistent with weak-form market efficiency.

### Attempts to raise accuracy

Both experiments below were run on the **walk-forward validation folds only**; the 2024+ test
set stayed sealed, so neither contaminates the table above.

**1. Regularising XGBoost — fixed a real bug, did not improve prediction.** The original
booster (400 trees, depth 4) scored **0.88 train / 0.35 validation** — it was memorising
individual trading days almost perfectly. Cutting to depth-2 stumps with `min_child_weight=40`
and `reg_lambda=20` closed the gap from +0.53 to +0.11 and lifted validation accuracy ~2
points. On the sealed test set accuracy then moved 0.366 → 0.358 (NIFTY) and 0.341 → 0.333
(TCS) — *down*, and within noise both times. The regularised settings were kept, because they
were selected on validation and because 0.88 train accuracy is indefensible regardless of what
the test set says. Reverting them *because* test accuracy dipped would be tuning on the test
set. The honest lesson: the overfitting was real, but removing it recovered no hidden signal,
because there was none to recover.

**2. Longer horizons — a trap, not an improvement.** Predicting the 3/5/10-day direction
instead of 1-day raises headline accuracy substantially (RF on TCS: 0.376 → 0.427 at 10 days).
But the majority-class baseline rises *faster*, because longer windows are more lopsided, so
the edge versus baseline gets **worse**:

| Horizon | RF accuracy (NIFTY) | Baseline | Edge |
|---------|--------------------|----------|------|
| 1-day | 0.379 | 0.431 | −0.052 |
| 5-day | 0.388 | 0.486 | −0.099 |
| 10-day | 0.399 | 0.513 | −0.115 |

This is the single easiest way to fool yourself in a project like this: the number on the slide
goes up while the model gets worse. It was rejected for that reason.

**What would actually be needed.** Not better hyperparameters — information these features do
not contain. Order-book or intraday microstructure, news and earnings sentiment, FII/DII flow
data, or options-chain positioning. Within price history alone, daily direction is close to a
coin flip, and no amount of model capacity changes that.

## Repository Layout
```
src/
  config.py        instrument registry, paths, feature lists
  data_loader.py   yfinance download
  data_cleaner.py  flatten and clean raw CSVs
  features.py      indicators + scale-invariant model features
  target.py        UP/DOWN/NEUTRAL labelling
  validation.py    walk-forward folds
  models/          base.py, rf_model.py, xgb_model.py, lstm_model.py
  backtest.py      trading simulation, Sharpe, drawdown
  evaluate.py      walk-forward training and scoring
  track_record.py  scores past predictions against what the market did
  recommend.py     direction -> BUY/SELL/HOLD
  predict.py       current call for an instrument
  run_pipeline.py  end-to-end orchestrator
  prepare_universe.py  bulk data prep for every instrument, no training
  fetch_universe.py    refresh the NIFTY 50 constituent list
dashboard/
  app.py           Streamlit UI, views, caching, job launching
  retro.py         styling, ASCII rendering, the Disha loading screen
  screener.py      cross-sectional RSI and Bollinger screens
  timeframes.py    multi-resolution chart data, including live intraday
data/              raw/, processed/, predictions/, constituent + status JSON
results/           metrics and confusion matrices
saved_models/      trained models
report/            project report (.docx/.pdf), figures, diagram generator
```

Note that `data/raw/`, `data/processed/` and `saved_models/` are gitignored. They are
rebuilt by `src.prepare_universe` and `src.run_pipeline`, as described under Setup.

## Tech Stack
Python 3.12 · yfinance · pandas · pandas-ta · scikit-learn · xgboost · tensorflow ·
plotly · streamlit

## Disclaimer
Educational project output, not investment advice. Measured on the sealed test set, sixteen
of twenty four model and instrument combinations are statistically indistinguishable from
simply predicting the most common class every day, and only seven of twenty four beat buy
and hold on risk-adjusted return once transaction costs are charged. Treat every call this
system produces as a demonstration of method, not a signal. Do not trade real money on it.

## Team
| Member | Role |
|--------|------|
| Arindam Pal | Pipeline Foundation |
| Sujal | Random Forest + XGBoost |
| Khushi | LSTM |
| Sneha | EDA + Backtesting + Dashboard + Documentation |

## Academic Context
BIT Mesra, Lalpur Campus — Semester 5 AI/ML Project (2026–2027)
Mentor: Dr. Binod Kumar Sharma Sir

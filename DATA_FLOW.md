# DATA_FLOW.md — what happens behind the scenes

This document traces every path through the system: how raw prices become a
trained model, and what happens when someone clicks around the dashboard. It is
written for someone who has just cloned the repo and wants to know where things
come from before changing them.

Read `APPLICATION_GUIDE.md` instead if you want the plain-language tour of the
features.

---

## 0. The shape of the thing

```
                         src/run_pipeline.py
                                 |
   download ---> clean ---> features ---> labels ---> train/validate ---> save
   (yfinance)    (OHLCV)    (26-28 cols)  (0/1/2)     (walk-forward)     (2 models)
      |             |            |            |             |               |
   data/raw   data/processed  data/processed data/processed results/   saved_models/

                         dashboard/app.py
                                 |
   read features ---> load LIVE model ---> predict tomorrow ---> recommend ---> render
```

Two completely separate entry points touch the same files:

| Entry point | Who runs it | Writes |
|---|---|---|
| `python -m src.run_pipeline` | a human, or the dashboard via subprocess | `data/`, `results/`, `saved_models/` |
| `streamlit run dashboard/app.py` | the user | nothing except the custom-symbol registry |

The dashboard is **read-only** over the artefacts the pipeline produces. It never
trains anything in-process. When you press a refresh or retrain button it spawns
`src.run_pipeline` as a child process and tails its log file. That separation is
deliberate: a TensorFlow fit inside the Streamlit process would block the
server's script thread and lose you the ability to show progress.

---

## 1. Configuration — `src/config.py`

Everything else imports its constants from here. The important ones:

- **Paths.** `data/raw`, `data/processed`, `data/predictions`, `results`,
  `saved_models`. `ensure_dirs()` creates them on demand.
- **Date range.** `START_DATE = "2015-01-01"`, `END_DATE = None` (meaning
  "today").
- **Class vocabulary.** `LABEL_NAMES = {0: DOWN, 1: NEUTRAL, 2: UP}` and
  `ACTION_NAMES = {0: SELL, 1: HOLD, 2: BUY}`. Labels are what the model
  predicts; actions are what gets displayed.
- **`Instrument`.** A frozen dataclass holding `key`, `symbol`, `name`, `kind`
  (`"index"` or `"stock"`) and `threshold`. It also owns its own filenames, so
  no other module ever builds a path by hand:

  ```python
  inst.raw_path()      # data/raw/sbin.csv
  inst.clean_path()    # data/processed/sbin_clean.csv
  inst.features_path() # data/processed/sbin_features.csv
  inst.labeled_path()  # data/processed/sbin_labeled.csv
  ```

  `slug` is the lowercased key and is what every results file is named after.
- **The eight built-ins.** NIFTY50 (`^NSEI`, index), then TCS, RELIANCE, INFY,
  HDFCBANK, ICICIBANK, SBIN, ITC (all `.NS` stocks).
- **Macro series.** `INDIA_VIX` (`^INDIAVIX`) and `USDINR` (`USDINR=X`). These
  are never predicted, only used as features.
- **Thresholds.** `_INDEX_THRESHOLD = 0.0025` (±0.25%),
  `_STOCK_THRESHOLD = 0.005` (±0.50%). A single stock is roughly twice as
  volatile as the index, so a shared band would make NEUTRAL nearly empty for
  stocks.
- **The custom registry.** `data/custom_instruments.json`. When a user types a
  Yahoo symbol into the sidebar, `register_instrument()` appends it here and
  `all_instruments()` merges it with the built-ins. `get_instrument(key)` is the
  single lookup used by both the pipeline and the dashboard, so a user-added
  symbol is indistinguishable from a built-in from then on.

### Feature list selection

```python
def model_features_for(instrument):
    if instrument.kind == "stock":
        return MODEL_FEATURES + MARKET_FEATURES   # 28 columns
    return list(MODEL_FEATURES)                   # 26 columns
```

A stock gets two extra columns (`Market_Return`, `Market_Return_5d`) describing
what NIFTY did. The index itself has no "market" to compare against, so it gets
26. This function is the only place the count is decided, and both training and
prediction call it — which is what guarantees the column order at predict time
matches the order the model was fitted on.

---

## 2. Download — `src/data_loader.py`

```python
yf.download(symbol, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False)
```

Written to `data/raw/{slug}.csv`. Two details matter:

- **`auto_adjust=True` is not optional.** Without it, TCS's stock split shows up
  as a genuine-looking −50% single-day return. A model trained on that learns
  that a specific date is catastrophic, which is nonsense. Adjusted prices
  restore continuity across splits and dividends.
- **Skip-if-exists.** `download_instrument()` returns early if the file is
  already on disk, unless `force=True`. This keeps repeated pipeline runs from
  hammering Yahoo. The dashboard's "refresh data" button passes `--force`.

`download_macro()` loops `MACRO_INSTRUMENTS` and does the same for VIX and
USDINR.

---

## 3. Clean — `src/data_cleaner.py`

yfinance hands back a three-row header (`Price` / `Ticker` / `Date`) when a
single symbol is requested, which `pd.read_csv` cannot parse into a flat frame.
`read_raw()` handles it:

1. `header=[0, 1]`, then flatten the MultiIndex down to the price level.
2. Coerce the index to datetime and **drop rows whose index will not parse** —
   that is how the leftover header junk gets removed.
3. `apply(pd.to_numeric, errors="coerce")` on every column.

`clean()` then:

1. Selects `["Open", "High", "Low", "Close", "Volume"]`.
2. Drops duplicate index entries, keeping the first.
3. Sorts by date.
4. `ffill()` to carry the last known value across holidays, then
   `dropna(subset=["Close"])` because a row with no close is useless.

Output: `data/processed/{slug}_clean.csv`.

---

## 4. Features — `src/features.py`

Three stages, all operating on the cleaned frame.

### 4a. `add_indicators()`

Standard technical indicators via `pandas_ta`: Return, Volatility_10/20,
SMA/EMA at 10/20/50, RSI_14, MACD(12,26,9), Bollinger Bands(20,2), ATR_14,
Momentum_10.

A small helper, `_pick(df, prefix)`, grabs a column by prefix because
`pandas_ta` changes its suffix conventions between versions
(`MACDh_12_26_9` vs `MACDh_12_26_9.0`). Matching on the prefix survives upgrades.

### 4b. `add_model_features()` — the scale-invariance step

This is the single most important design decision in the repo. **Raw price
levels are never fed to a model.** Instead everything is converted to a ratio or
a percentage:

| Feature | What it is |
|---|---|
| `Return_Lag1/2/3` | yesterday's, and the two days before, returns |
| `Return_5d` | week-scale momentum |
| `Close_vs_SMA10/20/50`, `Close_vs_EMA10/20/50` | price ÷ moving average |
| `SMA10_vs_SMA20`, `SMA20_vs_SMA50` | trend alignment |
| `MACD_Norm`, `MACD_Hist_Norm` | MACD ÷ close |
| `BB_Position` | where in the band the close sits, 0–1 |
| `BB_Width` | band width ÷ close |
| `ATR_Pct` | ATR ÷ close |
| `Momentum_10_Pct` | 10-day momentum ÷ close |
| `Volume_Ratio` | volume ÷ its own 20-day average |
| `RSI_14`, `Volatility_10/20` | already bounded or already a ratio |

Why: a tree that learns a split at `SMA_50 < 9000` from 2015 data is useless
once the series trades at 24000. Ratios are comparable across the whole decade,
which is the only way an expanding-window training scheme can work at all.

### 4c. `add_macro_features()` — and the leakage safeguard

VIX and USDINR trade on a slightly different calendar from an individual stock.
The function reindexes each macro series onto the instrument's own date index
and forward-fills. For a stock it also reindexes NIFTY to produce
`Market_Return` and `Market_Return_5d`.

**Forward-fill only ever carries a past observation onto a later date.** It
cannot pull a future value backwards, so no future information enters the
feature row. This is worth internalising before you add any new macro input: an
interpolation or a backfill here would quietly destroy the validity of every
number in `results/`.

`build_features()` finishes by dropping rows with NaN in any required column —
the first ~50 rows, where the 50-day averages have not warmed up yet.

Output: `data/processed/{slug}_features.csv`.

---

## 5. Label — `src/target.py`

```python
df["Next_Return"] = df["Close"].pct_change().shift(-1)
df["Target"] = 1
df.loc[df["Next_Return"] >  threshold, "Target"] = 2   # UP
df.loc[df["Next_Return"] < -threshold, "Target"] = 0   # DOWN
```

The `shift(-1)` is what makes this a forecasting problem: the row dated *t*
carries features known at *t*'s close and a label describing the move from *t*
to *t+1*. The final row has no `Next_Return` and is **dropped** — you cannot
label a day whose successor has not happened yet.

`class_balance()` prints the resulting distribution, and `threshold_report()`
sweeps `THRESHOLD_GRID = [0.001, 0.0025, 0.005]` so you can see how the balance
shifts if you retune the band.

Output: `data/processed/{slug}_labeled.csv`.

---

## 6. Splits — `src/validation.py`

> "Never a random split — time order is the point."

A random split would let the model train on 2024 and be tested on 2019, which
for a time series is simply reading the answer. Instead:

```
TRAIN_START = "2015-01-01"
TEST_START  = "2024-01-01"

fold 1   train <= 2019-12-31   validate 2020   "COVID year"
fold 2   train <= 2020-12-31   validate 2021   "Post-COVID recovery"
fold 3   train <= 2021-12-31   validate 2022   "Rate hike cycle"
fold 4   train <= 2022-12-31   validate 2023   "Stabilization year"
FINAL    train <= 2023-12-31   test     2024 -> today
```

Four **expanding** windows: each fold keeps all prior history and adds a year.
Each validation year is also a deliberately different market regime, so a model
that only works in a trending bull market has four chances to expose itself.

Everything from `TEST_START` onwards is sealed. `get_test_split()` is the only
function that opens it, and `run_model()` calls it exactly once per model.

---

## 7. Training — `src/evaluate.py` and `src/models/`

### The model interface — `src/models/base.py`

Every model implements the same abstract class:

```python
class DirectionModel:
    name: str
    lookback: int = 1          # 1 = tabular, >1 = needs a sequence
    def fit(self, X, y): ...
    def predict_proba(self, X, context=None): ...
    def predict(self, X, context=None): ...
    def save(self, stem); def load(stem)
```

Two helpers live here and are easy to overlook:

- **`align_proba()`.** If a training fold happens to contain no DOWN days,
  sklearn returns a two-column probability matrix. Handing that back as-is would
  silently relabel UP as NEUTRAL, because column 1 would be read as "NEUTRAL"
  when it actually means "UP". `align_proba` pads the missing class with zero
  probability so the output is always `(n, 3)` in class order 0/1/2.
- **`build_sequences(array, lookback)`.** Turns an `(n, f)` feature matrix into
  `(n - lookback + 1, lookback, f)` for the LSTM.

### The three models

**`rf_model.py` — Random Forest**

```python
RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=25,
                       max_features="sqrt", class_weight="balanced_subsample",
                       random_state=42, n_jobs=-1)
```

Depth 8 and a 25-sample leaf floor are both deliberate underfitting.
`class_weight="balanced_subsample"` stops NEUTRAL, the majority class, from
swallowing the other two.

**`xgb_model.py` — XGBoost**

```python
XGBClassifier(objective="multi:softprob", num_class=3,
              n_estimators=150, learning_rate=0.02, max_depth=2,
              subsample=0.6, colsample_bytree=0.5, min_child_weight=40,
              reg_lambda=20.0, reg_alpha=1.0, gamma=0.5,
              random_state=42, n_jobs=-1,
              eval_metric="mlogloss", tree_method="hist")
```

These numbers look timid because they are. The code records the earlier attempt:
400 trees at depth 4 reached 0.88 training accuracy against 0.35 validation —
pure memorisation. Depth 2 with heavy L1/L2 is what is left after that was
beaten out of it. Class balance comes from
`compute_sample_weight("balanced", y)`.

**`lstm_model.py` — LSTM**

`lookback=30`, `units=(64, 32)`, `dropout=0.2`, `learning_rate=1e-3`,
`epochs=60`, `batch_size=32`, `seed=42`, gradient `clip=5.0`.

```
LSTM(64, return_sequences=True) -> Dropout(0.2)
LSTM(32)                        -> Dropout(0.2)
Dense(16, relu) -> Dense(3, softmax)
Adam, sparse_categorical_crossentropy
```

Scaling uses `StandardScaler`, not `MinMaxScaler`: daily returns are
heavy-tailed, and min-max would compress the ordinary 99% of days into a sliver
of the range because one crash day defines the bounds.

For `EarlyStopping(patience=8, restore_best_weights=True)` the training block is
split **chronologically** at 85% — never shuffled, for the same reason the outer
folds are not shuffled.

Saving is split in two because Keras graphs do not survive joblib: the network
goes to `.keras`, the scaler and metadata to `.joblib`.

### The registry and lazy import

```python
MODEL_REGISTRY = {"rf": ..., "xgb": ..., "lstm": None}
```

`lstm` is `None` and gets imported inside `build_model()` on first use.
Importing TensorFlow costs about 15 seconds, and you should not pay it to run
`--model rf`.

### `evaluate_window()`

For one train/test pair: fit, `predict_proba`, write a predictions frame, and
backtest. Returns `accuracy`, `macro_f1`, `sharpe`, `cagr`, `max_drawdown`,
`n_trades`.

Two baselines are computed alongside:

- **`_majority_baseline()`** — always predict the most common class in the
  training window. The docstring says it plainly: *"This, not 1/3, is the number
  a model has to beat."* Because NEUTRAL is around 35% of days, a model scoring
  34% is worse than a constant.
- **`_buy_and_hold()`** — the return of just owning the thing.

### `run_model()` — the full sequence

1. Loop the four folds. Each writes
   `data/predictions/{slug}_{model_key}_fold{n}.csv` and prints
   `Fold N: acc ... f1 ...`.
2. Run the sealed final test. Writes `{slug}_{model_key}_finaltest.csv`, prints
   `FINAL TEST: ...`.
3. Save the fitted model as `saved_models/{slug}_{model_key}_final`.
4. Write `results/{slug}_{model_key}_confusion.csv`.
5. **Refit on every available day**, including the test window, and save as
   `saved_models/{slug}_{model_key}_live`. Prints
   `live model saved (trained through ...)`.

### Why two saved models per algorithm

| | `_final` | `_live` |
|---|---|---|
| Trained on | 2015–2023 | all history |
| Scored on | 2024→ , exactly once | never |
| Used for | every number in the MODELS and TRACK RECORD tabs | the call in the TODAY tab |

The `_final` model's score is honest because it never saw the test window. The
`_live` model is more current — which is what you want for tomorrow's call — but
it has seen everything, so any accuracy you computed for it would be
meaningless. Conflating these two is the most common way a project of this type
ends up claiming an edge that is not there, and keeping them in separate files
with separate names is how this repo avoids it.

Those printed markers (`Fold N:`, `FINAL TEST:`, `live model saved`) are not
just log noise — the dashboard's loading screen counts them to compute progress.
If you change the wording, update `dashboard/app.py` too.

---

## 8. Backtest — `src/backtest.py`

```python
TRANSACTION_COST = 0.0005        # 0.05% per position change
TRADING_DAYS_PER_YEAR = 252
POSITION_MAP = {0: -1, 1: 0, 2: 1}   # DOWN -> short, NEUTRAL -> cash, UP -> long
```

Predictions become positions, positions times next-day return give the strategy
return, and cost is charged on every change of position. Without the cost term a
model that flips long/short daily looks far better than it is. Output feeds
Sharpe, CAGR, max drawdown and trade count.

---

## 9. The orchestrator — `src/run_pipeline.py`

```
python -m src.run_pipeline --ticker SBIN --model all
```

Flags: `--ticker`, `--model`, `--skip-download`, `--force`, `--prepare-only`.

`prepare_data(inst, ...)`:

1. Download the instrument, then the macro series.
2. Clean everything needed — for a **stock** that includes `MARKET_INDEX_KEY`
   (NIFTY50), because the market-context features depend on it. Ask for SBIN
   with no NIFTY data on disk and the pipeline fetches NIFTY for you.
3. `build_features()` → `features_path()`.
4. `generate_target()` → `labeled_path()`.
5. Print the class balance.

`main()` then runs each requested model and concatenates the per-fold rows, the
final-test row, the majority-class baseline rows and the Buy & Hold row into a
single `results/{slug}_metrics.csv`. That one file is what the dashboard reads
for every metric it displays.

### Artefacts, in one place

```
data/raw/{slug}.csv
data/processed/{slug}_clean.csv
data/processed/{slug}_features.csv
data/processed/{slug}_labeled.csv
data/predictions/{slug}_{model}_fold{1..4}.csv
data/predictions/{slug}_{model}_finaltest.csv
results/{slug}_metrics.csv
results/{slug}_{model}_confusion.csv
saved_models/{slug}_{model}_final.{joblib,keras}
saved_models/{slug}_{model}_live.{joblib,keras}
data/custom_instruments.json
logs/{slug}_*.log
```

---

## 10. Prediction — `src/predict.py`

This is the path a dashboard page load takes.

```python
MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM"}
```

- `model_stem(inst, key, variant)` builds the path.
- `has_saved_model()` / `available_models()` — the dashboard uses these to draw
  the `[###]` badge showing which of the three are trained for a given symbol.
- `load_saved_model()`, `load_feature_history()` — read `saved_models/` and
  `features_path()`.

`predict_latest(inst, model_key)`:

1. Load the **live** model.
2. Load the feature history and select `model_features_for(inst)` — same
   function the trainer used, so column order cannot drift.
3. If the model needs a sequence, slice the context:
   ```python
   context = X.iloc[-model.lookback:-1] if model.lookback > 1 else None
   ```
   For the LSTM that is the 29 rows before today; today's row is passed
   separately and the model stacks them into one 30-step window.
4. `proba = model.predict_proba(X.iloc[[-1]], context)[0]`
5. Hand the probabilities and the last feature row to `recommend()`.

Note what is *not* here: no download, no training, no disk writes. A page load
is a model load plus one forward pass.

---

## 11. Recommendation — `src/recommend.py`

```python
MIN_CONFIDENCE = 0.45
```

```python
predicted = int(np.argmax(proba))
confidence = float(proba[predicted])
acted = predicted if confidence >= min_confidence else 1   # fall back to HOLD
```

A three-class model over noisy daily returns routinely puts its top class at
36%. That is technically a winner and nowhere near enough to act on. Below 45%
the action is downgraded to HOLD, `low_confidence` is set, and a `reason` string
is attached — but the predicted direction is still reported, so the UI can show
"the model leaned UP, too weakly to act".

`_indicator_notes()` turns the last feature row into readable context: RSI
overbought/oversold/neutral, MACD bullish/bearish, above or below the 50-day
EMA, and 10-day volatility annualised and bucketed high/low/moderate. These are
**descriptions of the inputs, not reasons for the output** — a tree ensemble has
no single reason, and the notes must not be read as an explanation of the call.

`DISCLAIMER` lives here and is rendered in the dashboard footer:

> Educational project output, not investment advice. Predictions are statistical
> estimates and are frequently wrong.

---

## 12. Track record — `src/track_record.py`

`load_track_record(inst, model_key)` joins
`data/predictions/{slug}_{model_key}_finaltest.csv` with the labeled CSV on
`Close` / `Next_Return` / `Target`, and adds `Predicted`, `Actual`, `Correct`
and `Confidence`.

The critical reading note is in the docstring: **a row dated *t* is the call made
after *t*'s close about the move from *t* to *t+1*.** Off-by-one here would
inflate or destroy the accuracy figure.

`hit_rates()` returns four numbers: `n`, `accuracy`, `n_directional`,
`directional_accuracy`. The last two exclude the days the model said NEUTRAL,
which answers a different and more interesting question — when it does commit to
a side, how often is it right?

Because this reads the **final-test** predictions, it is out-of-sample. It is
not a replay of the live model.

---

## 13. The dashboard request cycle — `dashboard/app.py`

Streamlit re-executes the whole script top to bottom on every interaction.
Without caching, each click would reload a model and re-read a CSV.

### Caching, keyed on file mtime

```python
@st.cache_resource(show_spinner=False)
def _prediction(ticker, model_key, data_mtime, model_mtime):
    return predict_latest(config.get_instrument(ticker), model_key)

def cached_prediction(inst, model_key):
    """Keyed on both the data and the model file, so retraining invalidates it."""
    return _prediction(
        inst.key, model_key,
        inst.features_path().stat().st_mtime,
        model_stem(inst, model_key, "live").with_suffix(".joblib").stat().st_mtime,
    )
```

The two mtimes are part of the cache key but unused in the body. That is the
whole trick: retraining rewrites the model file, its mtime changes, the key
changes, and the stale prediction is dropped automatically. No manual
invalidation, and no chance of showing yesterday's call after a retrain.

Data frames use `@st.cache_data` on the same mtime principle.

`baseline_accuracy(inst)` reads `results/{slug}_metrics.csv`, filters
`window == "Final Test"` and `model.str.startswith("Always ")`, and falls back
to `CHANCE = 1/3` if the file has no baseline row. The dashboard compares every
accuracy against this, not against 33%.

### What a click does

| Interaction | Work done |
|---|---|
| Change instrument or model | re-read cached frames, one cached `predict_latest` |
| Switch tab | render only; tabs are built in the same pass |
| Add a custom symbol | `register_instrument()` writes the JSON, then a retrain job |
| Refresh data | spawn `run_pipeline --force`, then clear caches |
| Retrain | spawn `run_pipeline`, then clear caches |

### The subprocess loading screen

`run_job(title, argv, slug, stages)` is the one place the dashboard causes work
to happen:

```python
env = {**os.environ,
       "PYTHONUNBUFFERED": "1",
       "PYTHONIOENCODING": "utf-8",
       "TF_CPP_MIN_LOG_LEVEL": "3"}
```

- Child stdout goes to a **log file** in `logs/`, not a pipe. A pipe would
  deadlock: TensorFlow is chatty, the 64 KB pipe buffer fills, and the child
  blocks on write while the parent is blocked waiting for the child.
- `-u` plus `PYTHONUNBUFFERED=1` means lines land in the file as they are
  printed rather than at exit, so progress is actually live.
- `TF_CPP_MIN_LOG_LEVEL=3` silences the CUDA/oneDNN banner that would otherwise
  dominate the log the user is reading.
- The parent polls `log_path.read_text(...)` every 0.4s while
  `proc.poll() is None`, counts the `Fold N:` / `FINAL TEST:` / `live model
  saved` markers against `len(models) * 6` expected stages, and animates Disha
  with the current percentage and the log tail.
- On a nonzero return code the error is stashed in
  `st.session_state["job_error"]` and surfaced after the rerun, so a failed
  training run shows a message instead of silently doing nothing.
- Finally both caches are cleared and `st.rerun()` redraws against the new
  artefacts.

### Rendering — `dashboard/retro.py`

Two font stacks exist on purpose: `FONT_UI` (a webfont) for chrome, and
`FONT_ART` (system monospace) for ASCII. A webfont with even slightly off
box-drawing metrics shears every frame in the app.

One Streamlit quirk is load-bearing: **ASCII must be emitted as `<div>`, never
`<pre>`.** Streamlit rewrites a raw `<pre>` into its own
`<div data-testid="stMarkdownPre">` and **strips the author's class**, so a
`pre.crt { white-space: pre }` rule never matches and every art block collapses
to one line. A raw `<div class="crt">` keeps its class. Hence:

```python
def crt(text, tone="", bare=False):
    classes = " ".join(filter(None, ["crt", tone, "bare" if bare else ""]))
    st.markdown(f'<div class="{classes}">{html.escape(text)}</div>',
                unsafe_allow_html=True)
```

Also worth knowing while developing: Streamlit re-executes only the **main
script** on rerun. `dashboard/retro.py` stays cached in `sys.modules`, so edits
to it need a full server restart, not a browser reload.

---

## 14. Leakage safeguards, collected

If you change anything in this pipeline, these are the invariants that keep the
reported numbers meaningful:

1. `shift(-1)` on the target, and the final unlabeled row dropped.
2. Expanding **chronological** folds; no shuffling, anywhere, including the
   LSTM's internal early-stopping split.
3. The window from 2024 onwards is touched exactly once per model.
4. Macro features are **forward**-filled only — never interpolated or
   backfilled.
5. Features are scale-invariant, so an expanding window is legitimate.
6. `_live` models are never scored; `_final` models are never refitted.
7. Scalers are fitted inside `fit()` on training data only, never on the full
   series.
8. `align_proba()` guarantees three probability columns in class order.

---

## 15. The honest finding

Across both the index and individual stocks, next-day direction predicted from
price history alone lands within roughly one standard error of always guessing
the most common class. More capacity did not help — the LSTM is the weakest of
the three. The value of this repo is the method: correct temporal validation, a
sealed test set, a real baseline, transaction costs, and a confidence gate.
Treat every call it produces as a demonstration, not a signal.

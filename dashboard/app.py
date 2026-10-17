"""
DishaNifty dashboard — retro terminal edition.

    streamlit run dashboard/app.py

Four views: today's call, how the model's past calls actually turned out, how the
three models compare on the held-out test set, and a plain-language primer.

Any instrument in the registry can be trained from the sidebar, so selecting a
symbol that has never been fitted shows a TRAIN button rather than an error.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import retro
from dashboard.retro import (ACTION_COLORS, DIRECTION_COLORS, DOWN_C, FLAT_C, GREEN,
                             MODEL_COLORS, MUTED, UP_C)
from src import config
from src.backtest import run_backtest
from src.config import LABEL_NAMES
from src.predict import MODEL_LABELS, available_models, model_stem, predict_latest
from src.recommend import DISCLAIMER, MIN_CONFIDENCE
from src.track_record import hit_rates, load_track_record

st.set_page_config(page_title="DISHA // NSE terminal", page_icon="▲", layout="wide")
retro.inject_css()

LOGS_DIR = config.PROJECT_ROOT / "logs"
MARKER_SYMBOLS = {"UP": "triangle-up", "DOWN": "triangle-down", "NEUTRAL": "circle"}
CHANCE = 1 / 3


# --------------------------------------------------------------------------
# Loading. Every cache key includes the file's mtime, so refreshed data
# invalidates the cache even if it is not cleared explicitly.
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float, index_col, parse_dates: bool) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)


def read_csv(path: Path, index_col=0, parse_dates: bool = True) -> pd.DataFrame:
    return _read(str(path), path.stat().st_mtime, index_col, parse_dates)


@st.cache_data(show_spinner=False)
def _track_record(ticker: str, model_key: str, mtime: float) -> pd.DataFrame | None:
    return load_track_record(config.get_instrument(ticker), model_key)


def track_record(ticker: str, model_key: str) -> pd.DataFrame | None:
    inst = config.get_instrument(ticker)
    path = config.PREDICTIONS_DIR / f"{inst.slug}_{model_key}_finaltest.csv"
    if not path.exists():
        return None
    return _track_record(ticker, model_key, path.stat().st_mtime)


@st.cache_resource(show_spinner=False)
def _prediction(ticker: str, model_key: str, data_mtime: float, model_mtime: float) -> dict:
    return predict_latest(config.get_instrument(ticker), model_key)


def cached_prediction(inst, model_key: str) -> dict:
    """Keyed on both the data and the model file, so retraining invalidates it."""
    return _prediction(
        inst.key, model_key,
        inst.features_path().stat().st_mtime,
        model_stem(inst, model_key, "live").with_suffix(".joblib").stat().st_mtime,
    )


def baseline_accuracy(inst) -> tuple[float, str]:
    """
    The accuracy a model actually has to beat: guess the most common class daily.

    Uniform 1/3 flatters the models, because the three classes are not equally
    frequent. Falls back to 1/3 only when the metrics file predates this row.
    """
    path = config.RESULTS_DIR / f"{inst.slug}_metrics.csv"
    if not path.exists():
        return CHANCE, "chance"
    m = read_csv(path, index_col=None, parse_dates=False)
    row = m[(m["window"] == "Final Test") & m["model"].str.startswith("Always ")]
    if row.empty:
        return CHANCE, "chance"
    return float(row["accuracy"].iloc[0]), str(row["model"].iloc[0])


def chart(fig: go.Figure, height: int = 320, legend: bool = True) -> None:
    st.plotly_chart(retro.terminal_chart(fig, height, legend), width="stretch")


# --------------------------------------------------------------------------
# Jobs — training and refreshing, both behind the Disha loading screen
# --------------------------------------------------------------------------

def run_job(title: str, argv: list[str], slug: str, stages: int = 0) -> None:
    rc, log = retro.run_job(title, argv, LOGS_DIR / f"{slug}.log",
                            config.PROJECT_ROOT, total_stages=stages)
    st.cache_data.clear()
    st.cache_resource.clear()
    if rc != 0:
        st.session_state["job_error"] = "\n".join(retro.log_lines(log)[-25:])
    else:
        st.session_state.pop("job_error", None)
    st.rerun()


def train(inst, download: bool = True) -> None:
    args = ["--ticker", inst.key, "--model", "all"]
    args += ["--force"] if download else ["--skip-download"]
    run_job(f"training all three models on {inst.name}",
            retro.python_cmd("src.run_pipeline", *args), f"{inst.slug}_train",
            stages=retro.train_stages(list(MODEL_LABELS)))


def refresh(inst) -> None:
    run_job(f"refreshing market data for {inst.name}",
            retro.python_cmd("src.run_pipeline", "--ticker", inst.key,
                             "--prepare-only", "--force"),
            f"{inst.slug}_refresh")


# --------------------------------------------------------------------------
# Sidebar — control panel
# --------------------------------------------------------------------------

with st.sidebar:
    retro.crt(retro.box(["D I S H A", "nse direction terminal"], title="v1.0"), bare=True)

    instruments = config.all_instruments()

    def _option_label(key: str) -> str:
        inst = instruments[key]
        n = len(available_models(inst))
        mark = "###"[:n].ljust(3, "-")
        return f"[{mark}] {inst.name}"

    default = "TCS" if "TCS" in instruments else next(iter(instruments))
    ticker = st.selectbox("> instrument", list(instruments),
                          index=list(instruments).index(default),
                          format_func=_option_label)
    instrument = config.get_instrument(ticker)
    st.caption("`[###]` = three models trained, `[---]` = none yet.")

    trained = available_models(instrument)
    missing = [k for k in MODEL_LABELS if k not in trained]

    if trained:
        model_key = st.selectbox("> model", trained, index=len(trained) - 1,
                                 format_func=lambda k: MODEL_LABELS[k])
    else:
        model_key = None
    if missing and trained:
        st.caption(f"not trained yet: {', '.join(MODEL_LABELS[k] for k in missing)}")

    st.divider()
    with st.expander("+ add another symbol"):
        with st.form("add_symbol", clear_on_submit=False):
            symbol = st.text_input("> yahoo symbol", placeholder="SBIN.NS")
            nice_name = st.text_input("> display name", placeholder="State Bank of India")
            kind = st.radio("> kind", ["stock", "index"], horizontal=True)
            submitted = st.form_submit_button("ADD + TRAIN", width="stretch")
        st.caption(
            "Any Yahoo Finance symbol works: `SBIN.NS`, `WIPRO.NS`, `^NSEBANK`. "
            "NSE equities end in `.NS`; indices start with `^`."
        )

    if submitted:
        try:
            added = config.register_instrument(symbol, nice_name, kind)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["pending_train"] = added.key

    st.divider()

    if instrument.features_path().exists():
        features_head = read_csv(instrument.features_path())
        last_bar = features_head.index[-1].date()
        behind = int(np.busday_count(last_bar, date.today()))
        state = "CURRENT" if behind <= 1 else f"{behind} SESSIONS BEHIND"
        retro.crt(retro.box([f"last bar  {last_bar:%d %b %Y}", f"status    {state}"],
                            title="DATA"),
                  tone="" if behind <= 1 else "amber", bare=True)

    if st.button("REFRESH MARKET DATA", width="stretch", disabled=not trained):
        refresh(instrument)
    if st.button("RETRAIN ALL MODELS", width="stretch", disabled=not trained):
        train(instrument)
    st.caption(
        "Refresh re-downloads prices and rebuilds indicators, then re-runs the saved "
        "models. Retrain redoes the whole walk-forward study — a few minutes."
    )

    st.divider()
    retro.crt(retro.box([
        f"symbol    {instrument.symbol}",
        f"kind      {instrument.kind}",
        f"threshold +/-{instrument.threshold:.2%}",
        f"custom    {'yes' if config.is_custom(instrument.key) else 'no'}",
    ], title="INSTRUMENT"), tone="dim", bare=True)


# --------------------------------------------------------------------------
# Masthead
# --------------------------------------------------------------------------

st.markdown(f'<div class="logo">{retro.LOGO}</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='tag'>next-day direction &middot; nse &middot; "
    "random forest / xgboost / lstm</div>",
    unsafe_allow_html=True,
)
st.write("")

if err := st.session_state.pop("job_error", None):
    st.error("The last job failed. Tail of the log:")
    retro.crt(err, tone="red")

# A symbol added from the sidebar has no data at all yet, so train immediately.
if pending := st.session_state.pop("pending_train", None):
    train(config.get_instrument(pending))


# --------------------------------------------------------------------------
# Not ready yet — offer to build it instead of dead-ending
# --------------------------------------------------------------------------

if not trained:
    has_data = instrument.labeled_path().exists()
    retro.crt(retro.box([
        f"instrument   {instrument.name} ({instrument.symbol})",
        f"price data   {'present' if has_data else 'not downloaded'}",
        "models       none on disk",
        "",
        "Nothing has been fitted for this symbol yet, so there is no call to",
        "show. Training runs the full walk-forward study: four expanding folds,",
        "one sealed final test, then a live refit on all history.",
        "",
        "Takes roughly three to six minutes. Disha will keep you company.",
    ], title="NOT TRAINED", width=76), tone="amber")

    left, right = st.columns([1, 3])
    if left.button("TRAIN NOW", width="stretch", type="primary"):
        train(instrument)
    st.stop()

model_label = MODEL_LABELS[model_key]
features = read_csv(instrument.features_path())
latest = features.iloc[-1]
change = latest["Close"] / features["Close"].iloc[-2] - 1

retro.crt(
    "  ".join([
        "SYS ONLINE", "│", instrument.symbol, "│", instrument.kind.upper(), "│",
        f"MODEL {model_label.upper()}", "│",
        f"LAST BAR {features.index[-1]:%d %b %Y}", "│",
        f"{len(features):,} SESSIONS",
    ]),
    tone="dim",
)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Last close", f"Rs {latest['Close']:,.2f}", f"{change:+.2%}")
h2.metric("Bars loaded", f"{len(features):,}")
h3.metric("RSI (14)", f"{latest['RSI_14']:.0f}")
h4.metric("Volatility", f"{latest['Volatility_10'] * (252 ** 0.5):.0%}",
          help="10-day, annualised")

st.write("")
closes = features["Close"].tail(180)
retro.crt(
    f"CLOSE, LAST {len(closes)} SESSIONS   {closes.min():,.0f} .. {closes.max():,.0f}\n"
    f"{retro.sparkline(closes, width=108)}",
    tone="cyan",
)

try:
    rec = cached_prediction(instrument, model_key)
except (FileNotFoundError, OSError) as exc:
    st.error(str(exc))
    st.stop()

record = track_record(ticker, model_key)
tab_today, tab_record, tab_models, tab_about = st.tabs(
    ["TODAY", "TRACK RECORD", "MODELS", "HOW IT WORKS"]
)

# --------------------------------------------------------------------------
# Today
# --------------------------------------------------------------------------

with tab_today:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        color = ACTION_COLORS[rec["action"]]
        st.markdown(
            f"""
            <div class="panel glow">
              <div class="eyebrow">call for the next session</div>
              <div class="big-call" style="color:{color}">&gt; {rec['action']}</div>
              <div style="font-size:13px;color:{MUTED};margin-top:6px">
                direction
                <b style="color:{DIRECTION_COLORS[rec['direction']]}">{rec['direction']}</b>
                &nbsp;·&nbsp; {rec['confidence']:.0%} confidence
              </div>
              <div style="font-size:12px;color:{MUTED};margin-top:14px;
                          border-top:1px solid #163a22;padding-top:12px">{rec['reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"{model_label} · data through {rec['as_of']:%d %b %Y}")

    with right:
        retro.crt(
            retro.labelled_bars(
                [("UP", rec["prob_up"]), ("NEUTRAL", rec["prob_neutral"]),
                 ("DOWN", rec["prob_down"])],
                width=34, marker=MIN_CONFIDENCE,
            ),
            tone="",
        )
        if record is not None and len(record):
            last = record.iloc[-1]
            verdict = "HIT " if last["Correct"] else "MISS"
            retro.crt(retro.box([
                f"settled   {last.name:%d %b %Y}",
                f"model     {last['Predicted']}",
                f"market    {last['Actual']}  ({last['Next_Return']:+.2%})",
                f"result    {verdict}",
            ], title="MOST RECENT SETTLED CALL"),
                tone="" if last["Correct"] else "red")

    with st.expander("indicator context"):
        retro.crt("\n".join(f"· {note}" for note in rec["notes"]), bare=True)
        i1, i2, i3 = st.columns(3)
        i1.metric("MACD", f"{latest['MACD']:.2f}",
                  "Bullish" if latest["MACD"] > latest["MACD_Signal"] else "Bearish")
        i2.metric("Close vs 50-day EMA", f"{latest['Close_vs_EMA50']:+.2%}")
        i3.metric("Bollinger position", f"{latest['BB_Position']:.2f}",
                  help="0 = lower band, 1 = upper band")

    st.subheader("Price")
    window = st.radio("Window", [60, 120, 250, 500], index=1, horizontal=True,
                      format_func=lambda n: f"{n}d", label_visibility="collapsed")
    recent = features.tail(window)

    price = go.Figure()
    price.add_trace(go.Candlestick(
        x=recent.index, open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Close"], name="Price",
        increasing=dict(line_color=UP_C, fillcolor=UP_C),
        decreasing=dict(line_color=DOWN_C, fillcolor=DOWN_C)))
    price.add_trace(go.Scatter(x=recent.index, y=recent["SMA_20"], name="SMA 20",
                               line=dict(width=1.2, color=FLAT_C)))
    price.add_trace(go.Scatter(x=recent.index, y=recent["EMA_50"], name="EMA 50",
                               line=dict(width=1.2, color=MUTED, dash="dot")))
    price.update_layout(xaxis_rangeslider_visible=False)
    chart(price, height=380)

    rsi = px.line(recent, y="RSI_14", color_discrete_sequence=[GREEN])
    rsi.add_hline(y=70, line_dash="dot", line_color=DOWN_C)
    rsi.add_hline(y=30, line_dash="dot", line_color=UP_C)
    rsi.update_layout(yaxis_title="RSI (14)", xaxis_title=None, yaxis_range=[0, 100])
    chart(rsi, height=190, legend=False)

# --------------------------------------------------------------------------
# Track record — predicted vs actual, day by day
# --------------------------------------------------------------------------

with tab_record:
    if record is None:
        st.info(f"No prediction history for {model_label}. Retrain from the sidebar.")
    else:
        st.caption(
            f"Every call below came from the final-test {model_label}, trained only on data "
            "before 2024 and never refitted, so these are genuine out-of-sample predictions "
            "rather than hindsight. A call dated t is about the move from t to the next session."
        )

        span = st.radio("Span", [30, 60, 120, 250, len(record)], index=1, horizontal=True,
                        format_func=lambda n: "All" if n == len(record) else f"Last {n}",
                        label_visibility="collapsed")
        view = record.tail(span)
        stats = hit_rates(view)
        base_acc, base_name = baseline_accuracy(instrument)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{stats['accuracy']:.1%}",
                  f"{stats['accuracy'] - base_acc:+.1%} vs {base_name.lower()}",
                  help=f"Compared against always guessing "
                       f"{base_name.replace('Always ', '')}, which scores {base_acc:.1%}")
        m2.metric("Calls", f"{stats['n']}")
        m3.metric("When taking a side", f"{stats['directional_accuracy']:.1%}",
                  help="Accuracy on days the model predicted UP or DOWN, ignoring NEUTRAL")
        m4.metric("Sided calls", f"{stats['n_directional']}")

        st.subheader("Each call against the outcome")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=view.index, y=view["Close"], mode="lines", name="Close",
            line=dict(width=1.1, color=MUTED), hoverinfo="skip"))

        for correct, color, name in [(True, UP_C, "Correct"), (False, DOWN_C, "Wrong")]:
            sub = view[view["Correct"] == correct]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub["Close"], mode="markers", name=name,
                marker=dict(size=9, color=color, line=dict(width=0),
                            symbol=[MARKER_SYMBOLS[p] for p in sub["Predicted"]]),
                customdata=np.stack([sub["Predicted"], sub["Actual"],
                                     sub["Next_Return"] * 100, sub["Confidence"] * 100], axis=-1),
                hovertemplate=("<b>%{x|%d %b %Y}</b><br>Close Rs %{y:,.2f}<br>"
                               "Predicted %{customdata[0]} (%{customdata[3]:.0f}%)<br>"
                               "Actual %{customdata[1]} (%{customdata[2]:+.2f}%)<extra></extra>")))
        fig.update_layout(yaxis_title="Close")
        chart(fig, height=400)
        st.caption(
            "Marker shape is the predicted direction: up triangle, down triangle, circle for "
            "NEUTRAL. Green means the market agreed, red means it did not."
        )

        st.subheader("Rolling accuracy, all models")
        rolling = go.Figure()
        for key, label in MODEL_LABELS.items():
            other = track_record(ticker, key)
            if other is None:
                continue
            series = other["Correct"].rolling(42).mean().dropna()
            rolling.add_trace(go.Scatter(
                x=series.index, y=series.values, name=label,
                line=dict(width=1.6, color=MODEL_COLORS[label]),
                hovertemplate="%{x|%d %b %Y}<br>" + label + " %{y:.1%}<extra></extra>"))
        rolling.add_hline(y=base_acc, line_dash="dot", line_color=MUTED,
                          annotation_text=f"{base_name.lower()} {base_acc:.1%}",
                          annotation_font_size=11, annotation_font_color=MUTED)
        rolling.update_layout(yaxis_tickformat=".0%", yaxis_title="42-day accuracy")
        chart(rolling, height=300)
        st.caption(
            "Crossings above and below the baseline are what a model without a durable edge "
            "looks like. A real edge would hold above the line."
        )

        st.subheader("Recent calls")
        table = view.tail(15).iloc[::-1].copy()
        table.insert(0, "Date", table.index.strftime("%d %b %Y"))
        table["Result"] = np.where(table["Correct"], "Hit", "Miss")
        st.dataframe(
            table[["Date", "Close", "Predicted", "Confidence", "Actual", "Next_Return", "Result"]]
            .rename(columns={"Predicted": "Model said", "Actual": "Market did",
                             "Next_Return": "Actual move"})
            .style.format({"Close": "Rs {:,.2f}", "Confidence": "{:.0%}",
                           "Actual move": "{:+.2%}"})
            .map(lambda v: f"color: {UP_C}" if v == "Hit" else f"color: {DOWN_C}",
                 subset=["Result"]),
            width="stretch", hide_index=True,
        )

# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------

with tab_models:
    metrics_path = config.RESULTS_DIR / f"{instrument.slug}_metrics.csv"
    if not metrics_path.exists():
        st.info("No results on disk yet. Retrain from the sidebar.")
    else:
        metrics = read_csv(metrics_path, index_col=None, parse_dates=False)
        final = metrics[metrics["window"] == "Final Test"]
        is_baseline = final["model"].str.startswith("Always ")
        scored = final[(final["model"] != "Buy & Hold") & ~is_baseline]
        base_acc, base_name = baseline_accuracy(instrument)

        st.caption(
            "Held-out test set: trained on 2015-2023, scored once on 2024 onwards. "
            "The backtest charges 0.05% per position change."
        )

        rows = [(str(r["model"]), float(r["accuracy"])) for _, r in scored.iterrows()]
        rows.append((base_name, base_acc))
        retro.crt(
            "ACCURACY ON THE SEALED TEST SET\n\n"
            + retro.labelled_bars(rows, width=40)
            + f"\n\nbar scale is 0..100%. one standard error is about "
              f"{(base_acc * (1 - base_acc) / float(final['n_rows'].iloc[0])) ** 0.5:.1%}, "
              f"so gaps smaller than that are noise.",
        )
        st.caption(
            f"`{base_name}` is the one-line strategy of guessing "
            f"{base_name.replace('Always ', '')} every single day. That, not 33.3%, is the bar "
            f"a model has to clear."
        )

        st.dataframe(
            final[["model", "accuracy", "macro_f1", "sharpe", "cagr", "max_drawdown", "n_trades"]]
            .rename(columns={"model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro F1",
                             "sharpe": "Sharpe", "cagr": "CAGR", "max_drawdown": "Max drawdown",
                             "n_trades": "Trades"})
            .style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}", "Sharpe": "{:.2f}",
                           "CAGR": "{:.2%}", "Max drawdown": "{:.2%}"}, na_rep="—"),
            width="stretch", hide_index=True,
        )

        labeled = read_csv(instrument.labeled_path())

        st.subheader("Equity curve")
        equity = go.Figure()
        for key, label in MODEL_LABELS.items():
            pred_path = config.PREDICTIONS_DIR / f"{instrument.slug}_{key}_finaltest.csv"
            if not pred_path.exists():
                continue
            result = run_backtest(labeled, read_csv(pred_path)["Predicted_Target"], label=label)
            equity.add_trace(go.Scatter(
                x=result.equity_curve.index, y=result.equity_curve.values,
                name=label, line=dict(width=1.6, color=MODEL_COLORS[label])))

        test = labeled[labeled.index >= "2024-01-01"]
        bh = run_backtest(labeled, pd.Series(2, index=test.index), label="Buy & Hold")
        equity.add_trace(go.Scatter(x=bh.equity_curve.index, y=bh.equity_curve.values,
                                    name="Buy & Hold",
                                    line=dict(dash="dash", width=1.4, color=MUTED)))
        equity.add_hline(y=1.0, line_dash="dot", line_color=retro.GRID)
        equity.update_layout(yaxis_title="Portfolio value (start = 1.0)")
        chart(equity, height=380)

        cm_path = config.RESULTS_DIR / f"{instrument.slug}_{model_key}_confusion.csv"
        if cm_path.exists():
            st.subheader(f"Confusion matrix — {model_label}")
            cm = read_csv(cm_path, parse_dates=False)
            names = [LABEL_NAMES[i] for i in range(3)]
            header = ("actual \\ pred │" + "".join(n.rjust(9) for n in names)
                      + "row %".rjust(10))
            lines = [header, "─" * len(header)]
            for i, name in enumerate(names):
                counts = [int(cm.values[i][j]) for j in range(3)]
                total = sum(counts) or 1
                hit = counts[i] / total
                lines.append(f"{name.ljust(14)}│"
                             + "".join(str(c).rjust(9) for c in counts)
                             + f"{hit:10.1%}")
            retro.crt("\n".join(lines) + "\n\nrow % is how often that actual class was "
                                         "correctly identified.")

        with st.expander("per-fold walk-forward results"):
            st.dataframe(metrics[metrics["window"] != "Final Test"],
                         width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# How it works
# --------------------------------------------------------------------------

with tab_about:
    retro.crt(r"""
  RAW PRICES          FEATURES              LABEL
  ----------          --------              -----
  Yahoo Finance  -->  26-28 scale-free  --> UP / NEUTRAL / DOWN
  OHLCV daily         indicators            based on tomorrow's
  2015 -> today       + VIX, USDINR         move vs a threshold
                      + NIFTY context
                              |
                              v
            +----------------------------------+
            |  WALK-FORWARD TRAINING           |
            |  fold 1  train 15-19  test 2020  |
            |  fold 2  train 15-20  test 2021  |
            |  fold 3  train 15-21  test 2022  |
            |  fold 4  train 15-22  test 2023  |
            |  FINAL   train 15-23  test 24->  |
            +----------------------------------+
                              |
              +---------------+---------------+
              v                               v
       FINAL-TEST MODEL                  LIVE MODEL
       scored once, never                refit on all history,
       refitted -> the honest            never scored -> the
       numbers in MODELS                 call in TODAY
""", bare=True)

    st.markdown(
        """
**What the app answers.** One question only: will this instrument close higher,
lower, or roughly flat tomorrow? Three classes, not a price target.

**Where NEUTRAL comes from.** Tomorrow's move is compared to a threshold —
±0.25% for the index, ±0.50% for a single stock. Anything inside that band is
NEUTRAL. Without it a stock would almost never be flat and the model would be
forced into a coin-flip between UP and DOWN.

**Why two models per algorithm.** The *final-test* model is trained on
2015-2023 and scored exactly once on 2024 onwards. That is the only number
worth quoting. The *live* model is refit on every day of history including the
test window, so it can call tomorrow — and because it has seen all the data, it
can never be scored honestly. Mixing the two up is the most common way to
accidentally claim an edge that is not there.

**Why the confidence gate.** A three-class model over noisy daily returns
routinely puts its top class at 36% — technically a winner, far too weak to act
on. Anything below 45% is downgraded to HOLD, while the predicted direction is
still shown so you can see what the model actually thought.

**The honest finding.** Across both the index and individual stocks, next-day
direction from price history alone sits within about one standard error of
simply guessing the most common class every day. More complexity did not help:
the LSTM is the worst of the three. Treat every call here as a demonstration of
method, not a signal.
        """
    )
    st.caption("Full write-ups: `DATA_FLOW.md` and `APPLICATION_GUIDE.md` in the repo root.")

retro.footer(DISCLAIMER)

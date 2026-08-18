"""
backtest.py — DishaNifty Step 10-11: Backtesting engine

Consumes model predictions (0=DOWN, 1=NEUTRAL, 2=UP, matching Target encoding)
and computes strategy performance: Sharpe, CAGR, Max Drawdown, equity curve.

Position mapping (per roadmap spec):
    UP (2)      -> long  (+1)
    DOWN (0)    -> short (-1)
    NEUTRAL (1) -> cash  (0)

Transaction cost: 0.05% (0.0005) charged only when position changes.

Usage:
    from src.validation import get_walk_forward_folds, get_test_split
    from src.backtest import run_backtest, run_walk_forward_backtest

    df = pd.read_csv("data/processed/nifty50_labeled.csv", index_col=0, parse_dates=True)
    preds = load_predictions("data/predictions/rf_fold1.csv")  # or pass a DataFrame directly
    result = run_backtest(df, preds)
    result.summary()
    result.plot_equity_curve()

Prediction file contract (until Sujal/Khushi finalize their own output):
    CSV indexed by Date (or with a 'Date' column), one column: 'Predicted_Target'
    values in {0, 1, 2}, same encoding as the real Target column.
    If a model instead outputs class probabilities, argmax them into this
    format before calling run_backtest — see predictions_from_probabilities().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

TRANSACTION_COST = 0.0005  # 0.05% per position change, per roadmap spec
TRADING_DAYS_PER_YEAR = 252

POSITION_MAP = {0: -1, 1: 0, 2: 1}  # DOWN, NEUTRAL, UP


# --------------------------------------------------------------------------
# Loading / preparing predictions
# --------------------------------------------------------------------------

def load_predictions(path: str, date_col: str = "Date") -> pd.Series:
    """
    Load a predictions CSV into a Series indexed by date.
    Expects a 'Predicted_Target' column with values in {0, 1, 2}.
    """
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index)

    if "Predicted_Target" not in df.columns:
        raise ValueError(
            f"Expected a 'Predicted_Target' column in {path}, "
            f"found columns: {list(df.columns)}"
        )
    return df["Predicted_Target"].astype(int)


def predictions_from_probabilities(prob_df: pd.DataFrame,
                                    class_cols=("prob_down", "prob_neutral", "prob_up")) -> pd.Series:
    """
    Convert a DataFrame of class probabilities into Predicted_Target (0/1/2)
    via argmax. class_cols order must match (DOWN, NEUTRAL, UP) = (0, 1, 2).
    """
    arr = prob_df[list(class_cols)].to_numpy()
    return pd.Series(np.argmax(arr, axis=1), index=prob_df.index, name="Predicted_Target")


# --------------------------------------------------------------------------
# Core backtest
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    positions: pd.Series
    sharpe: float
    cagr: float
    max_drawdown: float
    n_trades: int
    label: str = ""
    fold_results: list = field(default_factory=list)  # populated for walk-forward runs

    def summary(self):
        print(f"--- Backtest summary: {self.label or 'run'} ---")
        print(f"Sharpe (annualized): {self.sharpe:.3f}")
        print(f"CAGR:                {self.cagr:.2%}")
        print(f"Max Drawdown:        {self.max_drawdown:.2%}")
        print(f"Number of trades:    {self.n_trades}")
        print(f"Final equity (from 1.0): {self.equity_curve.iloc[-1]:.4f}")

    def plot_equity_curve(self, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.equity_curve.index, self.equity_curve.values, label=self.label or "Strategy")
        ax.set_title(f"Equity Curve — {self.label}" if self.label else "Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (start = 1.0)")
        ax.legend()
        ax.grid(alpha=0.3)
        if show:
            plt.tight_layout()
            plt.show()
        return ax


def _compute_metrics(strategy_returns: pd.Series) -> tuple[float, float, float]:
    """Returns (sharpe, cagr, max_drawdown) from a daily strategy return series."""
    if strategy_returns.std() == 0 or strategy_returns.empty:
        sharpe = 0.0
    else:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    equity = (1 + strategy_returns).cumprod()
    n_days = len(strategy_returns)
    if n_days == 0 or equity.iloc[-1] <= 0:
        cagr = 0.0
    else:
        years = n_days / TRADING_DAYS_PER_YEAR
        cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    return sharpe, cagr, max_drawdown


def run_backtest(df: pd.DataFrame,
                  predictions: pd.Series,
                  transaction_cost: float = TRANSACTION_COST,
                  return_col: str = "Return",
                  label: str = "") -> BacktestResult:
    """
    Run a single backtest over the dates present in `predictions`.

    df: the labeled dataframe (needs `return_col`, e.g. actual next-day 'Return')
    predictions: Series indexed by date, values in {0,1,2} (DOWN/NEUTRAL/UP)
    """
    aligned = df.loc[predictions.index].copy()
    aligned["Predicted_Target"] = predictions

    # Signal is decided using info available at t, applied to the return realized
    # from t to t+1 -> shift position forward one day to avoid lookahead.
    aligned["Position"] = aligned["Predicted_Target"].map(POSITION_MAP)
    applied_position = aligned["Position"].shift(1).fillna(0)

    gross_returns = applied_position * aligned[return_col]

    position_changes = applied_position.diff().abs().fillna(applied_position.abs())
    costs = position_changes * transaction_cost
    n_trades = int((position_changes > 0).sum())

    strategy_returns = gross_returns - costs
    equity_curve = (1 + strategy_returns).cumprod()
    equity_curve.iloc[0] = 1.0 if len(equity_curve) else equity_curve

    sharpe, cagr, max_dd = _compute_metrics(strategy_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        positions=applied_position,
        sharpe=sharpe,
        cagr=cagr,
        max_drawdown=max_dd,
        n_trades=n_trades,
        label=label,
    )


# --------------------------------------------------------------------------
# Walk-forward runner (uses real src/validation.py folds — do not redefine)
# --------------------------------------------------------------------------

def run_walk_forward_backtest(df: pd.DataFrame,
                               predictions_by_fold: dict[str, pd.Series],
                               transaction_cost: float = TRANSACTION_COST,
                               return_col: str = "Return") -> BacktestResult:
    """
    Run backtests across each walk-forward fold and stitch the equity curves
    into one continuous combined curve (each fold compounds on the last).

    predictions_by_fold: dict mapping fold label -> predictions Series for that
    fold's validation window, e.g.:
        {
            "Fold 1": preds_fold1,
            "Fold 2": preds_fold2,
            "Fold 3": preds_fold3,
            "Fold 4": preds_fold4,
            "Final Test": preds_test,
        }
    Get these labels/windows from src.validation.get_walk_forward_folds(df) /
    get_test_split(df) — this function doesn't recompute fold boundaries itself,
    it just expects predictions already sliced to each fold's val/test window.
    """
    fold_results = []
    combined_returns = []

    for fold_label, preds in predictions_by_fold.items():
        result = run_backtest(df, preds, transaction_cost=transaction_cost,
                               return_col=return_col, label=fold_label)
        fold_results.append(result)
        combined_returns.append(result.strategy_returns)

    all_returns = pd.concat(combined_returns).sort_index()
    combined_equity = (1 + all_returns).cumprod()

    sharpe, cagr, max_dd = _compute_metrics(all_returns)
    total_trades = sum(r.n_trades for r in fold_results)

    combined = BacktestResult(
        equity_curve=combined_equity,
        strategy_returns=all_returns,
        positions=pd.concat([r.positions for r in fold_results]).sort_index(),
        sharpe=sharpe,
        cagr=cagr,
        max_drawdown=max_dd,
        n_trades=total_trades,
        label="Combined (all folds)",
        fold_results=fold_results,
    )
    return combined


def plot_all_folds(walk_forward_result: BacktestResult):
    """Plot each fold's equity curve on its own subplot, plus the combined curve."""
    n = len(walk_forward_result.fold_results) + 1
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, fold_result in zip(axes[:-1], walk_forward_result.fold_results):
        fold_result.plot_equity_curve(ax=ax, show=False)

    walk_forward_result.plot_equity_curve(ax=axes[-1], show=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Smoke test with synthetic data so this file can be sanity-checked before
    # real model predictions exist. Not part of the actual pipeline.
    rng = pd.date_range("2024-01-01", periods=100, freq="B")
    fake_returns = pd.Series(np.random.normal(0, 0.01, size=100), index=rng, name="Return")
    fake_df = pd.DataFrame({"Return": fake_returns})
    fake_preds = pd.Series(np.random.choice([0, 1, 2], size=100), index=rng, name="Predicted_Target")

    res = run_backtest(fake_df, fake_preds, label="Synthetic smoke test")
    res.summary()
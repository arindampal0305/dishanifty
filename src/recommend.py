"""
Turn a model's UP/DOWN/NEUTRAL probabilities into a BUY/SELL/HOLD call.

The mapping is deliberately not just argmax. A three-class softmax over noisy
daily returns routinely puts its top class at 0.36 — technically a winner, but
far too weak to act on. Anything below MIN_CONFIDENCE is downgraded to HOLD,
while the predicted direction is still reported so the user can see what the
model actually thought.

This module produces an educational signal for a student project. It is not
investment advice and nothing here should be traded with real money.
"""

from __future__ import annotations

import pandas as pd

from src.config import ACTION_NAMES, LABEL_NAMES

# A top class below this is treated as "the model isn't sure" and sits out.
MIN_CONFIDENCE = 0.45

DISCLAIMER = (
    "Educational project output, not investment advice. "
    "Predictions are statistical estimates and are frequently wrong."
)


def _indicator_notes(row: pd.Series) -> list[str]:
    """Plain-language context from the raw indicators, for the dashboard."""
    notes = []

    rsi = row.get("RSI_14")
    if rsi is not None and pd.notna(rsi):
        if rsi > 70:
            notes.append(f"RSI {rsi:.0f} — overbought")
        elif rsi < 30:
            notes.append(f"RSI {rsi:.0f} — oversold")
        else:
            notes.append(f"RSI {rsi:.0f} — neutral range")

    macd, signal = row.get("MACD"), row.get("MACD_Signal")
    if pd.notna(macd) and pd.notna(signal):
        notes.append("MACD bullish" if macd > signal else "MACD bearish")

    close, ema50 = row.get("Close"), row.get("EMA_50")
    if pd.notna(close) and pd.notna(ema50):
        notes.append("Above 50-day EMA (uptrend)" if close > ema50
                     else "Below 50-day EMA (downtrend)")

    vol = row.get("Volatility_10")
    if pd.notna(vol):
        annual = vol * (252 ** 0.5)
        level = "high" if annual > 0.30 else "low" if annual < 0.15 else "moderate"
        notes.append(f"10-day volatility {annual:.0%} annualised — {level}")

    return notes


def recommend(probabilities, row: pd.Series | None = None,
              min_confidence: float = MIN_CONFIDENCE) -> dict:
    """
    probabilities: length-3 sequence ordered (DOWN, NEUTRAL, UP).
    row: optional raw-indicator row used to attach human-readable context.
    """
    probs = list(map(float, probabilities))
    predicted = int(max(range(3), key=lambda i: probs[i]))
    confidence = probs[predicted]

    acted = predicted if confidence >= min_confidence else 1  # 1 = HOLD
    gated = acted != predicted

    return {
        "direction": LABEL_NAMES[predicted],
        "action": ACTION_NAMES[acted],
        "confidence": confidence,
        "prob_down": probs[0],
        "prob_neutral": probs[1],
        "prob_up": probs[2],
        "low_confidence": gated,
        "reason": (
            f"Top class {LABEL_NAMES[predicted]} at {confidence:.0%} is below the "
            f"{min_confidence:.0%} threshold, so the call is downgraded to HOLD."
            if gated else
            f"Model predicts {LABEL_NAMES[predicted]} with {confidence:.0%} confidence."
        ),
        "notes": _indicator_notes(row) if row is not None else [],
    }


def recommend_series(proba_matrix, index, min_confidence: float = MIN_CONFIDENCE) -> pd.DataFrame:
    """Vectorised version over a full prediction history."""
    rows = [recommend(p, min_confidence=min_confidence) for p in proba_matrix]
    df = pd.DataFrame(rows, index=index).drop(columns=["notes", "reason"])
    df.index.name = "Date"
    return df


def format_recommendation(rec: dict, ticker: str, as_of, price: float | None = None) -> str:
    lines = [
        f"{ticker} — as of {as_of}",
        f"  Predicted direction : {rec['direction']}  ({rec['confidence']:.1%} confidence)",
        f"  Recommended action  : {rec['action']}",
    ]
    if price is not None:
        lines.insert(1, f"  Last close          : Rs {price:,.2f}")
    lines.append(
        f"  Probabilities       : DOWN {rec['prob_down']:.1%} | "
        f"NEUTRAL {rec['prob_neutral']:.1%} | UP {rec['prob_up']:.1%}"
    )
    lines.append(f"  {rec['reason']}")
    for note in rec["notes"]:
        lines.append(f"    - {note}")
    lines.append(f"  {DISCLAIMER}")
    return "\n".join(lines)

"""
DishaNifty - Random Forest Model (Task 8)
Owner: Sujal

Trains and tunes a Random Forest classifier on each walk-forward fold
defined in src/validation.py, and evaluates it on that fold's
validation period. Does NOT touch the 2024-2026 test set - that is
Phase 5 / task 15, done later on the full 2015-2023 data.
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# so "from validation import ..." works no matter where this script is run from
sys.path.append(str(Path(__file__).resolve().parents[1]))
from validation import get_walk_forward_folds

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Return', 'Volatility_10',
    'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_10', 'EMA_20', 'EMA_50',
    'RSI_14', 'MACD', 'MACD_Signal',
    'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width',
    'ATR_14', 'Momentum_10',
]
TARGET_COL = 'Target'

# Small-ish grid on purpose - GridSearchCV x TimeSeriesSplit already
# multiplies fits fast. Widen later if you have time/compute to spare.
PARAM_GRID = {
    'n_estimators': [200, 400],
    'max_depth': [5, 8, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample'],
}


def tune_and_evaluate_fold(train_df, val_df, fold_num):
    """Tune RF on train_df (using time-respecting inner CV), then
    evaluate the best model once on val_df."""
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]

    # inner CV must also respect time order - never use plain KFold here
    inner_cv = TimeSeriesSplit(n_splits=3)

    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=PARAM_GRID,
        scoring='f1_macro',
        cv=inner_cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    report = classification_report(
        y_val, y_pred, labels=[0, 1, 2], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

    print(f"\n=== Random Forest | Fold {fold_num} ===")
    print(f"Best params: {search.best_params_}")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted, order 0=DOWN,1=NEUTRAL,2=UP):")
    print(cm)

    predictions_df = pd.DataFrame({
        'Date': val_df.index,
        'Actual': y_val.values,
        'Predicted': y_pred,
        'Prob_Down': y_proba[:, 0],
        'Prob_Neutral': y_proba[:, 1],
        'Prob_Up': y_proba[:, 2],
    })

    result_row = {
        'fold': fold_num,
        'best_params': search.best_params_,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'precision_down': report['0']['precision'],
        'recall_down': report['0']['recall'],
        'precision_neutral': report['1']['precision'],
        'recall_neutral': report['1']['recall'],
        'precision_up': report['2']['precision'],
        'recall_up': report['2']['recall'],
    }

    return best_model, result_row, predictions_df


def run_random_forest(data_path="data/processed/nifty50_labeled.csv"):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    Path("results").mkdir(exist_ok=True)
    Path("saved_models").mkdir(exist_ok=True)

    folds = get_walk_forward_folds(df)

    all_results, all_predictions = [], []

    for f in folds:
        model, result_row, preds = tune_and_evaluate_fold(
            f['train'], f['val'], f['fold']
        )
        all_results.append(result_row)
        preds['fold'] = f['fold']
        all_predictions.append(preds)

        with open(f"saved_models/rf_fold{f['fold']}.pkl", "wb") as fh:
            pickle.dump(model, fh)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/rf_fold_results.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv("results/rf_predictions.csv", index=False)

    print("\n=== Random Forest: Summary across folds ===")
    print(results_df[['fold', 'accuracy', 'macro_f1']].to_string(index=False))
    print(f"\nMean Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Mean Macro F1:  {results_df['macro_f1'].mean():.4f}")
    print("\nSaved:")
    print("  results/rf_fold_results.csv   (metrics + best params per fold)")
    print("  results/rf_predictions.csv    (per-day predictions, for Sneha's backtest)")
    print("  saved_models/rf_fold*.pkl     (trained model per fold)")

    return results_df, predictions_df


if __name__ == "__main__":
    run_random_forest()

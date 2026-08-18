"""
DishaNifty - XGBoost Model (Task 9)
Owner: Sujal

Same walk-forward structure and evaluation as rf_model.py, so the
two models are directly comparable fold-by-fold. Only the model class,
its hyperparameter grid, and class-imbalance handling differ.
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

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

# XGBoost has no class_weight param for multiclass, so imbalance is
# handled via per-sample weights (compute_sample_weight) instead.
# This is now a sampling distribution for RandomizedSearchCV, not an
# exhaustive grid - it can stay wide without blowing up runtime.
PARAM_GRID = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}
N_ITER = 15  # how many random combos to try per fold - raise if you have time to spare


def tune_and_evaluate_fold(train_df, val_df, fold_num):
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]

    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    inner_cv = TimeSeriesSplit(n_splits=3)

    base_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        tree_method='hist',  # much faster than the 'exact' default, same accuracy
        random_state=42,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        scoring='f1_macro',
        cv=inner_cv,
        random_state=42,
        n_jobs=-1,
    )
    # sklearn slices sample_weight to match each inner CV split automatically
    search.fit(X_train, y_train, sample_weight=sample_weight)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    report = classification_report(
        y_val, y_pred, labels=[0, 1, 2], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

    print(f"\n=== XGBoost | Fold {fold_num} ===")
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


def run_xgboost(data_path="data/processed/nifty50_labeled.csv"):
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

        with open(f"saved_models/xgb_fold{f['fold']}.pkl", "wb") as fh:
            pickle.dump(model, fh)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/xgb_fold_results.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv("results/xgb_predictions.csv", index=False)

    print("\n=== XGBoost: Summary across folds ===")
    print(results_df[['fold', 'accuracy', 'macro_f1']].to_string(index=False))
    print(f"\nMean Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Mean Macro F1:  {results_df['macro_f1'].mean():.4f}")
    print("\nSaved:")
    print("  results/xgb_fold_results.csv  (metrics + best params per fold)")
    print("  results/xgb_predictions.csv   (per-day predictions, for Sneha's backtest)")
    print("  saved_models/xgb_fold*.pkl    (trained model per fold)")

    return results_df, predictions_df


if __name__ == "__main__":
    run_xgboost()
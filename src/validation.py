import pandas as pd

def get_walk_forward_folds(df):
    folds = [
        {
            "fold": 1,
            "train_end": "2019-12-31",
            "val_start": "2020-01-01",
            "val_end": "2020-12-31"
        },
        {
            "fold": 2,
            "train_end": "2020-12-31",
            "val_start": "2021-01-01",
            "val_end": "2021-12-31"
        },
        {
            "fold": 3,
            "train_end": "2021-12-31",
            "val_start": "2022-01-01",
            "val_end": "2022-12-31"
        },
        {
            "fold": 4,
            "train_end": "2022-12-31",
            "val_start": "2023-01-01",
            "val_end": "2023-12-31"
        }
    ]

    result = []
    for f in folds:
        train = df[df.index <= f["train_end"]]
        val = df[(df.index >= f["val_start"]) & (df.index <= f["val_end"])]
        result.append({
            "fold": f["fold"],
            "train": train,
            "val": val
        })
        print(f"Fold {f['fold']} | Train: 2015-01-01 to {f['train_end']} ({len(train)} rows) | Val: {f['val_start']} to {f['val_end']} ({len(val)} rows)")

    return result

def get_test_split(df):
    train = df[df.index <= "2023-12-31"]
    test = df[df.index >= "2024-01-01"]
    print(f"\nFinal Test | Train: 2015-01-01 to 2023-12-31 ({len(train)} rows) | Test: 2024-01-01 to present ({len(test)} rows)")
    return train, test

if __name__ == "__main__":
    df = pd.read_csv("data/processed/nifty50_labeled.csv", index_col=0, parse_dates=True)
    
    print("=== Walk-Forward Folds ===")
    folds = get_walk_forward_folds(df)
    
    print("\n=== Final Test Split ===")
    train, test = get_test_split(df)
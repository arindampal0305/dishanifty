import pandas as pd

def clean_nifty(path="data/raw/nifty50_raw.csv"):
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [col[0] for col in df.columns]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.ffill()
    df = df.dropna()
    print(f"Clean NIFTY shape: {df.shape}")
    print(df.head())
    return df

if __name__ == "__main__":
    df = clean_nifty()
    df.to_csv("data/processed/nifty50_clean.csv")
    print("Saved to data/processed/nifty50_clean.csv")
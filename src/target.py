import pandas as pd

def generate_target(df, threshold=0.0025):
    df['Next_Return'] = df['Close'].pct_change().shift(-1)

    def label(r):
        if r > threshold:
            return 2   
        elif r < -threshold:
            return 0   
        else:
            return 1  

    df['Target'] = df['Next_Return'].apply(label)


    df = df.dropna(subset=['Next_Return'])

    print(f"\nThreshold: {threshold}")
    print(df['Target'].value_counts())
    print(df['Target'].value_counts(normalize=True).round(3))
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/nifty50_features.csv", index_col=0, parse_dates=True)
    
    df = generate_target(df, threshold=0.0025)
    df.to_csv("data/processed/nifty50_labeled.csv")
    print("\nSaved to data/processed/nifty50_labeled.csv")
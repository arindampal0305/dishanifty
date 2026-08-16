import pandas as pd
import pandas_ta as ta

def add_indicators(df):

    df['Return'] = df['Close'].pct_change()


    df['Volatility_10'] = df['Return'].rolling(10).std()


    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)


    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)


    df['RSI_14'] = ta.rsi(df['Close'], length=14)


    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']


    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0_2.0']
    df['BB_Middle'] = bb['BBM_20_2.0_2.0']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)


    df['Momentum_10'] = ta.mom(df['Close'], length=10)


    df = df.dropna()

    print(f"Feature-engineered shape: {df.shape}")
    print(df.head())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/nifty50_clean.csv", index_col=0, parse_dates=True)
    df = add_indicators(df)
    df.to_csv("data/processed/nifty50_features.csv")
    print("Saved to data/processed/nifty50_features.csv")
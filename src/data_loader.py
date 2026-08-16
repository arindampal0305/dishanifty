import yfinance as yf
import pandas as pd

def download_data():
    print("Downloading NIFTY 50...")
    nifty = yf.download("^NSEI", start="2015-01-01", end="2026-01-01")
    nifty.to_csv("data/raw/nifty50_raw.csv")

    print("Downloading India VIX...")
    vix = yf.download("^INDIAVIX", start="2015-01-01", end="2026-01-01")
    vix.to_csv("data/raw/india_vix_raw.csv")

    print("Downloading USD/INR...")
    usdinr = yf.download("USDINR=X", start="2015-01-01", end="2026-01-01")
    usdinr.to_csv("data/raw/usdinr_raw.csv")

    print("Done. Files saved to data/raw/")

if __name__ == "__main__":
    download_data()
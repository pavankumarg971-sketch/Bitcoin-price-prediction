import pandas as pd

def create_features(df):
    # Convert numeric columns safely
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Feature Engineering
    df["HL_PCT"] = (df["High"] - df["Low"]) / df["Low"]
    df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"]

    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["STD21"] = df["Close"].rolling(21).std()

    # Target variable: next day's return
    df["Return"] = df["Close"].pct_change()
    df["Target"] = df["Return"].shift(-1)

    df = df.dropna()
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/btc_data.csv", index_col=0)
    df = create_features(df)
    df.to_csv("data/btc_features.csv")
    print("Feature dataset saved to data/btc_features.csv")

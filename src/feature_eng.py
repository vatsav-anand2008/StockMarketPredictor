import yfinance as yf
import pandas as pd
from numpy import mean

def add_features(df):
    def classify_return(x):
        if x < -5:
            return 0
        elif x < 0:
            return 1
        elif x < 5:
            return 2
        elif x < 10:
            return 3
        else:
            return 4
    
    #df=df.reset_index()

    spy=yf.download('SPY',period='10y',interval='1d')

    spy.columns = spy.columns.get_level_values(0)
    spy=spy.reset_index()
    #spy["Close"] = pd.to_numeric(spy["Close"], errors="coerce")

    #print(spy.columns)
    #print(df.head())

    df["Date"] = pd.to_datetime(df["Date"])
    spy["Date"] = pd.to_datetime(spy["Date"])

    spy = spy[["Date", "Close"]].rename(
        columns={"Close":"SPY_Close"}
    )

    df = df.merge(
        spy,
        on="Date",
        how="left"
    )

    df["SPY_5d_return"] = (
        df["SPY_Close"].shift(-5) - df["SPY_Close"]
    ) / df["SPY_Close"] * 100

    df["SPY_20d_return"] = (
        df["SPY_Close"].shift(-20) - df["SPY_Close"]
    ) / df["SPY_Close"] * 100

    df["SPY_daily_return"] = (
        df["SPY_Close"] - df["SPY_Close"].shift(1)
    ) / df["SPY_Close"].shift(1) * 100

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df['future_5d_return']=(
        df['Close'].shift(-5)-df['Close']
    )/df['Close']*100

    df['future_20d_return']=(
        df['Close'].shift(-20)-df['Close']
    )/df['Close']*100

    df['daily_return']=(
        df['Close']-df['Close'].shift(1)
    )/df['Close'].shift(1)*100

    df['rel_5d_return']=df['future_5d_return']-df['SPY_5d_return']
    df['rel_20d_return']=df['future_20d_return']-df['SPY_20d_return']
    df['rel_daily_return']=df['daily_return']-df['SPY_daily_return']

    df['5d_MA']=(
        df['Close'].rolling(window=5).mean()
    )

    df['20d_MA']=(
        df['Close'].rolling(window=20).mean()
    )

    df['MA_ratio']=df['5d_MA']/df['20d_MA']

    df["change"] = df["Close"].diff()

    df["gain"] = df["change"].clip(lower=0)
    df["loss"] = -df["change"].clip(upper=0)

    avg_gain=df['gain'].rolling(14).mean()
    avg_loss=df['loss'].rolling(14).mean()
    rs=avg_gain/avg_loss
    df['RSI']=100-(100/(1+rs))

    df['20d_volatility']=df['daily_return'].rolling(20).std()
    df['10d_volatility']=df['daily_return'].rolling(10).std()

    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)

    df["volume_ratio"] = (
        df["Volume"] /
        df["Volume"].rolling(20).mean()
    )

    df["close_MA20_ratio"] = (
        df["Close"] / df["20d_MA"]
    )

    df["SPY_volatility_20d"] = (
        df["SPY_daily_return"].rolling(20).std()
    )

    bins = [-float("inf"), -10, -5, 0, 5, 10, float("inf")]

    labels = [
        "Huge Loss",
        "Loss",
        "Small Loss",
        "Small Gain",
        "Gain",
        "Huge Gain"
    ]

    df["return_class_5d"] = pd.qcut(
        df["future_5d_return"],
        q=6,
        labels=False
    )

    df["return_class_20d"] = pd.qcut(
            df["future_20d_return"],
            q=6,
            labels=False
        )

    df["rel_class_5d"] = pd.qcut(
        df["rel_5d_return"],
        q=6,
        labels=False
    )
    
    df["rel_class_20d"] = pd.qcut(
        df["rel_20d_return"],
        q=6,
        labels=False
    )

    df["SPY_20d_MA"] = (
        df["SPY_Close"].rolling(20).mean()
    )

    df["SPY_MA_ratio"] = (
        df["SPY_Close"] / df["SPY_20d_MA"]
    )

    df["stock_5d_return"] = (
        df["Close"] / df["Close"].shift(5) - 1
    ) * 100

    df["SPY_5d_return_past"] = (
        df["SPY_Close"] / df["SPY_Close"].shift(5) - 1
    ) * 100

    df["SPY_20d_return_past"] = (
            df["SPY_Close"] / df["SPY_Close"].shift(20) - 1
        ) * 100

    df["rel_5d_performance"] = (
        (df["Close"] / df["Close"].shift(5))
        - (df["SPY_Close"] / df["SPY_Close"].shift(5))
    ) * 100

    df["rel_20d_performance"] = (
        (df["Close"] / df["Close"].shift(20))
        - (df["SPY_Close"] / df["SPY_Close"].shift(20))
    ) * 100

    df['outperform_20d']=(df['rel_20d_return']>0).astype(int)
    df['outperform_20d_past']=(df['rel_20d_performance']>0).astype(int)
    df['outperform_5d']=(df['rel_5d_return']>0).astype(int)
    df['outperform_5d_past']=(df['rel_5d_performance']>0).astype(int)

    df=df.dropna()

    return df
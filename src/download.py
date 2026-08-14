import yfinance as yf
import pandas as pd

TICKERS=['AAPL','MSFT','NVDA','AMZN','GOOG','META','SPY','MU','AMD','INTC','META','AVGO','TSLA','ORCL','CRM','QCOM','CSCO','ADBE','NFLX','COST','WMT','DIS','HD','MCD','JPM','BAC','GS','MS','V','MA','LLY','JNJ','UNH','PFE','CVX','XOM','PCG','RBLX','SNOW','UBER','SHOP','COIN','ABNB','QQQ','DIA','IWM']

for stock in TICKERS:
    df=yf.download(stock,period='10y',interval='1d')
    df.columns = df.columns.get_level_values(0)
    df.to_csv(f'../data/raw/{stock}.csv') 
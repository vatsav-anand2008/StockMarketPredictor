import yfinance as yf
import matplotlib.pyplot as plt

df = yf.download(
    "AAPL",
    period="10y",
    interval="1d"
)

print(df.head())

plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"])
plt.title("Apple Closing Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,brier_score_loss
#from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path
import numpy as np

PROCESSED_FOLDER = Path("../data/processed")

dfs=[]

for file in PROCESSED_FOLDER.glob("*.csv"):
    df = pd.read_csv(file)
    df["Ticker"] = file.stem
    if file.stem != "SPY":
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data=all_data.dropna()
all_data["Date"] = pd.to_datetime(all_data["Date"])

#removed SPY_Close
X=['Close',
    'High',
    'Low',
    'Open',
    'Volume',
    #'SPY_Close',
    'SPY_daily_return',
    'daily_return',
    'rel_daily_return',
    '5d_MA',
    '20d_MA',
    'MA_ratio',
    'RSI',
    '20d_volatility',
    '10d_volatility',
    'return_10d',
    'return_20d',
    'volume_ratio',
    'close_MA20_ratio',
    'SPY_volatility_20d',
    'SPY_20d_MA',
    'SPY_MA_ratio',
    'stock_5d_return',
    'SPY_5d_return_past',
    'SPY_20d_return_past',
    'rel_5d_performance',
    'rel_20d_performance',
    'outperform_20d_past',
    'outperform_5d_past']
y='outperform_5d'

train = all_data[all_data["Date"] < "2024-01-01"]
test = all_data[all_data["Date"] >= "2024-01-01"]

X_train,X_test,y_train,y_test=(
    train[X],
    test[X],
    train[y],
    test[y]
)

model=RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)

predictions=model.predict(X_test)
probabilities=model.predict_proba(X_test)
outperform_probability = probabilities[:, 1]

print("Accuracy:", accuracy_score(y_test, predictions))

print(
    classification_report(
        y_test,
        predictions
    )
)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

brier=brier_score_loss(
    y_test,
    probabilities
)
print('Brier Score:',brier)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from xgboost import XGBRegressor
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
    'SPY_volatility_20d']
y='rel_20d_return'

train = all_data[all_data["Date"] < "2024-01-01"]
test = all_data[all_data["Date"] >= "2024-01-01"]

X_train,X_test,y_train,y_test=(
    train[X],
    test[X],
    train[y],
    test[y]
)

baseline_prediction = np.full_like(
    y_test,
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_r2 = r2_score(
    y_test,
    baseline_prediction
)

print("Baseline R²:", baseline_r2)
print("Baseline MAE:", baseline_mae)
print()

'''print(all_data.columns.tolist())
print(X.dtypes)
print(X.head())'''

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    'XGBoost':XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(name)
    print("R²:", r2)
    print("MAE:", mae)
    print()
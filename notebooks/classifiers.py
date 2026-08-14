from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from xgboost import XGBClassifier
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

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    'XGBoost':XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)

    #predicts most likely classification
    predictions = model.predict(X_test)

    importance=pd.DataFrame({
        'Feature':X_train.columns,
        'Importance':model.feature_importances_
    })

    print(name)
    print(accuracy_score(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    print()
    print(importance.sort_values('Importance',ascending=False))
    print()
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True))
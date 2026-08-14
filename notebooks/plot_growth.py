import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#from src.feature_eng import add_features

RAW_FOLDER = Path("../data/raw")
PROCESSED_FOLDER = Path("../data/processed")

for csv_file in PROCESSED_FOLDER.glob("*.csv"):
    df=pd.read_csv(csv_file)

    fig,ax=plt.subplots(figsize=(10,7))

    ax.plot(df.index,df['future_5d_return'])

    ax.set_title(csv_file.stem)
    ax.set_ylabel('weekly growth %')

plt.show()
#plt.pause(1)
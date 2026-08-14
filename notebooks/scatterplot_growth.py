import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

#from src.feature_eng import add_features

RAW_FOLDER = Path("../data/raw")
PROCESSED_FOLDER = Path("../data/processed")

for csv_file in PROCESSED_FOLDER.glob("*.csv"):
    df=pd.read_csv(csv_file)

    fig,ax=plt.subplots(figsize=(10,7))

    ax.scatter(df.index,df['future_5d_return'])

    m,b=np.polyfit(df.index,df['future_5d_return'],1)

    ax.set_title(csv_file.stem)
    ax.set_ylabel('weekly growth %')

    plt.text(1,1,f'm={m:.3f}, b={b:.3f}',fontsize=13)

plt.show()
#plt.pause(1)
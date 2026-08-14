import matplotlib.pyplot as plt
import pandas as pd
import statistics
from pathlib import Path

PROCESSED_FOLDER = Path("../data/processed")

for csv_file in PROCESSED_FOLDER.glob("*.csv"):
    df=pd.read_csv(csv_file)

    fig,ax=plt.subplots(figsize=(10,7))

    ax.plot(df.index,df['5d_MA'],alpha=0.5)
    ax.plot(df.index,df['20d_MA'],alpha=0.5)

    ax.set_title(csv_file.stem)
    ax.set_ylabel('Moving Avg')
    #ax.legend()
    ax.text(1,1,f'5d: {statistics.mean(df['5d_MA'])}, 20d: {statistics.mean(df['20d_MA'])}')

plt.legend()
plt.show()
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.feature_eng import add_features

RAW_FOLDER = Path("../data/raw")
PROCESSED_FOLDER = Path("../data/processed")

# Create processed folder if it doesn't exist
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

for csv_file in RAW_FOLDER.glob("*.csv"):
    print(f"Processing {csv_file.name}...")

    df = pd.read_csv(csv_file)

    df = add_features(df)

    output_file = PROCESSED_FOLDER / csv_file.name
    df.to_csv(output_file, index=False)

print("Done!")
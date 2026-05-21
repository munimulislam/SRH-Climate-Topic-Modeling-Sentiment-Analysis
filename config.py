from pathlib import Path

SEED = 42

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DIR / "twitter_sentiment_data.csv"

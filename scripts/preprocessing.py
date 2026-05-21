import pandas as pd
import re
from config import RAW_DATA_FILE

URL_RE = re.compile(r"http\S+|www\.\S+|https\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
HTML_TAG_RE = re.compile(r"<.*?>")
EMAIL_RE = re.compile(r"\S+@\S+")
MULTISPACE = re.compile(r"\s+")
NON_PRINT = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")
REPEAT_CHR = re.compile(r"(.)\1{2,}")
PLACEHOLDER_RE = re.compile(r"<(url|email|user)>", re.IGNORECASE)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_FILE)


def senitize():
    pass


def main():
    load_dataset()
    senitize()


if __name__ == "__main__":
    main()

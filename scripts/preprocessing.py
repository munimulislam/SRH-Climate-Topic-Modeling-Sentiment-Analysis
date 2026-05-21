from pathlib import Path
import sys
import pandas as pd
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds root to path
from config import RAW_DATA_FILE, PROCESSED_DIR
import unicodedata

NEGATION_MAP = {
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
}


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_FILE)


def remove_duplicate(df: pd.DataFrame):
    df.drop_duplicates(subset="tweet", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)


def fix_encoding(text: str) -> str:
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        fixed = text

    fixed = unicodedata.normalize("NFKD", fixed)

    return fixed


def clean_text(text):
    """
    Base cleaning applied before task-specific processing.
    Safe for both BERT-based and LDA-based pipelines.
    """
    # 4a. Strip retweet prefix (RT @username:)
    text = re.sub(r"^RT\s+@\w+[:\s]+", "", text, flags=re.IGNORECASE)

    # 4b. Remove URLs (http, https, www, t.co)
    text = re.sub(r"http\S+|https\S+|www\.\S+", "", text)

    # 4c. Remove @mentions entirely
    text = re.sub(r"@\w+", "", text)

    # 4d. Remove # symbol but KEEP the hashtag word
    # #ClimateChange → ClimateChange (meaningful content)
    text = re.sub(r"#(\w+)", r"\1", text)

    # 4e. Remove HTML entities (&amp; &lt; &gt;)
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;|&#39;", " ", text)

    # 4f. Remove truncation artifacts (… and encoding remnants)
    text = re.sub(r"â€¦|Ã¢â‚¬Â¦|\.{3,}|…", "", text)

    # 4g. Remove leftover encoding garbage characters
    text = re.sub(r"[Ã¢â‚¬Â©®™°]+", "", text)

    # 4h. Collapse multiple spaces and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_short(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["tweet"].str.len() >= 10].reset_index(drop=True)


def expand_negations(text: str) -> str:
    for contraction, expansion in NEGATION_MAP.items():
        text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
    return text


def save_ds(df: pd.DataFrame):
    PROCESSED_DIR.mkdir(exist_ok=True)
    df.to_csv(PROCESSED_DIR / "data.csv", index=False)


def main():
    df = load_dataset()
    df["tweet"] = df["message"]
    df["tweet"] = (
        df["tweet"]
        .apply(fix_encoding)
        .apply(lambda txt: txt.encode("ascii", errors="ignore").decode("ascii"))
        .apply(clean_text)
        .apply(lambda text: text.lower())
        .apply(lambda txt: re.sub(r"\s+", " ", txt).strip())
        .apply(expand_negations)
    )

    remove_duplicate(df)
    df = remove_short(df)
    save_ds(df)


if __name__ == "__main__":
    main()

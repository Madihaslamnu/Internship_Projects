
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "Tweets.csv"
LABELS = ["negative", "neutral", "positive"]        
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
SEED = 42
TEST_SIZE = 0.2


def load_airline(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load + de-duplicate. Returns columns incl. text, airline_sentiment, label."""
    df = pd.read_csv(path)
    df = (
        df.sort_values("airline_sentiment_confidence", ascending=False)
        .drop_duplicates("text")
        .sort_index()
        .reset_index(drop=True)
    )
    df["label"] = df["airline_sentiment"].map(LABEL2ID)
    return df


def get_split(df: pd.DataFrame | None = None):
    """Deterministic stratified 80/20 split -> (train_df, test_df)."""
    df = load_airline() if df is None else df
    return train_test_split(
        df, test_size=TEST_SIZE, stratify=df["airline_sentiment"], random_state=SEED
    )
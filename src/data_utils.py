"""
data_utils.py
~~~~~~~~~~~~~
Handles all dataset loading, validation, cleaning, and EDA
computation for the Emolyzer application.
"""

import os
import pandas as pd
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_PATH = r"C:\Users\works\Desktop\emolyzer\merged_text.csv"

REQUIRED_COLUMNS = {"text", "label"}

EMOTION_MAP = {
    0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise", 6: "Neutral"
}

EMOTION_COLORS = {
    "Sadness":  "#5C85D6", 
    "Joy":      "#F5C542",
    "Love":     "#E75480", 
    "Anger":    "#E05252",
    "Fear":     "#9B59B6", 
    "Surprise": "#48C9B0",
    "Neutral":  "#A9A9A9"
}


# ─── Loading & Validation ─────────────────────────────────────────────────────

def load_and_validate(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the CSV dataset and validate its schema. Returns a clean DataFrame
    with an additional `emotion` column (string label).

    Raises:
        FileNotFoundError: if the CSV file is not found at `path`.
        ValueError: if required columns are missing, or label values are
                    outside the expected range [0, 5].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Ensure merged_text.csv is placed in the emolyzer project folder."
        )

    df = pd.read_csv(path)

    # ── Column presence check
    missing = REQUIRED_COLUMNS - set(df.columns.str.strip().str.lower())
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Your CSV must contain columns named 'text' and 'label'."
        )

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # ── Drop rows with nulls in critical columns
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    dropped = before - len(df)

    # ── Ensure correct types
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # ── Label range validation
    invalid_labels = df[~df["label"].isin(EMOTION_MAP.keys())]
    if not invalid_labels.empty:
        raise ValueError(
            f"Found {len(invalid_labels)} row(s) with labels outside [0–5]. "
            f"Valid labels are: {list(EMOTION_MAP.keys())}."
        )

    # ── Remove empty-text rows post-strip
    df = df[df["text"].str.len() > 0]

    # ── Map numeric labels to human-readable emotion strings
    df["emotion"] = df["label"].map(EMOTION_MAP)

    metadata = {
        "total_rows": len(df),
        "rows_dropped": dropped,
        "num_classes": df["label"].nunique(),
    }

    return df, metadata


# ─── EDA Helpers ─────────────────────────────────────────────────────────────

def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with emotion counts and percentage distribution.
    """
    counts = df["emotion"].value_counts().reset_index()
    counts.columns = ["Emotion", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)
    counts["Color"] = counts["Emotion"].map(EMOTION_COLORS)
    # Maintain consistent ordering by label value
    order = list(EMOTION_MAP.values())
    counts["_order"] = counts["Emotion"].map({e: i for i, e in enumerate(order)})
    counts = counts.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return counts


def text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with per-emotion text length statistics
    (word count mean, median, std).
    """
    df = df.copy()
    df["word_count"] = df["text"].str.split().str.len()
    stats = (
        df.groupby("emotion")["word_count"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    stats.columns = ["Emotion", "Mean Words", "Median Words", "Std Dev", "Min", "Max"]
    stats = stats.round(2)
    return stats


def sample_rows(df: pd.DataFrame, n: int = 8, random_state: int = 42) -> pd.DataFrame:
    """Returns a representative sample of n rows for display."""
    return (
        df[["text", "emotion"]]
        .sample(n=min(n, len(df)), random_state=random_state)
        .reset_index(drop=True)
    )

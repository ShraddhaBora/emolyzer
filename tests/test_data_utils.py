"""
test_data_utils.py
~~~~~~~~~~~~~~~~~~
Unit tests for src/data_utils.py
"""

import os
import pytest
import pandas as pd
import tempfile

from src.data_utils import (
    load_and_validate,
    class_distribution,
    text_length_stats,
    sample_rows,
    EMOTION_MAP,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

VALID_CSV_CONTENT = """text,label
I am so happy today,1
This is really sad,0
I love you so much,2
I am absolutely furious,3
Something scared me,4
Wow I did not expect that,5
Feeling joyful and alive,1
Deep sadness fills me,0
"""

@pytest.fixture
def valid_csv(tmp_path):
    path = tmp_path / "text.csv"
    path.write_text(VALID_CSV_CONTENT, encoding="utf-8")
    return str(path)


@pytest.fixture
def valid_df(valid_csv):
    df, _ = load_and_validate(valid_csv)
    return df


# ─── load_and_validate ────────────────────────────────────────────────────────

class TestLoadAndValidate:

    def test_loads_successfully(self, valid_csv):
        df, meta = load_and_validate(valid_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_adds_emotion_column(self, valid_csv):
        df, _ = load_and_validate(valid_csv)
        assert "emotion" in df.columns

    def test_emotion_mapping_correct(self, valid_csv):
        df, _ = load_and_validate(valid_csv)
        for label, emotion in EMOTION_MAP.items():
            rows = df[df["label"] == label]
            if not rows.empty:
                assert (rows["emotion"] == emotion).all()

    def test_metadata_structure(self, valid_csv):
        _, meta = load_and_validate(valid_csv)
        assert "total_rows" in meta
        assert "rows_dropped" in meta
        assert "num_classes" in meta

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_and_validate(r"C:\nonexistent\path\text.csv")

    def test_missing_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("sentence,emotion\nhello,joy\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required column"):
            load_and_validate(str(bad))

    def test_invalid_label_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("text,label\nhello,99\n", encoding="utf-8")
        with pytest.raises(ValueError, match="labels outside"):
            load_and_validate(str(bad))

    def test_nan_rows_are_dropped(self, tmp_path):
        content = "text,label\nhello,1\n,0\nhappy,1\n"
        p = tmp_path / "nan.csv"
        p.write_text(content, encoding="utf-8")
        df, meta = load_and_validate(str(p))
        assert meta["rows_dropped"] == 1
        assert df["text"].isna().sum() == 0

    def test_text_is_string_type(self, valid_csv):
        df, _ = load_and_validate(valid_csv)
        assert df["text"].dtype == object  # pandas stores strings as object


# ─── class_distribution ───────────────────────────────────────────────────────

class TestClassDistribution:

    def test_returns_dataframe(self, valid_df):
        result = class_distribution(valid_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, valid_df):
        result = class_distribution(valid_df)
        assert {"Emotion", "Count", "Percentage"}.issubset(result.columns)

    def test_percentages_sum_to_100(self, valid_df):
        result = class_distribution(valid_df)
        assert abs(result["Percentage"].sum() - 100.0) < 0.1


# ─── text_length_stats ────────────────────────────────────────────────────────

class TestTextLengthStats:

    def test_returns_dataframe(self, valid_df):
        result = text_length_stats(valid_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, valid_df):
        result = text_length_stats(valid_df)
        assert "Emotion" in result.columns
        assert "Mean Words" in result.columns


# ─── sample_rows ──────────────────────────────────────────────────────────────

class TestSampleRows:

    def test_returns_n_rows(self, valid_df):
        result = sample_rows(valid_df, n=3)
        assert len(result) == 3

    def test_does_not_exceed_dataset_size(self, valid_df):
        result = sample_rows(valid_df, n=9999)
        assert len(result) <= len(valid_df)

    def test_has_text_and_emotion_columns(self, valid_df):
        result = sample_rows(valid_df, n=2)
        assert "text" in result.columns
        assert "emotion" in result.columns

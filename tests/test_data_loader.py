"""Unit tests for dataset loading, cleaning, and validation."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    ALL_MODEL_FEATURES,
    TARGET_COLUMN,
    clean_music_dataframe,
    load_dataset,
    validate_uploaded_csv,
)


@pytest.fixture
def dirty_sample_df() -> pd.DataFrame:
    """Creates a dirty synthetic sample DataFrame covering all known anomalies."""
    return pd.DataFrame(
        {
            "instance_id": [1.0, 2.0, np.nan, 4.0, 5.0, 1.0],  # row 5 is duplicate of row 0
            "artist_name": ["Artist A", "Artist B", np.nan, "Artist C", "Artist D", "Artist A"],
            "track_name": ["Song 1", "Song 2", np.nan, "Song 3", "Song 4", "Song 1"],
            "popularity": [50.0, 60.0, np.nan, 30.0, 40.0, 50.0],
            "acousticness": [0.1, 0.2, np.nan, 0.8, 0.4, 0.1],
            "danceability": [0.6, 0.7, np.nan, 0.3, 0.5, 0.6],
            "duration_ms": [200000.0, -1.0, np.nan, 300000.0, 0.0, 200000.0],  # -1.0 and 0.0 anomalies
            "energy": [0.8, 0.9, np.nan, 0.2, 0.5, 0.8],
            "instrumentalness": [0.0, 0.5, np.nan, 0.9, 0.1, 0.0],
            "key": ["C", "G#", np.nan, "A", "F", "C"],
            "liveness": [0.1, 0.2, np.nan, 0.1, 0.15, 0.1],
            "loudness": [-5.0, -4.0, np.nan, -15.0, -8.0, -5.0],
            "mode": ["Major", "Minor", np.nan, "Minor", "Major", "Major"],
            "speechiness": [0.05, 0.08, np.nan, 0.03, 0.04, 0.05],
            "tempo": ["120.5", "?", np.nan, "80.0", "140.2", "120.5"],  # '?' string anomaly
            "obtained_date": ["4-Apr", "4-Apr", np.nan, "5-Apr", "4-Apr", "4-Apr"],
            "valence": [0.7, 0.4, np.nan, 0.2, 0.5, 0.7],
            "music_genre": ["Rock", "Electronic", np.nan, "Classical", "Jazz", "Rock"],
        }
    )


def test_clean_music_dataframe_removes_blank_rows(dirty_sample_df: pd.DataFrame):
    """Verifies that completely blank rows are removed."""
    cleaned = clean_music_dataframe(dirty_sample_df)
    assert not cleaned.isnull().all(axis=1).any(), "Blank rows must be removed"


def test_clean_music_dataframe_handles_question_marks(dirty_sample_df: pd.DataFrame):
    """Verifies that '?' in tempo is safely converted to float NaN."""
    cleaned = clean_music_dataframe(dirty_sample_df)
    assert not (cleaned["tempo"].astype(str) == "?").any(), "No string '?' should remain in tempo"
    assert pd.api.types.is_float_dtype(cleaned["tempo"]), "Tempo must be float dtype"
    # Row with '?' had index 1, its tempo should now be NaN
    assert np.isnan(cleaned.loc[1, "tempo"])


def test_clean_music_dataframe_cleans_duration_anomalies(dirty_sample_df: pd.DataFrame):
    """Verifies that negative and zero duration values are treated as NaN."""
    cleaned = clean_music_dataframe(dirty_sample_df)
    assert not (cleaned["duration_ms"] <= 0).any(), "No duration <= 0 should remain"
    assert np.isnan(cleaned.loc[1, "duration_ms"]), "Duration -1.0 should become NaN"
    assert np.isnan(cleaned.loc[4, "duration_ms"]), "Duration 0.0 should become NaN"


def test_clean_music_dataframe_removes_duplicates(dirty_sample_df: pd.DataFrame):
    """Verifies that identical duplicate rows are deduplicated."""
    cleaned = clean_music_dataframe(dirty_sample_df)
    assert cleaned.duplicated().sum() == 0, "Duplicate records must be removed"


def test_validate_uploaded_csv_valid():
    """Verifies that an uploaded DataFrame containing all model features is validated as True."""
    valid_df = pd.DataFrame(columns=ALL_MODEL_FEATURES + [TARGET_COLUMN])
    is_valid, msg, missing = validate_uploaded_csv(valid_df)
    assert is_valid is True
    assert len(missing) == 0


def test_validate_uploaded_csv_missing_columns():
    """Verifies that missing audio features are detected and flagged."""
    invalid_df = pd.DataFrame(columns=["danceability", "energy"])
    is_valid, msg, missing = validate_uploaded_csv(invalid_df)
    assert is_valid is False
    assert "tempo" in missing
    assert "acousticness" in missing


def test_load_real_dataset():
    """Verifies that the actual music_genre.csv loads cleanly from disk."""
    df = load_dataset()
    assert len(df) >= 49000, "Dataset should have ~50,000 records"
    assert df["music_genre"].nunique() == 10, "Dataset should have 10 genre classes"
    assert "tempo" in df.columns
    assert (df["tempo"].astype(str) == "?").sum() == 0
    assert (df["duration_ms"] <= 0).sum() == 0

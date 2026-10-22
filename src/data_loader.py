"""Data loading, validation, and cleaning module for the Music Genre Prediction System."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


RAW_COLUMNS: List[str] = [
    "instance_id",
    "artist_name",
    "track_name",
    "popularity",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "obtained_date",
    "valence",
    "music_genre",
]

NUMERIC_AUDIO_FEATURES: List[str] = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

CATEGORICAL_AUDIO_FEATURES: List[str] = ["key", "mode"]

ALL_MODEL_FEATURES: List[str] = NUMERIC_AUDIO_FEATURES + CATEGORICAL_AUDIO_FEATURES

TARGET_COLUMN: str = "music_genre"

METADATA_COLUMNS: List[str] = [
    "instance_id",
    "artist_name",
    "track_name",
    "popularity",
    "obtained_date",
]

DEFAULT_DATA_PATHS: List[str] = [
    os.path.join("data", "music_genre.csv"),
    "music_genre.csv",
]


def resolve_default_data_path() -> Optional[str]:
    """Finds the default CSV path in current workspace."""
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            return path
    return None


def inspect_raw_file(filepath: str) -> Dict[str, Union[int, float, List[str]]]:
    """Inspects raw CSV file on disk without mutations to document data anomalies.
    
    Identifies completely blank rows, occurrences of '?', invalid duration values,
    and raw record counts.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    df_raw = pd.read_csv(filepath)
    raw_shape = df_raw.shape
    completely_blank_rows = int(df_raw.isnull().all(axis=1).sum())

    # Count '?' strings
    q_counts: Dict[str, int] = {}
    for col in df_raw.columns:
        cnt = int((df_raw[col].astype(str).str.strip() == "?").sum())
        if cnt > 0:
            q_counts[col] = cnt

    # Count duration anomalies (duration_ms <= 0)
    duration_anomalies = 0
    if "duration_ms" in df_raw.columns:
        duration_numeric = pd.to_numeric(df_raw["duration_ms"], errors="coerce")
        duration_anomalies = int((duration_numeric <= 0).sum())

    duplicates_raw = int(df_raw.duplicated().sum())

    return {
        "raw_rows": int(raw_shape[0]),
        "raw_cols": int(raw_shape[1]),
        "blank_rows": completely_blank_rows,
        "question_mark_occurrences": q_counts,
        "duration_anomalies": duration_anomalies,
        "duplicates_count": duplicates_raw,
        "columns_present": list(df_raw.columns),
    }


def clean_music_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """Applies documented, reproducible data cleaning transformations.
    
    Cleaning decisions:
    1. Removes entirely blank / all-NaN rows (dropping spurious scraper artifacts).
    2. Removes duplicate song records.
    3. Converts tempo: replaces '?' and string sentinels with NaN and converts to float.
    4. Cleans duration_ms: replaces negative or zero values (-1.0) with NaN.
    5. Converts numeric acoustic columns to proper numeric dtypes.
    6. Strips extraneous whitespace from categorical fields (key, mode, music_genre).
    7. Drops records where the target 'music_genre' is null or empty.
    """
    df = df_input.copy()

    # 1. Drop rows where all elements are NaN
    df = df.dropna(how="all")

    # 2. Drop duplicates
    df = df.drop_duplicates()

    # 3. Clean tempo: replace '?' with NaN and cast
    if "tempo" in df.columns:
        df["tempo"] = df["tempo"].astype(str).str.strip()
        df["tempo"] = df["tempo"].replace({"nan": np.nan, "?": np.nan, "": np.nan})
        df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")

    # 4. Clean duration_ms: <= 0 is invalid
    if "duration_ms" in df.columns:
        df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
        df.loc[df["duration_ms"] <= 0, "duration_ms"] = np.nan

    # 5. Clean other numeric features
    for col in NUMERIC_AUDIO_FEATURES:
        if col in df.columns and col not in ["tempo", "duration_ms"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Clean categorical features
    for col in CATEGORICAL_AUDIO_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "?": np.nan, "": np.nan})

    # 7. Drop records without target variable if target column is present
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
        df = df[~df[TARGET_COLUMN].isin(["nan", "", "None", "NaN"]) & df[TARGET_COLUMN].notnull()]

    return df


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads and cleans the music genre dataset from disk.
    
    If filepath is None, searches default locations.
    """
    if filepath is None:
        resolved = resolve_default_data_path()
        if resolved is None:
            raise FileNotFoundError(
                "Could not find music_genre.csv in expected locations (data/ or root)."
            )
        filepath = resolved

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    raw_df = pd.read_csv(filepath)
    cleaned_df = clean_music_dataframe(raw_df)
    return cleaned_df


def validate_uploaded_csv(df_uploaded: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    """Validates an uploaded DataFrame against the model's expected schema.
    
    Returns:
        (is_valid, message, missing_columns)
    """
    missing_required = [col for col in ALL_MODEL_FEATURES if col not in df_uploaded.columns]

    if missing_required:
        return (
            False,
            f"Uploaded dataset is missing required model features: {', '.join(missing_required)}",
            missing_required,
        )

    has_target = TARGET_COLUMN in df_uploaded.columns
    target_info = "with target 'music_genre'" if has_target else "without target (inference-only)"

    return (
        True,
        f"Dataset successfully validated ({df_uploaded.shape[0]} rows, {df_uploaded.shape[1]} columns, {target_info}).",
        [],
    )

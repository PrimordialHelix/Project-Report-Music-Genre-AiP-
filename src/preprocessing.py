"""Preprocessing pipeline and feature transformation module.

Implements leak-free feature preprocessing using Scikit-learn Pipeline and
ColumnTransformer, ensuring all imputers and scalers are fitted exclusively on
the training set partition.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_AUDIO_FEATURES,
    NUMERIC_AUDIO_FEATURES,
    TARGET_COLUMN,
)


def get_feature_compatibility_notice() -> Dict[str, str]:
    """Provides a transparent notice regarding acoustic features vs pitch and rhythm.
    
    Academic Project Notice:
    The standard Spotify-derived music_genre.csv dataset contains 10 audio descriptors:
    tempo (rhythmic pulse/speed in BPM), danceability, energy, acousticness, instrumentalness,
    liveness, loudness, speechiness, valence, duration_ms, and musical theory keys/modes.
    It does not contain raw audio waveform DSP measurements for pitch salience or rhythm
    envelope variation. In accordance with scientific integrity, we do NOT fabricate pitch
    or rhythm columns, nor do we falsely claim other descriptors represent raw pitch.
    """
    return {
        "title": "Dataset & Audio Feature Compatibility Notice",
        "tempo_status": "Tempo (BPM) included as a verified numeric audio feature. Missing/anomalous values cleaned.",
        "pitch_rhythm_status": (
            "The supplied CSV contains high-level acoustic descriptors. Raw pitch and rhythm metrics "
            "(such as STFT chromagrams or spectral flux) require audio files (.wav/.mp3) and digital signal "
            "processing (librosa/Essentia). In accordance with academic honesty, those values are not fabricated."
        ),
        "defensible_feature_set": (
            "We utilize 12 intrinsic audio properties: acousticness, danceability, duration_ms, energy, "
            "instrumentalness, liveness, loudness, speechiness, tempo, valence, key (one-hot), and mode (one-hot). "
            "Song popularity and artist/track metadata are excluded to prevent target and metadata leakage."
        ),
    }


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Builds a Scikit-learn ColumnTransformer for leak-free preprocessing.
    
    Numeric pipeline:
    - Median imputation (robust to outliers and skewed distributions like duration_ms).
    - Standard scaling (zero mean, unit variance for SVM margin maximization).
    
    Categorical pipeline:
    - Most-frequent imputation for missing musical key or mode.
    - One-hot encoding with unknown handling enabled for generalization.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_AUDIO_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_AUDIO_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_columns: Optional[List[str]] = None,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into stratified training and testing partitions.
    
    Ensures feature selection excludes metadata identifiers and popularity by default.
    Applies stratified sampling so that all 10 genre classes are balanced identically
    in train and test splits.
    """
    if feature_columns is None:
        feature_columns = ALL_MODEL_FEATURES

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from DataFrame.")

    # Filter out rows with missing target
    valid_df = df[df[target_column].notnull()].copy()

    X = valid_df[feature_columns]
    y = valid_df[target_column]

    # Stratified split ensures equal class representation in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> List[str]:
    """Retrieves feature names produced by the ColumnTransformer after fitting."""
    if numeric_features is None:
        numeric_features = NUMERIC_AUDIO_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_AUDIO_FEATURES

    feature_names: List[str] = list(numeric_features)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cats = list(cat_encoder.get_feature_names_out(categorical_features))
        feature_names.extend(encoded_cats)
    except Exception:
        # Fallback if not yet fitted
        feature_names.extend([f"cat_{c}" for c in categorical_features])

    return feature_names

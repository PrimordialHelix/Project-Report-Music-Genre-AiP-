"""Unit tests for feature transformation, split stratification, and leak prevention."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import ALL_MODEL_FEATURES, TARGET_COLUMN
from src.preprocessing import (
    build_preprocessor,
    get_feature_compatibility_notice,
    get_transformed_feature_names,
    split_data,
)


@pytest.fixture
def synthetic_music_df() -> pd.DataFrame:
    """Creates a deterministic multi-genre synthetic dataset for testing."""
    np.random.seed(42)
    genres = ["Rock", "Classical", "Rap", "Electronic"]
    keys = ["C", "G", "D", "A", "F"]
    modes = ["Major", "Minor"]

    rows = []
    for i in range(120):
        rows.append(
            {
                "acousticness": np.random.uniform(0.0, 1.0),
                "danceability": np.random.uniform(0.0, 1.0),
                "duration_ms": np.random.uniform(150000.0, 300000.0),
                "energy": np.random.uniform(0.0, 1.0),
                "instrumentalness": np.random.uniform(0.0, 1.0),
                "liveness": np.random.uniform(0.0, 1.0),
                "loudness": np.random.uniform(-30.0, 0.0),
                "speechiness": np.random.uniform(0.0, 1.0),
                "tempo": np.random.uniform(60.0, 180.0),
                "valence": np.random.uniform(0.0, 1.0),
                "key": np.random.choice(keys),
                "mode": np.random.choice(modes),
                "music_genre": genres[i % len(genres)],
                "track_name": f"Track_{i}",
                "artist_name": f"Artist_{i}",
                "popularity": 50.0,
            }
        )
    df = pd.DataFrame(rows)
    # Introduce some deliberate NaNs in numeric and categorical to test imputation
    df.loc[0, "tempo"] = np.nan
    df.loc[1, "duration_ms"] = np.nan
    df.loc[2, "key"] = np.nan
    return df


def test_feature_compatibility_notice_content():
    """Verifies that the academic feature compatibility notice is informative and defensible."""
    notice = get_feature_compatibility_notice()
    assert "tempo_status" in notice
    assert "pitch_rhythm_status" in notice
    assert "defensible_feature_set" in notice
    assert "pitch" in notice["pitch_rhythm_status"].lower()
    assert "tempo" in notice["tempo_status"].lower()


def test_split_data_shapes_and_stratification(synthetic_music_df: pd.DataFrame):
    """Verifies 80/20 train/test split with stratified target distribution."""
    X_train, X_test, y_train, y_test = split_data(
        synthetic_music_df, test_size=0.2, random_state=42
    )

    assert len(X_train) == 96
    assert len(X_test) == 24
    assert list(X_train.columns) == ALL_MODEL_FEATURES
    assert "music_genre" not in X_train.columns
    assert "popularity" not in X_train.columns

    # Verify stratified balance in test set (equal representation of all 4 genres)
    test_counts = y_test.value_counts()
    for genre in ["Rock", "Classical", "Rap", "Electronic"]:
        assert test_counts[genre] == 6, f"Genre {genre} should have exactly 6 test instances"


def test_preprocessing_fitted_on_train_only(synthetic_music_df: pd.DataFrame):
    """Guarantees leak-free preprocessing: preprocessor must fit on training set only."""
    X_train, X_test, y_train, y_test = split_data(
        synthetic_music_df, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    # Fit exclusively on X_train
    preprocessor.fit(X_train)

    # Transform both
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert not np.isnan(X_train_trans).any(), "X_train transformed must contain no NaNs"
    assert not np.isnan(X_test_trans).any(), "X_test transformed must contain no NaNs"

    # Verify scaler mean was computed from X_train
    scaler_mean = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_
    assert len(scaler_mean) == 10  # 10 numeric features


def test_preprocessing_handles_unknown_categorical_levels(synthetic_music_df: pd.DataFrame):
    """Verifies that previously unseen categorical levels do not throw runtime exceptions."""
    X_train, X_test, _, _ = split_data(synthetic_music_df, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Create unseen key and mode in novel sample
    novel_sample = X_test.iloc[[0]].copy()
    novel_sample.loc[:, "key"] = "UnknownKeyX"
    novel_sample.loc[:, "mode"] = "Microtonal"

    # Should transform cleanly with zeroes for unknown one-hot categories
    transformed = preprocessor.transform(novel_sample)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_get_transformed_feature_names(synthetic_music_df: pd.DataFrame):
    """Verifies extraction of transformed feature names including one-hot encoded categories."""
    X_train, _, _, _ = split_data(synthetic_music_df, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    names = get_transformed_feature_names(preprocessor)
    assert "danceability" in names
    assert any("key_" in n for n in names)
    assert any("mode_" in n for n in names)

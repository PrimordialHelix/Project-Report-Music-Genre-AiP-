"""Unit tests for inference, input validation, presets, and song matching."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import ALL_MODEL_FEATURES
from src.model import build_svm_pipeline, train_model
from src.prediction import (
    SONG_PRESETS,
    find_similar_songs_in_dataset,
    predict_song_genre,
    validate_input_features,
)
from src.preprocessing import split_data


@pytest.fixture
def trained_dummy_model() -> tuple:
    """Provides a quickly trained pipeline and sample input."""
    np.random.seed(42)
    genres = ["Electronic", "Classical", "Rap"]
    keys = ["C", "G", "D"]
    modes = ["Major", "Minor"]

    rows = []
    for i in range(60):
        rows.append(
            {
                "acousticness": 0.9 if (i % 3 == 1) else 0.1,
                "danceability": 0.8 if (i % 3 == 0) else 0.3,
                "duration_ms": 200000.0,
                "energy": 0.85 if (i % 3 != 1) else 0.2,
                "instrumentalness": 0.8 if (i % 3 == 1) else 0.1,
                "liveness": 0.15,
                "loudness": -6.0 if (i % 3 != 1) else -18.0,
                "speechiness": 0.4 if (i % 3 == 2) else 0.05,
                "tempo": 128.0 if (i % 3 == 0) else 90.0,
                "valence": 0.5,
                "key": keys[i % len(keys)],
                "mode": modes[i % len(modes)],
                "music_genre": genres[i % len(genres)],
            }
        )
    df = pd.DataFrame(rows)
    X_train, _, y_train, _ = split_data(df, test_size=0.2, random_state=42)
    pipeline, _ = train_model(X_train, y_train, model_type="calibrated_linear", random_state=42)
    return pipeline, df


def test_validate_all_presets():
    """Ensures every built-in song preset is strictly valid."""
    for preset_name, feat_dict in SONG_PRESETS.items():
        is_valid, errors = validate_input_features(feat_dict)
        assert is_valid is True, f"Preset '{preset_name}' failed validation: {errors}"


def test_validate_input_features_invalid_ranges():
    """Verifies that out-of-bound inputs are rejected with informative error messages."""
    bad_features = SONG_PRESETS["Festival Club Banger (Electronic)"].copy()
    bad_features["danceability"] = 1.85  # Out of [0, 1]
    bad_features["tempo"] = -10.0  # Out of [30, 250]
    bad_features["key"] = "H#"  # Invalid musical key
    bad_features["mode"] = "SuperMajor"  # Invalid mode

    is_valid, errors = validate_input_features(bad_features)
    assert is_valid is False
    assert any("danceability" in e for e in errors)
    assert any("tempo" in e for e in errors)
    assert any("key" in e for e in errors)
    assert any("mode" in e for e in errors)


def test_validate_input_features_missing_column():
    """Verifies that omission of an acoustic feature is caught."""
    bad_features = SONG_PRESETS["Festival Club Banger (Electronic)"].copy()
    del bad_features["tempo"]

    is_valid, errors = validate_input_features(bad_features)
    assert is_valid is False
    assert any("Missing required audio feature: 'tempo'" in e for e in errors)


def test_predict_song_genre_calibrated(trained_dummy_model):
    """Verifies end-to-end genre inference with calibrated probabilities."""
    pipeline, _ = trained_dummy_model
    features = SONG_PRESETS["Beethoven Symphony (Classical)"]

    result = predict_song_genre(pipeline, features)

    assert "predicted_genre" in result
    assert result["predicted_genre"] in ["Electronic", "Classical", "Rap"]
    assert result["is_probability"] is True
    assert len(result["sorted_rankings"]) == 3

    # Check sum of probabilities equals ~1.0
    total_prob = sum(result["scores_dict"].values())
    assert abs(total_prob - 1.0) < 1e-4


def test_find_similar_songs(trained_dummy_model):
    """Verifies nearest-song matching in Euclidean space."""
    _, df = trained_dummy_model
    features = SONG_PRESETS["Beethoven Symphony (Classical)"]

    similar_df = find_similar_songs_in_dataset(df, features, top_n=3)
    assert len(similar_df) == 3
    assert "similarity_distance" in similar_df.columns
    # Distances should be sorted ascending
    assert (similar_df["similarity_distance"].is_monotonic_increasing)

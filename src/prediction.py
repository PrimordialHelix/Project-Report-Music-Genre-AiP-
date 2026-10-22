"""Inference, input validation, and song matching prediction module.

Performs genre prediction on single song input features, generates probability
breakdowns (or decision margin scores), and queries real dataset examples matching
the user's acoustic profile.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_AUDIO_FEATURES,
    NUMERIC_AUDIO_FEATURES,
)


FEATURE_BOUNDS: Dict[str, Tuple[float, float, float, str]] = {
    # feature_name: (min_val, max_val, default_val, unit/description)
    "tempo": (30.0, 250.0, 120.0, "Beats per Minute (BPM)"),
    "acousticness": (0.0, 1.0, 0.25, "Acoustic confidence (0.0 - 1.0)"),
    "danceability": (0.0, 1.0, 0.55, "Danceability rating (0.0 - 1.0)"),
    "duration_ms": (30000.0, 1200000.0, 215000.0, "Song duration in milliseconds"),
    "energy": (0.0, 1.0, 0.60, "Perceptual energy (0.0 - 1.0)"),
    "instrumentalness": (0.0, 1.0, 0.15, "Presence of no vocals (0.0 - 1.0)"),
    "liveness": (0.0, 1.0, 0.18, "Audience presence / live feel (0.0 - 1.0)"),
    "loudness": (-50.0, 5.0, -8.0, "Overall loudness in decibels (dB)"),
    "speechiness": (0.0, 1.0, 0.08, "Presence of spoken words (0.0 - 1.0)"),
    "valence": (0.0, 1.0, 0.48, "Musical positiveness/happiness (0.0 - 1.0)"),
}

MUSICAL_KEYS: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

MUSICAL_MODES: List[str] = ["Major", "Minor"]

SONG_PRESETS: Dict[str, Dict[str, Any]] = {
    "Beethoven Symphony (Classical)": {
        "acousticness": 0.94,
        "danceability": 0.28,
        "duration_ms": 380000.0,
        "energy": 0.18,
        "instrumentalness": 0.88,
        "liveness": 0.11,
        "loudness": -18.5,
        "speechiness": 0.04,
        "tempo": 78.0,
        "valence": 0.15,
        "key": "C",
        "mode": "Minor",
    },
    "Festival Club Banger (Electronic)": {
        "acousticness": 0.02,
        "danceability": 0.78,
        "duration_ms": 210000.0,
        "energy": 0.91,
        "instrumentalness": 0.76,
        "liveness": 0.22,
        "loudness": -4.2,
        "speechiness": 0.06,
        "tempo": 128.0,
        "valence": 0.58,
        "key": "F#",
        "mode": "Minor",
    },
    "Trap Rhythm & Flow (Rap/Hip-Hop)": {
        "acousticness": 0.12,
        "danceability": 0.84,
        "duration_ms": 195000.0,
        "energy": 0.73,
        "instrumentalness": 0.00,
        "liveness": 0.16,
        "loudness": -5.8,
        "speechiness": 0.38,
        "tempo": 140.0,
        "valence": 0.65,
        "key": "C#",
        "mode": "Major",
    },
    "Delta Blues Guitar (Blues)": {
        "acousticness": 0.62,
        "danceability": 0.52,
        "duration_ms": 240000.0,
        "energy": 0.49,
        "instrumentalness": 0.08,
        "liveness": 0.14,
        "loudness": -10.4,
        "speechiness": 0.05,
        "tempo": 95.0,
        "valence": 0.42,
        "key": "A",
        "mode": "Major",
    },
    "High Voltage Stadium (Rock)": {
        "acousticness": 0.04,
        "danceability": 0.46,
        "duration_ms": 250000.0,
        "energy": 0.89,
        "instrumentalness": 0.12,
        "liveness": 0.29,
        "loudness": -4.8,
        "speechiness": 0.07,
        "tempo": 132.0,
        "valence": 0.54,
        "key": "E",
        "mode": "Minor",
    },
    "Bluegrass Front Porch (Country)": {
        "acousticness": 0.55,
        "danceability": 0.61,
        "duration_ms": 185000.0,
        "energy": 0.62,
        "instrumentalness": 0.00,
        "liveness": 0.15,
        "loudness": -7.1,
        "speechiness": 0.03,
        "tempo": 112.0,
        "valence": 0.75,
        "key": "G",
        "mode": "Major",
    },
}


def validate_input_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates that input dictionary contains all expected features within reasonable bounds.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    # Check missing columns
    for feat in ALL_MODEL_FEATURES:
        if feat not in features:
            errors.append(f"Missing required audio feature: '{feat}'")

    if errors:
        return False, errors

    # Check numeric bounds
    for feat, (min_v, max_v, _, desc) in FEATURE_BOUNDS.items():
        val = features.get(feat)
        try:
            val_f = float(val)
            if np.isnan(val_f):
                errors.append(f"Feature '{feat}' cannot be NaN.")
            elif val_f < min_v or val_f > max_v:
                errors.append(
                    f"Feature '{feat}' value {val_f} is out of realistic bounds [{min_v}, {max_v}] ({desc})."
                )
        except (ValueError, TypeError):
            errors.append(f"Feature '{feat}' must be a numeric value, got '{val}'.")

    # Check categorical values
    key_val = str(features.get("key", "")).strip()
    if key_val not in MUSICAL_KEYS:
        errors.append(f"Invalid musical key '{key_val}'. Must be one of: {', '.join(MUSICAL_KEYS)}")

    mode_val = str(features.get("mode", "")).strip()
    if mode_val not in MUSICAL_MODES:
        errors.append(f"Invalid musical mode '{mode_val}'. Must be 'Major' or 'Minor'.")

    return (len(errors) == 0), errors


def predict_song_genre(
    pipeline: Pipeline,
    input_features: Dict[str, Any],
) -> Dict[str, Any]:
    """Performs inference using the fitted SVM pipeline.
    
    Returns:
        Dictionary containing:
        - predicted_genre
        - probabilities (if CalibratedClassifierCV) or decision_scores (if LinearSVC)
        - is_calibrated_probability (boolean flag)
        - sorted_genre_rankings (list of tuples (genre, score/prob))
    """
    is_valid, errors = validate_input_features(input_features)
    if not is_valid:
        raise ValueError("Invalid input features:\n" + "\n".join(errors))

    # Construct single-row DataFrame with strict column order
    df_sample = pd.DataFrame([input_features])[ALL_MODEL_FEATURES]

    # Predict class
    predicted_genre = str(pipeline.predict(df_sample)[0])
    classes = list(pipeline.classes_)

    scores_dict: Dict[str, float] = {}
    is_probability = False

    # Check if pipeline supports predict_proba (CalibratedClassifierCV)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_sample)[0]
        scores_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        is_probability = True
    elif hasattr(pipeline, "decision_function"):
        decision = pipeline.decision_function(df_sample)[0]
        scores_dict = {cls: float(score) for cls, score in zip(classes, decision)}
        is_probability = False
    else:
        scores_dict = {cls: (1.0 if cls == predicted_genre else 0.0) for cls in classes}

    sorted_rankings = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    return {
        "predicted_genre": predicted_genre,
        "is_probability": is_probability,
        "scores_dict": scores_dict,
        "sorted_rankings": sorted_rankings,
        "input_features": input_features,
    }


def find_similar_songs_in_dataset(
    df_cleaned: pd.DataFrame,
    input_features: Dict[str, Any],
    top_n: int = 5,
) -> pd.DataFrame:
    """Finds real songs from the dataset with similar acoustic properties.
    
    Uses normalized Euclidean distance over numeric audio features.
    """
    needed_cols = ["acousticness", "danceability", "energy", "instrumentalness", "valence"]
    for col in needed_cols:
        if col not in df_cleaned.columns:
            return pd.DataFrame()

    # Extract numeric matrix for distance computation
    sub_df = df_cleaned[needed_cols].dropna()
    if sub_df.empty:
        return pd.DataFrame()

    query_vec = np.array([float(input_features.get(col, 0.5)) for col in needed_cols])
    data_mat = sub_df.values

    # Normalized Euclidean distance
    distances = np.linalg.norm(data_mat - query_vec, axis=1)
    closest_indices = sub_df.index[np.argsort(distances)[:top_n]]

    display_cols = [
        col for col in ["artist_name", "track_name", "music_genre", "popularity", "tempo", "energy", "danceability"]
        if col in df_cleaned.columns
    ]

    result_df = df_cleaned.loc[closest_indices, display_cols].copy()
    result_df["similarity_distance"] = np.round(distances[np.argsort(distances)[:top_n]], 3)
    return result_df

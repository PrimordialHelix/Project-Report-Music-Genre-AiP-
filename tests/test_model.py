"""Unit tests for SVM pipeline construction, training, evaluation, and persistence."""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_cross_validation, evaluate_pipeline, get_metric_explanations
from src.model import (
    build_svm_pipeline,
    load_trained_pipeline,
    save_trained_pipeline,
    train_model,
)
from src.preprocessing import split_data


@pytest.fixture
def synthetic_train_test() -> tuple:
    """Creates a deterministic small dataset for fast model tests."""
    np.random.seed(42)
    genres = ["Electronic", "Classical", "Rap"]
    keys = ["C", "G", "D"]
    modes = ["Major", "Minor"]

    rows = []
    for i in range(90):
        rows.append(
            {
                "acousticness": 0.9 if (i % 3 == 1) else 0.1,  # Classical has high acousticness
                "danceability": 0.8 if (i % 3 == 0) else 0.3,  # Electronic has high danceability
                "duration_ms": 200000.0,
                "energy": 0.85 if (i % 3 != 1) else 0.2,
                "instrumentalness": 0.8 if (i % 3 == 1) else 0.1,
                "liveness": 0.15,
                "loudness": -6.0 if (i % 3 != 1) else -18.0,
                "speechiness": 0.4 if (i % 3 == 2) else 0.05,  # Rap has high speechiness
                "tempo": 128.0 if (i % 3 == 0) else 90.0,
                "valence": 0.5,
                "key": keys[i % len(keys)],
                "mode": modes[i % len(modes)],
                "music_genre": genres[i % len(genres)],
            }
        )
    df = pd.DataFrame(rows)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_build_svm_pipelines():
    """Verifies that SVM pipelines can be constructed for all supported model types."""
    pipe_cal = build_svm_pipeline("calibrated_linear")
    assert "preprocessor" in pipe_cal.named_steps
    assert "classifier" in pipe_cal.named_steps

    pipe_lin = build_svm_pipeline("linear_svc")
    assert "classifier" in pipe_lin.named_steps

    pipe_rbf = build_svm_pipeline("rbf_nystroem", nystroem_components=50)
    assert "kernel_approx" in pipe_rbf.named_steps


def test_train_model_and_predictions(synthetic_train_test):
    """Verifies training of Calibrated Linear SVM and valid prediction output."""
    X_train, X_test, y_train, y_test = synthetic_train_test

    pipeline, metadata = train_model(
        X_train, y_train, model_type="calibrated_linear", C=1.0, random_state=42
    )

    assert metadata["training_samples"] == len(X_train)
    assert set(metadata["classes"]) == {"Electronic", "Classical", "Rap"}

    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert all(p in metadata["classes"] for p in preds)


def test_evaluate_pipeline_metrics(synthetic_train_test):
    """Verifies that evaluation metrics are computed accurately on held-out test data."""
    X_train, X_test, y_train, y_test = synthetic_train_test

    pipeline, _ = train_model(X_train, y_train, model_type="calibrated_linear", random_state=42)
    results = evaluate_pipeline(pipeline, X_test, y_test)

    assert "accuracy" in results
    assert 0.0 <= results["accuracy"] <= 1.0
    assert "confusion_matrix_raw" in results
    assert "confusion_matrix_norm" in results
    assert results["confusion_matrix_raw"].shape == (3, 3)
    assert results["confusion_matrix_norm"].shape == (3, 3)
    assert not results["per_class_df"].empty


def test_cross_validation(synthetic_train_test):
    """Verifies stratified K-fold cross-validation calculation."""
    X_train, _, y_train, _ = synthetic_train_test
    pipe = build_svm_pipeline("linear_svc")
    cv_res = compute_cross_validation(pipe, X_train, y_train, cv_folds=3)

    assert cv_res["cv_folds"] == 3
    assert len(cv_res["fold_scores"]) == 3
    assert 0.0 <= cv_res["mean_accuracy"] <= 1.0


def test_save_and_load_trained_pipeline(synthetic_train_test):
    """Verifies persistence and loading of the trained pipeline and metadata."""
    X_train, X_test, y_train, _ = synthetic_train_test
    pipeline, metadata = train_model(X_train, y_train, random_state=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        m_path = os.path.join(tmpdir, "test_model.joblib")
        meta_path = os.path.join(tmpdir, "test_meta.joblib")

        save_trained_pipeline(pipeline, metadata, model_path=m_path, metadata_path=meta_path)
        assert os.path.exists(m_path)
        assert os.path.exists(meta_path)

        loaded_pipe, loaded_meta = load_trained_pipeline(m_path, meta_path)
        assert loaded_pipe is not None
        assert loaded_meta is not None
        assert loaded_meta["classes"] == metadata["classes"]

        # Check inference from loaded model
        test_pred = loaded_pipe.predict(X_test)
        assert len(test_pred) == len(X_test)


def test_load_nonexistent_model():
    """Verifies that loading a non-existent model returns None, None rather than crashing."""
    pipe, meta = load_trained_pipeline("non_existent_model_123.joblib", "non_existent_meta.joblib")
    assert pipe is None
    assert meta is None


def test_metric_explanations_exist():
    """Verifies that academic explanations exist for all core metrics."""
    explanations = get_metric_explanations()
    assert "Accuracy" in explanations
    assert "Precision" in explanations
    assert "Recall (Sensitivity)" in explanations
    assert "F1-Score" in explanations
    assert "Confusion Matrix" in explanations

"""Model definition, training, persistence, and evaluation orchestration module.

Implements multiclass Support Vector Machine (SVM) classification pipelines with:
- LinearSVC as a scalable baseline for the 50,000-record dataset.
- CalibratedClassifierCV wrapper to yield genuine probability estimates across classes.
- Nystroem kernel approximation pipeline for non-linear RBF SVM comparison.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_loader import ALL_MODEL_FEATURES
from src.preprocessing import build_preprocessor

MODEL_DIR: str = "models"
DEFAULT_MODEL_FILE: str = os.path.join(MODEL_DIR, "svm_genre_classifier.joblib")
DEFAULT_METADATA_FILE: str = os.path.join(MODEL_DIR, "model_metadata.joblib")

ModelType = Literal["calibrated_linear", "linear_svc", "rbf_nystroem"]


def build_svm_pipeline(
    model_type: ModelType = "calibrated_linear",
    C: float = 1.0,
    max_iter: int = 3000,
    random_state: int = 42,
    nystroem_components: int = 250,
) -> Pipeline:
    """Constructs a complete Scikit-learn Pipeline with preprocessing and SVM classifier.
    
    Args:
        model_type: Type of SVM classifier architecture:
            - 'calibrated_linear': CalibratedClassifierCV wrapping LinearSVC (yields class probabilities).
            - 'linear_svc': Pure LinearSVC with raw decision function margins.
            - 'rbf_nystroem': Nystroem RBF kernel approximation + LinearSVC (non-linear decision boundaries).
        C: Regularization parameter.
        max_iter: Maximum iterations for solver convergence.
        random_state: Reproducibility seed.
        nystroem_components: Number of basis components for kernel approximation.
    """
    preprocessor = build_preprocessor()

    base_svc = LinearSVC(
        dual=False,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-3,
    )

    if model_type == "calibrated_linear":
        clf = CalibratedClassifierCV(
            estimator=base_svc,
            cv=3,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
    elif model_type == "rbf_nystroem":
        nystroem = Nystroem(
            kernel="rbf",
            n_components=nystroem_components,
            random_state=random_state,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("kernel_approx", nystroem),
                ("classifier", base_svc),
            ]
        )
    elif model_type == "linear_svc":
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", base_svc),
            ]
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")

    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: ModelType = "calibrated_linear",
    C: float = 1.0,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Trains the SVM pipeline on training data partition only.
    
    Returns:
        (fitted_pipeline, training_metadata)
    """
    pipeline = build_svm_pipeline(
        model_type=model_type,
        C=C,
        random_state=random_state,
    )

    t_start = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - t_start

    classes_list = list(pipeline.classes_)

    metadata = {
        "model_type": model_type,
        "C": C,
        "random_state": random_state,
        "training_samples": int(len(X_train)),
        "training_time_seconds": round(training_time, 3),
        "classes": classes_list,
        "feature_columns": list(X_train.columns),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return pipeline, metadata


def save_trained_pipeline(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: str = DEFAULT_MODEL_FILE,
    metadata_path: str = DEFAULT_METADATA_FILE,
) -> None:
    """Persists trained pipeline and metadata to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    joblib.dump(metadata, metadata_path)


def load_trained_pipeline(
    model_path: str = DEFAULT_MODEL_FILE,
    metadata_path: str = DEFAULT_METADATA_FILE,
) -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    """Loads a previously trained pipeline and its metadata from disk."""
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None

    try:
        pipeline = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        return pipeline, metadata
    except Exception:
        return None, None

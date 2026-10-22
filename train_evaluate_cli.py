"""CLI script to train and evaluate the primary SVM Music Genre Classifier.

Saves the fitted pipeline to models/ and evaluation report to reports/.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.evaluation import evaluate_pipeline
from src.model import save_trained_pipeline, train_model
from src.preprocessing import split_data


def run_training_and_evaluation() -> None:
    print("=" * 60)
    print("MUSICVERSE: Training and Evaluating Support Vector Machine...")
    print("=" * 60)

    # 1. Load dataset
    print("[1/5] Loading and cleaning dataset from data/music_genre.csv...")
    df = load_dataset()
    print(f"       Cleaned records: {len(df)} across {df['music_genre'].nunique()} genres.")

    # 2. Split dataset (stratified 80/20)
    print("[2/5] Performing stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print(f"       Train set: {len(X_train)} samples, Test set: {len(X_test)} samples.")

    # 3. Train SVM pipeline
    print("[3/5] Fitting Calibrated Linear SVM (3-fold probability calibration)...")
    pipeline, metadata = train_model(
        X_train,
        y_train,
        model_type="calibrated_linear",
        C=1.0,
        random_state=42,
    )
    print(f"       Training completed in {metadata['training_time_seconds']}s.")

    # 4. Save model
    print("[4/5] Saving model pipeline and metadata to models/...")
    save_trained_pipeline(pipeline, metadata)

    # 5. Evaluate on held-out test data
    print("[5/5] Evaluating on held-out test set...")
    eval_results = evaluate_pipeline(pipeline, X_test, y_test)

    print("\n" + "=" * 60)
    print(f"VERIFIED TEST ACCURACY: {eval_results['accuracy']:.4%}")
    print(f"Macro Precision:        {eval_results['precision_macro']:.4f}")
    print(f"Macro Recall:           {eval_results['recall_macro']:.4f}")
    print(f"Macro F1-Score:         {eval_results['f1_macro']:.4f}")
    print("=" * 60)

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_to_save = {
        "timestamp": metadata["timestamp"],
        "model_type": metadata["model_type"],
        "train_samples": metadata["training_samples"],
        "test_samples": eval_results["test_samples"],
        "accuracy": eval_results["accuracy"],
        "precision_macro": eval_results["precision_macro"],
        "precision_weighted": eval_results["precision_weighted"],
        "recall_macro": eval_results["recall_macro"],
        "recall_weighted": eval_results["recall_weighted"],
        "f1_macro": eval_results["f1_macro"],
        "f1_weighted": eval_results["f1_weighted"],
        "per_class": eval_results["per_class_df"].to_dict(orient="records"),
        "confusion_matrix_raw": eval_results["confusion_matrix_raw"].tolist(),
        "confusion_matrix_norm": eval_results["confusion_matrix_norm"].tolist(),
        "classes": eval_results["classes"],
    }
    with open("reports/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(report_to_save, f, indent=2)

    print("Saved evaluation report to reports/evaluation_results.json.\n")


if __name__ == "__main__":
    run_training_and_evaluation()

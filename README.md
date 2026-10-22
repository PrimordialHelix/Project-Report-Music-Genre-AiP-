# Music Genre Prediction System Using Support Vector Machine (SVM)

A local machine-learning application for classifying music tracks into one of ten genres using structured audio descriptors. The project includes a Scikit-learn preprocessing and classification pipeline, an interactive Streamlit dashboard, model evaluation visualizations, dataset exploration, and automated tests.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Algorithm](https://img.shields.io/badge/ML-Support%20Vector%20Machine-7C3AED.svg)
![Interface](https://img.shields.io/badge/GUI-Streamlit-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Data note:** This project works with high-level audio descriptors from `music_genre.csv`. It does not directly analyze raw sound unless a compatible audio-feature extraction workflow is added. The available dataset includes `tempo`, but does not provide explicit pitch-salience or rhythm-variation measurements; the app must not represent other columns as those measurements.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Machine-learning workflow](#machine-learning-workflow)
- [Reported evaluation results](#reported-evaluation-results)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the application](#run-the-application)
- [Use the application](#use-the-application)
- [Run tests](#run-tests)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [References](#references)

## Overview

Music genre classification is a multiclass classification task: a model uses a track's available audio descriptors to predict one of the genre labels. The project uses a Support Vector Machine (SVM) classifier and a local Streamlit interface to make the prediction and evaluation process accessible.

The dataset contains ten genre labels: Electronic, Anime, Jazz, Alternative, Country, Rap, Blues, Rock, Classical, and Hip-Hop. The primary target column is `music_genre`.

The earlier popularity-prediction notebook addressed a different task. This application focuses on genre classification rather than predicting a numeric popularity score.

## Features

- **Genre classification:** Predicts a genre using a trained multiclass SVM pipeline.
- **Data validation and cleaning:** Handles blank records, invalid numeric values, `?` placeholders, non-positive duration values, and duplicate records where appropriate.
- **Leakage-aware preprocessing:** Uses a Scikit-learn pipeline so imputers, scaling, and categorical encoding are fitted on training data rather than the held-out test set.
- **Evaluation dashboard:** Reports accuracy, precision, recall, F1-score, and a confusion matrix from held-out predictions.
- **Dataset explorer:** Inspects feature columns, genre distribution, missing values, and relevant feature summaries.
- **Genre insights:** Visualizes feature patterns and classification performance by genre.
- **Local execution:** Core data processing, training, and inference run locally; no generative-AI API is required.
- **Automated tests:** Includes tests for data processing, model creation, evaluation, and prediction workflows.

## Dataset

Place the dataset at `data/music_genre.csv` unless the application is configured to use a different location. The expected target column is `music_genre`.

The source data includes fields such as:

| Column | Description | Primary-model use |
|---|---|---|
| `instance_id` | Track identifier | Excluded; not a musical feature |
| `artist_name` | Artist metadata | Excluded from the primary model |
| `track_name` | Track-title metadata | Excluded from the primary model |
| `popularity` | Popularity score | Excluded from the primary acoustic model |
| `acousticness` | Acoustic-versus-electronic descriptor | Numeric feature |
| `danceability` | Danceability descriptor | Numeric feature |
| `duration_ms` | Duration in milliseconds | Numeric feature after validation |
| `energy` | Perceived intensity descriptor | Numeric feature |
| `instrumentalness` | Instrumental-content descriptor | Numeric feature |
| `key` | Musical key | Categorical feature where supported |
| `liveness` | Live-performance likelihood descriptor | Numeric feature |
| `loudness` | Loudness in decibels | Numeric feature |
| `mode` | Musical mode | Categorical feature where supported |
| `speechiness` | Speech-content descriptor | Numeric feature |
| `tempo` | Tempo in beats per minute | Numeric feature after cleaning |
| `obtained_date` | Source or collection metadata | Excluded from the primary model |
| `valence` | Musical positivity descriptor | Numeric feature |
| `music_genre` | Genre label | **Target variable** |

The exact columns and data types should be checked against the CSV supplied with the repository. If the dataset version differs, update the feature configuration rather than assuming that every column is present.

### Data-cleaning considerations

The preprocessing workflow should account for:

- Fully blank rows.
- String placeholders such as `?` in numeric columns.
- Numeric fields imported as text.
- Missing or invalid feature values.
- Non-positive or otherwise invalid track durations.
- Duplicate rows.
- Unexpected or missing genre labels.

Cleaning operations should be documented and should not silently fabricate values. The model target is `music_genre`; metadata and arbitrary identifiers should not be used as predictors in the primary acoustic model.

## Machine-learning workflow

1. Load and validate the CSV dataset.
2. Clean malformed values and prepare the target labels.
3. Select relevant musical descriptors and exclude target leakage and arbitrary identifiers.
4. Split data into stratified training and test partitions when class counts allow it.
5. Apply numeric imputation and feature scaling; encode categorical fields where needed.
6. Fit an SVM classifier on the training partition.
7. Evaluate predictions on the held-out test partition.
8. Save reports and visualizations and use the fitted pipeline for interactive predictions.

The primary implementation described in the project report uses a `LinearSVC` with probability calibration. A Nystroem RBF feature approximation followed by `LinearSVC` is also reported as an optional non-linear comparison. Use the actual model configuration present in the code and generated artifacts when reproducing results.

## Reported evaluation results

The accompanying project report records the following evaluation on a stratified held-out test partition of 10,000 tracks. These are **reported project results**, not a guarantee that every environment or dataset version will reproduce the same numbers. Re-run the training and evaluation workflow to verify them against the current source data and code.

| Model configuration | Reported test accuracy | Reported macro F1-score |
|---|---:|---:|
| Random-guess baseline | 10.00% | 0.1000 |
| Base LinearSVC | 39.00% | 0.3600 |
| Calibrated Linear SVM | 39.31% | 0.3808 |
| Nystroem RBF + LinearSVC | 45.55% | 0.4462 |

The report also records **39.31% accuracy** for the calibrated linear SVM and **45.55% accuracy** for the Nystroem RBF approximation. The latter is a separate configuration; do not describe its score as the calibrated linear model's score.

Accuracy alone does not fully describe a multiclass model. Use the confusion matrix and per-class precision, recall, and F1-score to understand which genres are easier or harder to distinguish. Results can change if the dataset, cleaning decisions, feature set, random seed, or library versions change.

## Project structure

The project may contain files similar to the following; check the actual repository for the exact layout.

```text
music-genre-prediction/
├── app.py
├── train_evaluate_cli.py
├── requirements.txt
├── req.bat
├── run.bat
├── music_genre.csv
├── data/
│   └── music_genre.csv
├── models/
│   └── trained model artifacts
├── reports/
│   └── evaluation reports and charts
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── prediction.py
│   └── visualization.py
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_prediction.py
```

Some files may be absent or named differently in a particular checkout. Follow the actual project files and `README` commands rather than assuming this illustrative tree is exact.

## Requirements

- Windows, macOS, or Linux.
- Python 3.11 recommended.
- Dependencies listed in `requirements.txt`.
- Node.js is optional and required only for the optional `npm run dev` launcher.

The application uses Python and Streamlit. NPM, when configured, is only a convenience command for launching Streamlit; it is not a separate application server.

## Installation

### Windows PowerShell

```powershell
# Open the repository folder
cd path\to\music-genre-prediction

# Create a virtual environment
py -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install project dependencies
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

If the repository provides `req.bat`, it can be used for dependency setup after reviewing its commands.

## Run the application

From the project root, activate the virtual environment and run:

```powershell
py -m streamlit run app.py
```

Streamlit should print a local URL, normally `http://localhost:8501`, which can be opened in a browser.

If provided and configured, the Windows launcher can also be run with:

```powershell
.\run.bat
```

### Optional: launch via NPM

If `package.json` defines the `dev` script and Node.js is installed, use:

```bash
npm run dev
```

This command should call the Streamlit Python process. It does not mean the project has a separate Node.js web server. Python and project dependencies must already be installed.

## Use the application

1. **Dashboard:** Review dataset statistics, model status, and evaluation summaries.
2. **Predict Genre:** Enter supported song-feature values or select an available preset, then request a prediction.
3. **Dataset Explorer:** Inspect records, feature columns, genre counts, and missing values. Validate uploaded CSVs before using them.
4. **Model Evaluation:** View the confusion matrix and per-genre evaluation metrics.
5. **Genre Insights:** Compare selected audio descriptors across genres.
6. **About Project:** Review the methodology, implementation choices, and limitations.

Only use prediction fields that match the feature schema used to train the model. A numeric output from unrelated features should not be treated as a meaningful genre prediction.

## Run tests

If the project includes the `tests/` suite, run:

```powershell
py -m pytest tests/ -v
```

The project report records 24 passing tests in one execution. Test status should be confirmed by running the tests in the current checkout; do not assume that historical results guarantee the current tests pass.

If an NPM test script is provided, it may also be run with:

```bash
npm test
```

## Troubleshooting

| Problem | Possible solution |
|---|---|
| `No module named streamlit` | Activate the virtual environment and run `py -m pip install -r requirements.txt`. |
| Python command is not recognized | On Windows, try `py --version` and `py -m streamlit run app.py`. Install Python if it is missing. |
| Port 8501 is already in use | Run `py -m streamlit run app.py --server.port 8502`. |
| Model file is missing | Follow the repository's training command, such as `py train_evaluate_cli.py`, if that script exists. |
| Dataset validation fails | Confirm that the CSV has a `music_genre` target column and the feature columns expected by the application. |
| Predictions fail after uploading a CSV | The uploaded columns must match the trained pipeline's required input schema; validate or retrain the model for the new dataset. |

## Limitations

- The project classifies genres from structured audio descriptors; it does not necessarily listen to or analyze raw audio files.
- The dataset has no explicit raw pitch-salience or rhythm-variation measurements. `tempo` is available, but it is not a substitute for every aspect of rhythm or pitch.
- Genre boundaries are subjective and can overlap, especially between related categories such as Rap and Hip-Hop or Rock and Alternative.
- Classification results depend on dataset quality, feature coverage, genre definitions, class balance, and evaluation design.
- Model probabilities, if displayed, should come from a calibrated estimator and should not be interpreted as guaranteed certainty.
- The reported scores must be revalidated against the current code and dataset before being cited as newly reproduced results.

## Future improvements

- Extract features such as MFCCs, chroma, spectral descriptors, and rhythm statistics from audio files using a properly validated audio-processing workflow.
- Evaluate additional classification algorithms and tune parameters using cross-validation.
- Improve genre separation with a larger and more diverse labeled dataset.
- Add prediction uncertainty and clearer explanations of the model's limitations.
- Improve accessibility and exportable evaluation reports.
- Package the application for convenient local use or deployment when appropriate.

## References

1. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.
2. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing, 10*(5), 293–302.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
4. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*.
5. McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference*.
6. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362.
7. [Streamlit documentation](https://docs.streamlit.io/)
8. [Plotly Python documentation](https://plotly.com/python/)
9. [Scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html)

---

## License

A license badge alone does not grant a license. Add a `LICENSE` file with the terms you intend to use before describing this repository as MIT-licensed or otherwise open source.

# ACADEMIC PROJECT REPORT

## Music Genre Prediction System Using Support Vector Machine (SVM)

---

### Cover Page Information

- **Project Title:** Music Genre Prediction System Using Support Vector Machine (SVM)
- **Course Name:** Artificial Intelligence Labs (AI Labs)
- **Degree Program:** Bachelor of Engineering in Computer Science & Engineering
- **Institution:** Department of Computer Science & Engineering, Chandigarh University
- **Student Authors:**
  - **Anurag Verma** — Roll No: **26BCS10805**
  - **Ranjeet Kumar** — Roll No: **26BCS10859**
  - **Gursimranjeet Singh** — Roll No: **26BCS10843**
- **Date:** October 2026
- **Domain:** Artificial Intelligence & Machine Learning
- **Repository / Submission:** Local Academic Project

---

## 1. Abstract

Music Genre Prediction is a foundational problem within Music Information Retrieval (MIR). In this project, we design, implement, and evaluate a multiclass classification system that predicts song genres from acoustic and musical descriptors using Support Vector Machines (SVM). Addressing data quality deficiencies found in scraped tabular music datasets—including blank scraping artifacts, string sentinels (`?`) in tempo columns, and non-physical negative duration recordings—we implement a leak-free preprocessing pipeline utilizing Scikit-learn's `Pipeline` and `ColumnTransformer`. 

The classification core employs a scalable `LinearSVC` with probability calibration (`CalibratedClassifierCV`) across ten balanced musical genres: Electronic, Anime, Jazz, Alternative, Country, Rap, Blues, Rock, Classical, and Hip-Hop (50,000 total songs). Evaluated on a strictly held-out 20% test partition (10,000 records), the calibrated linear SVM achieves an overall test accuracy of **39.31%** (versus a 10.0% random-guess baseline) and a macro F1-score of **0.3808**, with non-linear RBF kernel approximation achieving **45.55%**. We deploy the solution via an interactive, locally running Streamlit application (**MUSICVERSE — Genre Intelligence Studio**) equipped with single-song inference, musical presets, nearest dataset song lookup, and visual exploratory dashboards.

---

## 2. Introduction

Music categorization has historically relied on human curation and subjective musicological conventions. With the exponential growth of digital streaming platforms hosting tens of millions of tracks, automated genre identification has become essential for catalog organization, query search, and user recommendation systems.

While earlier exploratory coursework in this repository experimented with Random Forest regression to predict song popularity scores, song popularity is an extrinsic commercial outcome heavily influenced by marketing, artist celebrity, and platform playlist placement rather than intrinsic musical structure. In this work, we restructure the engineering pipeline to solve the authentic core challenge: **multiclass genre classification directly from acoustic and musical theory properties**.

---

## 3. Problem Statement

Given an audio recording represented by a feature vector $\mathbf{x} \in \mathbb{R}^d$ comprising continuous acoustic measurements (tempo, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, duration) and categorical tonal descriptors (key, mode), determine the discrete music genre $y \in \mathcal{Y}$ where:
$$\mathcal{Y} = \{\text{Alternative}, \text{Anime}, \text{Blues}, \text{Classical}, \text{Country}, \text{Electronic}, \text{Hip-Hop}, \text{Jazz}, \text{Rap}, \text{Rock}\}$$

The challenge is exacerbated by:
1. Significant sonic overlap between adjacent genres (e.g., Rap and Hip-Hop, Rock and Alternative).
2. Data quality anomalies present in real-world scraped datasets.
3. Strict academic constraints requiring zero data leakage between training and evaluation phases.

---

## 4. Objectives

1. **Task Reformation:** Replace arbitrary popularity regression with multiclass genre classification.
2. **Data Sanitization:** Detect, isolate, and systematically handle blank rows, string `'?'` values in tempo, invalid duration records, and duplicate entries.
3. **Leakage-Free Feature Engineering:** Ensure all imputations (medians, modes) and standard scalers are fitted solely on the training partition.
4. **Support Vector Machine Implementation:** Implement a high-performance multiclass SVM with probability calibration.
5. **Empirical Evaluation:** Measure generalization via held-out test accuracy, precision, recall, F1-scores, and normalized confusion matrices.
6. **Application Deployment:** Build an accessible, local Streamlit dashboard enabling real-time manual testing and exploratory data visualization.

---

## 5. Literature Review and Existing Approaches

Early approaches in Music Information Retrieval utilized hand-crafted audio descriptors such as Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroids, and Zero Crossing Rates paired with classical statistical classifiers:
- **K-Nearest Neighbors (KNN):** Simple but computationally sluggish during inference on 50,000 records and sensitive to high dimensionality.
- **Decision Trees & Random Forests:** Robust to unscaled data, but prone to axis-aligned decision boundary limitations in continuous acoustic spaces.
- **Support Vector Machines (SVM):** Widely recognized in MIR literature (Cortes & Vapnik, 1995; Tzanetakis & Cook, 2002) for their ability to find optimal separating hyperplanes with maximum geometric margin, rendering them highly resilient to overfitting in continuous feature spaces.

---

## 6. Dataset Description and Preprocessing

### 6.1 Dataset Characteristics
The dataset `music_genre.csv` contains 50,005 raw records and 18 columns. After removing 5 completely empty scraping artifacts, 50,000 songs remain. Each of the ten genre classes contains exactly 5,000 songs, establishing a balanced multiclass classification benchmark.

### 6.2 Data Quality Anomalies Identified and Cleaned
1. **Empty Records:** Exactly 5 rows with all columns as `NaN` dropped.
2. **Non-Numeric Tempo Strings:** 4,980 rows contained `'?'` as the tempo entry. These were converted to `np.nan` and imputed with the training median.
3. **Negative Duration Anomalies:** 4,939 records recorded duration as `-1.0` ms. These were marked as `np.nan` and imputed with the training median.
4. **Duplicate Entries:** 4 duplicate records removed.
5. **Metadata Exclusion:** `instance_id`, `track_name`, `artist_name`, and `obtained_date` were excluded from the model to prevent identity memorization and data leakage. Song `popularity` was excluded from the primary model to focus exclusively on acoustic characteristics.

### 6.3 Defensible Audio Feature Set
- **Numeric (10):** `acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`.
- **Categorical (2):** `key` (12 pitch classes), `mode` (Major/Minor).
- **Pitch and Rhythm Statement:** Raw pitch salience and rhythm variations require waveform DSP processing (STFT/chromagrams). As our tabular dataset supplies high-level audio descriptors, we transparently use `tempo` (BPM) as the rhythmic pace metric and do not fabricate synthetic pitch values.

---

## 7. System Architecture and Methodology

The system is organized into modular Python components:

```
src/
├── data_loader.py       # Data ingestion, schema validation, anomaly cleaning
├── preprocessing.py     # Leak-free ColumnTransformer & stratified splitting
├── model.py             # SVM pipeline construction, training, persistence
├── evaluation.py        # Metrics, classification reports, confusion matrix
├── prediction.py        # Single-song inference, presets, nearest-song matching
└── visualization.py     # Plotly interactive charting and custom theme
```

### 7.1 Mathematical Formulation of SVM

For a binary classification subproblem with training pairs $(\mathbf{x}_i, y_i)$, where $\mathbf{x}_i \in \mathbb{R}^p$ and $y_i \in \{-1, +1\}$, the soft-margin Support Vector Classifier solves:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$

subject to the constraints:
$$y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 \quad \forall i$$

Where:
- $\mathbf{w}$ represents the normal vector to the separating hyperplane.
- $C > 0$ is the regularization parameter governing the penalty for margin violations.
- $\xi_i$ are slack variables permitting bounded margin errors.
- $\phi(\mathbf{x})$ denotes the feature mapping function.

For multiclass classification across 10 genres, the system utilizes a **One-vs-Rest (OvR)** scheme, fitting 10 separate binary classifiers each separating one genre from the remaining nine.

### 7.2 Probability Calibration
To transform uncalibrated decision margin scores $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ into true posterior probabilities, we wrap the linear SVM inside Platt's calibration (`CalibratedClassifierCV`) using 3-fold internal cross-validation:
$$P(y = c \mid \mathbf{x}) = \frac{1}{1 + \exp(A f_c(\mathbf{x}) + B)}$$
where $A$ and $B$ are scalar parameters optimized via maximum likelihood on validation folds.

---

## 8. Experimental Setup and Evaluation

- **Hardware/Environment:** Local execution on Python 3.11 with Scikit-learn 1.8, Windows 64-bit.
- **Train/Test Partition:** Stratified 80% train (40,000 samples) and 20% test (10,000 samples).
- **Random Seed:** Reproducible evaluation with fixed seed `42`.

### 8.1 Measured Results

| Evaluation Metric | Calibrated Linear SVM | LinearSVC (Uncalibrated) | Nystroem RBF + LinearSVC |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **39.31%** | 39.00% | **45.55%** |
| **Macro Precision** | **0.3867** | 0.3700 | 0.4480 |
| **Macro Recall** | **0.3931** | 0.3900 | 0.4555 |
| **Macro F1-Score** | **0.3808** | 0.3600 | 0.4462 |
| **Training Time** | **1.35 seconds** | 0.59 seconds | 34.8 seconds |

### 8.2 Per-Class Performance Analysis

```
               precision    recall  f1-score   support
 Alternative       0.251     0.227     0.239      1000
       Anime       0.326     0.175     0.228      1000
       Blues       0.329     0.296     0.312      1000
   Classical       0.738     0.814     0.774      1000
     Country       0.323     0.608     0.422      1000
  Electronic       0.484     0.506     0.495      1000
     Hip-Hop       0.409     0.511     0.455      1000
        Jazz       0.414     0.339     0.373      1000
         Rap       0.334     0.225     0.269      1000
        Rock       0.259     0.230     0.244      1000
```

---

## 9. Discussion and Scientific Insights

1. **Classical Dominance (F1 = 77.4%):** Classical music occupies a distinct region in acoustic feature space characterized by high acousticness (>0.85), high instrumentalness, and low energy.
2. **Electronic Precision (F1 = 49.5%):** Strong synthetic production, high energy, and consistent tempo grids allow the SVM to identify electronic music with solid discriminative reliability.
3. **The Rap / Hip-Hop Boundary:** Rap and Hip-Hop exhibit substantial bidirectional confusion. Both styles feature high danceability (0.70-0.85), high speechiness (0.20-0.45), and similar drum machine tempo signatures. Distinguishing them purely from audio descriptors without lyrical semantic analysis or artist identity is naturally bounded.
4. **Rock / Alternative Overlap:** Guitars, drum kits, and vocal arrangements share acoustic timbral properties, resulting in expected mutual classification spillover.

---

## 10. Conclusion and Future Work

This project demonstrates a disciplined, leak-free machine-learning engineering methodology for music genre classification. By replacing popularity regression with multiclass classification and deploying a calibrated Support Vector Machine, the system achieves strong predictive power (39.31% - 45.55%) on a challenging 10-class dataset with 50,000 records.

**Future Extensions:**
1. Ingest raw audio files and extract digital signal processing (DSP) features including Mel-Frequency Cepstral Coefficients (MFCCs) and chromagram pitch representations using `librosa`.
2. Implement deep audio architectures such as Convolutional Neural Networks (CNNs) trained on Mel-spectrograms.
3. Incorporate NLP models (BERT) to analyze transcribed song lyrics as multimodal input features.

---

## 11. References

1. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine learning, 20(3), 273-297.
2. Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals*. IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
4. Platt, J. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. Advances in Large Margin Classifiers.

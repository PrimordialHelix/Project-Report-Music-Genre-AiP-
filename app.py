"""MUSICVERSE — Genre Intelligence Studio

Music Genre Prediction System using Support Vector Machine (SVM)
AI Labs Academic Project

Student Authors:
- Anurag Verma (26BCS10805)
- Ranjeet Kumar (26BCS10859)
- Gursimranjeet Singh (26BCS10843)
Chandigarh University
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Add workspace to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_AUDIO_FEATURES,
    DEFAULT_DATA_PATHS,
    NUMERIC_AUDIO_FEATURES,
    TARGET_COLUMN,
    clean_music_dataframe,
    inspect_raw_file,
    load_dataset,
    resolve_default_data_path,
    validate_uploaded_csv,
)
from src.evaluation import (
    compute_cross_validation,
    evaluate_pipeline,
    get_metric_explanations,
)
from src.model import (
    DEFAULT_METADATA_FILE,
    DEFAULT_MODEL_FILE,
    build_svm_pipeline,
    load_trained_pipeline,
    save_trained_pipeline,
    train_model,
)
from src.prediction import (
    FEATURE_BOUNDS,
    MUSICAL_KEYS,
    MUSICAL_MODES,
    SONG_PRESETS,
    find_similar_songs_in_dataset,
    predict_song_genre,
    validate_input_features,
)
from src.preprocessing import (
    get_feature_compatibility_notice,
    split_data,
)
from src.visualization import (
    COLOR_BG_DARK,
    COLOR_CARD_DARK,
    COLOR_EMERALD,
    COLOR_MAGENTA,
    COLOR_PURPLE,
    COLOR_TEXT_LIGHT,
    COLOR_TEXT_MUTED,
    COLOR_TURQUOISE,
    plot_confusion_matrix,
    plot_feature_by_genre,
    plot_feature_correlation,
    plot_genre_distribution,
    plot_per_genre_metrics,
    plot_prediction_confidence,
    plot_song_radar,
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="MUSICVERSE — Genre Intelligence Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Studio Theme with Magenta, Purple, Turquoise, and Emerald accents)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}

/* Background & Main App */
.stApp {
    background-color: #0B0F19;
    color: #F9FAFB;
}

/* Custom Header Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.25) 0%, rgba(232, 67, 147, 0.2) 50%, rgba(45, 212, 191, 0.15) 100%);
    border: 1px solid rgba(124, 58, 237, 0.4);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #E84393, #7C3AED, #2DD4BF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #D1D5DB;
    margin-bottom: 12px;
}

.badge-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
    letter-spacing: 0.5px;
}

.badge-purple {
    background: rgba(124, 58, 237, 0.2);
    color: #A78BFA;
    border: 1px solid rgba(124, 58, 237, 0.4);
}

.badge-turquoise {
    background: rgba(45, 212, 191, 0.2);
    color: #2DD4BF;
    border: 1px solid rgba(45, 212, 191, 0.4);
}

.badge-magenta {
    background: rgba(232, 67, 147, 0.2);
    color: #F472B6;
    border: 1px solid rgba(232, 67, 147, 0.4);
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

/* Metric Cards */
.metric-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(124, 58, 237, 0.4);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9CA3AF;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #F9FAFB;
}

.metric-highlight {
    color: #2DD4BF;
}

/* Prediction Result Card */
.prediction-box {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(31, 41, 55, 0.8));
    border: 2px solid #7C3AED;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.3);
}

.prediction-genre {
    font-size: 2.6rem;
    font-weight: 800;
    color: #2DD4BF;
    margin: 10px 0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Sidebar navigation styling */
.css-1d391kg, [data-testid="stSidebar"] {
    background-color: #0E1322;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #7C3AED 0%, #E84393 100%);
    color: #FFFFFF;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.6);
    transform: translateY(-1px);
}

/* Tab & Expander styling */
.streamlit-expanderHeader {
    background-color: #111827;
    border-radius: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Caching Layer for Data & Model
@st.cache_data(show_spinner=False)
def get_cached_dataset() -> pd.DataFrame:
    """Loads and caches the cleaned dataset."""
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Loads and caches the pre-trained SVM pipeline and metadata."""
    pipeline, metadata = load_trained_pipeline()
    return pipeline, metadata


@st.cache_data(show_spinner=False)
def get_cached_evaluation_report() -> Optional[Dict[str, Any]]:
    """Loads cached evaluation metrics from disk."""
    report_path = "reports/evaluation_results.json"
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Sidebar Navigation & Model Info
def render_sidebar(df: Optional[pd.DataFrame], model_meta: Optional[Dict[str, Any]]) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0 16px 0;">
                <div style="font-size: 2.4rem;">🎵</div>
                <h2 style="font-size: 1.4rem; font-weight: 800; margin: 0; background: linear-gradient(90deg, #E84393, #2DD4BF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MUSICVERSE</h2>
                <div style="font-size: 0.78rem; color: #9CA3AF; letter-spacing: 1px;">GENRE INTELLIGENCE STUDIO</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "Predict Genre",
                "Dataset Explorer",
                "Model Evaluation",
                "Genre Insights",
                "About Project",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Sidebar Status Indicators
        st.markdown("##### ⚙️ System Status")
        if df is not None:
            st.success(f"Dataset Loaded ({len(df):,} songs)")
        else:
            st.error("Dataset Missing")

        if model_meta is not None:
            st.info(f"SVM Pipeline Ready ({model_meta.get('model_type', 'SVM')})")
        else:
            st.warning("Model Not Trained")

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #6B7280; line-height: 1.4;">
                <b>Department of Computer Science & Engineering</b><br>
                Chandigarh University<br>
                Authors: Anurag Verma, Ranjeet Kumar, Gursimranjeet Singh<br>
                Algorithm: Support Vector Machine (SVM)<br>
                Environment: Local Python Execution
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# Main App Component Rendering
def main():
    # Load cached data and model
    try:
        df = get_cached_dataset()
    except Exception as e:
        df = None

    pipeline, model_meta = get_cached_model()
    eval_report = get_cached_evaluation_report()

    # Sidebar Navigation
    current_page = render_sidebar(df, model_meta)

    # Top Hero Banner on All Pages
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">MUSICVERSE — Genre Intelligence Studio</div>
            <div class="hero-subtitle">High-Dimensional Music Genre Classification via Support Vector Machine (SVM)</div>
            <div>
                <span class="badge-tag badge-purple">AI Labs Project</span>
                <span class="badge-tag badge-turquoise">Multiclass SVM</span>
                <span class="badge-tag badge-magenta">10 Music Genres</span>
                <span class="badge-tag badge-emerald">Local Python Inference</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Dashboard Page
    if current_page == "Dashboard":
        render_dashboard_page(df, pipeline, model_meta, eval_report)

    # 2. Predict Genre Page
    elif current_page == "Predict Genre":
        render_prediction_page(df, pipeline, model_meta)

    # 3. Dataset Explorer Page
    elif current_page == "Dataset Explorer":
        render_dataset_explorer_page(df)

    # 4. Model Evaluation Page
    elif current_page == "Model Evaluation":
        render_model_evaluation_page(df, pipeline, model_meta, eval_report)

    # 5. Genre Insights Page
    elif current_page == "Genre Insights":
        render_genre_insights_page(df, eval_report)

    # 6. About Project Page
    elif current_page == "About Project":
        render_about_page()


# PAGE 1: DASHBOARD
def render_dashboard_page(
    df: Optional[pd.DataFrame],
    pipeline: Any,
    model_meta: Optional[Dict[str, Any]],
    eval_report: Optional[Dict[str, Any]],
):
    st.markdown("### 📊 Studio Executive Dashboard")

    # Metrics Summary Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        records_val = f"{len(df):,}" if df is not None else "0"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Usable Songs</div>
                <div class="metric-value">{records_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        classes_val = str(df["music_genre"].nunique()) if df is not None else "0"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Genre Classes</div>
                <div class="metric-value metric-highlight">{classes_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        features_val = str(len(ALL_MODEL_FEATURES))
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Audio Features</div>
                <div class="metric-value">{features_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        acc_str = f"{eval_report['accuracy']:.1%}" if eval_report else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value metric-highlight">{acc_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        model_status_str = "Trained & Evaluated" if (pipeline and eval_report) else ("Trained" if pipeline else "Not Trained")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Status</div>
                <div class="metric-value" style="font-size: 1.15rem; color: #10B981; padding-top: 6px;">{model_status_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Genre Distribution & System Workflow
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown("#### 🎧 Balanced Multiclass Genre Distribution")
        if df is not None:
            fig_dist = plot_genre_distribution(df)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("Dataset not available to generate distribution.")

    with c_right:
        st.markdown("#### 🔄 End-to-End SVM Architecture")
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px;">
                <div style="display: flex; align-items: flex-start; margin-bottom: 14px;">
                    <div style="background: #7C3AED; color: white; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px; flex-shrink: 0;">1</div>
                    <div>
                        <div style="font-weight: 600; color: #F9FAFB;">Data Ingestion & Anomaly Sanitization</div>
                        <div style="font-size: 0.83rem; color: #9CA3AF;">Removes blank rows, deduplicates, converts tempo '?' strings to float, and sanitizes negative duration values (-1.0).</div>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-start; margin-bottom: 14px;">
                    <div style="background: #E84393; color: white; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px; flex-shrink: 0;">2</div>
                    <div>
                        <div style="font-weight: 600; color: #F9FAFB;">Stratified 80/20 Partitioning</div>
                        <div style="font-size: 0.83rem; color: #9CA3AF;">Splits 50,000 records into 40,000 training and 10,000 held-out test records with strict equal class representation.</div>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-start; margin-bottom: 14px;">
                    <div style="background: #2DD4BF; color: #111827; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px; flex-shrink: 0;">3</div>
                    <div>
                        <div style="font-weight: 600; color: #F9FAFB;">Leak-Free Preprocessing Pipeline</div>
                        <div style="font-size: 0.83rem; color: #9CA3AF;">ColumnTransformer fitted solely on training data: Median imputation, StandardScaler for continuous features, OneHotEncoder for key/mode.</div>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-start;">
                    <div style="background: #10B981; color: white; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px; flex-shrink: 0;">4</div>
                    <div>
                        <div style="font-weight: 600; color: #F9FAFB;">Multiclass SVM with Probability Calibration</div>
                        <div style="font-size: 0.83rem; color: #9CA3AF;">Scalable LinearSVC with 3-fold CalibratedClassifierCV yields maximal margin decision hyperplanes and genuine posterior probabilities.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Academic Compatibility Notice
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📌 Academic Notice: Audio Descriptors vs. Pitch and Rhythm Features", expanded=False):
        notice = get_feature_compatibility_notice()
        st.markdown(f"**{notice['title']}**")
        st.write(f"• **Tempo Status:** {notice['tempo_status']}")
        st.write(f"• **Pitch & Rhythm Analysis:** {notice['pitch_rhythm_status']}")
        st.write(f"• **Chosen Feature Set:** {notice['defensible_feature_set']}")


# PAGE 2: PREDICT GENRE
def render_prediction_page(df: Optional[pd.DataFrame], pipeline: Any, model_meta: Optional[Dict[str, Any]]):
    st.markdown("### 🎙️ Predict Music Genre from Audio Characteristics")
    st.write(
        "Enter song acoustic properties or load an authentic musical preset to predict its genre using the trained Support Vector Machine."
    )

    if pipeline is None:
        st.error("No trained SVM pipeline is available. Please train the model from the Model Evaluation page first.")
        return

    # Presets Selector
    preset_choice = st.selectbox(
        "⚡ Quick Preset Loader (Select to prefill realistic acoustic values):",
        options=["-- Manual Custom Input --"] + list(SONG_PRESETS.keys()),
        index=0,
    )

    current_values = {}
    if preset_choice in SONG_PRESETS:
        current_values = SONG_PRESETS[preset_choice]

    # Input Form
    with st.form("genre_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🎚️ Dynamics & Energy")
            tempo = st.slider(
                "Tempo (BPM)",
                min_value=30.0,
                max_value=250.0,
                value=float(current_values.get("tempo", 120.0)),
                step=0.5,
                help="Rhythmic speed in beats per minute.",
            )
            energy = st.slider(
                "Energy (0.0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_values.get("energy", 0.60)),
                step=0.01,
                help="Perceptual intensity and activity.",
            )
            loudness = st.slider(
                "Loudness (dB)",
                min_value=-50.0,
                max_value=5.0,
                value=float(current_values.get("loudness", -8.0)),
                step=0.5,
                help="Overall sound loudness in decibels.",
            )
            duration_s = st.slider(
                "Duration (seconds)",
                min_value=30,
                max_value=900,
                value=int(current_values.get("duration_ms", 215000.0) / 1000.0),
                step=5,
                help="Total track duration.",
            )

        with col2:
            st.markdown("##### 💃 Movement & Timbre")
            danceability = st.slider(
                "Danceability (0.0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_values.get("danceability", 0.55)),
                step=0.01,
                help="How suitable a track is for dancing based on tempo, rhythm stability, and beat strength.",
            )
            valence = st.slider(
                "Valence / Positiveness (0.0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_values.get("valence", 0.48)),
                step=0.01,
                help="Musical positiveness conveyed by a track (high valence = happy, cheerful).",
            )
            acousticness = st.slider(
                "Acousticness (0.0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_values.get("acousticness", 0.25)),
                step=0.01,
                help="Confidence measure of whether the track is acoustic.",
            )
            instrumentalness = st.slider(
                "Instrumentalness (0.0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_values.get("instrumentalness", 0.15)),
                step=0.01,
                help="Predicts whether a track contains no vocals.",
            )

        with col3:
            st.markdown("##### 🎼 Voice, Space & Theory")
            speechiness = st.slider(
                "Speechiness (0.0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_values.get("speechiness", 0.08)),
                step=0.01,
                help="Presence of spoken words in a track (rap, spoken word vs music).",
            )
            liveness = st.slider(
                "Liveness (0.0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_values.get("liveness", 0.18)),
                step=0.01,
                help="Detects the presence of an audience in the recording.",
            )
            key_choice = st.selectbox(
                "Musical Key",
                options=MUSICAL_KEYS,
                index=MUSICAL_KEYS.index(current_values.get("key", "C")),
                help="Overall key of the track using standard pitch notation.",
            )
            mode_choice = st.selectbox(
                "Modality (Mode)",
                options=MUSICAL_MODES,
                index=MUSICAL_MODES.index(current_values.get("mode", "Major")),
                help="Major or Minor modality.",
            )

        submit_prediction = st.form_submit_button("🔮 Predict Music Genre", use_container_width=True)

    # Prediction Execution and Output
    if submit_prediction:
        input_dict = {
            "acousticness": acousticness,
            "danceability": danceability,
            "duration_ms": float(duration_s * 1000.0),
            "energy": energy,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "loudness": loudness,
            "speechiness": speechiness,
            "tempo": tempo,
            "valence": valence,
            "key": key_choice,
            "mode": mode_choice,
        }

        # Validate input features
        is_valid, validation_errors = validate_input_features(input_dict)
        if not is_valid:
            for err in validation_errors:
                st.error(f"Validation error: {err}")
            return

        with st.spinner("Classifying audio signature with SVM hyperplane..."):
            pred_result = predict_song_genre(pipeline, input_dict)

        top_genre = pred_result["predicted_genre"]
        sorted_rankings = pred_result["sorted_rankings"]
        is_prob = pred_result["is_probability"]
        top_score = sorted_rankings[0][1]

        # Display Result
        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            score_desc = f"{top_score:.1%} Confidence" if is_prob else f"Margin Score: {top_score:.2f}"
            st.markdown(
                f"""
                <div class="prediction-box">
                    <div style="font-size: 0.9rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;">Predicted Music Genre</div>
                    <div class="prediction-genre">{top_genre}</div>
                    <div style="font-size: 1.15rem; font-weight: 600; color: #E84393;">{score_desc}</div>
                    <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 10px;">
                        Model: {model_meta.get('model_type', 'Support Vector Machine')} | Classes: 10
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            # Radar acoustic fingerprint
            st.markdown("##### 🕸️ Song Acoustic Fingerprint")
            fig_radar = plot_song_radar(input_dict)
            st.plotly_chart(fig_radar, use_container_width=True)

        with res_col2:
            st.markdown("##### 📈 Multiclass Ranking Breakdown")
            fig_conf = plot_prediction_confidence(sorted_rankings, is_probability=is_prob)
            st.plotly_chart(fig_conf, use_container_width=True)

        # Real Matching Songs from Dataset
        if df is not None:
            st.markdown("---")
            st.markdown("#### 🔍 Closest Matching Songs in Dataset")
            st.caption("Tracks from the verified dataset with nearest Euclidean acoustic distance to your input parameters:")
            similar_df = find_similar_songs_in_dataset(df, input_dict, top_n=5)
            if not similar_df.empty:
                st.dataframe(
                    similar_df,
                    use_container_width=True,
                    hide_index=True,
                )


# PAGE 3: DATASET EXPLORER
def render_dataset_explorer_page(df: Optional[pd.DataFrame]):
    st.markdown("### 📂 Dataset Explorer & Schema Inspector")
    st.write(
        "Inspect the raw structure, cleaned records, audio feature distributions, and correlations in the provided music dataset."
    )

    if df is None:
        st.error("No dataset is loaded.")
        return

    # Data Quality Summary
    st.markdown("#### 🛡️ Data Quality & Anomaly Report")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("🧹 **Blank Rows Removed:** 5 rows")
    with c2:
        st.info("❓ **Tempo '?' Converted:** 4,980 rows")
    with c3:
        st.info("⏱️ **Duration Anomalies (<=0):** 4,939 rows")
    with c4:
        st.info("✂️ **Duplicate Rows Removed:** 4 rows")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Table with Search & Filter
    st.markdown("#### 🔎 Interactive Dataset Viewer")
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1.2])

    with f_col1:
        selected_genres = st.multiselect(
            "Filter by Genre:",
            options=sorted(df["music_genre"].unique()),
            default=sorted(df["music_genre"].unique())[:3],
        )

    with f_col2:
        search_query = st.text_input("Search Artist or Track Title:", "")

    with f_col3:
        display_columns = st.multiselect(
            "Select Columns:",
            options=list(df.columns),
            default=["artist_name", "track_name", "music_genre", "tempo", "energy", "danceability", "valence", "popularity"],
        )

    filtered_df = df.copy()
    if selected_genres:
        filtered_df = filtered_df[filtered_df["music_genre"].isin(selected_genres)]
    if search_query:
        mask = (
            filtered_df["artist_name"].astype(str).str.contains(search_query, case=False, na=False)
            | filtered_df["track_name"].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    cols_to_show = display_columns if display_columns else list(df.columns)
    st.dataframe(filtered_df[cols_to_show].head(500), use_container_width=True, height=350)
    st.caption(f"Showing up to 500 of {len(filtered_df):,} matching rows.")

    # Summary Statistics & Feature Distributions
    st.markdown("---")
    st.markdown("#### 📐 Numeric Audio Feature Summaries")
    num_summary = df[NUMERIC_AUDIO_FEATURES].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    st.dataframe(num_summary.style.format("{:.3f}"), use_container_width=True)

    # Correlation Heatmap
    st.markdown("---")
    st.markdown("#### 🔗 Acoustic Feature Correlation Heatmap")
    fig_corr = plot_feature_correlation(df)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Upload New Dataset
    st.markdown("---")
    st.markdown("#### 📤 Upload Custom Music Dataset (CSV)")
    st.caption("Upload an alternate dataset to test schema compatibility and inspect feature distributions.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            is_valid, msg, missing = validate_uploaded_csv(up_df)
            if is_valid:
                st.success(f"✅ {msg}")
                st.dataframe(up_df.head(10), use_container_width=True)
            else:
                st.error(f"❌ {msg}")
        except Exception as ex:
            st.error(f"Failed to parse uploaded CSV: {ex}")


# PAGE 4: MODEL EVALUATION
def render_model_evaluation_page(
    df: Optional[pd.DataFrame],
    pipeline: Any,
    model_meta: Optional[Dict[str, Any]],
    eval_report: Optional[Dict[str, Any]],
):
    st.markdown("### 🎯 Model Evaluation & Scientific Metrics")
    st.write(
        "Performance of the Support Vector Machine evaluated exclusively on the held-out 20% test partition (10,000 songs) with zero data leakage."
    )

    if eval_report is None:
        st.warning("No evaluation report found on disk. Please trigger model training and evaluation below.")
        if st.button("🚀 Train & Evaluate Primary SVM Pipeline Now"):
            if df is None:
                st.error("Dataset not found. Cannot train.")
                return
            with st.spinner("Training Calibrated Linear SVM on 40,000 samples and evaluating on 10,000 test samples..."):
                X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
                pipe, meta = train_model(X_train, y_train, model_type="calibrated_linear", C=1.0, random_state=42)
                save_trained_pipeline(pipe, meta)
                eval_res = evaluate_pipeline(pipe, X_test, y_test)
                # Save report
                os.makedirs("reports", exist_ok=True)
                with open("reports/evaluation_results.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "timestamp": meta["timestamp"],
                            "model_type": meta["model_type"],
                            "train_samples": meta["training_samples"],
                            "test_samples": eval_res["test_samples"],
                            "accuracy": eval_res["accuracy"],
                            "precision_macro": eval_res["precision_macro"],
                            "precision_weighted": eval_res["precision_weighted"],
                            "recall_macro": eval_res["recall_macro"],
                            "recall_weighted": eval_res["recall_weighted"],
                            "f1_macro": eval_res["f1_macro"],
                            "f1_weighted": eval_res["f1_weighted"],
                            "per_class": eval_res["per_class_df"].to_dict(orient="records"),
                            "confusion_matrix_raw": eval_res["confusion_matrix_raw"].tolist(),
                            "confusion_matrix_norm": eval_res["confusion_matrix_norm"].tolist(),
                            "classes": eval_res["classes"],
                        },
                        f,
                        indent=2,
                    )
                st.success("Model trained and evaluated successfully! Please refresh or switch pages.")
                st.rerun()
        return

    # Key Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value metric-highlight">{eval_report['accuracy']:.2%}</div>
                <div style="font-size: 0.72rem; color: #9CA3AF;">10-class baseline: 10%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Macro Precision</div>
                <div class="metric-value">{eval_report['precision_macro']:.4f}</div>
                <div style="font-size: 0.72rem; color: #9CA3AF;">Weighted: {eval_report['precision_weighted']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Macro Recall</div>
                <div class="metric-value">{eval_report['recall_macro']:.4f}</div>
                <div style="font-size: 0.72rem; color: #9CA3AF;">Weighted: {eval_report['recall_weighted']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Macro F1-Score</div>
                <div class="metric-value metric-highlight">{eval_report['f1_macro']:.4f}</div>
                <div style="font-size: 0.72rem; color: #9CA3AF;">Weighted: {eval_report['f1_weighted']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sample Split</div>
                <div class="metric-value" style="font-size: 1.15rem; padding-top: 6px;">40k / 10k</div>
                <div style="font-size: 0.72rem; color: #9CA3AF;">Stratified 80/20</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix Visualization
    st.markdown("#### 🧩 Confusion Matrix")
    cm_toggle = st.radio(
        "Matrix Display Mode:",
        options=["Normalized (Class Proportions)", "Raw (Song Sample Counts)"],
        horizontal=True,
    )
    is_norm = "Normalized" in cm_toggle
    cm_data = np.array(eval_report["confusion_matrix_norm"] if is_norm else eval_report["confusion_matrix_raw"])
    classes_list = eval_report["classes"]

    fig_cm = plot_confusion_matrix(cm_data, classes_list, normalized=is_norm)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-Genre Metrics Table & Chart
    st.markdown("---")
    st.markdown("#### 📊 Per-Genre Classification Report")
    df_per_class = pd.DataFrame(eval_report["per_class"])

    col_t, col_g = st.columns([1, 1.4])
    with col_t:
        st.dataframe(
            df_per_class.style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}", "Support": "{:,}"}
            ),
            use_container_width=True,
            height=390,
        )
    with col_g:
        fig_per_genre = plot_per_genre_metrics(df_per_class)
        st.plotly_chart(fig_per_genre, use_container_width=True)

    # Cross-Validation & Architecture Comparison
    st.markdown("---")
    st.markdown("#### ⚖️ SVM Architectural Comparison")
    st.markdown(
        """
        | Model Variant | Kernel / Strategy | Scalability / Training Time | Test Accuracy | Probability Calibration |
        | :--- | :--- | :--- | :--- | :--- |
        | **LinearSVC (Base)** | Linear Hyperplane (Primal/Dual) | ~0.60 seconds | **39.00%** | Raw decision margins |
        | **Calibrated Linear SVM (Primary)** | Linear + 3-Fold Sigmoid Calibrator | ~1.35 seconds | **39.31%** | ✅ Posterior $P(\\text{genre} \\mid x)$ |
        | **Nystroem + LinearSVC** | Non-Linear RBF Kernel Approx (300 comps) | ~34.8 seconds | **45.55%** | Raw decision margins |
        """
    )

    # Metric Explanations for Academic Defense
    st.markdown("---")
    with st.expander("📚 Plain Academic English Explanations for AI Labs Defense", expanded=False):
        explanations = get_metric_explanations()
        for metric_name, expl in explanations.items():
            st.markdown(f"**{metric_name}:** {expl}")


# PAGE 5: GENRE INSIGHTS
def render_genre_insights_page(df: Optional[pd.DataFrame], eval_report: Optional[Dict[str, Any]]):
    st.markdown("### 💡 Genre Insights & Acoustic Boundary Analysis")
    st.write(
        "Empirical analysis of acoustic characteristics across musical genres based on the 50,000 song dataset."
    )

    if df is None:
        st.error("Dataset not available for genre insights.")
        return

    # Select Feature for Comparison
    selected_feature = st.selectbox(
        "Choose an Audio Feature to Compare Across Genres:",
        options=NUMERIC_AUDIO_FEATURES,
        index=NUMERIC_AUDIO_FEATURES.index("danceability"),
    )

    fig_box = plot_feature_by_genre(df, selected_feature)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔬 Empirical Confusion Matrix Observations")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px;">
                <h5 style="color: #2DD4BF; margin-top: 0;">✅ Sharp Acoustic Separations</h5>
                <ul style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.6;">
                    <li><b>Classical:</b> Exhibits the highest accuracy (F1 = 77.4%), driven by extremely high acousticness (mean > 0.85) and instrumentalness with low loudness and energy.</li>
                    <li><b>Electronic:</b> Distinguishes cleanly (F1 = 49.5%) via high synthetic energy, loud production (-5 dB), and strict tempo quantisation.</li>
                    <li><b>Country:</b> High recall (60.8%) due to balanced acousticness, moderate danceability, and high valence (happiness).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px;">
                <h5 style="color: #E84393; margin-top: 0;">⚠️ Acoustic Overlaps & Confusions</h5>
                <ul style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.6;">
                    <li><b>Rap vs. Hip-Hop:</b> Share nearly identical rhythmic danceability, tempo (120-140 BPM), and high speechiness, creating bidirectional confusion in pure acoustic space.</li>
                    <li><b>Rock vs. Alternative:</b> Sonic overlap in distorted electric guitars, dynamic loudness, and band instrumentation.</li>
                    <li><b>Anime:</b> High variance because anime soundtracks span j-rock, orchestral scores, pop, and electronic themes.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Example Tracks per Genre from Dataset
    st.markdown("---")
    st.markdown("#### 🎵 Representative Songs from Dataset by Genre")
    inspect_genre = st.selectbox(
        "Select Genre to View Example Songs from Dataset:",
        options=sorted(df["music_genre"].unique()),
    )
    genre_samples = df[df["music_genre"] == inspect_genre][
        ["artist_name", "track_name", "popularity", "tempo", "energy", "danceability", "valence"]
    ].dropna().head(10)

    st.dataframe(genre_samples, use_container_width=True, hide_index=True)


# PAGE 6: ABOUT PROJECT
def render_about_page():
    st.markdown("### 📖 About the Project")

    st.markdown(
        """
        <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; margin-bottom: 24px;">
            <h4 style="color: #2DD4BF; margin-top: 0;">Academic Project Information</h4>
            <p style="color: #D1D5DB; line-height: 1.6;">
                <b>Project Title:</b> Music Genre Prediction System Using Support Vector Machine<br>
                <b>Course:</b> AI Labs Academic Project<br>
                <b>Domain:</b> Artificial Intelligence and Machine Learning<br>
                <b>Problem Type:</b> 10-Class Multiclass Classification<br>
                <b>Primary Algorithm:</b> Support Vector Machine (SVM) with Probability Calibration<br>
                <b>Execution:</b> 100% Local Processing (Scikit-learn, Streamlit, Pandas, NumPy, Plotly)
            </p>
            <h5 style="color: #E84393;">Student Research Team (Chandigarh University)</h5>
            <p style="color: #F9FAFB; font-size: 1.05rem;">
                • <b>Anurag Verma</b> — 26BCS10805<br>
                • <b>Ranjeet Kumar</b> — 26BCS10859<br>
                • <b>Gursimranjeet Singh</b> — 26BCS10843
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎯 Problem Statement & Scientific Objective")
    st.write(
        "Music genre classification is a fundamental task in Music Information Retrieval (MIR). "
        "Given the subjective and overlapping nature of musical genres, this project investigates whether "
        "intrinsic acoustic and musical-theory descriptors (tempo, danceability, energy, acousticness, instrumentalness, "
        "liveness, loudness, speechiness, valence, key, and mode) can reliably classify songs into ten distinct genres "
        "using Support Vector Machines."
    )

    st.markdown("#### 🛡️ Why Support Vector Machine (SVM)?")
    st.markdown(
        """
        1. **Effective in High-Dimensional Continuous Spaces:** SVM searches for the optimal separating hyperplane that maximizes the margin of separation between classes, preventing overfitting even when features exhibit complex correlations.
        2. **Robust Regularization:** The soft-margin formulation ($C$ parameter) balances training error minimization against margin width, preventing the classifier from memorizing noisy outliers.
        3. **Scalability:** `LinearSVC` based on `liblinear` converges in seconds on 40,000 training records, making it ideal for local academic labs compared to compute-heavy $O(N^3)$ full kernel Gram matrices.
        4. **Probability Calibration:** Wrapping the linear SVM with `CalibratedClassifierCV` applies Platt scaling / isotonic regression, producing authentic posterior probabilities across all 10 genres.
        """
    )

    st.markdown("#### 🛠️ Technology Stack")
    st.markdown(
        """
        - **Language:** Python 3.11
        - **Machine Learning:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `LinearSVC`, `CalibratedClassifierCV`)
        - **Data Manipulation:** Pandas, NumPy
        - **Dashboard Framework:** Streamlit (Custom Theme)
        - **Interactive Plotting:** Plotly Express & Graph Objects
        - **Testing:** Pytest (Unit & Integration Tests)
        """
    )


if __name__ == "__main__":
    main()

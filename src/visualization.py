"""Interactive data visualization module using Plotly.

Adheres strictly to the requested aesthetic identity:
- Magenta: #E84393
- Purple: #7C3AED
- Turquoise: #2DD4BF
- Emerald green: #10B981
- Dark Background / Slate: #111827, #1F2937
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design Palette Constants
COLOR_MAGENTA = "#E84393"
COLOR_PURPLE = "#7C3AED"
COLOR_TURQUOISE = "#2DD4BF"
COLOR_EMERALD = "#10B981"
COLOR_BG_DARK = "#111827"
COLOR_CARD_DARK = "#1F2937"
COLOR_TEXT_LIGHT = "#F9FAFB"
COLOR_TEXT_MUTED = "#9CA3AF"

GENRE_PALETTE = [
    "#7C3AED",  # Purple
    "#2DD4BF",  # Turquoise
    "#E84393",  # Magenta
    "#10B981",  # Emerald
    "#3B82F6",  # Blue
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#8B5CF6",  # Violet
    "#14B8A6",  # Teal
    "#6366F1",  # Indigo
]


def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Applies modern dark styling to any Plotly chart."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "color": COLOR_TEXT_LIGHT},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.6)",
        font={"family": "Inter, system-ui, sans-serif", "color": COLOR_TEXT_LIGHT},
        height=height,
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
        xaxis={
            "gridcolor": "rgba(255,255,255,0.08)",
            "zerolinecolor": "rgba(255,255,255,0.15)",
            "tickfont": {"color": COLOR_TEXT_MUTED},
        },
        yaxis={
            "gridcolor": "rgba(255,255,255,0.08)",
            "zerolinecolor": "rgba(255,255,255,0.15)",
            "tickfont": {"color": COLOR_TEXT_MUTED},
        },
        legend={"font": {"color": COLOR_TEXT_LIGHT}},
    )
    return fig


def plot_genre_distribution(df: pd.DataFrame, target_col: str = "music_genre") -> go.Figure:
    """Plots genre distribution bar chart showing record counts and balance."""
    if target_col not in df.columns:
        fig = go.Figure()
        return apply_dark_theme(fig, "No Genre Column Found")

    counts = df[target_col].value_counts().reset_index()
    counts.columns = ["Genre", "Count"]

    fig = px.bar(
        counts,
        x="Genre",
        y="Count",
        color="Genre",
        color_discrete_sequence=GENRE_PALETTE,
        text="Count",
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker={"line": {"width": 1.5, "color": "rgba(255,255,255,0.2)"}},
    )
    return apply_dark_theme(fig, "Music Genre Distribution Across Dataset", height=420)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    normalized: bool = True,
) -> go.Figure:
    """Plots interactive confusion matrix heatmap with custom gradient."""
    z_values = cm
    z_text = (
        [[f"{val:.1%}" for val in row] for row in cm]
        if normalized
        else [[f"{int(val)}" for val in row] for row in cm]
    )

    custom_colorscale = [
        [0.0, "#111827"],
        [0.2, "#312E81"],
        [0.5, "#7C3AED"],
        [0.8, "#E84393"],
        [1.0, "#2DD4BF"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=classes,
            y=classes,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#FFFFFF"},
            colorscale=custom_colorscale,
            hoverongaps=False,
            colorbar={"title": "Proportion" if normalized else "Count"},
        )
    )

    fig.update_layout(
        xaxis={"title": "Predicted Genre", "tickangle": -35},
        yaxis={"title": "Actual Genre", "autorange": "reversed"},
    )
    title_str = (
        "Normalized Confusion Matrix (Row Proportion)"
        if normalized
        else "Raw Test Confusion Matrix (Counts)"
    )
    return apply_dark_theme(fig, title_str, height=520)


def plot_per_genre_metrics(df_per_class: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing Precision, Recall, and F1-Score per genre."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Precision",
            x=df_per_class["Genre"],
            y=df_per_class["Precision"],
            marker_color=COLOR_TURQUOISE,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Recall",
            x=df_per_class["Genre"],
            y=df_per_class["Recall"],
            marker_color=COLOR_PURPLE,
        )
    )
    fig.add_trace(
        go.Bar(
            name="F1-Score",
            x=df_per_class["Genre"],
            y=df_per_class["F1-Score"],
            marker_color=COLOR_MAGENTA,
        )
    )

    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1.05], "title": "Score (0.0 - 1.0)"},
        xaxis={"tickangle": -30},
    )
    return apply_dark_theme(fig, "Per-Genre Performance Metrics (Test Set)", height=450)


def plot_feature_correlation(df: pd.DataFrame, features: Optional[List[str]] = None) -> go.Figure:
    """Plots correlation heatmap among numeric audio features."""
    if features is None:
        features = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]

    available = [f for f in features if f in df.columns]
    corr = df[available].dropna().corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=available,
            y=available,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#FFFFFF"},
            colorscale="Viridis",
            zmin=-1,
            zmax=1,
            colorbar={"title": "Pearson r"},
        )
    )
    fig.update_layout(xaxis={"tickangle": -35})
    return apply_dark_theme(fig, "Acoustic Feature Correlation Heatmap", height=480)


def plot_feature_by_genre(
    df: pd.DataFrame,
    feature_name: str,
    target_col: str = "music_genre",
) -> go.Figure:
    """Boxplot displaying distributions of a single audio feature across all 10 genres."""
    if feature_name not in df.columns or target_col not in df.columns:
        return go.Figure()

    clean_subset = df[[target_col, feature_name]].dropna()

    fig = px.box(
        clean_subset,
        x=target_col,
        y=feature_name,
        color=target_col,
        color_discrete_sequence=GENRE_PALETTE,
        points=False,  # Avoid over-rendering 50k points
    )
    fig.update_layout(xaxis={"tickangle": -30}, showlegend=False)
    return apply_dark_theme(
        fig, f"Distribution of '{feature_name}' Across Music Genres", height=430
    )


def plot_prediction_confidence(
    sorted_rankings: List[Tuple[str, float]],
    is_probability: bool = True,
) -> go.Figure:
    """Horizontal bar chart showing ranked confidence scores for user prediction."""
    genres = [item[0] for item in reversed(sorted_rankings)]
    scores = [item[1] for item in reversed(sorted_rankings)]

    # Highlight top prediction with turquoise / emerald
    colors = [
        COLOR_TURQUOISE if i == len(scores) - 1 else COLOR_PURPLE
        for i in range(len(scores))
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=genres,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1%}" if is_probability else f"{s:.2f}" for s in scores],
            textposition="outside",
        )
    )

    x_title = "Posterior Probability" if is_probability else "SVM Decision Margin Score"
    fig.update_layout(
        xaxis={"title": x_title},
        yaxis={"title": ""},
        margin={"l": 90, "r": 60, "t": 50, "b": 40},
    )
    title_text = "Genre Confidence Breakdown" if is_probability else "SVM Decision Margins"
    return apply_dark_theme(fig, title_text, height=380)


def plot_song_radar(features: Dict[str, Any]) -> go.Figure:
    """Radar polar chart showing normalized acoustic fingerprint of the song."""
    radar_keys = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "valence",
    ]
    values = [float(features.get(k, 0.5)) for k in radar_keys]
    # Close the loop
    radar_keys_closed = radar_keys + [radar_keys[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_closed,
            theta=radar_keys_closed,
            fill="toself",
            fillcolor="rgba(45, 212, 191, 0.25)",
            line={"color": COLOR_TURQUOISE, "width": 2.5},
            marker={"size": 6, "color": COLOR_MAGENTA},
        )
    )

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 1.0],
                "tickfont": {"color": COLOR_TEXT_MUTED},
                "gridcolor": "rgba(255,255,255,0.1)",
            },
            "angularaxis": {
                "tickfont": {"color": COLOR_TEXT_LIGHT, "size": 11},
                "gridcolor": "rgba(255,255,255,0.1)",
            },
            "bgcolor": "rgba(17, 24, 39, 0.5)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 30, "r": 30, "t": 40, "b": 30},
        height=320,
    )
    return fig

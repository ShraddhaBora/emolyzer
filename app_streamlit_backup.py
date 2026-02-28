"""
app.py
~~~~~~
Emolyzer – Emotion Classification Dashboard
A research-style Streamlit application for training and demonstrating
a TF-IDF + Logistic Regression text emotion classifier.
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Ensure src is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import (
    load_and_validate,
    class_distribution,
    text_length_stats,
    sample_rows,
    EMOTION_MAP,
    EMOTION_COLORS,
)
from src.model_pipeline import (
    train_and_cross_validate,
    evaluate_model,
    get_feature_importance,
    get_top_misclassifications,
    predict_emotion,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emolyzer | Emotion Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Main header */
  .main-header {
    background: linear-gradient(135deg, #1a1f35 0%, #2d1b69 50%, #1a1f35 100%);
    border: 1px solid rgba(124, 111, 247, 0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(124,111,247,0.08) 0%, transparent 60%);
    pointer-events: none;
  }
  .main-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #7C6FF7 60%, #B09FFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }
  .main-subtitle {
    color: #a0aec0;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.02em;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(145deg, #1e2235, #252b42);
    border: 1px solid rgba(124, 111, 247, 0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #7C6FF7;
    line-height: 1;
    margin-bottom: 0.25rem;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }

  /* Section headers */
  .section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 3px solid #7C6FF7;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Emotion badge */
  .emotion-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
  }

  /* Prediction result card */
  .prediction-card {
    background: linear-gradient(145deg, #1a1f35, #252b42);
    border-radius: 14px;
    padding: 1.8rem;
    border: 1px solid rgba(124, 111, 247, 0.25);
    text-align: center;
  }
  .prediction-emotion {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.5rem 0;
  }
  .prediction-confidence {
    font-size: 0.9rem;
    color: #a0aec0;
    margin-top: 0.3rem;
  }

  /* Warning banner */
  .oov-warning {
    background: rgba(245, 197, 66, 0.1);
    border: 1px solid rgba(245, 197, 66, 0.4);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #F5C542;
    font-size: 0.88rem;
    margin-top: 1rem;
  }

  /* Info chips */
  .info-chip {
    display: inline-block;
    background: rgba(124, 111, 247, 0.15);
    border: 1px solid rgba(124, 111, 247, 0.3);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    color: #9f8fff;
    font-weight: 500;
    margin: 0.2rem;
  }

  /* Tab styling */
  [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
  }
  [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
  }

  /* Dataframe styling fix */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* Sidebar logo area */
  .sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(124, 111, 247, 0.2);
    margin-bottom: 1.2rem;
  }
  .sidebar-logo {
    font-size: 2.5rem;
    margin-bottom: 0.3rem;
  }
  .sidebar-name {
    font-size: 1.2rem;
    font-weight: 700;
    color: #7C6FF7;
    letter-spacing: -0.5px;
  }
  .sidebar-tagline {
    font-size: 0.75rem;
    color: #718096;
    margin-top: 0.2rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── Cached Functions ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_data():
    return load_and_validate()


@st.cache_resource(show_spinner=False)
def cached_train_and_cv(max_features: int, C: float):
    df, _ = cached_load_data()
    return train_and_cross_validate(df, max_features=max_features, C=C)


# ─── Plotly Theme Helper ───────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a0aec0", size=12),
    margin=dict(t=40, b=30, l=30, r=20),
    coloraxis_showscale=False,
)


def apply_dark_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-logo">🧠</div>
          <div class="sidebar-name">Emolyzer</div>
          <div class="sidebar-tagline">Emotion Classification System</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚙️ Model Configuration")
        max_features = st.slider(
            "TF-IDF Max Features",
            min_value=5_000, max_value=60_000, value=30_000, step=5_000,
            help="Vocabulary size for the TF-IDF vectorizer. Higher → richer features, slower training.",
        )
        C = st.select_slider(
            "Regularisation Strength (C)",
            options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            value=1.0,
            help="Inverse of regularisation strength. Lower C → stronger regularisation and simpler model.",
        )

        st.markdown("---")
        st.markdown("#### 📋 Research Methodology")
        st.markdown("""
        <p style='color:#718096; font-size:0.8rem; line-height:1.6'>
        Emolyzer functions as a comparative study in supervised text classification.
        Instead of relying on a single algorithm, the pipeline rigorously evaluates
        <strong>Logistic Regression</strong>, <strong>Multinomial Naive Bayes</strong>, and 
        <strong>Linear SVM</strong> using Stratified 5-Fold Cross Validation.
        <br><br>
        The model demonstrating the highest Mean F1-Score is automatically championed 
        for live inference, ensuring maximum generalisation.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("#### 🏷️ Emotion Classes")
        for label, emotion in EMOTION_MAP.items():
            color = EMOTION_COLORS[emotion]
            st.markdown(
                f'<span class="info-chip" style="border-color:{color}40; color:{color};">'
                f'{label} — {emotion}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("""
        <p style='color:#4a5568; font-size:0.72rem; text-align:center; margin-top:1rem'>
        Framework: Scikit-learn · Streamlit<br>
        Evaluation: 5-Fold Stratified CV
        </p>
        """, unsafe_allow_html=True)

    return max_features, C


# ─── Tab 1: Data & EDA ────────────────────────────────────────────────────────

def render_eda_tab(df, meta):
    # Top metrics
    dist_df = class_distribution(df)
    majority_class = dist_df.iloc[0]["Emotion"]
    majority_pct   = dist_df.iloc[0]["Percentage"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{meta['total_rows']:,}</div>
          <div class="metric-label">Total Samples</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{meta['num_classes']}</div>
          <div class="metric-label">Emotion Classes</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{meta['rows_dropped']}</div>
          <div class="metric-label">Rows Dropped (NaN)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{majority_class}</div>
          <div class="metric-label">Majority Class ({majority_pct}%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=dist_df["Emotion"],
            y=dist_df["Count"],
            marker_color=dist_df["Color"],
            marker_line_width=0,
            text=dist_df["Percentage"].apply(lambda x: f"{x}%"),
            textposition="outside",
            textfont=dict(size=11, color="#a0aec0"),
        ))
        fig.update_layout(
            title=dict(text="Samples per Emotion Class", font=dict(size=13, color="#e2e8f0")),
            xaxis_title=None, yaxis_title="Count",
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Proportion Breakdown</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=dist_df["Emotion"],
            values=dist_df["Count"],
            marker=dict(colors=dist_df["Color"], line=dict(color="#0E1117", width=2)),
            hole=0.52,
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value} samples<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Text Length Analysis
    st.markdown('<div class="section-header">Text Length Statistics (Word Count)</div>', unsafe_allow_html=True)
    len_df = text_length_stats(df)
    fig_box = go.Figure()
    for _, row in dist_df.iterrows():
        subset = df[df["emotion"] == row["Emotion"]]["text"].str.split().str.len()
        fig_box.add_trace(go.Box(
            y=subset,
            name=row["Emotion"],
            marker_color=row["Color"],
            line=dict(color=row["Color"]),
            boxmean=True,
        ))
    fig_box.update_layout(
        title=dict(text="Word Count Distribution per Emotion", font=dict(size=13, color="#e2e8f0")),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    fig_box.update_yaxes(title="Word Count")
    st.plotly_chart(fig_box, use_container_width=True)

    # Data sample
    st.markdown('<div class="section-header">Dataset Sample</div>', unsafe_allow_html=True)
    sample = sample_rows(df, n=8)
    # Style the emotion column
    st.dataframe(
        sample.rename(columns={"text": "Text", "emotion": "Emotion"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("📊 Full Statistics Table"):
        st.dataframe(len_df, use_container_width=True, hide_index=True)


# ─── Tab 2: Model Performance ─────────────────────────────────────────────────

def render_performance_tab(pipeline, best_model_name, cv_results, X_test, y_test):
    eval_results = evaluate_model(pipeline, X_test, y_test)

    accuracy  = eval_results["accuracy"]
    macro_f1  = eval_results["macro_f1"]
    report    = eval_results["report_dict"]
    cm        = eval_results["confusion_matrix"]
    cls_names = eval_results["class_names"]

    # Champion Model Banner
    st.markdown(f"""
    <div style='background: rgba(124, 111, 247, 0.1); border-left: 4px solid #7C6FF7; padding: 1rem 1.5rem; border-radius: 4px; margin-bottom: 2rem;'>
        <h3 style='margin:0; color:#e2e8f0; font-size:1.2rem;'>🏆 Champion Model: {best_model_name}</h3>
        <p style='color:#a0aec0; font-size:0.85rem; margin:0.4rem 0 0 0;'>
            This model achieved the highest Mean F1-Score during 5-Fold Cross Validation. <br>
            The metrics below represent its performance on a strictly isolated 20% holdout test set ({len(X_test):,} samples).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Cross Validation Comparison Table
    st.markdown('<div class="section-header">Cross-Validation Study (Train Set)</div>', unsafe_allow_html=True)
    cv_df_rows = []
    for model, res in cv_results.items():
        is_best = "★ " if model == best_model_name else ""
        cv_df_rows.append({
            "Algorithm": f"{is_best}{model}",
            "CV Mean F1": f"{res['mean_f1']:.4f} (± {res['std_f1']:.3f})",
            "CV Mean Accuracy": f"{res['mean_accuracy']:.4f} (± {res['std_accuracy']:.3f})",
        })
    cv_df = pd.DataFrame(cv_df_rows)
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Holdout Set Performance (Test Set)</div>', unsafe_allow_html=True)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{accuracy:.1%}</div>
            <div class="metric-label">Holdout Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{macro_f1:.3f}</div>
            <div class="metric-label">Macro F1-Score</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        wa_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{wa_f1:.3f}</div>
            <div class="metric-label">Weighted F1-Score</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(X_test):,}</div>
            <div class="metric-label">Test Samples</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    # Confusion Matrix
    with col_left:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        # Normalize by true label (rows) for readability
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_norm,
            x=cls_names, y=cls_names,
            colorscale=[
                [0.0, "#1a1f35"],
                [0.5, "#4a3fa5"],
                [1.0, "#7C6FF7"],
            ],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: %{text}<extra></extra>",
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(
                thickness=10,
                tickfont=dict(color="#718096", size=10),
                outlinewidth=0,
            ),
        ))
        fig_cm.update_layout(
            title=dict(text="Normalised by True Class", font=dict(size=12, color="#718096")),
            xaxis=dict(title="Predicted", tickfont=dict(size=11)),
            yaxis=dict(title="Actual", tickfont=dict(size=11), autorange="reversed"),
            height=800,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class F1 bar chart
    with col_right:
        st.markdown('<div class="section-header">Per-Class F1 Scores</div>', unsafe_allow_html=True)
        class_f1s = [(cls, report[cls]["f1-score"]) for cls in cls_names if cls in report]
        class_f1s.sort(key=lambda x: x[1], reverse=True)
        cls_sorted, f1_sorted = zip(*class_f1s)
        colors = [EMOTION_COLORS.get(c, "#7C6FF7") for c in cls_sorted]
        fig_f1 = go.Figure(go.Bar(
            x=list(f1_sorted),
            y=list(cls_sorted),
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.3f}" for v in f1_sorted],
            textposition="outside",
            textfont=dict(size=11, color="#a0aec0"),
        ))
        fig_f1.update_layout(
            title=dict(text="F1-Score per Emotion Class (Test Set)", font=dict(size=13, color="#e2e8f0")),
            xaxis=dict(range=[0, 1.1], title="F1-Score"),
            yaxis=dict(title=None, tickfont=dict(size=11)),
            showlegend=False,
            height=800,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # Misclassification Insights
    st.markdown('<div class="section-header">Model Limitations: Misclassification Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#718096;font-size:0.85rem'>Analysis of the top confused class pairs reveals "
        "ambiguities inherent in human-labelled text datasets. High confusion between semantically similar "
        "emotions (e.g. Annoyance vs Anger) suggests label subjectivity rather than purely algorithmic failure.</p>",
        unsafe_allow_html=True,
    )
    misclassifications = get_top_misclassifications(cm, cls_names, top_k=5)
    misc_df = pd.DataFrame(misclassifications)
    misc_df.columns = ["True Emotion", "Predicted As", "Error Count"]
    st.dataframe(misc_df, use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown('<div class="section-header">Top TF-IDF Features per Emotion Class</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#718096;font-size:0.82rem'>Features with the highest learned coefficients "
        "reveal the linguistic patterns dominating each class.</p>",
        unsafe_allow_html=True,
    )

    importance = get_feature_importance(pipeline, top_n=12)
    
    selected_emotion = None
    if "Note" in importance:
        st.info("Feature interpretation via coefficients is not supported for the championed algorithm.")
    else:
        selected_emotion = st.selectbox("Select an emotion to view its top features:", list(importance.keys()))
    
    if selected_emotion:
        features = importance[selected_emotion]
        words, weights = zip(*features)
        color = EMOTION_COLORS.get(selected_emotion, "#7C6FF7")
        fig_feat = go.Figure(go.Bar(
            x=list(weights),
            y=list(words),
            orientation="h",
            marker_color=color,
            marker_line_width=0,
            opacity=0.85,
        ))
        fig_feat.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Coefficient Weight"),
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # Raw report
    with st.expander("📋 Full Classification Report (Text)"):
        st.code(eval_results["report_str"], language=None)


# ─── Live Analysis (Landing Page) ───────────────────────────────────────────

EMOTION_EMOJI = {
    "Sadness": "😢", "Joy": "😄", "Love": "❤️",
    "Anger": "😠", "Fear": "😨", "Surprise": "😲",
}


def render_inference_tab(pipeline):
    """
    Live Analysis tab. Uses st.form so the Streamlit script only reruns when
    the user explicitly submits — preventing lag on every keystroke.
    Prediction result is stored in st.session_state to survive reruns.
    """
    col_input, col_result = st.columns([1.1, 1])

    with col_input:
        st.markdown('<div class="section-header">Emotion Predictor</div>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#718096;font-size:0.82rem;margin-bottom:0.5rem'>"
            "Enter a Twitter-style message. The model predicts its primary emotion "
            "and shows confidence across all six classes.</p>",
            unsafe_allow_html=True,
        )
        # ── st.form prevents the app from rerunning on every keystroke ──────
        with st.form(key="inference_form", clear_on_submit=False):
            user_text = st.text_area(
                label="Input text",
                label_visibility="collapsed",
                placeholder="e.g. I don't like this at all...",
                height=140,
            )
            submitted = st.form_submit_button(
                "🔍 Analyse Emotion",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if not user_text.strip():
                st.warning("⚠️ Please enter some text before running the analysis.")
                st.session_state.pop("inference_result", None)
            else:
                with st.spinner("Running inference…"):
                    # Run prediction once and cache result in session_state
                    st.session_state["inference_result"] = predict_emotion(pipeline, user_text)
                    st.session_state["inference_text"] = user_text

    # ── Render result from session_state (persists across reruns) ──────────
    result    = st.session_state.get("inference_result")
    last_text = st.session_state.get("inference_text", "")

    with col_result:
        if result:
            emotion    = result["predicted_emotion"]
            confidence = result["confidence"]
            is_oov     = result["is_oov"]
            color      = EMOTION_COLORS.get(emotion, "#7C6FF7")
            emoji      = EMOTION_EMOJI.get(emotion, "🤔")

            st.markdown(f"""
            <div class="prediction-card">
              <div style='font-size:3.2rem;line-height:1.1'>{emoji}</div>
              <div class="prediction-emotion" style='color:{color}'>{emotion}</div>
              <div class="prediction-confidence">
                Confidence: <strong style='color:{color}'>{confidence:.1%}</strong>
              </div>
            </div>""", unsafe_allow_html=True)

            if is_oov:
                st.markdown("""
                <div class="oov-warning">
                  ⚠️ <strong>Low Vocabulary Match</strong> — Most words were not seen
                  during training. The prediction may be unreliable.
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding:3rem 1.5rem; text-align:center; color:#4a5568;
                        border:1px dashed #2d3748; border-radius:14px;'>
              <div style='font-size:2.5rem;margin-bottom:0.8rem'>🧪</div>
              <div style='font-size:0.9rem'>Enter a message and click
              <strong>Analyse Emotion</strong><br>to see the prediction here.</div>
            </div>""", unsafe_allow_html=True)

    # ── Probability Distribution (rendered below both columns) ──────────────
    if result:
        probs   = result["probabilities"]
        emotion = result["predicted_emotion"]

        st.markdown(
            f"<div class='section-header'>Probability Distribution · "
            f"<span style='color:#a0aec0;font-weight:400;font-size:0.9rem'>"
            f"Input: <em>\"{last_text[:80]}{'…' if len(last_text) > 80 else ''}\""
            f"</em></span></div>",
            unsafe_allow_html=True,
        )

        prob_df = pd.DataFrame({
            "Emotion": list(probs.keys()),
            "Probability": list(probs.values()),
        }).sort_values("Probability", ascending=True)

        colors_bar = [
            EMOTION_COLORS.get(e, "#7C6FF7") if e == emotion else "#2d3748"
            for e in prob_df["Emotion"]
        ]

        fig_prob = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Emotion"],
            orientation="h",
            marker_color=colors_bar,
            marker_line_width=0,
            text=[f"{p:.1%}" for p in prob_df["Probability"]],
            textposition="outside",
            textfont=dict(size=12, color="#a0aec0"),
        ))
        fig_prob.update_layout(
            xaxis=dict(range=[0, 1.15], title="Probability", tickformat=".0%"),
            yaxis=dict(title=None, tickfont=dict(size=11)),
            showlegend=False,
            height=700,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        with st.expander("🔬 Methodology Note"):
            st.markdown("""
            **How this prediction was made:**
            1. The input is tokenised and transformed by the **TF-IDF vectorizer** (preserving contractions like *don't*, *can't*).
            2. The sparse feature vector is passed to **Logistic Regression**.
            3. Class probabilities are computed via softmax of the decision function scores.
            4. The class with the highest probability is returned.

            **Limitations:**
            - Short, ambiguous, or heavily sarcastic text may not be classified reliably.
            - Words absent from the training vocabulary are silently ignored.
            """)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    # Sidebar (returns hyperparameters)
    max_features, C = render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🧠 Emolyzer</div>
      <div class="main-subtitle">
        Natural Language Processing · Emotion Classification Dashboard ·
        TF-IDF + Logistic Regression Pipeline
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load dataset
    with st.spinner("Loading dataset…"):
        try:
            df, meta = cached_load_data()
        except FileNotFoundError as e:
            st.error(f"**Dataset Not Found**\n\n{e}")
            st.stop()
        except ValueError as e:
            st.error(f"**Dataset Validation Error**\n\n{e}")
            st.stop()

    # Train models and cross-validate
    with st.spinner("Running 5-Fold Cross Validation... this may take a moment."):
        try:
            pipeline, best_model_name, cv_results, X_test, y_test = cached_train_and_cv(max_features, C)
        except Exception as e:
            st.error(f"**Model Training Failed**\n\n{e}")
            st.stop()

    # Live Analysis is Tab 1 (landing tab) — most important user-facing feature
    tab1, tab2, tab3 = st.tabs([
        "  🔬  Live Analysis  ",
        "  📊  Dataset & EDA  ",
        "  📈  Model Performance  ",
    ])

    with tab1:
        render_inference_tab(pipeline)

    with tab2:
        render_eda_tab(df, meta)

    with tab3:
        render_performance_tab(pipeline, best_model_name, cv_results, X_test, y_test)


if __name__ == "__main__":
    main()

# Trigger reload

# Trigger reload for OOV fix

# Trigger reload for 7 classes

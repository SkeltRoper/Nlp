"""
NLP Project Dashboard — Streamlit Frontend
==========================================
A modern, AI-style dashboard for a text-classification NLP project.

It loads YOUR existing artifacts (no training happens here):
    - a trained scikit-learn model        (.pkl)
    - a fitted vectorizer (e.g. TF-IDF)   (.pkl)
    - a cleaned dataset                   (.csv)

Run it with:
    streamlit run app.py

Dependencies (install once):
    pip install streamlit scikit-learn pandas numpy plotly wordcloud matplotlib joblib streamlit-option-menu

Everything you need to change for your own project lives in the CONFIG block below.
"""

import os
from pydoc import text
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  ── edit these for your project
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_TITLE = "App Review Sentiment Analyzer"
PROJECT_SUBTITLE = "Universiti Teknologi Malaysia · Faculty of Artificial Intelligence"
PROJECT_DESCRIPTION = (
    "This application analyzes user reviews and predicts their sentiment as "
    "Positive, Neutral, or Negative using Natural Language Processing and a "
    "Linear SVM classifier trained on TF-IDF features."
)

# File paths (place these next to app.py, or give absolute paths)
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "cleaned_dataset.csv"

# Dataset column names
TEXT_COLUMN = "cleaned_text"  # the column holding the cleaned text the model was trained on
LABEL_COLUMN = "label"        # the column holding the class label

# Optional: human-readable class names. Leave as None to use the model's own labels.
CLASS_NAMES = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}

# Team members shown on the Home page
TEAM_MEMBERS = [
    {"name": "Ahmed", "role": "Data Preprocessing & Feature Engineering", "id": "A24AI4007"},
    {"name": "Osama",   "role": "Model Training & Evaluation", "id": "A24AI4011"},
    {"name": "Abobakr",   "role": "Frontend & Visualization", "id": "A24AI0002"},
]

# Model-comparison numbers. Person 2 trained Naive Bayes + SVM.
# "Linear SVM" below = the saved model, measured on the full dataset (≈ training data,
#   so it is optimistic). ⚠️ Replace BOTH rows with Person 2's proper TEST-SET scores.
MODEL_COMPARISON = {
    "Naive Bayes (MultinomialNB)": {"Accuracy": 0.718437, "F1": 0.653634},
    "Linear SVM":                  {"Accuracy": 0.863727, "F1": 0.851370},
}

# How many rows to use when computing live metrics / confusion matrix.
# Keep it modest so the app stays snappy. Set to None to use the full dataset.
EVAL_SAMPLE_SIZE = 3000

# ──────────────────────────────────────────────────────────────────────────────
# PAGE SETUP & THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens — dark theme
INK = "#E8EAF0"        # primary text (light)
MUTED = "#9AA0AE"      # secondary text
ACCENT = "#818CF8"     # indigo (lightened for dark bg)
ACCENT_2 = "#A78BFA"   # violet
SURFACE = "#161922"    # card background
CANVAS = "#0E1117"     # app background
BORDER = "#272B36"     # subtle dark border
POSITIVE = "#34D399"   # green
PLOTLY_SEQ = ["#818CF8", "#A78BFA", "#F472B6", "#FBBF24", "#34D399", "#22D3EE", "#F87171"]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {CANVAS}; }}

    /* Hide default chrome for a cleaner look */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

    h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; letter-spacing: -0.02em; }}
    p, li, span, label {{ color: {INK}; }}

    /* Generic card */
    .card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25), 0 1px 2px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }}
    .card h3 {{ margin-top: 0; }}

    /* Metric card */
    .metric {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    }}
    .metric .label {{ color: {MUTED}; font-size: 0.82rem; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric .value {{ color: {INK}; font-size: 2rem; font-weight: 800; line-height: 1.1;
                      margin-top: 0.35rem; }}
    .metric .delta {{ color: {POSITIVE}; font-size: 0.85rem; font-weight: 600; }}

    /* Hero */
    .hero {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%);
        border-radius: 20px;
        padding: 2.4rem 2.6rem;
        color: #fff;
        box-shadow: 0 12px 30px rgba(99,102,241,0.25);
        margin-bottom: 1.6rem;
    }}
    .hero h1 {{ color: #fff; font-size: 2.1rem; margin: 0 0 0.4rem 0; }}
    .hero p {{ color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0; max-width: 720px; }}
    .hero .eyebrow {{ color: rgba(255,255,255,0.85); font-size: 0.8rem; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.08em; }}

    /* Prediction result card */
    .result {{
        border-radius: 18px; padding: 1.8rem 2rem; color: #fff;
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%);
        box-shadow: 0 10px 24px rgba(99,102,241,0.22);
    }}
    .result .pred-label {{ font-size: 1.9rem; font-weight: 800; margin: 0.2rem 0; }}
    .result .pred-sub {{ color: rgba(255,255,255,0.9); font-size: 0.85rem;
                         text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}

    /* Pills / badges */
    .badge {{ display:inline-block; background: rgba(129,140,248,0.16); color: {ACCENT};
              border: 1px solid rgba(129,140,248,0.3);
              border-radius: 999px; padding: 0.25rem 0.75rem; font-size: 0.8rem;
              font-weight: 600; margin: 0.15rem 0.2rem 0.15rem 0; }}

    /* Team member */
    .member {{ display:flex; align-items:center; gap: 0.9rem; padding: 0.6rem 0; }}
    .avatar {{ width:44px; height:44px; border-radius:12px; flex:none;
               background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
               color:#fff; display:flex; align-items:center; justify-content:center;
               font-weight:700; font-size:1rem; }}
    .member .m-name {{ font-weight:700; color:{INK}; }}
    .member .m-role {{ color:{MUTED}; font-size:0.85rem; }}

    /* Word highlight */
    .hl {{ padding: 1px 4px; border-radius: 5px; margin: 1px;
           display:inline-block; line-height: 1.9; }}

    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
        color:#fff; border:none; border-radius:12px;
        padding: 0.6rem 1.4rem; font-weight:600; font-size:0.95rem;
        box-shadow: 0 4px 12px rgba(99,102,241,0.25);
    }}
    .stButton > button:hover {{ filter: brightness(1.05); }}
    .small {{ color:{MUTED}; font-size:0.85rem; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOADERS  (cached so they only run once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pickle(path):
    # The .pkl files were saved with joblib (joblib.dump), so try joblib first
    # and fall back to plain pickle for files saved the other way.
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)


def safe_load():
    """Load all artifacts, returning (model, vectorizer, df, errors)."""
    errors = []
    model = vectorizer = df = None
    for path, name in [(MODEL_PATH, "model"), (VECTORIZER_PATH, "vectorizer"), (DATA_PATH, "dataset")]:
        if not os.path.exists(path):
            errors.append(f"Could not find the {name} at `{path}`.")
    if errors:
        return None, None, None, errors
    try:
        model = load_pickle(MODEL_PATH)
    except Exception as e:
        errors.append(f"Failed to load model: {e}")
    try:
        vectorizer = load_pickle(VECTORIZER_PATH)
    except Exception as e:
        errors.append(f"Failed to load vectorizer: {e}")
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        errors.append(f"Failed to load dataset: {e}")
    return model, vectorizer, df, errors


def class_label(raw):
    if CLASS_NAMES and raw in CLASS_NAMES:
        return CLASS_NAMES[raw]
    return str(raw)


# ──────────────────────────────────────────────────────────────────────────────
# CORE NLP HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _nlp_tools():
    """Load NLTK lemmatizer + stopwords (downloading data on first run).
    Returns (lemmatizer, stopwords_set, ok). Falls back to (None, sklearn_stopwords, False)
    if NLTK is unavailable, so the app still runs."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        for res in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
        sw = set(stopwords.words("english")) - {'not', 'no', 'nor'}
        return WordNetLemmatizer(), sw, True
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return None, set(ENGLISH_STOP_WORDS), False


NEGATION_TRIGGERS = {
    'not', 'no', 'nor', 'never',
    "don't", "doesn't", "didn't", "won't",
    "can't", "isn't", "aren't", "wasn't",
    "weren't", "couldn't", "shouldn't", "wouldn't"
}

def _negation_tagging(text):
    tokens = text.split()
    result = []
    negate = False
    for token in tokens:
        clean = re.sub(r"[^\w']", '', token).lower()
        if clean in NEGATION_TRIGGERS:
            negate = True
        elif negate:
            result.append('NOT_' + token)
            negate = False
        else:
            result.append(token)
    return ' '.join(result)

def preprocess(text):
    """Replicates the team's 8-step cleaning pipeline including negation tagging."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE).strip()
    text = _negation_tagging(text)                  # ← step 2b: BEFORE punctuation removal
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text).strip()
    lemmatizer, stop_words, ok = _nlp_tools()
    if ok:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    else:
        tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


def predict_text(model, vectorizer, text):
    """Clean the input the same way as training, then return (label, confidence, prob_dict)."""
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    probs = None
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        probs = {class_label(c): float(p) for c, p in zip(classes, proba)}
        confidence = float(np.max(proba))
    elif hasattr(model, "decision_function"):
        scores = np.atleast_1d(model.decision_function(X)[0])
        if scores.ndim == 0 or len(np.shape(scores)) == 0:
            scores = np.array([scores])
        # squash margins to pseudo-probabilities (LinearSVC has no predict_proba)
        exp = np.exp(scores - np.max(scores))
        soft = exp / exp.sum() if exp.sum() else exp
        classes = model.classes_
        if len(classes) == len(soft):
            probs = {class_label(c): float(p) for c, p in zip(classes, soft)}
            confidence = float(np.max(soft))
    return class_label(pred), confidence, probs


def highlight_words(model, vectorizer, text, pred_label):
    """
    Importance highlighter. For linear models we use coef_ on the cleaned tokens
    to estimate each word's contribution to the predicted class, then map it back
    onto the original words for display.
    """
    try:
        vocab = vectorizer.vocabulary_
    except Exception:
        return None

    cleaned = preprocess(text)
    contributions = {}

    has_coef = hasattr(model, "coef_")
    coef_row = None
    if has_coef:
        classes = list(getattr(model, "classes_", []))
        # map predicted label back to a class index
        target_idx = 0
        for i, c in enumerate(classes):
            if class_label(c) == pred_label:
                target_idx = i
                break
        coef = model.coef_
        coef_row = coef[target_idx] if coef.shape[0] > 1 else coef[0]

    analyzer = vectorizer.build_analyzer()
    for term in analyzer(cleaned):
        if term in vocab:
            idx = vocab[term]
            score = abs(coef_row[idx]) if coef_row is not None else 1.0
            contributions[term] = max(contributions.get(term, 0), score)

    if not contributions:
        return None

    max_score = max(contributions.values()) or 1.0
    chunks = []
    cleaned_tokens = preprocess(text).split()   # preprocess the full text once
    display_tokens = text.split()
    for tok, key in zip(display_tokens, cleaned_tokens[:len(display_tokens)]):
        weight = contributions.get(key, 0)
        if weight > 0:
            intensity = 0.25 + 0.55 * (weight / max_score)
            chunks.append(
                f"<span class='hl' style='background: rgba(129,140,248,{intensity:.2f}); "
                f"color:#FFFFFF; font-weight:600;'>{tok}</span>"
            )
        else:
            chunks.append(f"<span class='hl'>{tok}</span>")
    return " ".join(chunks)


@st.cache_data(show_spinner=False)
def top_tfidf_words(_vectorizer, _df, text_col, n=20):
    """Top-n terms by summed TF-IDF weight across the corpus."""
    try:
        feature_names = np.array(_vectorizer.get_feature_names_out())
    except Exception:
        return None
    texts = _df[text_col].astype(str).tolist()
    X = _vectorizer.transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    top_idx = sums.argsort()[::-1][:n]
    return pd.DataFrame({"word": feature_names[top_idx], "weight": sums[top_idx]})


@st.cache_data(show_spinner=False)
def evaluate(_model, _vectorizer, _df, text_col, label_col, sample_size):
    """Run the model over the dataset and return metrics + confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    data = _df.dropna(subset=[text_col, label_col])
    if sample_size and len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)
    X = _vectorizer.transform(data[text_col].astype(str))
    y_true = data[label_col].values
    y_pred = _model.predict(X)
    labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics, cm, [class_label(l) for l in labels]


# ──────────────────────────────────────────────────────────────────────────────
# SMALL UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None):
    delta_html = f"<div class='delta'>▲ {delta}</div>" if delta else ""
    st.markdown(
        f"<div class='metric'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=INK, size=13),
        colorway=PLOTLY_SEQ,
        title_font=dict(size=16, color=INK),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────────────────────────────────────
def page_home(df):
    st.markdown(
        f"""
        <div class='hero'>
            <div class='eyebrow'>{PROJECT_SUBTITLE}</div>
            <h1>🧠 {PROJECT_TITLE}</h1>
            <p>{PROJECT_DESCRIPTION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Dataset Size", f"{len(df):,}" if df is not None else "—")
    with c2:
        n_classes = df[LABEL_COLUMN].nunique() if df is not None and LABEL_COLUMN in df else "—"
        metric_card("Classes", n_classes)
    with c3:
        metric_card("Model", "Loaded ✓")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            "<div class='card'><h3>📋 Instructions</h3>"
            "<ol style='margin:0; padding-left:1.1rem; line-height:1.9;'>"
            "<li>Go to <b>Sentiment Analyzer</b> and paste a review to classify its sentiment.</li>"
            "<li>Open <b>Data Explorer</b> to browse the dataset and label balance.</li>"
            "<li>Visit <b>Visualizations</b> for the word cloud, distributions and the confusion matrix.</li>"
            "<li>Check <b>Model Information</b> for preprocessing details and performance scores.</li>"
            "</ol></div>",
            unsafe_allow_html=True,
        )
    with right:
        members_html = "<div class='card'><h3>👥 Team Members</h3>"
        for m in TEAM_MEMBERS:
            initials = "".join([w[0] for w in m["name"].split()[:2]]).upper()
            members_html += (
                f"<div class='member'><div class='avatar'>{initials}</div>"
                f"<div><div class='m-name'>{m['name']}</div>"
                f"<div class='m-role'>{m['role']} · {m['id']}</div></div></div>"
            )
        members_html += "</div>"
        st.markdown(members_html, unsafe_allow_html=True)


def page_analyzer(model, vectorizer):
    st.markdown("## ✍️ Sentiment Analyzer")
    st.markdown("<p class='small'>Paste a review below and the model will predict its sentiment "
                "(Positive, Neutral, or Negative) with a confidence score.</p>",
                unsafe_allow_html=True)

    text = st.text_area("Your text", height=200,
                        placeholder="e.g. The app keeps crashing every time I open it…",
                        label_visibility="collapsed")
    go_btn = st.button("🔮 Predict Sentiment")

    if go_btn:
        if not text.strip():
            st.warning("Please enter some text first.")
            return
        label, confidence, probs = predict_text(model, vectorizer, text)

        col1, col2 = st.columns([1, 1])
        with col1:
            conf_txt = f"{confidence*100:.1f}% confidence" if confidence is not None else "confidence n/a"
            st.markdown(
                f"<div class='result'><div class='pred-sub'>Predicted class</div>"
                f"<div class='pred-label'>{label}</div>"
                f"<div class='pred-sub'>{conf_txt}</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            if probs:
                pdf = pd.DataFrame({"class": list(probs.keys()), "prob": list(probs.values())})
                pdf = pdf.sort_values("prob", ascending=True)
                fig = px.bar(pdf, x="prob", y="class", orientation="h",
                            text=pdf["prob"].map(lambda v: f"{v*100:.0f}%"))
                fig.update_traces(marker_color=ACCENT, textposition="outside")
                fig.update_layout(xaxis_title="Confidence (margin-based)", yaxis_title="")
                st.plotly_chart(style_fig(fig, height=260), use_container_width=True)
            else:
                st.markdown("<div class='card small'>This model does not expose class "
                            "probabilities, so only the top prediction is shown.</div>",
                            unsafe_allow_html=True)

        st.caption("Your text is automatically cleaned with the same 8-step pipeline used "
           "during training (lowercase, remove URLs, negation tagging, remove "
           "punctuation/numbers, stopwords, lemmatization) before it reaches the model.")

        st.markdown("### 🔦 Important words "
                    "<span class='badge'>coef-based</span>", unsafe_allow_html=True)
        hl = highlight_words(model, vectorizer, text, label)
        if hl:
            st.markdown(f"<div class='card'>{hl}</div>", unsafe_allow_html=True)
            st.markdown("<p class='small'>Darker words pushed the prediction toward "
                        "<b>{}</b> most strongly (from the model's coefficients). "
                        "Swap in SHAP or LIME later for a more rigorous explanation.</p>".format(label),
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='card small'>Word-importance highlighting needs a vectorizer "
                        "vocabulary (and ideally a linear model). Hook up SHAP/LIME here.</div>",
                        unsafe_allow_html=True)


def page_explorer(df):
    st.markdown("## 📊 Data Explorer")

    if df is None:
        st.error("Dataset not loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Rows", f"{len(df):,}")
    with c2: metric_card("Columns", df.shape[1])
    with c3:
        n_classes = df[LABEL_COLUMN].nunique() if LABEL_COLUMN in df else "—"
        metric_card("Classes", n_classes)
    with c4:
        missing = int(df.isna().sum().sum())
        metric_card("Missing values", f"{missing:,}")

    st.markdown("### 🗂️ Dataset preview")
    st.dataframe(df.head(200), use_container_width=True, height=360)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### 🏷️ Label distribution")
        if LABEL_COLUMN in df:
            counts = df[LABEL_COLUMN].value_counts().reset_index()
            counts.columns = ["label", "count"]
            counts["label"] = counts["label"].map(class_label)
            fig = px.bar(counts, x="label", y="count", text="count")
            fig.update_traces(marker_color=ACCENT, textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="Count")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info(f"No `{LABEL_COLUMN}` column found.")
    with cols[1]:
        st.markdown("### 📈 Numeric summary")
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            st.dataframe(numeric.describe().T, use_container_width=True, height=320)
        else:
            st.markdown("<div class='card small'>No numeric columns to summarize. "
                        "Below is text-length distribution instead.</div>", unsafe_allow_html=True)
            if TEXT_COLUMN in df:
                lengths = df[TEXT_COLUMN].astype(str).str.split().map(len)
                fig = px.histogram(lengths, nbins=40)
                fig.update_traces(marker_color=ACCENT_2)
                fig.update_layout(xaxis_title="Words per document", yaxis_title="Count",
                                showlegend=False)
                st.plotly_chart(style_fig(fig), use_container_width=True)


def page_visualizations(model, vectorizer, df):
    st.markdown("## 📉 Visualizations")

    tabs = st.tabs(["☁️ Word Cloud", "🏷️ Class Distribution", "🔲 Confusion Matrix",
                    "⚖️ Model Comparison", "🔠 Top 20 Words"])

    # Word Cloud
    with tabs[0]:
        if TEXT_COLUMN not in df:
            st.info(f"No `{TEXT_COLUMN}` column found.")
        else:
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                text_blob = " ".join(df[TEXT_COLUMN].astype(str).sample(
                    min(len(df), 4000), random_state=1))
                wc = WordCloud(width=1000, height=420, background_color=None, mode="RGBA",
                            colormap="plasma", max_words=150).generate(text_blob)
                fig, ax = plt.subplots(figsize=(10, 4.2))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)
            except ImportError:
                st.warning("Install the `wordcloud` package to see this: `pip install wordcloud`")

    # Class Distribution
    with tabs[1]:
        if LABEL_COLUMN in df:
            counts = df[LABEL_COLUMN].value_counts().reset_index()
            counts.columns = ["label", "count"]
            counts["label"] = counts["label"].map(class_label)
            fig = px.pie(counts, names="label", values="count", hole=0.55)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info(f"No `{LABEL_COLUMN}` column found.")

    # Confusion Matrix
    with tabs[2]:
        try:
            _, cm, labels = evaluate(model, vectorizer, df, TEXT_COLUMN,
                                    LABEL_COLUMN, EVAL_SAMPLE_SIZE)
            fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                            color_continuous_scale="Purples",
                            labels=dict(x="Predicted", y="Actual", color="Count"))
            st.plotly_chart(style_fig(fig, height=430), use_container_width=True)
            st.markdown("<p class='small'>Computed live on a sample of your dataset.</p>",
                        unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not compute the confusion matrix: {e}")

    # Model Comparison
    with tabs[3]:
        rows = []
        for name, m in MODEL_COMPARISON.items():
            for metric, val in m.items():
                rows.append({"Model": name, "Metric": metric, "Score": val})
        comp = pd.DataFrame(rows)
        fig = px.bar(comp, x="Model", y="Score", color="Metric", barmode="group",
                    text=comp["Score"].map(lambda v: f"{v:.2f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[0, 1.05], yaxis_title="Score", xaxis_title="")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.markdown("<p class='small'>Edit <code>MODEL_COMPARISON</code> at the top of "
                    "app.py with your real numbers.</p>", unsafe_allow_html=True)

    # Top 20 Words
    with tabs[4]:
        top = top_tfidf_words(vectorizer, df, TEXT_COLUMN, n=20)
        if top is None or top.empty:
            st.info("Could not extract feature names from the vectorizer.")
        else:
            top = top.sort_values("weight", ascending=True)
            fig = px.bar(top, x="weight", y="word", orientation="h")
            fig.update_traces(marker_color=ACCENT)
            fig.update_layout(xaxis_title="Total TF-IDF weight", yaxis_title="")
            st.plotly_chart(style_fig(fig, height=520), use_container_width=True)


def page_model_info(model, vectorizer, df):
    st.markdown("## 🤖 Model Information")

    # Live performance metrics
    try:
        metrics, _, _ = evaluate(model, vectorizer, df, TEXT_COLUMN,
                                LABEL_COLUMN, EVAL_SAMPLE_SIZE)
        cols = st.columns(4)
        for col, (name, val) in zip(cols, metrics.items()):
            with col:
                metric_card(name, f"{val*100:.1f}%")
        st.markdown("<p class='small'>Computed live on a sample of your dataset "
                    "(weighted averages).</p>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not compute live metrics: {e}")

    st.markdown("### 🧩 How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='card'><h3>🧹 Preprocessing</h3>"
            "<p class='small'>Raw text is cleaned in 8 steps: lowercasing, URL removal, "
            "negation tagging (not good → NOT_good), punctuation and digit removal, "
            "tokenization, stopword removal, and lemmatization.</p></div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div class='card'><h3>🔢 TF-IDF</h3>"
            "<p class='small'>Term Frequency–Inverse Document Frequency turns text into numbers. "
            "It scores a word highly when it appears often in one document but rarely across the "
            "whole corpus, so common-everywhere words are down-weighted and distinctive words "
            "stand out.</p></div>",
            unsafe_allow_html=True)
    with c3:
        model_name = type(model).__name__
        st.markdown(
            f"<div class='card'><h3>🧠 Model</h3>"
            f"<p class='small'>The active classifier is <b>{model_name}</b>. It learns a "
            "decision boundary over the TF-IDF features and predicts the most likely sentiment. "
            "A Linear SVM reports decision-function margins rather than true probabilities, so the "
            "confidence shown is a margin-based estimate. Naive Bayes and SVM were compared and "
            "the SVM was selected.</p></div>",
            unsafe_allow_html=True)

    # Vectorizer details
    try:
        vocab_size = len(vectorizer.get_feature_names_out())
        st.markdown(
            f"<div class='card'><h3>📐 Feature space</h3>"
            f"<span class='badge'>{vocab_size:,} features</span>"
            f"<span class='badge'>{type(vectorizer).__name__}</span>"
            f"<span class='badge'>{type(model).__name__}</span>"
            f"<span class='badge'>{df[LABEL_COLUMN].nunique() if LABEL_COLUMN in df else '?'} classes</span>"
            "</div>",
            unsafe_allow_html=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION + ROUTER
# ──────────────────────────────────────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:1.4rem;'>"
            f"<div style='width:40px;height:40px;border-radius:11px;"
            f"background:linear-gradient(135deg,{ACCENT},{ACCENT_2});display:flex;"
            f"align-items:center;justify-content:center;font-size:1.3rem;'>🧠</div>"
            f"<div><div style='font-weight:800;color:{INK};line-height:1.1;'>NLP Studio</div>"
            f"<div style='font-size:0.72rem;color:{MUTED};'>Text Classification</div></div></div>",
            unsafe_allow_html=True,
        )
        pages = {
            "🏠 Home": "home",
            "✍️ Sentiment Analyzer": "analyzer",
            "📊 Data Explorer": "explorer",
            "📉 Visualizations": "viz",
            "🤖 Model Information": "model",
        }
        try:
            from streamlit_option_menu import option_menu
            choice = option_menu(
                menu_title=None,
                options=list(pages.keys()),
                icons=["", "", "", "", ""],  # emojis already in labels
                default_index=0,
                styles={
                    "container": {"padding": "0", "background-color": "transparent"},
                    "nav-link": {"font-size": "0.95rem", "font-weight": "500",
                                "color": INK, "border-radius": "10px",
                                "margin": "3px 0", "--hover-color": "#212531"},
                    "nav-link-selected": {"background-color": ACCENT, "color": "white",
                                        "font-weight": "600"},
                },
            )
        except ImportError:
            choice = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

        st.markdown("<hr style='border:none;border-top:1px solid #E8EAF0;margin:1.2rem 0;'>",
                    unsafe_allow_html=True)
        st.markdown("<p class='small'>Built with Streamlit · UTM AI Faculty</p>",
                    unsafe_allow_html=True)
        return pages[choice]


def main():
    model, vectorizer, df, errors = safe_load()

    page = sidebar_nav()

    if errors:
        st.markdown("## ⚠️ Setup needed")
        for e in errors:
            st.error(e)
        st.markdown(
            "<div class='card'><h3>Quick fix</h3><p class='small'>Place your three files "
            "next to <code>app.py</code> (or update the paths in the CONFIG block at the top "
            f"of the file):</p><span class='badge'>{MODEL_PATH}</span>"
            f"<span class='badge'>{VECTORIZER_PATH}</span>"
            f"<span class='badge'>{DATA_PATH}</span></div>",
            unsafe_allow_html=True,
        )
        return

    if page == "home":
        page_home(df)
    elif page == "analyzer":
        page_analyzer(model, vectorizer)
    elif page == "explorer":
        page_explorer(df)
    elif page == "viz":
        page_visualizations(model, vectorizer, df)
    elif page == "model":
        page_model_info(model, vectorizer, df)


if __name__ == "__main__":
    main()

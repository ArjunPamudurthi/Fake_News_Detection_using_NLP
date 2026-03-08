
#!/usr/bin/env python3
from pathlib import Path
import re
import joblib
import streamlit as st

# ---------- text cleaning ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- path helpers ----------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

_OUTPUT_DIR = project_root() / "outputs"

@st.cache_resource(show_spinner=False)
def load_model():
    pipe_path = _OUTPUT_DIR / "pipeline.joblib"
    if pipe_path.exists():
        return joblib.load(pipe_path), None, None
    model_path = _OUTPUT_DIR / "model.joblib"
    vec_path = _OUTPUT_DIR / "vectorizer.joblib"
    if model_path.exists() and vec_path.exists():
        return None, joblib.load(model_path), joblib.load(vec_path)
    return None, None, None

# ---------- custom CSS ----------
CUSTOM_CSS = """
<style>
/* ── Simple background ── */
.stApp {
    background: #1e293b;
    min-height: 100vh;
}

/* ── Glass card (simplified) ── */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1.2rem;
    margin: 1.2rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
}

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 2.5rem 0 0.5rem 0;
}
.main-header h1 {
    color: #f1f5f9;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
}

/* ── Section dividers ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin: 1.8rem 0;
}

/* ── Section titles (original size) ── */
.section-title {
    color: #e2e8f0;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
/* ── Text area inside glass (slightly larger font) ── */
.stTextArea textarea {
    border-radius: 8px !important;
    padding: 11px !important;
    font-size: 19px !important;
    background: rgba(255, 255, 255, 0.03) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    min-height: 120px !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
}

/* ── Primary detect button (simple) ── */
div.stButton > button[kind="primary"] {
    background: #3B82F6 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.2px !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #2563eb !important;
}

/* ── Sample buttons (simple) ── */
div.stButton > button:not([kind="primary"]) {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
div.stButton > button:not([kind="primary"]):hover {
    background: rgba(255, 255, 255, 0.13) !important;
}

/* ── Result glass card (simple) ── */
.result-glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1.2rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
    text-align: center;
}
.result-glass-fake {
    border-top: 3px solid #EF4444;
}
.result-glass-real {
    border-top: 3px solid #22C55E;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    letter-spacing: 0.3px;
}
.result-label-fake { color: #EF4444; }
.result-label-real { color: #22C55E; }
.result-stats {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-top: 0.3rem;
}
.result-stat-item {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    min-width: 100px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.result-stat-item .stat-label {
    color: #94a3b8;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 0.2rem;
    font-weight: 600;
}
.result-stat-item .stat-value {
    color: #f1f5f9;
    font-size: 1.5rem;
    font-weight: 700;
}

/* ── Metric cards (glass) ── */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}

/* ── Section titles ── */
.section-title {
    color: #e2e8f0;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 2.5rem;
}
.footer strong { color: #94a3b8; }

/* ── Slider ── */
div[data-testid="stSlider"] label {
    color: #94a3b8 !important;
}
</style>
"""

# ---------- streamlit app ----------
def main():
    st.set_page_config(page_title="AI Fake News Detector", page_icon="🔎", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header ──
    st.markdown(
        '<div class="main-header">'
        '<h1>🔎 Fake News Detection System</h1>'
        '<p>Detect whether a news article is Real or Fake</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Input Section ──
    st.markdown('<p class="section-title">📝 Paste News Article</p>', unsafe_allow_html=True)
    default_text = st.session_state.get("sample_text", "")
    txt = st.text_area(
        "Enter the news headline or full article below:",
        value=default_text,
        height=200,
        label_visibility="collapsed",
        placeholder="Paste your news article here...",
    )
    threshold = st.slider("FAKE decision threshold", 0.05, 0.95, 0.50, 0.01)
    detect_clicked = st.button("🔍 Detect News Authenticity", type="primary", use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Prediction Result ──
    if detect_clicked and txt.strip():
        pipe, clf, vec = load_model()
        if pipe is None and (clf is None or vec is None):
            st.error("Model artifacts not found. Ensure you trained and saved files to `outputs/`.")
            st.stop()

        s = clean_text(txt)
        word_count = len(s.split())

        if pipe is not None:
            prob_fake = float(pipe.predict_proba([s])[0, 1])
        else:
            X = vec.transform([s])
            prob_fake = float(clf.predict_proba(X)[0, 1])

        label = "FAKE" if prob_fake >= threshold else "REAL"
        prob_real = 1 - prob_fake

        # Confidence level
        winning_prob = prob_fake if label == "FAKE" else prob_real
        if winning_prob >= 0.80:
            confidence = "High"
        elif winning_prob >= 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Short text warnings
        if word_count < 10:
            st.warning(
                f"Your text is very short ({word_count} words). "
                "The model works best with news articles (20+ words). "
                "The prediction below may not be reliable."
            )
        elif word_count < 20:
            st.info(
                f"Your text has only {word_count} words. "
                "For better accuracy, paste a longer news article or headline."
            )

        # Result glass card
        if label == "FAKE":
            glass_cls = "result-glass result-glass-fake"
            lbl_cls = "result-label result-label-fake"
            lbl_txt = "❌ FAKE NEWS"
        else:
            glass_cls = "result-glass result-glass-real"
            lbl_cls = "result-label result-label-real"
            lbl_txt = "✅ REAL NEWS"

        st.markdown(
            f'<div class="{glass_cls}">'
            f'<p class="section-title">🤖 Prediction</p>'
            f'<div class="{lbl_cls}">{lbl_txt}</div>'
            f'<div class="result-stats">'
            f'  <div class="result-stat-item">'
            f'    <div class="stat-label">Fake Probability</div>'
            f'    <div class="stat-value">{prob_fake:.1%}</div>'
            f'  </div>'
            f'  <div class="result-stat-item">'
            f'    <div class="stat-label">Real Probability</div>'
            f'    <div class="stat-value">{prob_real:.1%}</div>'
            f'  </div>'
            f'  <div class="result-stat-item">'
            f'    <div class="stat-label">Confidence</div>'
            f'    <div class="stat-value">{confidence}</div>'
            f'  </div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.progress(prob_fake, text=f"Fake: {prob_fake:.1%} | Threshold: {threshold:.0%}")
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Quick Test Glass Card ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">⚡ Quick Test</p>', unsafe_allow_html=True)
    sample_real = (
        "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, "
        "who voted this month for a huge expansion of the national debt to pay for tax cuts, "
        "called himself a fiscal conservative on Sunday and urged budget restraint in 2018."
    )
    sample_fake = (
        "BREAKING: Government exposed for hiding alien technology in secret underground base. "
        "Whistleblower reveals shocking documents that prove decades of cover-up. "
        "The mainstream media refuses to report the truth about what is really going on."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📗 Load Real Sample", use_container_width=True):
            st.session_state["sample_text"] = sample_real
            st.rerun()
    with col2:
        if st.button("📕 Load Fake Sample", use_container_width=True):
            st.session_state["sample_text"] = sample_fake
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(
        '<div class="footer">'
        'Powered by <strong>Machine Learning</strong> &nbsp;|&nbsp; TF-IDF + Logistic Regression Model'
        '</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

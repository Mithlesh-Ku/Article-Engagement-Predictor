"""app.py - Streamlit UI for the Article Engagement Predictor.
Run `python model_training.py` first, then `streamlit run app.py`.
"""
import json
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Engagement Predictor", page_icon="🔥", layout="centered")

st.markdown("""
<style>
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #ff7a18 0%, #af002d 100%); color: white;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: .3rem 0 0; opacity: .92;}
.card {padding: 1.1rem 1.3rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);}
</style>
""", unsafe_allow_html=True)

EXAMPLES = [
    "10 Advanced Python Tips Every Developer Should Know",
    "Release Notes for Docker Version 4.2",
    "5 React Hooks Mistakes You're Probably Making",
    "An Analysis of SQL Performance Characteristics",
]


@st.cache_resource
def load_artifacts():
    vec_path, model_path = Path("vectorizer.pkl"), Path("model.pkl")
    if not (vec_path.exists() and model_path.exists()):
        return None, None
    return joblib.load(vec_path), joblib.load(model_path)


def load_metrics():
    p = Path("metrics.json")
    return json.loads(p.read_text()) if p.exists() else None


vectorizer, model = load_artifacts()

st.markdown(
    '<div class="hero"><h1>🔥 Article Engagement Predictor</h1>'
    "<p>Estimate whether your article title will earn high likes &amp; retweets.</p></div>",
    unsafe_allow_html=True,
)

if model is None:
    st.error("Model files not found. Run `python model_training.py` first, then reload this page.")
    st.stop()

# Sidebar: model info
with st.sidebar:
    st.header("📊 Model info")
    m = load_metrics()
    if m:
        st.write(f"**Model:** {m['model']}")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']:.1%}")
        c2.metric("F1 (High)", f"{m['f1_high']:.2f}")
        if m.get("synthetic_data"):
            st.warning("Trained on **synthetic** data. Retrain with `--csv` on real "
                       "article data for meaningful predictions.")
    st.caption("Features: TF-IDF (1–2 grams, sublinear TF)")

# Example buttons set the text box value
if "title" not in st.session_state:
    st.session_state.title = ""
st.caption("Try an example:")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.title = ex

title = st.text_input("Article title", key="title",
                      placeholder="e.g. 10 Advanced Python Tips Every Developer Should Know")

if st.button("🚀 Predict engagement", type="primary", use_container_width=True):
    if not title.strip():
        st.warning("Please enter an article title first.")
    else:
        X = vectorizer.transform([title.strip()])
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        p_high = float(proba[list(model.classes_).index(1)])
        confidence = p_high if pred == 1 else 1 - p_high

        if pred == 1:
            st.success("🔥 **High Engagement Potential**")
        else:
            st.warning("⚠️ **Low Engagement Potential**")

        c1, c2 = st.columns(2)
        c1.metric("Model confidence", f"{confidence:.1%}")
        c2.metric("P(high engagement)", f"{p_high:.1%}")
        st.progress(p_high, text="Probability of high engagement")

        # Tell the user when the title shares nothing with the training vocabulary
        if X.nnz == 0:
            st.info("None of these words appeared in the training data, so this "
                    "prediction is essentially a guess.")

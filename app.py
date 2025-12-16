import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Will Your YikYak Yak?", page_icon="🔥")

@st.cache_resource
def load_pipeline():
    return joblib.load("artifacts/yikyak_pipeline.joblib")

pipe = load_pipeline()

st.title("Will Your YikYak Yak?")
st.write(
    "Hi! This is a demo app that employs a pipeline developed in a CSC 371 (Machine Learning) class final. " \
    "We  explored engagement prediction on Davidson College's YikYak data through several ML models. Our goal was to predict whether or not a post would receive high engagement (top 10% of upvotes). "
    "Our most successful model was a TF-IDF and Logistic Regression model that is employed on this app."
)  # :contentReference[oaicite:1]{index=1}

with st.expander("⚠️ A Brief Warning on Accuracy"):
    st.write(
        "This is a class project demo, not a guarantee your post will blow up. "
        "In our evaluation, the logistic regression model had ROC AUC ≈ 0.71, and it struggles with low precision/recall. Basically, it has trouble predicting the minority (high enagagement class). Plus, virality is completely randomized, so take that into account when yakking!"
    )  # :contentReference[oaicite:2]{index=2}

# Optional: include your paper in the repo as a file (recommended)
# Put it at: assets/research_paper.pdf
paper_path = Path("assets/research_paper.pdf")
if paper_path.exists():
    with paper_path.open("rb") as f:
        st.download_button(
            label="Download our research paper (PDF)",
            data=f,
            file_name="Predicting_Controversiality_and_Engagement_YikYak.pdf",
            mime="application/pdf",
        )

st.divider()

text = st.text_area("Type your YikYak draft:", height=140, placeholder="This dining hall food is actually criminal...")

if st.button("Predict") and text.strip():
    X = pd.DataFrame({"text": [text]})
    proba = float(pipe.predict_proba(X)[0, 1])

    st.metric("Viral probability", f"{proba:.2%}")
    st.progress(proba)

    # Fun tiered messaging
    if proba < 0.33:
        st.info("Low likelihood. This is probably a quiet yak. Try being more specific?")
    elif proba < 0.66:
        st.warning("Medium likelihood. Could flop, but then again...could soar. Add a hook or a question.")
    else:
        st.success("High likelihood!!!! this one might yak. Proceed responsibly :)")

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Will Your YikYak Yak?", page_icon="🔥")

@st.cache_resource
def load_pipeline():
    return joblib.load("artifacts/yikyak_pipeline.joblib")

pipe = load_pipeline()

st.title("Will Your YikYak Yak?")

text = st.text_area("Type a YikYak post:", height=140)

if st.button("Predict") and text.strip():
    X = pd.DataFrame({"text": [text]})
    proba = pipe.predict_proba(X)[0, 1]
    st.metric("Viral probability", f"{proba:.2%}")

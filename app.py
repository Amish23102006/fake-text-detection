"""Streamlit demo:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from src.predict import predict_text

st.set_page_config(page_title="Fake Text Detector", page_icon="🕵️")
st.title("AI vs Human Text Detector")
st.caption("Stylometric features + classical ML. A probabilistic signal, not proof of authorship.")


@st.cache_resource
def load():
    return joblib.load("models/best_model.joblib")


text = st.text_area("Paste text (50+ words works best)", height=220)
if st.button("Analyze") and text.strip():
    r = predict_text(load(), text)
    st.subheader(r["label"])
    st.progress(r["p_ai"], text=f"P(AI-generated) = {r['p_ai']:.2f}")
    st.dataframe(pd.DataFrame(r["features"], index=["value"]).T)
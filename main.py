import streamlit as st
from app.dashboard import run_dashboard
st.set_page_config(
    page_title="Disease Outbreak Prediction",
    page_icon="🦠",
    layout="wide"
)
run_dashboard()

import streamlit as st
import re

def inject_css():
    st.markdown("""
    <style>
    header[data-testid="stHeader"] { height: 0px; }
    footer { visibility: hidden; }
    .main .block-container {
      padding-top: 0.75rem;
      padding-bottom: 0.75rem;
      max-width: 1200px;
    }
    section[data-testid="stSidebar"] { width: 300px !important; }
    </style>
    """, unsafe_allow_html=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[:<>\"/\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:200]
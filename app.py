# app.py
import streamlit as st
from app_core import main
   

st.set_page_config(page_title="Quant AI Assistant", page_icon="🤖", layout="wide")


if __name__ == "__main__":
    main()
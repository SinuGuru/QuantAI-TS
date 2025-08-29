# app.py
import streamlit as st
import app_core
   

st.set_page_config(page_title="Quant AI Assistant", page_icon="🤖", layout="wide")


if __name__ == "__main__":
    app_core.main()
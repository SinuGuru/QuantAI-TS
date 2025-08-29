# app.py
import streamlit as st
from app_core import (
    init_session_state, init_db, inject_css, auth_gate,
    render_sidebar, render_topbar, render_chat_messages, response, main
)

st.set_page_config(page_title="Quant AI Assistant", page_icon="🤖", layout="wide")


if __name__ == "__main__":
    main()
import streamlit as st
from db import init_db
from auth import auth_gate
from sidebar import render_sidebar
from utils import inject_css

def init_session_state():
    from datetime import datetime
    defaults = {
        "authenticated": False,
        "user_id": None,
        "username": None,
        "user_role": None,
        "api_key": "",
        "client_initialized": False,
        "messages": [
            {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?"}
        ],
        "model": "gpt-4o",
        "conversation_name": f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}",
        "usage_stats": {"tokens": 0, "requests": 0, "cost": 0.0},
        "temperature": 0.7,
        "_pricing_map": None,
        "_pricing_warned": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    inject_css()
    init_session_state()
    conn = init_db()
    auth_gate(conn)
    render_sidebar(conn)
    # Main content is handled in Streamlit pages

if __name__ == "__main__":
    main()
import streamlit as st
from db import save_conversation_db, get_user_conversations, load_conversation_db, init_db
from datetime import datetime
import json

def new_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help you today?"}]
    st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}"

def save_conversation():
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        st.warning("Please log in before saving conversations.")
        return
    try:
        conversation_name = st.session_state.get("conversation_name", f"conv_{int(datetime.now().timestamp())}")
        if not conversation_name:
            conversation_name = f"conv_{int(datetime.now().timestamp())}"
        conn = init_db()
        save_conversation_db(conn, st.session_state["user_id"], conversation_name, st.session_state.messages)
        st.success("Conversation saved.")
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

def load_conversation(name: str):
    try:
        conn = init_db()
        msgs = load_conversation_db(conn, st.session_state["user_id"], name)
        if msgs is None:
            st.error("Conversation not found.")
            return
        st.session_state.messages = msgs
        st.session_state.conversation_name = name
        st.success("Conversation loaded.")
    except Exception as e:
        st.error(f"Failed to load conversation: {e}")

def get_user_conversations(conn, user_id):
    return get_user_conversations(conn, user_id)
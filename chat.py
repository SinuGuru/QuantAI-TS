import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

from db import (
    init_db,
    save_conversation_db,
    load_conversation_db,
    get_user_conversations as db_get_user_conversations,
)

logger = logging.getLogger(__name__)

def new_chat():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?"}
    ]
    st.session_state["conversation_name"] = f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}"

def save_conversation():
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        st.warning("Please log in before saving conversations.")
        return
    try:
        name = st.session_state.get("conversation_name") or f"conv_{int(datetime.utcnow().timestamp())}"
        conn = init_db()
        save_conversation_db(conn, st.session_state["user_id"], name, st.session_state["messages"])
        st.success("Conversation saved.")
    except Exception as e:
        logger.exception("Failed to save conversation")
        st.error(f"Failed to save conversation: {e}")

def load_conversation(name: str):
    try:
        conn = init_db()
        msgs = load_conversation_db(conn, st.session_state["user_id"], name)
        if msgs is None:
            st.error("Conversation not found.")
            return
        st.session_state["messages"] = msgs
        st.session_state["conversation_name"] = name
        st.success("Conversation loaded.")
    except Exception as e:
        logger.exception("Failed to load conversation")
        st.error(f"Failed to load conversation: {e}")

def get_user_conversations(conn, user_id: int):
    """Wrapper to the db function (no recursion)."""
    return db_get_user_conversations(conn, user_id)

# Minimal/fallback response()
def create_openai_client(api_key: str):
    if not api_key:
        return None
    try:
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            import openai as _openai  # type: ignore
            OpenAI = getattr(_openai, "OpenAI", None)  # type: ignore
        if OpenAI is None:
            return None
        client = OpenAI(api_key=api_key)
        return client
    except Exception:
        logger.exception("OpenAI client initialization failed")
        return None

def response(messages: List[Dict[str, Any]], model: str = "gpt-4o") -> str:
    api_key = st.session_state.get("api_key") or ""
    client = create_openai_client(api_key)

    if not messages:
        return "No messages provided."

    if client is None:
        # Simple fallback echo/summary
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content")
                break
        if not last_user:
            return "I'm ready — send a message."
        mock = f"(Mock reply) Received your message ({min(len(last_user),200)} chars). Provide an OpenAI API key in the sidebar for full responses."
        return mock

    # Attempt best-effort real call (handle SDK differences)
    try:
        try:
            resp = client.chat.completions.create(model=model, messages=[{"role": m["role"], "content": m["content"]} for m in messages])
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                msg = choices[0].message
                return getattr(msg, "content", "") or ""
        except Exception:
            try:
                resp = client.chat.create(model=model, messages=[{"role": m["role"], "content": m["content"]} for m in messages])
                c = getattr(resp, "choices", None)
                if c and len(c) > 0:
                    m = c[0]
                    return getattr(getattr(m, "message", m), "content", getattr(m, "text", "")) or ""
            except Exception:
                logger.exception("OpenAI call failed")
                return "OpenAI call failed. Check API key and model."
        return "No reply from OpenAI."
    except Exception as e:
        logger.exception("Unexpected error in response()")
        return f"Unexpected error: {e}"
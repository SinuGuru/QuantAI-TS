import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

from db import (
    init_db,
    save_conversation_db,
    load_conversation_db,
    get_user_conversations as db_get_user_conversations,
)

logger = logging.getLogger(__name__)

# --- OpenAI client helper (best-effort) ---
def create_openai_client(api_key: str):
    """
    Try to create an OpenAI client. Return None if not available or key missing.
    This is non-fatal; response() will fallback to a local mock reply.
    """
    if not api_key:
        return None
    try:
        # prefer the newer import style if available
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            import openai as _openai  # type: ignore
            OpenAI = getattr(_openai, "OpenAI", None)  # type: ignore

        if OpenAI is None:
            return None
        client = OpenAI(api_key=api_key)
        # quick check (may raise if invalid)
        try:
            client.models.list()
        except Exception:
            # don't fail app startup; invalid keys will be reported at call time
            pass
        return client
    except Exception:
        logger.exception("OpenAI client initialization failed")
        return None


# --- Conversation helpers ---
def new_chat():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?"}
    ]
    st.session_state["conversation_name"] = f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}"


def save_conversation():
    """Persist current conversation for the logged-in user."""
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
    """Load a named conversation for the current user into session state."""
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
    """Return conversations list from DB (wrapper to db layer)."""
    return db_get_user_conversations(conn, user_id)


# --- Minimal/fallback response() ---
def response(messages: List[Dict[str, Any]], model: str = "gpt-4o") -> str:
    """
    Send messages to OpenAI if available; otherwise return a simple fallback reply.
    Always returns assistant content as string.
    """
    api_key = st.session_state.get("api_key") or ""
    client = create_openai_client(api_key)

    # Basic validation
    if not messages:
        return "No messages provided."

    # If no client, return a safe mock reply so UI remains functional
    if client is None:
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content")
                break
        if not last_user:
            return "I'm ready — send a message."
        # Very simple echo/summary fallback
        mock = f"(Mock reply) I received your message of {min(len(last_user), 200)} chars. Provide an OpenAI API key in the sidebar for full responses."
        return mock

    # If an OpenAI client exists, attempt a real chat completion (best-effort).
    try:
        # Attempt to use the client in a generic way. Different OpenAI SDKs have different signatures;
        # we try a common pattern and fallback to a simple echo on unexpected failures.
        try:
            resp = client.chat.completions.create(model=model, messages=[{"role": m["role"], "content": m["content"]} for m in messages])
            choice = getattr(resp, "choices", None)
            if choice and len(choice) > 0:
                msg = choice[0].message
                return getattr(msg, "content", "") or ""
        except Exception:
            # Try older API shape
            try:
                resp = client.chat.create(model=model, messages=[{"role": m["role"], "content": m["content"]} for m in messages])
                # resp.choices[0].message.content or resp.choices[0].text
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
import streamlit as st
from db import init_db, get_user_conversations as db_get_user_conversations
from chat import new_chat, save_conversation, load_conversation

def logout():
    keys_to_clear = ["authenticated", "user_id", "username", "user_role", "messages", "conversation_name", "client_initialized"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

def render_sidebar(conn):
    with st.sidebar:
        st.subheader("Account")
        st.markdown(f"Signed in as: **{st.session_state.get('username','Guest')}**")
        if st.button("🚪 Logout", on_click=logout, use_container_width=True):
            pass
        st.markdown("---")

        st.subheader("Conversations")
        if st.button("🆕 Start New Chat", use_container_width=True):
            new_chat()
            st.experimental_rerun()
        if st.button("💾 Save Conversation", use_container_width=True):
            save_conversation()

        user_id = st.session_state.get("user_id")
        conv_rows = []
        if conn and user_id:
            try:
                conv_rows = db_get_user_conversations(conn, user_id)
            except Exception:
                conv_rows = []
        conv_names = [r[0] for r in conv_rows]
        if conv_names:
            sel = st.selectbox("Load conversation", options=conv_names, index=0)
            if sel and sel != st.session_state.get("conversation_name"):
                load_conversation(sel)
                st.experimental_rerun()
        else:
            st.caption("No saved conversations yet.")
        st.markdown("---")

        st.subheader("Usage")
        # compact usage placeholders
        total_tokens = st.session_state.get("usage_stats", {}).get("tokens", 0)
        total_requests = st.session_state.get("usage_stats", {}).get("requests", 0)
        total_cost = st.session_state.get("usage_stats", {}).get("cost", 0.0)
        col1, col2, col3 = st.columns(3)
        col1.metric("Tok", f"{total_tokens:,}")
        col2.metric("Req", f"{total_requests}")
        col3.metric("$", f"{total_cost:.2f}")
        st.markdown("---")

        st.subheader("OpenAI API")
        api_text = st.text_input("API Key", type="password", value=st.session_state.get("api_key", ""))
        if api_text and api_text != st.session_state.get("api_key", ""):
            st.session_state.api_key = api_text
            st.success("API key configured.")
        st.markdown("---")

        st.markdown("### ⚙️ Settings")
        token = st.text_input("API Token", type="password", value=st.session_state.get("api_token", ""), help="Enter your API token here.")
        if st.button("Update Token", key="update_token_sidebar"):
            st.session_state["api_token"] = token
            st.success("Token updated!")
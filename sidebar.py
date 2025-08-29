import streamlit as st
from chat import new_chat, save_conversation, get_user_conversations, load_conversation
from analytics import display_usage_stats_block

def logout():
    keys_to_clear = ["authenticated", "user_id", "username", "user_role", "messages", "conversation_name", "client_initialized"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def render_sidebar(conn):
    with st.sidebar:
        st.subheader("Account")
        st.markdown(f"Signed in as: **{st.session_state.get('username','Guest')}**")
        st.button("🚪 Logout", on_click=logout, use_container_width=True)
        st.markdown("---")

        st.subheader("Conversations")
        if st.button("🆕 Start New Chat", use_container_width=True):
            new_chat()
            st.rerun()
        if st.button("💾 Save Conversation", use_container_width=True):
            save_conversation()

        conv_rows = get_user_conversations(conn, st.session_state.get("user_id") or -1)
        conv_names = [r[0] for r in conv_rows]
        if conv_names:
            sel = st.radio("Load", options=conv_names, label_visibility="collapsed")
            if sel and sel != st.session_state.get("conversation_name"):
                load_conversation(sel)
                st.rerun()
        else:
            st.caption("No saved conversations yet.")
        st.markdown("---")

        st.subheader("Usage")
        display_usage_stats_block()
        st.markdown("---")

        st.subheader("OpenAI API")
        api_text = st.text_input("API Key", type="password", value=st.session_state.get("api_key", ""))
        if api_text and api_text != st.session_state.get("api_key", ""):
            st.session_state.api_key = api_text
            st.success("API key configured.")

        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        st.markdown("### 🔑 Token Management")
        token = st.text_input(
            "API Token",
            type="password",
            value=st.session_state.get("api_token", ""),
            help="Enter your API token here."
        )
        if st.button("Update Token", key="update_token_sidebar"):
            st.session_state["api_token"] = token
            st.success("Token updated!")

        st.markdown("---")
        st.markdown("### 📄 Navigation")
        st.info("Use the tabs in the main area to switch between workflows.")
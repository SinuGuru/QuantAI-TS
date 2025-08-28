# app.py
import streamlit as st
from app_core import (
    init_session_state, init_db, inject_css, auth_gate,
    render_sidebar, render_topbar, render_chat_messages, response
)

st.set_page_config(page_title="Quant AI Assistant", page_icon="🤖", layout="wide")

def main():
    init_session_state()
    conn = init_db()
    inject_css()

    # Auth gate across all pages
    auth_gate(conn)

    # Sidebar and top bar
    render_sidebar(conn)
    render_topbar("Quant AI Assistant")

    # Chat area
    if not st.session_state.get("api_key"):
        st.warning("Please enter your OpenAI API key in the sidebar to interact with the model.")
        return

    st.markdown(f"**Conversation:** {st.session_state.conversation_name} • Model: {st.session_state.model}")
    render_chat_messages()
    prompt = st.chat_input("Type a message...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            _ = response(st.session_state.messages, st.session_state.model)
        st.rerun()

if __name__ == "__main__":
    main()
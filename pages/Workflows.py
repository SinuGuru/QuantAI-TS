# pages/Workflows.py
import streamlit as st
import pandas as pd
from db import init_db
from auth import auth_gate
from sidebar import render_sidebar
from chat import response, new_chat
from utils import inject_css

st.set_page_config(page_title="Workflows • Quant AI", page_icon="🛠️", layout="wide")

def ensure_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?"}
        ]
    if "model" not in st.session_state:
        st.session_state["model"] = "gpt-4o"
    if "conversation_name" not in st.session_state:
        from datetime import datetime
        st.session_state["conversation_name"] = f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}"

def main():
    inject_css()

    conn = init_db()
    # auth_gate will stop rendering until user logs in
    auth_gate(conn)
    render_sidebar(conn)

    ensure_session()

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                min-width: 270px !important;
                width: 320px !important;
                max-width: 400px !important;
            }
            @media (max-width: 600px) {
                section[data-testid="stSidebar"] {
                    min-width: 150px !important;
                    width: 90vw !important;
                    max-width: 98vw !important;
                }
            }
            .topbar { padding: .5rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="topbar"><h2>🛠️ Workflows</h2></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Code Review", "📊 Data Analysis", "💬 Chat"])

    # Code Review tab
    with tab1:
        st.markdown("### Code Review")
        code_blob = st.text_area("Paste code to review", height=200, help="Paste your code here for AI review.")
        if st.button("Review Code", key="review_code"):
            if not code_blob.strip():
                st.warning("Provide code to review.")
            else:
                st.session_state.messages.append({"role": "user", "content": f"Please review this code:\n\n```{code_blob}```"})
                with st.spinner("Analyzing..."):
                    _ = response(st.session_state.messages, st.session_state.get("model", "gpt-4o"))
                st.experimental_rerun()

    # Data Analysis tab
    with tab2:
        st.markdown("### Data Analysis")
        uploaded = st.file_uploader("CSV file", type=["csv"], help="Upload a CSV file for analysis.")
        question = st.text_input("Question about your data", help="Ask a question about your uploaded data.")
        if st.button("Analyze", key="analyze_data"):
            if not uploaded or not question.strip():
                st.warning("Provide CSV and a question.")
            else:
                try:
                    df = pd.read_csv(uploaded)
                    csv_text = df.to_csv(index=False)
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Analyze the following data:\n\n```csv\n{csv_text}\n```\nQuestion: {question}"
                    })
                    with st.spinner("Analyzing..."):
                        _ = response(st.session_state.messages, st.session_state.get("model", "gpt-4o"))
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

    # Chat tab
    with tab3:
        st.markdown("### 💬 Chat")
        for msg in st.session_state["messages"]:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**AI:** {content}")

        user_input = st.text_input("Type your message...", key="chat_input")
        col1, col2, col3 = st.columns([6,1,1])
        with col2:
            if st.button("Send", key="send_chat"):
                if user_input.strip():
                    st.session_state["messages"].append({"role": "user", "content": user_input})
                    with st.spinner("AI is thinking..."):
                        ai_reply = response(st.session_state["messages"], st.session_state.get("model", "gpt-4o"))
                    st.session_state["messages"].append({"role": "assistant", "content": ai_reply})
                    st.experimental_rerun()
        with col3:
            if st.button("New Chat", key="new_chat_btn"):
                new_chat()
                st.experimental_rerun()

    # small footer/navigation hint
    st.info("Use the sidebar to save/load conversations and configure your OpenAI API key.")

if __name__ == "__main__":
    main()
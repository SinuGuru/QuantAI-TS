# pages/Workflows.py
import streamlit as st
import pandas as pd
from chat import response
from utils import inject_css

st.set_page_config(page_title="Workflows • Quant AI", page_icon="🛠️", layout="wide")

def main():
    inject_css()

    # Responsive sidebar width
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="topbar"><h2>🛠️ Workflows</h2></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Code Review", "📊 Data Analysis", "💬 Chat"])

    with tab1:
        st.markdown("### Code Review")
        code_blob = st.text_area("Paste code to review", height=200, help="Paste your code here for AI review.")
        if st.button("Review Code", key="review_code"):
            if not code_blob.strip():
                st.warning("Provide code to review.")
            else:
                st.session_state.messages.append({"role": "user", "content": f"Please review this code:\n\n```{code_blob}```"})
                try:
                    st.toast("Code sent to chat for review.", icon="🔎")
                except Exception:
                    st.success("Code sent to chat for review.")
                with st.spinner("Analyzing..."):
                    _ = response(st.session_state.messages, st.session_state.model)
                st.rerun()

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
                    try:
                        st.toast("Data sent to chat for analysis.", icon="📊")
                    except Exception:
                        st.success("Data sent to chat for analysis.")
                    with st.spinner("Analyzing..."):
                        _ = response(st.session_state.messages, st.session_state.model)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

    with tab3:
        st.markdown("### 💬 Chat")
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?"}
            ]

        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")

        user_input = st.text_input("Type your message...", key="chat_input")
        if st.button("Send", key="send_chat"):
            if user_input.strip():
                st.session_state["messages"].append({"role": "user", "content": user_input})
                with st.spinner("AI is thinking..."):
                    ai_reply = response(st.session_state["messages"], st.session_state.get("model", "gpt-4o"))
                st.session_state["messages"].append({"role": "assistant", "content": ai_reply})
                st.experimental_rerun()

if __name__ == "__main__":
    main()
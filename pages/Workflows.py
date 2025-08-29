# pages/Workflows.py
import pandas as pd
import streamlit as st
from app_core import (
    init_session_state, init_db, inject_css, auth_gate,
    render_sidebar, render_topbar, response
)

st.set_page_config(page_title="Workflows • Quant AI", page_icon="🛠️", layout="wide")

def main():
    init_session_state()
    conn = init_db()
    inject_css()
    auth_gate(conn)

    # Use centralized sidebar
    render_sidebar(conn)

    render_topbar("Workflows")

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                min-width: 350px;
                width: 350px;
                max-width: 400px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📝 Code Review", "📊 Data Analysis"])

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

    st.info("Switch back to the Chat page to continue the conversation.")

if __name__ == "__main__":
    main()
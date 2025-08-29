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

    render_sidebar(conn)
    render_topbar("Workflows")

    st.markdown("### Code Review")
    code_blob = st.text_area("Paste code to review", height=200)
    if st.button("Review Code"):
        if not code_blob.strip():
            st.warning("Provide code to review.")
        else:
            st.session_state.messages.append({"role": "user", "content": f"Please review this code:\n\n```{code_blob}```"})
            try:
                st.toast("Code sent to chat for review.", icon="🔎")
            except Exception:
                st.success("Code sent to chat for review.")
            # Optionally trigger a response immediately:
            with st.spinner("Analyzing..."):
                _ = response(st.session_state.messages, st.session_state.model)
            st.rerun()

    st.markdown("---")
    st.markdown("### Data Analysis")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    question = st.text_input("Question about your data")
    if st.button("Analyze"):
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
                # Optionally trigger response immediately:
                with st.spinner("Analyzing..."):
                    _ = response(st.session_state.messages, st.session_state.model)
                st.rerun()
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    st.info("Switch back to the Chat page to continue the conversation.")

if __name__ == "__main__":
    main()
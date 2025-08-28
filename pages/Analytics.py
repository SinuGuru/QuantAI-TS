# pages/Analytics.py
import streamlit as st
import plotly.express as px
from app_core import (
    init_session_state, init_db, inject_css, auth_gate,
    render_sidebar, render_topbar, load_usage_dataframe
)

st.set_page_config(page_title="Analytics • Quant AI", page_icon="📈", layout="wide")

def main():
    init_session_state()
    conn = init_db()
    inject_css()
    auth_gate(conn)

    render_sidebar(conn)
    render_topbar("Usage Analytics")

    st.markdown("### Usage Analytics")
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("Please log in to view your analytics.")
        return

    try:
        df_usage = load_usage_dataframe(user_id)
        if not df_usage.empty:
            df_usage["date"] = px.to_datetime(df_usage["date"])
            st.plotly_chart(px.line(df_usage, x="date", y="tokens", title="Daily Token Usage"), use_container_width=True)
            st.plotly_chart(px.line(df_usage, x="date", y="cost", title="Estimated Daily Cost"), use_container_width=True)
            st.plotly_chart(px.line(df_usage, x="date", y="requests", title="Daily API Requests"), use_container_width=True)
        else:
            st.info("No usage data available yet. Start a conversation to see your analytics!")
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")

if __name__ == "__main__":
    main()
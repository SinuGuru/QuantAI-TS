import streamlit as st
from db import init_db
from typing import Optional, Tuple

def get_usage_totals(user_id: Optional[int]) -> Tuple[int, int, float]:
    try:
        if not user_id:
            return 0, 0, 0.0
        conn = init_db()
        c = conn.cursor()
        c.execute("SELECT SUM(tokens), SUM(requests), SUM(cost) FROM usage WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        total_tokens = int(result[0] or 0)
        total_requests = int(result[1] or 0)
        total_cost = float(result[2] or 0.0)
        return total_tokens, total_requests, total_cost
    except Exception:
        return 0, 0, 0.0

def display_usage_stats_block():
    total_tokens, total_requests, total_cost = get_usage_totals(st.session_state.get("user_id"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Tok", f"{total_tokens:,}")
    col2.metric("Req", f"{total_requests}")
    col3.metric("$", f"{total_cost:.2f}")
import streamlit as st
import re
import time

def inject_css():
    st.markdown("""
    <style>
    header[data-testid="stHeader"] { height: 0px; }
    footer { visibility: hidden; }
    .main .block-container {
      padding-top: 0.75rem;
      padding-bottom: 0.75rem;
      max-width: 1200px;
    }
    section[data-testid="stSidebar"] { width: 300px !important; }
    </style>
    """, unsafe_allow_html=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[:<>\"/\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:200]

def safe_rerun():
    """
    Compatibility wrapper for reloading the app/page.
    Tries st.experimental_rerun(), falls back to setting a query param or toggling a sentinel and calling st.stop().
    """
    try:
        st.experimental_rerun()
        return
    except Exception:
        # experimental_rerun may be unavailable in some Streamlit builds
        try:
            # try to force a client reload via query params
            st.experimental_set_query_params(_refresh=str(time.time()))
            return
        except Exception:
            # last resort: set a sentinel and stop execution so user can manually refresh
            st.session_state["_needs_rerun"] = not st.session_state.get("_needs_rerun", False)
            st.stop()
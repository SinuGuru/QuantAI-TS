import sys
import tempfile
import os

# ensure project root on sys.path when running from tests folder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def run():
    import db, auth, chat, sidebar, app_core, utils
    print("Imported modules OK")

    # DB init smoke
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    try:
        conn = db.init_db(db_path=tmp_db.name)  # type: ignore
        print("DB init OK:", tmp_db.name)
    finally:
        conn.close()

    # auth functions smoke
    h = auth.hash_password("TestPass123!")
    ok = auth.verify_password(h, "TestPass123!")
    print("Password hashing/verify OK:", ok)

    # chat fallback response (no API key)
    import streamlit as st
    # ensure session_state is available
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    msg = chat.response([{"role":"user","content":"hello"}], model="gpt-4o")
    print("Chat response (fallback) OK:", msg[:80])

    # sidebar render function import-check
    try:
        # render_sidebar requires a conn; pass None for import smoke
        sidebar.render_sidebar(None)
    except Exception as e:
        print("Sidebar render call raised (expected in non-UI env):", type(e).__name__)

    print("SMOKE TESTS FINISHED")

if __name__ == "__main__":
    run()
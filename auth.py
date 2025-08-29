import os
import binascii
import hashlib
import streamlit as st
import time
import re
from db import create_user, get_user_by_username

def hash_password(password: str) -> str:
    try:
        import bcrypt
        salt = bcrypt.gensalt(rounds=12)
        return "bcrypt$" + bcrypt.hashpw(password.encode("utf-8"), salt).decode()
    except Exception:
        salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return f"pbkdf2${binascii.hexlify(salt).decode()}${binascii.hexlify(dk).decode()}"

def verify_password(stored: str, provided: str) -> bool:
    try:
        if stored.startswith("bcrypt$"):
            _, hash_str = stored.split("$", 1)
            import bcrypt
            return bcrypt.checkpw(provided.encode("utf-8"), hash_str.encode())
        if stored.startswith("pbkdf2$"):
            _, salt_hex, hash_hex = stored.split("$")
            salt = binascii.unhexlify(salt_hex)
            dk = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, 200_000)
            return binascii.hexlify(dk).decode() == hash_hex
        parts = stored.split("$")
        if len(parts) == 2:
            salt_hex, hash_hex = parts
            salt = binascii.unhexlify(salt_hex)
            dk = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, 200_000)
            return binascii.hexlify(dk).decode() == hash_hex
        return False
    except Exception:
        return False

def authenticate_user(conn, username: str, password: str):
    user = get_user_by_username(conn, username)
    if not user:
        return None
    if verify_password(user["password_hash"], password):
        return {"id": user["id"], "username": user["username"], "role": user["role"]}
    return None

def password_strength(password: str) -> str:
    import re
    if len(password) < 8:
        return "Weak"
    if not re.search(r"[A-Z]", password) or not re.search(r"[0-9]", password):
        return "Medium"
    if re.search(r"[!@#$%^&*]", password):
        return "Strong"
    return "Medium"

def auth_gate(conn):
    if st.session_state.get("authenticated", False):
        return
    st.markdown(
        """
        <style>
            .block-container { padding-top: 0.2rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex;justify-content:center;align-items:flex-start;">',
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<h1 style="text-align:center;">🤖 Quant AI — Sign in</h1>', unsafe_allow_html=True)
        with st.form("auth_form"):
            mode = st.radio("Mode", ["Login", "Register"])
            username = st.text_input("Username", key="auth_username")
            password = st.text_input("Password", type="password", key="auth_password")
            if mode == "Register":
                st.caption(f"Password strength: **{password_strength(password)}**")
            submitted = st.form_submit_button("Continue")
            if submitted:
                if not username or not password:
                    st.error("Provide username and password")
                else:
                    if mode == "Register":
                        try:
                            _user = create_user(conn, username, password)
                            st.success("Account created. Please login.")
                        except ValueError:
                            st.error("Username already taken.")
                        except Exception as e:
                            st.error(f"Registration failed: {e}")
                    else:
                        user = authenticate_user(conn, username, password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.user_id = user["id"]
                            st.session_state.username = user["username"]
                            st.session_state.user_role = user["role"]
                            st.success(f"Welcome, {username}!")
                            time.sleep(0.4)
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
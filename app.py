import streamlit as st
import openai
import os
import json
import requests
import pandas as pd
import plotly.express as px
import time
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from streamlit_lottie import st_lottie
import re
import sqlite3
import hashlib
import binascii
from pathlib import Path

# --- CONFIG / CONSTANTS ---
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are Enterprise AI Assistant 2025, a professional assistant with knowledge up to December 2025. "
    "You are an expert in using tools to perform tasks."
)
MAX_CONTEXT_MESSAGES = 12
DB_PATH = Path("app.db")

# --- UTILITIES ---

def sanitize_filename(name: str) -> str:
    """Sanitize filenames by removing / replacing characters invalid on many systems."""
    name = re.sub(r"[:<>\"/\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:200]

def safe_json_dumps(obj: Any) -> str:
    """Safely dump a JSON object, handling non-serializable types gracefully."""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

@st.cache_data
def load_lottieurl(url: str) -> Optional[Dict[str, Any]]:
    """Lazy-load Lottie JSON from a URL (cached)."""
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        st.warning(f"Could not load Lottie animation from {url}")
        return None

# --- DATABASE / AUTH HELPERS ---

@st.cache_resource
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize and return a SQLite connection (cached resource)."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user'
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
        UNIQUE(user_id, name),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        tokens INTEGER DEFAULT 0,
        requests INTEGER DEFAULT 0,
        cost REAL DEFAULT 0.0,
        FOREIGN KEY(user_id) REFERENCES users(id),
        UNIQUE(user_id, date)
    );
    """)
    conn.commit()
    return conn

def hash_password(password: str) -> str:
    """Hash a password using a random salt and PBKDF2."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{binascii.hexlify(salt).decode()}${binascii.hexlify(dk).decode()}"

def verify_password(stored: str, provided: str) -> bool:
    """Verify a provided password against a stored hash."""
    try:
        salt_hex, hash_hex = stored.split("$")
        salt = binascii.unhexlify(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, 200_000)
        return binascii.hexlify(dk).decode() == hash_hex
    except Exception:
        return False

def create_user(conn: sqlite3.Connection, username: str, password: str, role: str = "user") -> Dict[str, Any]:
    """Create a new user in the database."""
    pw_hash = hash_password(password)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  (username, pw_hash, role))
        conn.commit()
        return {"id": c.lastrowid, "username": username, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError("User exists")

def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    """Retrieve a user by their username."""
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}

def authenticate_user(conn: sqlite3.Connection, username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user with a username and password."""
    user = get_user_by_username(conn, username)
    if not user:
        return None
    if verify_password(user["password_hash"], password):
        return {"id": user["id"], "username": user["username"], "role": user["role"]}
    return None

def save_conversation_db(conn: sqlite3.Connection, user_id: int, name: str, messages: list) -> None:
    """Save or update a conversation in the database."""
    c = conn.cursor()
    payload = json.dumps(messages, ensure_ascii=False, indent=2)
    try:
        c.execute("""
        INSERT INTO conversations (user_id, name, data) VALUES (?, ?, ?)
        ON CONFLICT(user_id, name) DO UPDATE SET data=excluded.data, created_at=(DATETIME('now'))
        """, (user_id, name, payload))
        conn.commit()
    except sqlite3.OperationalError:
        c.execute("DELETE FROM conversations WHERE user_id = ? AND name = ?", (user_id, name))
        c.execute("INSERT INTO conversations (user_id, name, data) VALUES (?, ?, ?)", (user_id, name, payload))
        conn.commit()

def get_user_conversations(conn: sqlite3.Connection, user_id: int):
    """Retrieve all conversations for a specific user."""
    c = conn.cursor()
    c.execute("SELECT name, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    return c.fetchall()

def load_conversation_db(conn: sqlite3.Connection, user_id: int, name: str):
    """Load a specific conversation from the database."""
    c = conn.cursor()
    c.execute("SELECT data FROM conversations WHERE user_id = ? AND name = ? LIMIT 1", (user_id, name))
    row = c.fetchone()
    if not row:
        return None
    return json.loads(row[0])

def add_usage(conn: sqlite3.Connection, user_id: int, tokens: int = 0, requests: int = 1, cost: float = 0.0):
    """Add or update usage statistics for a user on the current day."""
    today = date.today().isoformat()
    c = conn.cursor()
    try:
        c.execute("""
        INSERT INTO usage (user_id, date, tokens, requests, cost)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, date) DO UPDATE SET
          tokens = tokens + excluded.tokens,
          requests = requests + excluded.requests,
          cost = cost + excluded.cost
        """, (user_id, today, tokens, requests, cost))
    except sqlite3.OperationalError:
        c.execute("SELECT id FROM usage WHERE user_id = ? AND date = ?", (user_id, today))
        if c.fetchone():
            c.execute("UPDATE usage SET tokens = tokens + ?, requests = requests + ?, cost = cost + ? WHERE user_id = ? AND date = ?",
                      (tokens, requests, cost, user_id, today))
        else:
            c.execute("INSERT INTO usage (user_id, date, tokens, requests, cost) VALUES (?, ?, ?, ?, ?)",
                      (user_id, today, tokens, requests, cost))
    conn.commit()

# --- SESSION STATE HELPERS ---

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "authenticated": False,
        "user_id": None,
        "username": None,
        "user_role": None,
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "client_initialized": False,
        "messages": [
            {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?"}
        ],
        "model": "gpt-4o",
        "conversation_name": f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}",
        "usage_stats": {"tokens": 0, "requests": 0, "cost": 0.0},
        "temperature": 0.7,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def logout():
    """Clear session keys for a clean logout."""
    keys_to_clear = ["authenticated", "user_id", "username", "user_role", "messages", "conversation_name", "client_initialized"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# --- OPENAI CLIENT SETUP ---

@st.cache_resource
def create_openai_client(api_key: str):
    """Create an OpenAI client resource."""
    if not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        st.session_state["client_initialized"] = True
        return client
    except Exception as e:
        st.error(f"Failed to create OpenAI client: {e}")
        st.session_state["client_initialized"] = False
        return None

# --- LOCAL TOOL FUNCTIONS ---

def web_search(query: str) -> str:
    results = [
        {"title": "AI Trends 2025: GPT-5", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-5's improved reasoning ..."},
        {"title": "OpenAI Releases GPT-5", "url": "https://example.com/gpt5-update", "snippet": "GPT-5 features enhanced multimodal..."},
    ]
    return safe_json_dumps({"query": query, "results": results})

def code_review(code: str, language: str = "Python") -> str:
    review = {
        "language": language,
        "summary": "Basic static review completed.",
        "recommendations": [
            "Add docstrings and type annotations.",
            "Add tests for edge cases.",
            "Consider splitting long functions into smaller ones.",
        ],
    }
    if "TODO" in code or "FIXME" in code:
        review["recommendations"].append("Remove TODO/FIXME comments or address them.")
    return safe_json_dumps(review)

def data_analysis(query: str, data: str) -> str:
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(data))
        insights = []
        insights.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            summary = df[numeric_cols].describe().to_dict()
            insights.append({"numeric_summary": summary})
        if "sales" in query.lower():
            insights.append("Detected 'sales' in query: consider grouping by date/region to analyze trends.")
        return safe_json_dumps({"query": query, "insights": insights})
    except Exception as e:
        return safe_json_dumps({"error": str(e)})

def get_current_datetime() -> str:
    return datetime.utcnow().isoformat() + "Z"

FUNCTIONS = [
    {
        "name": "web_search",
        "description": "Searches the web for up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "code_review",
        "description": "Reviews a block of code for errors, style, and efficiency.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "data_analysis",
        "description": "Analyzes CSV data and answers questions about it.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "data": {"type": "string"}},
            "required": ["query", "data"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time in ISO format.",
        "parameters": {"type": "object", "properties": {}},
    },
]

LOCAL_TOOL_MAP = {
    "web_search": web_search,
    "code_review": code_review,
    "data_analysis": data_analysis,
    "get_current_datetime": get_current_datetime,
}

# --- MODEL / TOOL INVOCATION (function-calling pattern) ---

def _normalize_message(msg: Any) -> Dict[str, Any]:
    """Normalize a message into a dict with role/content and include tool_calls when present."""
    if isinstance(msg, dict):
        return msg
    try:
        normalized = {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", str(msg)),
        }
        # Preserve tool_calls for assistant messages so UI can render them
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            try:
                normalized["tool_calls"] = [
                    {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", None),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                        },
                    }
                    for tc in tool_calls
                ]
            except Exception:
                pass
        return normalized
    except Exception:
        return {"role": "assistant", "content": str(msg)}

def response(messages: List[Dict[str, Any]], model: str) -> str:
    """Generate a response using OpenAI function-calling."""
    if not st.session_state.get("api_key"):
        return "Please enter your OpenAI API key in the Settings tab."

    client = create_openai_client(st.session_state.api_key)
    if client is None:
        return "OpenAI client could not be initialized. Check your API key."

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[-MAX_CONTEXT_MESSAGES:]

    try:
        response_obj = client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=[{"type": "function", "function": f} for f in FUNCTIONS],
            tool_choice="auto",
            temperature=st.session_state.get("temperature", 0.7),
        )
        st.session_state.usage_stats["requests"] += 1

        choice = response_obj.choices[0]
        message = choice.message

        tool_calls = message.tool_calls
        if tool_calls:
            st.session_state.messages.append(message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments_str = tool_call.function.arguments

                try:
                    args = json.loads(arguments_str)
                except json.JSONDecodeError:
                    args = {"raw_arguments": arguments_str}

                tool_func = LOCAL_TOOL_MAP.get(function_name)
                if not tool_func:
                    func_result = f"Error: Tool '{function_name}' not implemented."
                else:
                    try:
                        func_result = tool_func(**args)
                    except Exception as e:
                        func_result = f"Error when executing tool '{function_name}': {e}"

                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": func_result
                })

            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages[-MAX_CONTEXT_MESSAGES:]
            response2 = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=st.session_state.get("temperature", 0.7),
            )
            st.session_state.usage_stats["requests"] += 1
            final_message = response2.choices[0].message
            st.session_state.messages.append(final_message)

            if st.session_state.get("user_id"):
                conn = init_db()
                tokens = response2.usage.total_tokens
                add_usage(conn, st.session_state["user_id"], tokens=tokens)

            return final_message.content or ""
        else:
            st.session_state.messages.append(message)

            if st.session_state.get("user_id"):
                conn = init_db()
                tokens = response_obj.usage.total_tokens
                add_usage(conn, st.session_state["user_id"], tokens=tokens)

            return message.content or ""
    except openai.APIError as e:
        return f"OpenAI API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# --- UI RENDER HELPERS ---

def render_chat_messages():
    """Render all messages with avatars and compact tool outputs."""
    avatar_map = {"assistant": "🤖", "user": "🧑", "tool": "🧰"}
    for raw in st.session_state.messages:
        msg = _normalize_message(raw)
        role = msg.get("role", "user")
        avatar = avatar_map.get(role, "")
        with st.chat_message(role, avatar=avatar):
            content = msg.get("content", "")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    func_name = tc.get("function", {}).get("name", "unknown")
                    with st.expander(f"Tool requested: {func_name}", icon="🧰"):
                        st.code(tc.get("function", {}).get("arguments", ""), language="json")
                if content:
                    st.markdown(f"<div class='chat-bubble assistant'>{content}</div>", unsafe_allow_html=True)
            elif role == "tool":
                name = msg.get("name", "tool")
                with st.expander(f"Tool output: {name}", expanded=False):
                    try:
                        parsed = json.loads(content)
                        st.json(parsed)
                    except Exception:
                        st.code(content)
            else:
                st.markdown(f"<div class='chat-bubble {role}'>{content}</div>", unsafe_allow_html=True)

def display_usage_stats():
    """Display usage statistics in a clean card format."""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT SUM(tokens), SUM(requests), SUM(cost) FROM usage WHERE user_id = ?", (st.session_state.get("user_id"),))
    result = c.fetchone()

    # FIX: Safely handle the case where a user has no usage data
    if result and result[0] is not None:
        total_tokens, total_requests, total_cost = result
    else:
        total_tokens, total_requests, total_cost = 0, 0, 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{int(total_tokens):,}</div><div class='stat-label'>Total Tokens</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{int(total_requests)}</div><div class='stat-label'>API Requests</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>${total_cost:.2f}</div><div class='stat-label'>Estimated Cost</div></div>", unsafe_allow_html=True)

# --- CONVERSATION SAVE / LOAD (DB-backed) ---

def save_conversation():
    """Save the current conversation."""
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        st.warning("Please log in before saving conversations.")
        return
    try:
        conversation_name = sanitize_filename(st.session_state.get("conversation_name", f"conv_{int(time.time())}"))
        if not conversation_name:
            conversation_name = f"conv_{int(time.time())}"
        conn = init_db()
        save_conversation_db(conn, st.session_state["user_id"], conversation_name, st.session_state.messages)
        st.success("Conversation saved.")
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

def load_conversation(name: str):
    """Load a conversation from the database."""
    try:
        conn = init_db()
        msgs = load_conversation_db(conn, st.session_state["user_id"], name)
        if msgs is None:
            st.error("Conversation not found.")
            return
        st.session_state.messages = msgs
        st.session_state.conversation_name = name
        st.success("Conversation loaded.")
    except Exception as e:
        st.error(f"Failed to load conversation: {e}")

def new_chat():
    """Start a new chat session."""
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help you today?"}]
    st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}"

# --- APP LAYOUT / MAIN ---

def main():
    st.set_page_config(page_title="Quant AI Assistant", page_icon="🤖", layout="wide")
    init_session_state()
    conn = init_db()

    # --- Compact, responsive CSS for Streamlit Cloud ---
    st.markdown("""
    <style>
    /* Hide Streamlit default header/footer for a cleaner app look */
    header[data-testid="stHeader"] { height: 0px; }
    footer { visibility: hidden; }

    /* App container width and padding */
    .main .block-container { 
      padding-top: 0.75rem; 
      padding-bottom: 0.75rem; 
      max-width: 1200px; 
    }

    /* Sidebar width on desktop */
    section[data-testid="stSidebar"] { width: 300px !important; }

    /* Gradient title */
    .main-header { 
      font-size: 1.8rem; 
      font-weight: 700; 
      margin: .25rem 0 .35rem 0;
      background: -webkit-linear-gradient(135deg, #667eea, #764ba2); 
      -webkit-background-clip: text; 
      -webkit-text-fill-color: transparent;
    }

    /* Sticky topbar */
    .topbar {
      position: sticky;
      top: 0;
      z-index: 1000;
      background: #0f172a;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: .6rem .75rem;
      margin-bottom: .6rem;
    }

    /* Chat message bubble styles */
    .chat-bubble {
      padding: .6rem .75rem; 
      border-radius: 10px; 
      border: 1px solid rgba(0,0,0,0.05);
    }
    .chat-bubble.assistant { background: #f8fafc; }
    .chat-bubble.user { background: #eef2ff; }
    .chat-bubble.tool { background: #fefce8; }

    /* Tool output expander style */
    .tool-expander > div[role="button"] {
      background: #fff7ed; border: 1px solid #fed7aa;
    }

    /* Stat cards */
    .stat-card { 
      background:white; border-radius:8px; padding:.8rem; 
      box-shadow:0 2px 6px rgba(0,0,0,0.05); text-align:center; 
    }
    .stat-value { font-size:1.2rem; font-weight:700; }
    .stat-label { color:#718096; }

    /* Mobile tweaks */
    @media (max-width: 640px) {
      section[data-testid="stSidebar"] { width: 260px !important; }
      .main .block-container { padding-left:.5rem; padding-right:.5rem; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Lottie animations are commented out to prevent network-related warnings
    # lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_uz5cqu1b.json")
    # lottie_robot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sk5h1kfn.json")

    # --- AUTH SCREEN ---
    if not st.session_state.get("authenticated", False):
        st.markdown('<div style="display:flex;justify-content:center;align-items:center;height:80vh">', unsafe_allow_html=True)
        with st.container():
            st.markdown('<h1 style="text-align:center;">🤖 Quant AI — Sign in</h1>', unsafe_allow_html=True)
            # if lottie_ai:
            #     st_lottie(lottie_ai, height=180)
            with st.form("auth_form"):
                mode = st.radio("Mode", ["Login", "Register"])
                username = st.text_input("Username", key="auth_username")
                password = st.text_input("Password", type="password", key="auth_password")
                submitted = st.form_submit_button("Continue")
                if submitted:
                    if not username or not password:
                        st.error("Provide username and password")
                    else:
                        if mode == "Register":
                            try:
                                user = create_user(conn, username, password)
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
        return

    # --- SIDEBAR (Account, Conversations, Usage, API) ---
    with st.sidebar:
        st.subheader("Account")
        st.markdown(f"Signed in as: **{st.session_state.get('username','Guest')}**")
        st.button("🚪 Logout", on_click=logout, use_container_width=True)
        st.markdown("---")

        # Conversations
        st.subheader("Conversations")
        if st.button("🆕 Start New Chat", use_container_width=True):
            new_chat()
            st.rerun()
        if st.button("💾 Save Conversation", use_container_width=True):
            save_conversation()

        conv_rows = get_user_conversations(conn, st.session_state.get("user_id") or -1)
        conv_names = [r[0] for r in conv_rows]
        if conv_names:
            sel = st.radio("Load", options=conv_names, label_visibility="collapsed")
            if sel and sel != st.session_state.get("conversation_name"):
                load_conversation(sel)
                st.rerun()
        else:
            st.caption("No saved conversations yet.")

        st.markdown("---")

        # Compact usage metrics
        st.subheader("Usage")
        try:
            conn2 = init_db()
            c2 = conn2.cursor()
            c2.execute("SELECT SUM(tokens), SUM(requests), SUM(cost) FROM usage WHERE user_id = ?", (st.session_state.get("user_id"),))
            result = c2.fetchone()
            total_tokens = int(result[0] or 0)
            total_requests = int(result[1] or 0)
            total_cost = float(result[2] or 0.0)
        except Exception:
            total_tokens, total_requests, total_cost = 0, 0, 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Tokens", f"{total_tokens:,}")
        m2.metric("Reqs", f"{total_requests}")
        m3.metric("Cost", f"${total_cost:.2f}")

        st.markdown("---")

        # API settings moved here from Settings tab
        st.subheader("OpenAI API")
        api_text = st.text_input("API Key", type="password", value=st.session_state.get("api_key", ""))
        if api_text and api_text != st.session_state.get("api_key", ""):
            st.session_state.api_key = api_text
            client = create_openai_client(api_text)
            if client:
                st.success("API key configured.")
            else:
                st.error("Failed to initialize client with the provided API key.")

    # --- STICKY TOPBAR (Title, Model, Temp, Primary Actions) ---
    with st.container():
        st.markdown('<div class="topbar">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
        with c1:
            st.markdown("<div class='main-header'>Quant AI Assistant</div>", unsafe_allow_html=True)
            st.caption(f"Welcome, {st.session_state.username or 'Guest'} • {st.session_state.conversation_name}")
        with c2:
            model_options = ["gpt-5", "gpt-5-pro", "gpt-5-mini", "gpt-4o", "gpt-4-turbo-preview"]
            current_model = st.session_state.get("model", "gpt-4o")
            st.selectbox(
                "Model",
                model_options,
                index=model_options.index(current_model) if current_model in model_options else model_options.index("gpt-4o"),
                key="model",
                label_visibility="collapsed",
            )
        with c3:
            st.slider("Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.7), key="temperature", label_visibility="collapsed")
        with c4:
            colA, colB = st.columns(2)
            with colA:
                if st.button("🆕 New", use_container_width=True):
                    new_chat()
                    st.rerun()
            with colB:
                st.button("💾 Save", on_click=save_conversation, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # We now use the sticky topbar title; keep main area clean.
    st.title("")

    # --- MAIN CONTENT TABS (Chat, Workflows, Analytics) ---
    tab_chat, tab_tools, tab_analytics = st.tabs(["Chat", "Workflows", "Analytics"])

    with tab_chat:
        if not st.session_state.get("api_key"):
            st.warning("Please enter your OpenAI API key in the sidebar to interact with the model.")
        else:
            st.markdown(f"**Conversation:** {st.session_state.conversation_name} • Model: {st.session_state.model}")
            render_chat_messages()
            prompt = st.chat_input("Type a message...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("Thinking..."):
                    _ = response(st.session_state.messages, st.session_state.model)
                st.rerun()

    with tab_tools:
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
                    st.session_state.messages.append({"role": "user", "content": f"Analyze the following data:\n\n```csv\n{csv_text}\n```\nQuestion: {question}"})
                    try:
                        st.toast("Data sent to chat for analysis.", icon="📊")
                    except Exception:
                        st.success("Data sent to chat for analysis.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

    with tab_analytics:
        st.markdown("### Usage Analytics")
        if st.session_state.get("user_id"):
            try:
                conn_analytics = init_db()
                query = "SELECT date, tokens, requests, cost FROM usage WHERE user_id = ? ORDER BY date ASC"
                df_usage = pd.read_sql_query(query, conn_analytics, params=(st.session_state["user_id"],))

                if not df_usage.empty:
                    df_usage["date"] = pd.to_datetime(df_usage["date"])
                    st.plotly_chart(px.line(df_usage, x="date", y="tokens", title="Daily Token Usage"), use_container_width=True)
                    st.plotly_chart(px.line(df_usage, x="date", y="cost", title="Estimated Daily Cost"), use_container_width=True)
                    st.plotly_chart(px.line(df_usage, x="date", y="requests", title="Daily API Requests"), use_container_width=True)
                else:
                    st.info("No usage data available yet. Start a conversation to see your analytics!")
            except Exception as e:
                st.error(f"Failed to load analytics: {e}")
        else:
            st.warning("Please log in to view your analytics.")

if __name__ == "__main__":
    main()

# app_core.py
import os
import re
import json
import time
import binascii
import hashlib
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple, Callable

import pandas as pd
import requests
import streamlit as st

# OpenAI client import (v1)
try:
    from openai import OpenAI
    from openai import APIError  # type: ignore
except Exception:  # fallback to older import style if needed
    import openai  # type: ignore
    OpenAI = getattr(openai, "OpenAI", None)
    APIError = getattr(openai, "APIError", Exception)  # type: ignore

# Optional animation import; not used by default but kept for compatibility
try:
    from streamlit_lottie import st_lottie  # noqa: F401
except Exception:
    pass

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIG / CONSTANTS ---
SYSTEM_PROMPT = (
    "You are Enterprise AI Assistant 2025, a professional assistant with knowledge up to December 2025. "
    "You are an expert in using tools to perform tasks."
)
MAX_CONTEXT_MESSAGES = 12  # Note: this limits messages, not tokens.
DB_PATH = Path("app.db")
PRICING_ENV_VAR = "OPENAI_MODEL_PRICING_JSON"

# Centralize model options used in UI
MODEL_OPTIONS: List[str] = ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4-turbo-preview"]


# --- PRICING ---
@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float  # price per 1K input tokens
    output_per_1k: float  # price per 1K output tokens


def _parse_pricing_map(raw: str) -> Dict[str, ModelPricing]:
    """
    Parse a JSON string into a {model: ModelPricing} mapping.
    Expected JSON format:
      {"model-id":{"in":0.005,"out":0.015}, ...}
    """
    try:
        obj = json.loads(raw)
        out: Dict[str, ModelPricing] = {}
        for model, prices in obj.items():
            inp = float(prices.get("in", 0.0))
            outp = float(prices.get("out", 0.0))
            out[model] = ModelPricing(input_per_1k=inp, output_per_1k=outp)
        return out
    except Exception as e:
        logger.warning("Failed to parse pricing JSON: %s", e)
        return {}


def _load_pricing_map() -> Dict[str, ModelPricing]:
    """
    Load pricing map from:
      1) st.secrets[OPENAI_MODEL_PRICING_JSON] if present
      2) env var OPENAI_MODEL_PRICING_JSON
    Returns empty dict if not configured.
    """
    # from st.secrets if available
    try:
        secrets_val = st.secrets.get(PRICING_ENV_VAR)  # type: ignore[attr-defined]
        if secrets_val:
            return _parse_pricing_map(secrets_val)
    except Exception:
        pass

    # from env var
    env_val = os.getenv(PRICING_ENV_VAR, "")
    if env_val:
        return _parse_pricing_map(env_val)

    return {}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate cost for a single call based on model pricing and token counts.
    Returns 0.0 if pricing for the model is not configured.
    """
    pricing_map: Dict[str, ModelPricing] = st.session_state.get("_pricing_map")  # type: ignore
    if pricing_map is None:
        pricing_map = _load_pricing_map()
        st.session_state["_pricing_map"] = pricing_map  # cache for session

    mp = pricing_map.get(model)
    if not mp:
        # Show a one-time info about missing pricing
        if not st.session_state.get("_pricing_warned", False):
            st.info(
                f"No pricing configured for model '{model}'. "
                f"Set {PRICING_ENV_VAR} (JSON) to enable cost estimates."
            )
            st.session_state["_pricing_warned"] = True
        return 0.0

    cost = (prompt_tokens / 1000.0) * mp.input_per_1k + (completion_tokens / 1000.0) * mp.output_per_1k
    return round(cost, 6)


# --- UTILITIES ---
def inject_css() -> None:
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

    .main-header {
      font-size: 1.8rem;
      font-weight: 700;
      margin: .25rem 0 .35rem 0;
      background: -webkit-linear-gradient(135deg, #667eea, #764ba2);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
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

    .chat-bubble {
      padding: .6rem .75rem;
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.05);
    }
    .chat-bubble.assistant { background: #f8fafc; }
    .chat-bubble.user { background: #eef2ff; }
    .chat-bubble.tool { background: #fefce8; }

    .tool-expander > div[role="button"] {
      background: #fff7ed; border: 1px solid #fed7aa;
    }

    .stat-card {
      background:white; border-radius:8px; padding:.8rem;
      box-shadow:0 2px 6px rgba(0,0,0,0.05); text-align:center;
    }
    .stat-value { font-size:1.2rem; font-weight:700; }
    .stat-label { color:#718096; }

    @media (max-width: 640px) {
      section[data-testid="stSidebar"] { width: 260px !important; }
      .main .block-container { padding-left:.5rem; padding-right:.5rem; }
    }
    </style>
    """, unsafe_allow_html=True)


def sanitize_filename(name: str) -> str:
    """Sanitize a filename-like string for safe storage."""
    name = re.sub(r"[:<>\"/\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:200]


def safe_json_dumps(obj: Any) -> str:
    """Dump to JSON safely; fallback to str on failure."""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


@st.cache_data
def load_lottieurl(url: str) -> Optional[Dict[str, Any]]:
    """Load Lottie animation JSON from a URL (cached)."""
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
    """Initialize or open the SQLite database and ensure tables exist."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user'
    );""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
        UNIQUE(user_id, name),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );""")
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
    );""")
    conn.commit()
    return conn


def _hash_password_pbkdf2(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2${binascii.hexlify(salt).decode()}${binascii.hexlify(dk).decode()}"


def _hash_password_bcrypt(password: str) -> Optional[str]:
    try:
        import bcrypt  # type: ignore
        salt = bcrypt.gensalt(rounds=12)
        return "bcrypt$" + bcrypt.hashpw(password.encode("utf-8"), salt).decode()
    except Exception:
        return None


def hash_password(password: str) -> str:
    """
    Hash a password. Prefer bcrypt if available; fallback to PBKDF2.
    Prefix format used to allow verify_password to detect algorithm.
    """
    b = _hash_password_bcrypt(password)
    if b:
        return b
    return _hash_password_pbkdf2(password)


def verify_password(stored: str, provided: str) -> bool:
    """Verify password against stored hash supporting both bcrypt and PBKDF2."""
    try:
        if stored.startswith("bcrypt$"):
            _, hash_str = stored.split("$", 1)
            import bcrypt  # type: ignore
            return bcrypt.checkpw(provided.encode("utf-8"), hash_str.encode())
        if stored.startswith("pbkdf2$"):
            _, salt_hex, hash_hex = stored.split("$")
            salt = binascii.unhexlify(salt_hex)
            dk = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, 200_000)
            return binascii.hexlify(dk).decode() == hash_hex
        # Backward compatibility if no prefix was stored (legacy)
        parts = stored.split("$")
        if len(parts) == 2:
            salt_hex, hash_hex = parts
            salt = binascii.unhexlify(salt_hex)
            dk = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, 200_000)
            return binascii.hexlify(dk).decode() == hash_hex
        return False
    except Exception:
        return False


def create_user(conn: sqlite3.Connection, username: str, password: str, role: str = "user") -> Dict[str, Any]:
    """Create a new user in DB."""
    pw_hash = hash_password(password)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", (username, pw_hash, role))
        conn.commit()
        return {"id": c.lastrowid, "username": username, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError("User exists")


def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    """Fetch a user by username."""
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}


def authenticate_user(conn: sqlite3.Connection, username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validate credentials and return user object if valid."""
    user = get_user_by_username(conn, username)
    if not user:
        return None
    if verify_password(user["password_hash"], password):
        return {"id": user["id"], "username": user["username"], "role": user["role"]}
    return None


def save_conversation_db(conn: sqlite3.Connection, user_id: int, name: str, messages: List[Dict[str, Any]]) -> None:
    """Insert or update a conversation for the user."""
    c = conn.cursor()
    payload = json.dumps(messages, ensure_ascii=False, indent=2)
    try:
        c.execute("""
        INSERT INTO conversations (user_id, name, data) VALUES (?, ?, ?)
        ON CONFLICT(user_id, name) DO UPDATE SET data=excluded.data, created_at=(DATETIME('now'))
        """, (user_id, name, payload))
        conn.commit()
    except sqlite3.OperationalError:
        # Fallback path for older SQLite versions
        c.execute("DELETE FROM conversations WHERE user_id = ? AND name = ?", (user_id, name))
        c.execute("INSERT INTO conversations (user_id, name, data) VALUES (?, ?, ?)", (user_id, name, payload))
        conn.commit()


def get_user_conversations(conn: sqlite3.Connection, user_id: int) -> List[Tuple[str, str]]:
    """Get (name, created_at) for conversations of a user."""
    c = conn.cursor()
    c.execute("SELECT name, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    return c.fetchall()


def load_conversation_db(conn: sqlite3.Connection, user_id: int, name: str) -> Optional[List[Dict[str, Any]]]:
    """Load a conversation's message list for a user."""
    c = conn.cursor()
    c.execute("SELECT data FROM conversations WHERE user_id = ? AND name = ? LIMIT 1", (user_id, name))
    row = c.fetchone()
    if not row:
        return None
    return json.loads(row[0])


def add_usage(conn: sqlite3.Connection, user_id: int, tokens: int = 0, requests: int = 1, cost: float = 0.0) -> None:
    """Aggregate and persist daily usage for a user."""
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
        # Fallback path for older SQLite versions
        c.execute("SELECT id FROM usage WHERE user_id = ? AND date = ?", (user_id, today))
        if c.fetchone():
            c.execute("UPDATE usage SET tokens = tokens + ?, requests = requests + ?, cost = cost + ? WHERE user_id = ? AND date = ?",
                      (tokens, requests, cost, user_id, today))
        else:
            c.execute("INSERT INTO usage (user_id, date, tokens, requests, cost) VALUES (?, ?, ?, ?, ?)",
                      (user_id, today, tokens, requests, cost))
    conn.commit()


# --- SESSION STATE ---
def init_session_state() -> None:
    """Initialize default session state keys."""
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
        "_pricing_map": None,
        "_pricing_warned": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def logout() -> None:
    """Clear auth-related session keys and rerun."""
    keys_to_clear = ["authenticated", "user_id", "username", "user_role", "messages", "conversation_name", "client_initialized"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


# --- OPENAI CLIENT ---
@st.cache_resource
def create_openai_client(api_key: str):
    """Create and validate an OpenAI client instance."""
    if not api_key:
        return None
    try:
        if OpenAI is None:
            raise RuntimeError("OpenAI client class not available. Update the openai package.")
        client = OpenAI(api_key=api_key)
        # Simple call to validate key
        client.models.list()
        st.session_state["client_initialized"] = True
        return client
    except Exception as e:
        st.error(f"Failed to create OpenAI client: {e}")
        st.session_state["client_initialized"] = False
        logger.exception("OpenAI client initialization failed.")
        return None


# --- LOCAL TOOL FUNCTIONS & SCHEMA ---
def tool_web_search(query: str) -> str:
    results = [
        {"title": "AI Trends 2025: GPT-5", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-5's improved reasoning ..."},
        {"title": "OpenAI Releases GPT-5", "url": "https://example.com/gpt5-update", "snippet": "GPT-5 features enhanced multimodal..."},
    ]
    return safe_json_dumps({"query": query, "results": results})


def tool_code_review(code: str, language: str = "Python") -> str:
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


def tool_data_analysis(query: str, data: str) -> str:
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(data))
        insights: List[Any] = []
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


def tool_get_current_datetime() -> str:
    return datetime.utcnow().isoformat() + "Z"


FUNCTIONS = [
    {"name": "web_search", "description": "Searches the web for up-to-date information.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "code_review", "description": "Reviews a block of code for errors, style, and efficiency.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "language": {"type": "string"}}, "required": ["code"]}},
    {"name": "data_analysis", "description": "Analyzes CSV data and answers questions about it.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "data": {"type": "string"}}, "required": ["query", "data"]}},
    {"name": "get_current_datetime", "description": "Returns the current date and time in ISO format.",
     "parameters": {"type": "object", "properties": {}}}
]
LOCAL_TOOL_MAP: Dict[str, Callable[..., str]] = {
    "web_search": tool_web_search,
    "code_review": tool_code_review,
    "data_analysis": tool_data_analysis,
    "get_current_datetime": tool_get_current_datetime,
}


# --- MODEL / TOOL INVOCATION ---
def _normalize_message(msg: Any) -> Dict[str, Any]:
    """Normalize messages into dict form suitable for rendering."""
    if isinstance(msg, dict):
        return msg
    try:
        normalized = {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", str(msg)),
        }
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
                    } for tc in tool_calls
                ]
            except Exception:
                pass
        return normalized
    except Exception:
        return {"role": "assistant", "content": str(msg)}


def _record_usage(user_id: Optional[int], model: str, usage_obj: Any) -> Tuple[int, float]:
    """
    Extract tokens from OpenAI usage object and record usage in DB and session.
    Returns (total_tokens, cost).
    """
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens) or 0)

    cost = estimate_cost(model, prompt_tokens, completion_tokens)

    # Update session stats
    st.session_state.usage_stats["tokens"] += total_tokens
    st.session_state.usage_stats["cost"] += cost

    if user_id:
        conn = init_db()
        add_usage(conn, user_id, tokens=total_tokens, requests=1, cost=cost)

    return total_tokens, cost


def response(messages: List[Dict[str, Any]], model: str) -> str:
    """Send messages to OpenAI, handle tool calls, track usage, and return assistant content."""
    if not st.session_state.get("api_key"):
        return "Please enter your OpenAI API key in the sidebar."

    client = create_openai_client(st.session_state.api_key)
    if client is None:
        return "OpenAI client could not be initialized. Check your API key."

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[-MAX_CONTEXT_MESSAGES:]

    try:
        # First call: allow tools
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
            # Save assistant msg with tool calls
            st.session_state.messages.append(message)

            # Execute tool calls locally
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                try:
                    args = json.loads(arguments_str)
                    if not isinstance(args, dict):
                        args = {"raw_arguments": arguments_str}
                except json.JSONDecodeError:
                    args = {"raw_arguments": arguments_str}

                tool_func = LOCAL_TOOL_MAP.get(function_name)
                if not tool_func:
                    func_result = f"Error: Tool '{function_name}' not implemented."
                else:
                    try:
                        func_result = tool_func(**args)
                    except Exception as e:
                        logger.exception("Tool '%s' execution error", function_name)
                        func_result = f"Error when executing tool '{function_name}': {e}"

                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": func_result
                })

            # Second call: respond with tool outputs
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages[-MAX_CONTEXT_MESSAGES:]
            response2 = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=st.session_state.get("temperature", 0.7),
            )
            st.session_state.usage_stats["requests"] += 1
            final_message = response2.choices[0].message
            st.session_state.messages.append(final_message)

            # Record usage for both calls
            try:
                if getattr(response_obj, "usage", None):
                    _record_usage(st.session_state.get("user_id"), model, response_obj.usage)
                if getattr(response2, "usage", None):
                    _record_usage(st.session_state.get("user_id"), model, response2.usage)
            except Exception:
                logger.exception("Failed to record usage.")

            return final_message.content or ""
        else:
            # No tool calls
            st.session_state.messages.append(message)
            try:
                if getattr(response_obj, "usage", None):
                    _record_usage(st.session_state.get("user_id"), model, response_obj.usage)
            except Exception:
                logger.exception("Failed to record usage.")
            return message.content or ""

    except APIError as e:
        logger.exception("OpenAI API error")
        return f"OpenAI API error: {e}"
    except Exception as e:
        logger.exception("Unexpected error in response()")
        return f"Unexpected error: {e}"


# --- UI HELPERS ---
def render_chat_messages() -> None:
    """Render chat messages with basic avatars and tool call visualization."""
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


def get_usage_totals(user_id: Optional[int]) -> Tuple[int, int, float]:
    """Return total tokens, total requests, total cost for a user."""
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
        logger.exception("Failed to get usage totals")
        return 0, 0, 0.0


def display_usage_stats_block() -> None:
    """Show a small usage summary block."""
    total_tokens, total_requests, total_cost = get_usage_totals(st.session_state.get("user_id"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Tokens", f"{total_tokens:,}")
    col2.metric("Reqs", f"{total_requests}")
    col3.metric("Cost", f"${total_cost:.2f}")


def save_conversation() -> None:
    """Persist the current conversation under its name for the logged-in user."""
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
        logger.exception("Failed to save conversation")
        st.error(f"Failed to save conversation: {e}")


def load_conversation(name: str) -> None:
    """Load a named conversation for the current user into session state."""
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
        logger.exception("Failed to load conversation")
        st.error(f"Failed to load conversation: {e}")


def new_chat() -> None:
    """Reset session messages to begin a new chat."""
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help you today?"}]
    st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}"


def render_sidebar(conn: sqlite3.Connection) -> None:
    """Render the application sidebar UI."""
    with st.sidebar:
        st.subheader("Account")
        st.markdown(f"Signed in as: **{st.session_state.get('username','Guest')}**")
        st.button("🚪 Logout", on_click=logout, use_container_width=True)
        st.markdown("---")

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

        st.subheader("Usage")
        display_usage_stats_block()
        st.markdown("---")

        st.subheader("OpenAI API")
        api_text = st.text_input("API Key", type="password", value=st.session_state.get("api_key", ""))
        if api_text and api_text != st.session_state.get("api_key", ""):
            st.session_state.api_key = api_text
            client = create_openai_client(api_text)
            if client:
                st.success("API key configured.")
            else:
                st.error("Failed to initialize client with the provided API key.")

        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        # Token management section
        st.markdown("### 🔑 Token Management")
        token = st.text_input(
            "API Token",
            type="password",
            value=st.session_state.get("api_token", ""),
            help="Enter your API token here."
        )
        if st.button("Update Token", key="update_token_sidebar"):
            st.session_state["api_token"] = token
            st.success("Token updated!")

        st.markdown("---")
        st.markdown("### 📄 Navigation")
        st.info("Use the tabs in the main area to switch between workflows.")

        # Add any additional sidebar controls here as needed


def render_topbar(title: str) -> None:
    """Render the top bar with title, model selector, temperature, and controls."""
    with st.container():
        st.markdown('<div class="topbar">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
        with c1:
            st.markdown(f"<div class='main-header'>{title}</div>", unsafe_allow_html=True)
            st.caption(f"Welcome, {st.session_state.username or 'Guest'} • {st.session_state.conversation_name}")
        with c2:
            current_model = st.session_state.get("model", "gpt-4o")
            st.selectbox(
                "Model",
                MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(current_model) if current_model in MODEL_OPTIONS else MODEL_OPTIONS.index("gpt-4o"),
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
    st.title("")  # Keep main area clean


def auth_gate(conn: sqlite3.Connection) -> None:
    """Block main UI until the user is authenticated."""
    if st.session_state.get("authenticated", False):
        return
    st.markdown('<div style="display:flex;justify-content:center;align-items:center;height:80vh">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<h1 style="text-align:center;">🤖 Quant AI — Sign in</h1>', unsafe_allow_html=True)
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
                            _user = create_user(conn, username, password)
                            st.success("Account created. Please login.")
                        except ValueError:
                            st.error("Username already taken.")
                        except Exception as e:
                            logger.exception("Registration failed")
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


# --- ANALYTICS HELPER (used on Analytics page) ---
def load_usage_dataframe(user_id: Optional[int]) -> pd.DataFrame:
    """Load daily usage rows for a user as a DataFrame."""
    if not user_id:
        return pd.DataFrame(columns=["date", "tokens", "requests", "cost"])
    conn = init_db()
    query = "SELECT date, tokens, requests, cost FROM usage WHERE user_id = ? ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=(user_id,))
    # Ensure types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("tokens", "requests"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "cost" in df.columns:
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0).astype(float)
    return df
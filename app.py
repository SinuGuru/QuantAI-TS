# fixed_neuralink_app.py
import streamlit as st
import openai
import os
import json
import requests
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from streamlit_lottie import st_lottie
import re

# --- CONFIG / CONSTANTS ---
CONVERSATIONS_DIR = "conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
SYSTEM_PROMPT = (
    "You are Enterprise AI Assistant 2025, a professional assistant with knowledge up to December 2025. "
    "You are an expert in using tools to perform tasks."
)
MAX_CONTEXT_MESSAGES = 12  # keep context bounded for token limits

# --- UTILITIES ---

def sanitize_filename(name: str) -> str:
    """Sanitize filenames by removing / replacing characters invalid on many systems."""
    name = re.sub(r"[:<>\"/\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:200]

def safe_json_dumps(obj: Any) -> str:
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
        return None

# --- SESSION STATE HELPERS ---

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "authenticated": False,
        "user_id": None,
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
    """Clear session keys that represent the current user session (safe logout)."""
    keys_to_clear = ["authenticated", "user_id", "user_role", "messages", "conversation_name", "client_initialized"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# --- OPENAI CLIENT SETUP ---

@st.cache_resource
def create_openai_client(api_key: str):
    """Create an OpenAI client resource. Use Streamlit cached resource for reuse."""
    if not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        # Optionally test by listing available models once
        # client.models.list()
        return client
    except Exception as e:
        st.error(f"Failed to create OpenAI client: {e}")
        return None

# --- TOOL FUNCTIONS (local "functions" that model can call) ---

def web_search(query: str) -> str:
    # Placeholder: in production you'd run a real web search (Bing/Google API/Enterprise crawl).
    results = [
        {"title": "AI Trends 2025: GPT-5", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-5's improved reasoning ..."},
        {"title": "OpenAI Releases GPT-5", "url": "https://example.com/gpt5-update", "snippet": "GPT-5 features enhanced multimodal..."},
    ]
    return safe_json_dumps({"query": query, "results": results})

def code_review(code: str, language: str = "Python") -> str:
    # A simple static review template; replace with a more powerful analyzer if desired.
    review = {
        "language": language,
        "summary": "Basic static review completed.",
        "recommendations": [
            "Add docstrings and type annotations.",
            "Add tests for edge cases.",
            "Consider splitting long functions into smaller ones.",
        ],
    }
    # Quick lint-style checks:
    if "TODO" in code or "FIXME" in code:
        review["recommendations"].append("Remove TODO/FIXME comments or address them.")
    return safe_json_dumps(review)

def data_analysis(query: str, data: str) -> str:
    # Expect data is CSV text; we attempt to provide a quick insight.
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(data))
        insights = []
        insights.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        # Example quick check: numeric columns summary
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            summary = df[numeric_cols].describe().to_dict()
            insights.append({"numeric_summary": summary})
        # Try to match keywords in the query for canned responses
        if "sales" in query.lower():
            insights.append("Detected 'sales' in query: consider grouping by date/region to analyze trends.")
        return safe_json_dumps({"query": query, "insights": insights})
    except Exception as e:
        return safe_json_dumps({"error": str(e)})

def get_current_datetime() -> str:
    return datetime.utcnow().isoformat() + "Z"

# Convert to "functions" descriptors for the model
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

def response(messages: List[Dict[str, Any]], model: str) -> str:
    """
    Generate a response using OpenAI function-calling:
    - Send messages + functions to the model.
    - If the model requests a function, call it, append the function result as a message with role "tool",
      then call the model again to get the final assistant message.
    """
    if not st.session_state.get("api_key"):
        return "Please enter your OpenAI API key in the Settings tab."

    client = create_openai_client(st.session_state.api_key)
    if client is None:
        return "OpenAI client could not be initialized. Check your API key."

    # Build the API messages (system + last N messages)
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[-MAX_CONTEXT_MESSAGES:]

    try:
        # First call: ask model; allow it to request calling a tool (function)
        response_obj = client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=[{"type": "function", "function": f} for f in FUNCTIONS], # Use the correct 'tools' parameter
            tool_choice="auto", # Use the correct 'tool_choice' parameter
            temperature=st.session_state.get("temperature", 0.7),
        )
        st.session_state.usage_stats["requests"] += 1
        
        choice = response_obj.choices[0]
        message = choice.message # assistant message (may include tool_calls)
        tool_calls = message.tool_calls
        
        # If model requested a tool (function):
        if tool_calls:
            # Append the assistant message that requested the tool to the session messages
            # Note: The raw message object from the API is now a ToolCall object.
            # We must convert it to a dictionary for Streamlit's chat history.
            st.session_state.messages.append(message)

            # Re-call the model with the tool output
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments_str = tool_call.function.arguments or "{}"
                
                try:
                    args = json.loads(arguments_str)
                except json.JSONDecodeError:
                    args = {"raw_arguments": arguments_str}

                # Call the corresponding local tool function
                tool_func = LOCAL_TOOL_MAP.get(function_name)
                if not tool_func:
                    func_result = f"Error: Tool '{function_name}' not implemented."
                else:
                    try:
                        func_result = tool_func(**args)
                    except Exception as e:
                        func_result = f"Error when executing tool '{function_name}': {e}"

                # Append function result as a message with role "tool"
                st.session_state.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": func_result
                })

            # Re-call the model to produce a final assistant response using the tool output
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages[-MAX_CONTEXT_MESSAGES:]
            response2 = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=st.session_state.get("temperature", 0.7),
            )
            st.session_state.usage_stats["requests"] += 1
            final_choice = response2.choices[0]
            final_message = final_choice.message
            st.session_state.messages.append(final_message) # Append final response
            return final_message.content or ""
        else:
            # Normal assistant reply (no function required)
            st.session_state.messages.append(message)
            return message.content or ""
    except openai.APIError as e:
        return f"OpenAI API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# --- UI RENDER HELPERS ---

def render_chat_messages():
    """Render all messages in the session with better formatting for code and JSON."""
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        with st.chat_message(role):
            content = msg.get("content", "")
            # If assistant included tool_calls, render them
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.tool_calls:
                    st.markdown(f"**Tool request:** `{tc.function.name}` with args:")
                    try:
                        st.code(json.dumps(json.loads(tc.function.arguments), indent=2), language="json")
                    except json.JSONDecodeError:
                        st.code(tc.function.arguments)
                if content:
                    st.markdown(content)
            elif role == "tool":
                name = msg.get("name", "tool")
                st.info(f"📦 Tool output from `{name}`")
                try:
                    parsed = json.loads(content)
                    st.json(parsed)
                except Exception:
                    st.code(content)
            else:
                st.markdown(content)

def display_usage_stats():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{st.session_state.usage_stats['tokens']:,}</div><div class='stat-label'>Total Tokens</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{st.session_state.usage_stats['requests']}</div><div class='stat-label'>API Requests</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-card'><div class='stat-value'>${st.session_state.usage_stats['cost']:.2f}</div><div class='stat-label'>Estimated Cost</div></div>", unsafe_allow_html=True)

# --- CONVERSATION SAVE / LOAD ---

def save_conversation():
    if not st.session_state.get("authenticated") or not st.session_state.get("user_id"):
        st.warning("Please log in before saving conversations.")
        return
    try:
        conversation_name = sanitize_filename(st.session_state.get("conversation_name", f"conv_{int(time.time())}"))
        filename = f"{st.session_state['user_id']}-{conversation_name}.json"
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        
        # Save messages in a JSON-serializable format
        messages_to_save = []
        for m in st.session_state.messages:
            if hasattr(m, "model_dump"):
                messages_to_save.append(m.model_dump())
            else:
                messages_to_save.append(m)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(messages_to_save, f, indent=2, ensure_ascii=False)
        st.success("Conversation saved.")
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

def load_conversation(filename: str):
    filepath = os.path.join(CONVERSATIONS_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            msgs = json.load(f)
            st.session_state.messages = msgs
            st.session_state.conversation_name = filename.replace(f"{st.session_state.user_id}-", "").replace(".json", "")
            st.success("Conversation loaded.")
    except Exception as e:
        st.error(f"Failed to load conversation: {e}")

def new_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help you today?"}]
    st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H-%M')}"

# --- APP LAYOUT / MAIN ---

def main():
    st.set_page_config(page_title="NeuraLink AI Assistant", page_icon="🤖", layout="wide")
    init_session_state()

    # CSS (kept from original)
    st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .main-header { font-size: 2.4rem; font-weight:700; background: -webkit-linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .stat-card { background:white; border-radius:8px; padding:1rem; box-shadow:0 2px 6px rgba(0,0,0,0.05); text-align:center;}
    .stat-value { font-size:1.6rem; font-weight:700;}
    .stat-label { color:#718096; }
    </style>
    """, unsafe_allow_html=True)

    lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_uz5cqu1b.json")
    lottie_robot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sk5h1kfn.json")

    # --- Login screen ---
    if not st.session_state.get("authenticated", False):
        st.markdown('<div style="display:flex;justify-content:center;align-items:center;height:80vh">', unsafe_allow_html=True)
        with st.container():
            st.markdown('<h1 style="text-align:center;">🤖 NeuraLink AI</h1>', unsafe_allow_html=True)
            if lottie_ai:
                st_lottie(lottie_ai, height=200)
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    demo_users = {"admin": "admin2025", "analyst": "analyst2025", "manager": "manager2025"}
                    if username in demo_users and demo_users[username] == password:
                        st.session_state.authenticated = True
                        st.session_state.user_id = username
                        st.session_state.user_role = "demo"
                        st.success(f"Welcome, {username}!")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # --- Main UI ---
    sidebar = st.sidebar
    sidebar.button("🚪 Logout", on_click=logout)
    sidebar.markdown("### Conversation")
    if sidebar.button("🆕 Start New Chat"):
        new_chat()
        st.rerun()
    if sidebar.button("💾 Save Conversation"):
        save_conversation()
    sidebar.markdown("---")
    conv_files = [f for f in os.listdir(CONVERSATIONS_DIR) if f.startswith(f"{st.session_state.user_id}-")]
    if conv_files:
        sel = sidebar.selectbox("Load a conversation", options=[""] + conv_files, index=0, format_func=lambda x: (x.replace(f"{st.session_state.user_id}-", "").replace(".json", "") if x else "Select..."))
        if sel:
            load_conversation(sel)
            st.rerun()

    sidebar.markdown("---")
    sidebar.markdown("### Usage")
    display_usage_stats()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<div class="main-header">NeuraLink AI Assistant</div>', unsafe_allow_html=True)
        st.caption(f"Welcome, {st.session_state.user_id} • v7.0.0")
    with col2:
        if lottie_robot:
            st_lottie(lottie_robot, height=80)

    tab_chat, tab_tools, tab_analytics, tab_settings = st.tabs(["Chat", "Workflows", "Analytics", "Settings"])

    with tab_chat:
        if not st.session_state.get("api_key"):
            st.warning("Please enter your OpenAI API key in Settings to interact with the model.")
        else:
            st.markdown(f"**Conversation:** {st.session_state.conversation_name} • Model: {st.session_state.model}")
            render_chat_messages()
            prompt = st.chat_input("Type a message...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("Thinking..."):
                    response(st.session_state.messages, st.session_state.model)
                st.rerun()

    with tab_tools:
        st.markdown("### Code Review")
        code_blob = st.text_area("Paste code to review", height=200)
        if st.button("Review Code"):
            if not code_blob.strip():
                st.warning("Provide code to review.")
            else:
                st.session_state.messages.append({"role": "user", "content": f"Please review this code:\n\n```{code_blob}```"})
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
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

    with tab_analytics:
        st.markdown("### Usage Analytics (mock)")
        df_usage = pd.DataFrame({
            "Date": pd.date_range(start="2025-01-01", periods=30, freq="D"),
            "Tokens": [1000 + i * 150 for i in range(30)],
            "Cost": [5 + i * 0.5 for i in range(30)],
            "Requests": [10 + i * 2 for i in range(30)],
        })
        st.plotly_chart(px.line(df_usage, x="Date", y="Tokens", title="Daily Token Usage"), use_container_width=True)
        st.plotly_chart(px.line(df_usage, x="Date", y="Cost", title="Estimated Daily Cost"), use_container_width=True)
        st.plotly_chart(px.line(df_usage, x="Date", y="Requests", title="Daily API Requests"), use_container_width=True)

    with tab_settings:
        st.markdown("### OpenAI API Settings")
        api_text = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("api_key", ""))
        if api_text and api_text != st.session_state.get("api_key", ""):
            st.session_state.api_key = api_text
            client = create_openai_client(api_text)
            if client:
                st.success("API key configured. You may now interact with the model.")
            else:
                st.error("Failed to initialize client with the provided API key.")
        if st.session_state.get("api_key"):
            model_choice = st.selectbox("Model", ["gpt-5", "gpt-5-pro", "gpt-5-mini", "gpt-4o", "gpt-4-turbo-preview"], index=3, key="model")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, key="temperature")

if __name__ == "__main__":
    main()

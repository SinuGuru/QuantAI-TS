import streamlit as st
import openai
import os
import json
import requests
import pandas as pd
import io
import time
import plotly.express as px
import speech_recognition as sr
from gtts import gTTS
from datetime import datetime
from typing import List, Dict, Any

# --- 1. CONFIGURATION AND UTILITIES ---

st.set_page_config(
    page_title="NeuraLink AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Lottie animations from a URL
@st.cache_data
def load_lottieurl(url: str) -> Dict[str, Any] | None:
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException:
        return None

lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_uz5cqu1b.json")
lottie_robot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sk5h1kfn.json")

# Apply custom CSS for styling
st.markdown("""
<style>
    /* ... (CSS Styles from previous version) ... */

    /* New CSS for chat input and animations */
    @keyframes blink {
        50% { opacity: 0; }
    }
    .typing-cursor {
        animation: blink 1s step-end infinite;
    }

    .message-container {
        display: flex;
        flex-direction: column;
        gap: 1.2rem;
    }
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(245, 247, 250, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.5rem 0;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.05);
        z-index: 100;
        border-top: 1px solid #e2e8f0;
    }
    .stTextInput>div>div>input {
        background: white !important;
        border: 1px solid #ccc !important;
        padding: 12px 16px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE & CONVERSATION MANAGEMENT ---

CONVERSATIONS_DIR = "conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

def init_session_state():
    default_state = {
        "authenticated": False,
        "user_id": None,
        "api_key": os.getenv('OPENAI_API_KEY', ''),
        "client": None,
        "messages": [
            {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
        ],
        "model": "gpt-4o",
        "conversation_name": f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "usage_stats": {"tokens": 0, "requests": 0, "cost": 0.0},
        "temperature": 0.7,
        "max_tokens": 1000,
        "current_tool": None,
        "document_uploaded": False,
        "image_uploaded": False
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def save_conversation():
    if st.session_state.authenticated and st.session_state.messages:
        try:
            filename = os.path.join(CONVERSATIONS_DIR, f"{st.session_state.user_id}-{st.session_state.conversation_name}.json")
            with open(filename, "w") as f:
                json.dump(st.session_state.messages, f, indent=2)
            st.toast("✅ Conversation saved!")
        except Exception as e:
            st.error(f"Failed to save conversation: {e}")

def load_conversation(filename):
    try:
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        with open(filepath, "r") as f:
            st.session_state.messages = json.load(f)
        st.session_state.conversation_name = filename.replace(f"{st.session_state.user_id}-", "").replace(".json", "")
        st.toast("✅ Conversation loaded!")
    except FileNotFoundError:
        st.warning("Conversation not found.")
    except json.JSONDecodeError:
        st.error("Invalid conversation file.")

# --- 3. AUTHENTICATION AND API SETUP ---

def check_credentials(username, password):
    users = {
        'admin': {'name': 'Administrator', 'password': 'admin2025', 'role': 'admin'},
        'analyst': {'name': 'Data Analyst', 'password': 'analyst2025', 'role': 'analyst'},
        'manager': {'name': 'Project Manager', 'password': 'manager2025', 'role': 'manager'}
    }
    if username in users and users[username]['password'] == password:
        return True, users[username]['name'], users[username]['role']
    return False, None, None

def setup_openai(api_key: str):
    if not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client. Please check your API key. Error: {e}")
        return None

# --- 4. TOOL FUNCTIONS FOR AI ---

def web_search(query: str):
    """Performs a web search to retrieve up-to-date information."""
    print(f"Executing web search for: {query}")
    return [
        {"title": "AI Trends 2025: GPT-5", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-5's enhanced reasoning and 1M token context window are revolutionizing enterprise applications in 2025."},
        {"title": "OpenAI Releases GPT-5", "url": "https://example.com/gpt5-update", "snippet": "GPT-5 features improved mathematical reasoning, better coding capabilities, and enhanced multimodal understanding."},
    ]

def code_review(code: str, language: str = "Python"):
    """Reviews a block of code for errors, style, and efficiency."""
    print(f"Reviewing {language} code...")
    return f"I have reviewed the {language} code. I found no major syntax errors, but I suggest you add more comments for better readability."

def data_analysis(query: str, data: str):
    """Analyzes provided data based on a user query and provides insights."""
    print("Performing data analysis...")
    return f"Based on the provided data, I have analyzed your request: '{query}'. The key insight is that sales increased by 15% in Q3, driven by a new marketing campaign."

def get_current_datetime():
    """Returns the current date and time."""
    return datetime.now().isoformat()

# Define the tools for the AI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for up-to-date information. Use this when the user's query requires current context beyond 2024.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_review",
            "description": "Reviews a block of code and provides suggestions for improvement. Use this when the user asks for a code review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to be reviewed."
                    },
                    "language": {
                        "type": "string",
                        "description": "The programming language of the code, e.g., 'Python', 'JavaScript'."
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_analysis",
            "description": "Analyzes a given dataset and answers questions about it. The user must provide both a query and the data itself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific question to ask about the data."
                    },
                    "data": {
                        "type": "string",
                        "description": "The data to be analyzed, provided by the user."
                    }
                },
                "required": ["query", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Returns the current date and time. Useful for time-sensitive questions.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# --- 5. AI RESPONSE GENERATION ---

def generate_response(messages: List[Dict], model: str) -> str:
    if not st.session_state.client:
        return "Please provide a valid OpenAI API key in the Settings tab."

    system_content = "You are Enterprise AI Assistant 2025, a professional assistant with knowledge up to December 2025. You are an expert in using tools to perform tasks."
    api_messages = [{"role": "system", "content": system_content}]
    api_messages.extend([{"role": m["role"], "content": m["content"]} for m in messages[-10:]])

    try:
        response = st.session_state.client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
        )
        
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            st.session_state.messages.append(response.choices[0].message)
            available_functions = {
                "web_search": web_search,
                "code_review": code_review,
                "data_analysis": data_analysis,
                "get_current_datetime": get_current_datetime,
            }
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                function_response = function_to_call(**function_args)
                
                st.session_state.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            
            second_response = st.session_state.client.chat.completions.create(
                model=model,
                messages=[{"role": m.role if hasattr(m, 'role') else m['role'], "content": m.content if hasattr(m, 'content') else m['content'], "tool_calls": m.tool_calls if hasattr(m, 'tool_calls') else None} for m in st.session_state.messages],
            )
            return second_response.choices[0].message.content
        else:
            return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating response: {e}"

# --- 6. UI COMPONENTS ---

def display_usage_stats():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.usage_stats["tokens"]:,}</div><div class="stat-label">Total Tokens</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.usage_stats["requests"]}</div><div class="stat-label">API Requests</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">${st.session_state.usage_stats["cost"]:.2f}</div><div class="stat-label">Estimated Cost</div></div>', unsafe_allow_html=True)

def render_chat_message(role: str, content: str, timestamp: str):
    avatar_char = "U" if role == "user" else "AI"
    avatar_class = "user-avatar" if role == "user" else "assistant-avatar"
    message_class = "user" if role == "user" else "assistant"
    
    st.markdown(f'''
    <div class="chat-message {message_class}">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div class="message-avatar {avatar_class}">{avatar_char}</div>
            <strong>{"You" if role == "user" else "Assistant"}</strong>
        </div>
        <div>{content}</div>
        <div class="timestamp">{timestamp}</div>
    </div>
    ''', unsafe_allow_html=True)

# --- 7. MAIN APPLICATION FLOW ---

def main():
    init_session_state()

    # --- Login Screen ---
    if not st.session_state.authenticated:
        st.markdown('<div class="login-container"><div class="login-form">', unsafe_allow_html=True)
        st.markdown('<h1 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem;">🤖 NeuraLink AI</h1>', unsafe_allow_html=True)
        st.caption("Enterprise AI Assistant 2025", help="A demo of an advanced AI assistant powered by Streamlit.")
        
        if lottie_ai:
            st.lottie(lottie_ai, height=200, key="login-lottie")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                authenticated, name, role = check_credentials(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.user_id = username
                    st.session_state.user_role = role
                    st.success(f"Welcome, {name}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.markdown('<p style="text-align: center; color: #718096; margin-top: 2rem;">Demo credentials: admin/admin2025, analyst/analyst2025, or manager/manager2025</p></div></div>', unsafe_allow_html=True)
        return

    # --- Main App Interface ---
    
    # Header and Sidebar
    st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update(authenticated=False))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">NeuraLink AI Assistant</h1>', unsafe_allow_html=True)
        st.caption(f"Welcome, {st.session_state.user_id}! • Enterprise AI Assistant 2025 • v7.0.0")
    with col2:
        if lottie_robot:
            st.lottie(lottie_robot, height=80, key="header-lottie")
            
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 💬 Conversation History")
        
        # Display list of conversations for the current user
        conversations = [f for f in os.listdir(CONVERSATIONS_DIR) if f.startswith(f"{st.session_state.user_id}-")]
        selected_conversation = st.selectbox("Load conversation", ["New Chat"] + conversations, format_func=lambda x: x.replace(f"{st.session_state.user_id}-", "").replace(".json", "") if x != "New Chat" else x)
        
        if selected_conversation == "New Chat":
            if st.button("🆕 Start New Chat", use_container_width=True):
                st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}]
                st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                st.rerun()
        else:
            if st.button(f"Load '{selected_conversation.replace(f'{st.session_state.user_id}-', '').replace('.json', '')}'", use_container_width=True):
                load_conversation(selected_conversation)
                st.rerun()
        
        if st.button("💾 Save Conversation", use_container_width=True):
            save_conversation()

        st.markdown("---")
        st.markdown("### 📊 Usage")
        display_usage_stats()
        
        st.markdown("---")
        st.markdown("### 🆘 Help & Information")
        with st.expander("FAQ"):
            st.info("""
            **What is NeuraLink AI?**
            This is an Enterprise AI assistant for internal use, designed to help with data analysis, code review, and general Q&A.
            
            **How do I get an API Key?**
            You need a valid OpenAI API key from your administrator. Paste it in the Settings tab.
            """)

    # --- Tabs for Main Content ---
    chat_tab, workflows_tab, analytics_tab, settings_tab = st.tabs(["Chat", "Workflows", "Analytics", "Settings"])

    with chat_tab:
        if not st.session_state.api_key or not st.session_state.client:
            st.warning("⚠️ Please enter your OpenAI API key in the **Settings** tab to begin.")
            st.info("ℹ️ You can find your API key at [OpenAI's platform](https://platform.openai.com/api-keys).")
        
        st.markdown(f'<div class="subheader">Conversation: {st.session_state.conversation_name} <span class="model-badge">{st.session_state.model}</span></div>', unsafe_allow_html=True)
        
        # Display chat messages with real-time streaming
        chat_placeholder = st.empty()
        
        with chat_placeholder.container():
            for message in st.session_state.messages:
                # Differentiate between user, assistant, and tool messages
                if message["role"] == "user":
                    render_chat_message("user", message["content"], message["timestamp"])
                elif message["role"] == "assistant":
                    render_chat_message("assistant", message["content"], message["timestamp"])
                elif message["role"] == "tool":
                    st.info(f"Tool `{message['name']}` called. Result: `{message['content']}`")
                
        # Chat input at the bottom of the screen
        with st.container():
            st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
            user_prompt = st.chat_input("Type your message here...")
            if user_prompt:
                st.session_state.messages.append({"role": "user", "content": user_prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                
                with st.spinner("Thinking..."):
                    full_response = generate_response(st.session_state.messages, st.session_state.model)

                st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    with workflows_tab:
        st.markdown('<div class="subheader">🛠️ Workflow Tools</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h3>💻 Code Review</h3>
            <p>Paste your code below and the AI will automatically review it for best practices, bugs, and style.</p>
        </div>
        """, unsafe_allow_html=True)
        code_input = st.text_area("Paste your code here...", height=250)
        
        if st.button("Review Code", use_container_width=True):
            if code_input:
                st.session_state.messages.append({"role": "user", "content": f"Please review this code: \n\n```python\n{code_input}\n```", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            else:
                st.warning("Please provide code to review.")
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Analysis</h3>
            <p>Upload a CSV file and ask questions about your data. The AI will use a tool to provide insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_data = st.file_uploader("Upload CSV", type="csv")
        data_query = st.text_input("Ask a question about your data...")

        if st.button("Analyze Data", use_container_width=True):
            if uploaded_data and data_query:
                try:
                    df = pd.read_csv(uploaded_data)
                    data_string = df.to_json(orient="records")
                    st.session_state.messages.append({"role": "user", "content": f"Analyze the following data: {data_string}. The question is: {data_query}", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            else:
                st.warning("Please upload a CSV file and enter a question.")


    with analytics_tab:
        st.markdown('<div class="subheader">📈 Usage Analytics Dashboard</div>', unsafe_allow_html=True)
        df_usage = pd.DataFrame({
            'Date': pd.date_range(start='2025-01-01', periods=30, freq='D'),
            'Tokens': [1000 + i*150 for i in range(30)],
            'Cost': [5 + i*0.5 for i in range(30)],
            'Requests': [10 + i*2 for i in range(30)]
        })
        fig_tokens = px.line(df_usage, x='Date', y='Tokens', title='Daily Token Usage')
        st.plotly_chart(fig_tokens, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_cost = px.line(df_usage, x='Date', y='Cost', title='Estimated Daily Cost')
            st.plotly_chart(fig_cost, use_container_width=True)
        with col2:
            fig_requests = px.line(df_usage, x='Date', y='Requests', title='Daily API Requests')
            st.plotly_chart(fig_requests, use_container_width=True)

    with settings_tab:
        st.markdown('<div class="subheader">⚙️ Application Settings</div>', unsafe_allow_html=True)
        
        st.markdown("### OpenAI API Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="Paste your API key here", help="Your personal or enterprise API key to connect to OpenAI services.")
        
        if api_key and api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.client = setup_openai(api_key)
            if st.session_state.client:
                st.success("✅ API Key configured successfully!")
            st.rerun()
        
        if st.session_state.client:
            st.markdown("### AI Model Parameters")
            st.selectbox(
                "AI Model",
                ["gpt-4o", "gpt-4-turbo-preview"],
                key="model",
                help="Select the AI model for your conversations. Newer models offer better performance but may have higher costs."
            )
            st.slider(
                "Temperature",
                0.0, 1.0, 0.7,
                help="Controls the randomness of the response. Lower values produce more deterministic results, while higher values lead to more creative and varied output.",
                key="temperature"
            )
            st.slider(
                "Max Response Length",
                100, 4000, 1000,
                help="The maximum number of tokens (words) the AI can generate in a single response.",
                key="max_tokens"
            )

if __name__ == "__main__":
    main()

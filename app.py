import streamlit as st
import openai
import os
from datetime import datetime
import json
from typing import List, Dict, Any
import pandas as pd
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
import time
import tempfile
import hashlib
import tiktoken
import speech_recognition as sr
from gtts import gTTS
import uuid
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

# --- 1. CONFIGURATION AND UTILITIES ---

# Set page configuration with a modern, clean look
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

# Load the animations at the start
lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_uz5cqu1b.json")
lottie_robot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sk5h1kfn.json")

# Apply custom CSS for styling
st.markdown("""
<style>
    /* Global Styles */
    .main, .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        font-size: 3.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subheader {
        font-size: 1.5rem;
        color: #2d3748;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 15px rgba(0,0,0,0.07);
        max-width: 80%;
        position: relative;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .chat-message.assistant {
        background: white;
        color: #2d3748;
        margin-right: auto;
        border-bottom-left-radius: 5px;
        border: 1px solid #e2e8f0;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
    }
    
    .user-avatar {
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .timestamp {
        font-size: 0.75rem;
        color: #718096;
        text-align: right;
        margin-top: 0.5rem;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.8rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: rgba(255, 255, 255, 0.7); 
        border-radius: 12px;
        gap: 8px;
        padding: 16px 24px;
        font-weight: 600;
        margin: 0 4px;
        border: 1px solid rgba(255, 255, 255, 0.5); 
        backdrop-filter: blur(10px);
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
        left: 25%;
        background: white;
        padding: 15px;
        border-radius: 16px;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .login-form {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 3rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15); 
        width: 450px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.12);
    }
    
    .model-badge {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 600;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message {
        animation: fadeIn 0.3s ease;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .stChatInput {
            width: 90%;
            left: 5%;
        }
        .chat-message {
            max-width: 95%;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE MANAGEMENT ---

def init_session_state():
    """Initializes session state variables if they don't exist."""
    default_state = {
        "authenticated": False,
        "user_id": None,
        "user_role": None,
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
        "knowledge_base": {},
        "current_tool": None,
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- 3. AUTHENTICATION AND API SETUP ---

def check_credentials(username, password):
    """Checks user credentials against a pre-defined list or Streamlit secrets."""
    try:
        if hasattr(st, 'secrets') and 'users' in st.secrets:
            users = st.secrets["users"]
            if username in users and users[username]["password"] == password:
                return True, users[username]["name"], users[username].get("role", "user")
    except Exception:
        pass
    
    users = {
        'admin': {'name': 'Administrator', 'password': 'admin2025', 'role': 'admin'},
        'analyst': {'name': 'Data Analyst', 'password': 'analyst2025', 'role': 'analyst'},
        'manager': {'name': 'Project Manager', 'password': 'manager2025', 'role': 'manager'}
    }
    if username in users and users[username]['password'] == password:
        return True, users[username]['name'], users[username]['role']
    return False, None, None

def setup_openai(api_key: str):
    """Initializes and caches the OpenAI client."""
    if not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client. Check your API key. Error: {e}")
        return None

# --- 4. DATA PROCESSING AND API CALLS ---

@st.cache_data(ttl=300)
def web_search(query: str) -> List[Dict[str, str]]:
    """Simulates a web search for up-to-date context."""
    return [
        {"title": "AI Trends 2025: GPT-5", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-5's enhanced reasoning and 1M token context window are revolutionizing enterprise applications in 2025."},
        {"title": "OpenAI Releases GPT-5", "url": "https://example.com/gpt5-update", "snippet": "GPT-5 features improved mathematical reasoning, better coding capabilities, and enhanced multimodal understanding."},
    ]

@st.cache_data(show_spinner="Processing document...")
def process_document(uploaded_file) -> str:
    """Extracts text from uploaded PDF or text files."""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or "" + "\n"
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type.startswith("image/"):
            text = f"[Image file: {uploaded_file.name}]"
    except Exception as e:
        return f"Error processing document: {e}"
    return text

@st.cache_data(show_spinner="Processing image...")
def process_image(uploaded_image) -> str:
    """Uses OpenAI's vision capabilities to describe an image."""
    try:
        base64_image = base64.b64encode(uploaded_image.read()).decode('utf-8')
        response = st.session_state.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "What is in this image? Describe it in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing image: {e}"

def estimate_tokens(text: str, model: str) -> int:
    """Estimates token count for a string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text.split()) * 1.5

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates estimated cost based on OpenAI's 2025 pricing."""
    pricing = {
        "gpt-5": {"input": 0.015, "output": 0.06},
        "gpt-4.5": {"input": 0.012, "output": 0.045},
        "gpt-4.1": {"input": 0.008, "output": 0.025},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }
    
    model_key = next((k for k in pricing if k in model.lower()), None)
    if not model_key:
        return 0.0
    
    cost = (prompt_tokens / 1000 * pricing[model_key]["input"] + 
            completion_tokens / 1000 * pricing[model_key]["output"])
    return cost

def generate_response(messages: List[Dict], model: str, use_web_search: bool, doc_context: str | None, img_context: str | None) -> str:
    """Generates an AI response with enhanced context."""
    if not st.session_state.client:
        return "Please provide a valid OpenAI API key in the sidebar."
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_content = (f"""You are Enterprise AI Assistant 2025, a professional assistant with knowledge up to December 2025.
    Current date: {current_date}
    
    Key 2025 Context:
    - GPT-5 has been released with 1M token context window.
    - AI regulations have evolved with the EU AI Act fully implemented.
    - Quantum computing is beginning to impact cryptography.
    
    Always provide accurate, up-to-date information. Be professional and concise.""")
    
    if doc_context:
        system_content += f"\n\nDocument Context:\n{doc_context}\n"
    if img_context:
        system_content += f"\n\nImage Context:\n{img_context}\n"
    if use_web_search and messages and messages[-1]["role"] == "user":
        search_results = web_search(messages[-1]["content"])
        if search_results:
            web_context = "\n\nWeb Search Results:\n"
            for res in search_results:
                web_context += f"- {res['title']}: {res['snippet']}\n"
            system_content += web_context
    
    api_messages = [{"role": "system", "content": system_content}]
    api_messages.extend([{"role": m["role"], "content": m["content"]} for m in messages[-15:]])
    
    try:
        response_stream = st.session_state.client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens
        )
        
        full_response = ""
        for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
        
        prompt_tokens = estimate_tokens(system_content + " ".join([m["content"] for m in messages[-15:]]), model)
        completion_tokens = estimate_tokens(full_response, model)
        
        st.session_state.usage_stats["tokens"] += prompt_tokens + completion_tokens
        st.session_state.usage_stats["requests"] += 1
        st.session_state.usage_stats["cost"] += calculate_cost(model, prompt_tokens, completion_tokens)
        
        return full_response
    except Exception as e:
        return f"❌ Error generating response: {e}"

# --- 5. UI COMPONENTS ---

def create_model_comparison() -> pd.DataFrame:
    """Generates a comparison dataframe for available models."""
    data = {
        "Model": ["GPT-5", "GPT-4.5", "GPT-4.1", "GPT-4o"],
        "Context Window": ["1M", "128K", "256K", "128K"],
        "Intelligence": [10.0, 9.5, 9.3, 9.2],
        "Speed": [7, 8, 8.5, 9],
        "Cost per 1K tokens (Input/Output)": ["$15/$60", "$12/$45", "$8/$25", "$5/$15"]
    }
    return pd.DataFrame(data)

def display_usage_stats():
    """Displays user usage statistics in a clean card layout."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.usage_stats["tokens"]:,}</div><div class="stat-label">Total Tokens</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.usage_stats["requests"]}</div><div class="stat-label">API Requests</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">${st.session_state.usage_stats["cost"]:.2f}</div><div class="stat-label">Estimated Cost</div></div>', unsafe_allow_html=True)

def render_chat_message(role: str, content: str, timestamp: str):
    """Renders a single chat message with a user/assistant avatar."""
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

def speech_to_text() -> str:
    """Converts speech to text using the microphone."""
    try:
        recognizer = sr.Recognizer()
        with st.spinner("Listening..."):
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        return "No speech detected"
    except Exception as e:
        return f"Error with speech recognition: {e}"

def text_to_speech(text: str) -> io.BytesIO | None:
    """Converts text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def render_tool_ui(tool_id: str):
    """Renders the UI for a specific workflow tool."""
    tools = {
        "code_review": {"name": "Code Review", "icon": "💻", "prompt": "Please review this code for errors, style, and efficiency:"},
        "data_analysis": {"name": "Data Analysis", "icon": "📊", "prompt": "Help me analyze this data and provide insights:"},
    }
    tool_info = tools.get(tool_id)
    if tool_info:
        st.markdown(f'<div class="subheader">{tool_info["icon"]} {tool_info["name"]} Assistant</div>', unsafe_allow_html=True)
        user_input = st.text_area(f"{tool_info['prompt']}", height=200)
        if st.button(f"Run {tool_info['name']}", use_container_width=True):
            if user_input:
                st.session_state.messages.append({"role": "user", "content": f"{tool_info['name']} Request: {user_input}", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            else:
                st.warning("Please provide input for this tool.")
        if st.button("← Back to Tools", use_container_width=True):
            st.session_state.current_tool = None
            st.rerun()

# --- 6. MAIN APPLICATION FLOW ---

def main():
    """Main function to run the Streamlit app."""
    init_session_state()

    # --- Login Screen ---
    if not st.session_state.authenticated:
        st.markdown('<div class="login-container"><div class="login-form">', unsafe_allow_html=True)
        st.markdown('<h1 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem;">🤖 NeuraLink AI</h1>', unsafe_allow_html=True)
        st.caption("Enterprise AI Assistant 2025", help="A demo of an advanced AI assistant powered by Streamlit.")
        
        if lottie_ai:
            st_lottie(lottie_ai, height=200, key="login-lottie")
        
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
            st_lottie(lottie_robot, height=80, key="header-lottie")
            
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        if not st.session_state.api_key:
            api_key = st.text_input("OpenAI API Key", type="password", placeholder="Paste your API key here")
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.client = setup_openai(api_key)
                if st.session_state.client:
                    st.success("✓ API Key Configured")
        
        if st.session_state.api_key:
            st.selectbox("AI Model", ["gpt-5", "gpt-4.5", "gpt-4.1", "gpt-4o"], key="model")
            st.slider("Temperature", 0.0, 1.0, 0.7, key="temperature")
            st.slider("Max Response Length", 100, 4000, 1000, key="max_tokens")

        st.markdown("---")
        st.markdown("### 🌟 Features")
        use_web_search = st.checkbox("Enable Web Search", value=True)
        document_upload = st.file_uploader("Upload Document", type=["pdf", "txt"], help="Provide context from documents")
        image_upload = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], help="Analyze images with AI vision")
        
        if st.button("Start Voice Input", use_container_width=True):
            voice_text = speech_to_text()
            if voice_text not in ["No speech detected", "Could not understand audio", "Error with speech recognition: "]:
                st.session_state.messages.append({"role": "user", "content": voice_text, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            else:
                st.warning(voice_text)

        st.markdown("---")
        st.markdown("### 💬 Conversation")
        st.session_state.conversation_name = st.text_input("Conversation Name", value=st.session_state.conversation_name)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}]
                st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = st.session_state.messages[:1]
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Usage")
        display_usage_stats()

    # --- Tabs for Main Content ---
    chat_tab, workflows_tab, analytics_tab = st.tabs(["💬 Chat", "🛠️ Workflows", "📈 Analytics"])

    with chat_tab:
        if not st.session_state.api_key or not st.session_state.client:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to begin.")
            st.info("ℹ️ You can find your API key at [OpenAI's platform](https://platform.openai.com/api-keys).")
            st.markdown("---")
            st.markdown('<div class="subheader">Latest Model Comparison</div>', unsafe_allow_html=True)
            st.dataframe(create_model_comparison(), use_container_width=True, hide_index=True)
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f'<div class="subheader">Conversation: {st.session_state.conversation_name} <span class="model-badge">{st.session_state.model}</span></div>', unsafe_allow_html=True)
            for message in st.session_state.messages:
                render_chat_message(message["role"], message["content"], message["timestamp"])
                if message["role"] == "assistant":
                    audio_bytes = text_to_speech(message["content"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

        with col2:
            st.markdown('<div class="subheader">🚀 Quick Actions</div>', unsafe_allow_html=True)
            if st.button("📊 Market Analysis Summary", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Provide a brief market analysis summary for Q2 2025 focusing on tech sector trends.", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()

    # Chat input at the bottom of the screen
    user_prompt = st.chat_input("Type your message here...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
        
        doc_context = process_document(document_upload) if document_upload else None
        img_context = process_image(image_upload) if image_upload else None
        
        with st.spinner(f"Analyzing with {st.session_state.model}..."):
            response = generate_response(
                st.session_state.messages,
                st.session_state.model,
                use_web_search,
                doc_context,
                img_context
            )
        
        st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
        st.rerun()

    with workflows_tab:
        if st.session_state.current_tool:
            render_tool_ui(st.session_state.current_tool)
        else:
            st.markdown('<div class="subheader">🛠️ Workflow Tools</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💻 Use Code Review", use_container_width=True):
                    st.session_state.current_tool = "code_review"
                    st.rerun()
            with col2:
                if st.button("📊 Use Data Analysis", use_container_width=True):
                    st.session_state.current_tool = "data_analysis"
                    st.rerun()

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

if __name__ == "__main__":
    main()

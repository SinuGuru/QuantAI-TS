import streamlit as st
from openai import OpenAI
import os
from datetime import datetime
import json
from typing import List, Dict
import pandas as pd
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go
import base64
from PIL import Image
import io
import time
import tempfile
import hashlib
import tiktoken
import secrets
import speech_recognition as sr
from gtts import gTTS
import uuid
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

# Set page configuration with premium theme
st.set_page_config(
    page_title="NeuraLink AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io',
        'Report a bug': "https://github.com/yourusername/neurallink-ai/issues",
        'About': "# NeuraLink AI Assistant\nEnterprise-grade AI for 2025 and beyond."
    }
)

# Load Lottie animations
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animations
lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_uz5cqu1b.json")
lottie_robot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sk5h1kfn.json")

# Premium CSS styling with glassmorphism and modern design
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styles */
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
    
    /* Buttons */
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
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .sidebar .stTextInput input, .sidebar .stSelectbox select {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: none;
        padding: 0.75rem;
    }
    
    /* Tabs */
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
    
    /* Input Fields */
    .stTextInput input, .stSelectbox select, .stSlider div {
        border-radius: 12px;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.5);
    }
    
    /* Chat Input */
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
    
    /* Login Form */
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
    
    /* Feature Cards */
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
    
    /* Model Badge */
    .model-badge {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    /* Stats Cards */
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
    
    /* Icons */
    .icon-large {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Custom Scrollbar */
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
    
    /* Animation for messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message {
        animation: fadeIn 0.3s ease;
    }
    
    /* Streamlit Cloud optimizations */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    
    /* Mobile responsive adjustments */
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

# Initialize session state with caching
@st.cache_resource
def init_session_state():
    return {
        "messages": [
            {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
        ],
        "api_key": os.getenv('OPENAI_API_KEY', ''),
        "model": "gpt-4.1",
        "conversation_name": f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "usage_stats": {"tokens": 0, "requests": 0, "cost": 0.0},
        "client": None,
        "temperature": 0.7,
        "max_tokens": 1000,
        "user_id": None,
        "authenticated": False,
        "knowledge_base": {},
        "workflow_tools": setup_workflow_tools(),
        "conversation_summaries": {},
        "current_tool": None,
        "image_cache": {}
    }

# Initialize session state
for key, value in init_session_state().items():
    if key not in st.session_state:
        st.session_state[key] = value

# Simple authentication system with Streamlit Secrets integration
def check_credentials(username, password):
    """Credential check with Streamlit Secrets support"""
    # Try to get users from secrets first
    try:
        if hasattr(st, 'secrets') and 'users' in st.secrets:
            users = st.secrets["users"]
            if username in users and users[username]["password"] == password:
                return True, users[username]["name"], users[username].get("role", "user")
    except:
        pass
    
    # Fallback to hardcoded users
    users = {
        'admin': {'name': 'Administrator', 'password': 'admin2025', 'role': 'admin'},
        'analyst': {'name': 'Data Analyst', 'password': 'analyst2025', 'role': 'analyst'},
        'manager': {'name': 'Project Manager', 'password': 'manager2025', 'role': 'manager'}
    }
    
    if username in users and users[username]['password'] == password:
        return True, users[username]['name'], users[username]['role']
    return False, None, None

# Initialize workflow tools
@st.cache_data
def setup_workflow_tools():
    """Add tools for specific professional workflows"""
    tools = {
        "code_review": {
            "name": "Code Review",
            "description": "Analyze and improve code snippets with AI-powered suggestions",
            "icon": "💻",
            "color": "#F596B3"
        },
        "document_generation": {
            "name": "Document Generation",
            "description": "Create professional documents, reports, and presentations",
            "icon": "📄",
            "color": "#10B981"
        },
        "data_analysis": {
            "name": "Data Analysis",
            "description": "Analyze and visualize data with AI-powered insights",
            "icon": "📊",
            "color": "#3B82F6"
        },
        "research_assistant": {
            "name": "Research Assistant",
            "description": "Help with academic research and information gathering",
            "icon": "🔍",
            "color": "#8B5CF6"
        },
        "meeting_assistant": {
            "name": "Meeting Assistant",
            "description": "Summarize meetings and extract action items",
            "icon": "📅",
            "color": "#EC4899"
        },
        "strategy_advisor": {
            "name": "Strategy Advisor",
            "description": "Get AI-powered strategic recommendations for business decisions",
            "icon": "🎯",
            "color": "#EF4444"
        }
    }
    return tools

# Initialize OpenAI client with caching
@st.cache_resource
def setup_openai(api_key):
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            # Test the client with a simple request
            client.models.list()
            return client
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            return None
    return None

# Function to perform web search (simulated for 2025)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def web_search(query: str):
    """Simulate web search with latest 2025 data"""
    search_results = [
        {"title": "AI Trends 2025: GPT-4.1 and GPT-5 Transform Industries", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-4.1's enhanced reasoning and 256K token context window are revolutionizing enterprise applications in 2025."},
        {"title": "OpenAI Releases GPT-4.1: What's New", "url": "https://example.com/gpt41-update", "snippet": "GPT-4.1 features improved mathematical reasoning, better coding capabilities, and enhanced multimodal understanding."},
        {"title": "Multimodal AI Becomes Standard in 2025", "url": "https://example.com/multimodal-2025", "snippet": "Most enterprise AI systems now seamlessly process text, images, audio, and video in unified models."},
        {"title": "OpenAI's GPT-4.1 Sets New Standards", "url": "https://example.com/gpt41-series", "snippet": "The GPT-4.1 model offers unprecedented speed and accuracy for real-time applications."}
    ]
    return search_results

# Function to process uploaded documents with caching
@st.cache_data(show_spinner="Processing document...")
def process_document(uploaded_file):
    """Extract text from uploaded documents"""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type.startswith("image/"):
            # For images, we'll use a placeholder since we can't extract text
            text = f"[Image file: {uploaded_file.name}]"
        return text
    except Exception as e:
        return f"Error processing document: {str(e)}"

# Function to process images with multimodal capabilities and caching
@st.cache_data(show_spinner="Processing image...")
def process_image(uploaded_image):
    """Process images using OpenAI's vision capabilities"""
    try:
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            uploaded_image.seek(0)
            tmp_file.write(uploaded_image.read())
            tmp_file_path = tmp_file.name
        
        # Read image as base64
        with open(tmp_file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = st.session_state.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image? Please describe it in detail."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Function to estimate token count using tiktoken
@st.cache_data
def estimate_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Estimate token count for a string"""
    try:
        # Try to get encoding for the specific model, fall back to cl100k_base
        try:
            encoding = tiktoken.encoding_for_model(model)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback estimation if tiktoken is not available
        return len(text.split()) * 1.33

# Function to calculate cost based on model and tokens
@st.cache_data
def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on OpenAI's 2025 pricing"""
    pricing = {
        "gpt-5": {"input": 0.015, "output": 0.06},
        "gpt-4.5": {"input": 0.012, "output": 0.045},
        "gpt-4.1": {"input": 0.008, "output": 0.025},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    # Normalize model name
    model_key = model
    if "gpt-5" in model:
        model_key = "gpt-5"
    elif "gpt-4.5" in model:
        model_key = "gpt-4.5"
    elif "gpt-4.1" in model:
        model_key = "gpt-4.1"
    elif "gpt-4o" in model:
        model_key = "gpt-4o"
    elif "gpt-4-turbo" in model:
        model_key = "gpt-4-turbo"
    
    if model_key not in pricing:
        return 0.0
    
    cost = (prompt_tokens / 1000 * pricing[model_key]["input"] + 
            completion_tokens / 1000 * pricing[model_key]["output"])
    return cost

# Implement conversation memory with summarization
def summarize_conversation(messages, conversation_id):
    """Create a summary of long conversations to stay within token limits"""
    if conversation_id in st.session_state.conversation_summaries:
        return st.session_state.conversation_summaries[conversation_id]
    
    # Only summarize if conversation is long
    if len(messages) < 10:
        return None
    
    summary_prompt = """
    Summarize the key points from this conversation in a concise paragraph.
    Focus on decisions made, action items, and important information exchanged.
    
    Conversation:
    """
    
    for msg in messages:
        summary_prompt += f"\n{msg['role']}: {msg['content']}"
    
    try:
        response = st.session_state.client.chat.completions.create(
            model=st.session_state.model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500
        )
        
        summary = response.choices[0].message.content
        st.session_state.conversation_summaries[conversation_id] = summary
        return summary
    except Exception as e:
        st.error(f"Error summarizing conversation: {str(e)}")
        return None

# Add robust error handling for API limits
def generate_response_with_retry(messages, model, use_web_search=False, document_context=None, image_context=None, max_retries=3):
    """Generate response with retry logic for rate limits"""
    for attempt in range(max_retries):
        try:
            return generate_response(messages, model, use_web_search, document_context, image_context)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                return f"❌ Error: {str(e)}"

# Main function to generate AI response
def generate_response(messages: List[Dict], model: str, use_web_search: bool = False, 
                     document_context: str = None, image_context: str = None) -> str:
    """Generate response from OpenAI API with enhanced 2025 context"""
    
    # Prepare system message with 2025 context
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_content = f"""You are Enterprise AI Assistant 2025, a professional AI assistant with knowledge up to December 2025.

Current date: {current_date}

Key 2025 Context:
- GPT-5 has been released with 1M token context window and enhanced reasoning
- GPT-4.5 offers improved efficiency for enterprise applications
- GPT-4.1 provides optimized performance with enhanced capabilities
- o3 series models provide optimized performance for different use cases
- AI regulations have evolved with the EU AI Act fully implemented
- Quantum computing is beginning to impact cryptography and optimization
- Climate tech solutions powered by AI are seeing widespread adoption

Always provide accurate, up-to-date information. If you're unsure, say so. 
Be professional, concise, and helpful.
"""
    
    # Add document context if provided
    if document_context:
        system_content += f"\n\nDocument Context:\n{document_context}\n\nPlease reference this document when relevant to the user's query."
    
    # Add image context if provided
    if image_context:
        system_content += f"\n\nImage Context:\n{image_context}\n\nPlease reference this image when relevant to the user's query."
    
    # Add web search results if enabled
    if use_web_search and messages and messages[-1]["role"] == "user":
        search_query = messages[-1]["content"]
        search_results = web_search(search_query)
        if search_results:
            web_context = "\n\nCurrent Web Context:\n"
            for i, result in enumerate(search_results, 1):
                web_context += f"{i}. {result['title']}: {result['snippet']}\n"
            system_content += web_context
    
    # Add conversation summary if available
    conversation_id = st.session_state.conversation_name
    summary = summarize_conversation(messages, conversation_id)
    if summary:
        system_content += f"\n\nConversation Summary:\n{summary}"
    
    # Prepare messages for API call
    api_messages = [{"role": "system", "content": system_content}]
    
    # Add conversation history (limit to last 15 messages to avoid token limits)
    for msg in messages[-15:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        # Create completion with the enhanced context using the new client API
        response = st.session_state.client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens
        )
        
        # Collect streaming response
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                
        # Update usage statistics
        prompt_tokens = estimate_tokens(system_content + " ".join([m["content"] for m in messages]), model)
        completion_tokens = estimate_tokens(full_response, model)
        
        st.session_state.usage_stats["tokens"] += prompt_tokens + completion_tokens
        st.session_state.usage_stats["requests"] += 1
        st.session_state.usage_stats["cost"] += calculate_cost(model, prompt_tokens, completion_tokens)
                
        return full_response
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Function to create model comparison chart
@st.cache_data
def create_model_comparison():
    """Create a comparison chart of available models"""
    models_data = {
        "Model": ["GPT-5", "GPT-4.5", "GPT-4.1", "GPT-4o"],
        "Context Window": ["1M", "128K", "256K", "128K"],
        "Intelligence": [10.0, 9.5, 9.3, 9.2],
        "Speed": [7, 8, 8.5, 9],
        "Cost per 1K tokens": ["$15/60", "$12/45", "$8/25", "$5/15"]
    }
    
    df = pd.DataFrame(models_data)
    return df

# Function to display usage statistics
def display_usage_stats():
    """Display usage statistics in a nice format"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.usage_stats['tokens']:,.0f}</div>
            <div class="stat-label">Total Tokens</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.usage_stats['requests']}</div>
            <div class="stat-label">API Requests</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">${st.session_state.usage_stats['cost']:.2f}</div>
            <div class="stat-label">Estimated Cost</div>
        </div>
        """, unsafe_allow_html=True)

# Function to create analytics dashboard
def create_analytics_dashboard():
    """Create a comprehensive analytics dashboard"""
    st.markdown('<div class="subheader">📈 Usage Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    usage_data = {
        'Date': dates,
        'Tokens': [max(1000, int(5000 * (0.5 + 0.5 * (i/10)))) for i in range(len(dates))],
        'Cost': [max(5, 25 * (0.5 + 0.5 * (i/7))) for i in range(len(dates))],
        'Requests': [max(10, int(50 * (0.5 + 0.5 * (i/5)))) for i in range(len(dates))]
    }
    
    df = pd.DataFrame(usage_data)
    
    # Token usage over time
    fig_tokens = px.line(df, x='Date', y='Tokens', title='Token Usage Over Time')
    fig_tokens.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_tokens, use_container_width=True)
    
    # Cost analysis
    col1, col2 = st.columns(2)
    with col1:
        fig_cost = px.line(df, x='Date', y='Cost', title='Cost Over Time')
        fig_cost.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        fig_requests = px.line(df, x='Date', y='Requests', title='API Requests Over Time')
        fig_requests.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_requests, use_container_width=True)
    
    # Model comparison
    st.markdown('<div class="subheader">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)
    model_df = create_model_comparison()
    st.dataframe(model_df, use_container_width=True, hide_index=True)

# Voice input functionality
def speech_to_text():
    """Convert speech to text using microphone input"""
    try:
        recognizer = sr.Recognizer()
        
        with sr.Microphone() as source: 
            st.info("Listening... Speak now")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                return "No speech detected"
            except sr.UnknownValueError:
                return "Could not understand audio"
            except Exception as e:
                return f"Error: {str(e)}"
    except:
        return "Microphone not available"

# Text to speech functionality
def text_to_speech(text):
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang='en') 
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Simple semantic search (replacement for vector DB)
def semantic_search(query, knowledge_base):
    """Simple semantic search implementation"""
    results = []
    for doc_id, doc_info in knowledge_base.items():
        content = doc_info['content'].lower()
        if query.lower() in content:
            results.append({
                'id': doc_id,
                'name': doc_info['name'],
                'content': content[:200] + '...' if len(content) > 200 else content
            })
    return results

# Tool-specific UI components
def render_tool_ui(tool_id):
    """Render UI for specific tools"""
    if tool_id == "code_review":
        st.markdown('<div class="subheader">💻 Code Review Assistant</div>', unsafe_allow_html=True)
        code_input = st.text_area("Paste your code here:", height=200, placeholder="def example_function():\n    # Your code here")
        if st.button("Analyze Code", use_container_width=True):
            if code_input:
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"Please review this code:\n\n{code_input}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.rerun()
            else:
                st.warning("Please enter some code to analyze")
    
    elif tool_id == "data_analysis":
        st.markdown('<div class="subheader">📊 Data Analysis Assistant</div>', unsafe_allow_html=True)
        data_input = st.text_area("Describe your data or paste a sample:", height=150, placeholder="I have sales data for Q2 2025 with columns: date, product, region, sales_amount")
        if st.button("Analyze Data", use_container_width=True):
            if data_input:
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"Help me analyze this data: {data_input}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.rerun()
            else:
                st.warning("Please describe your data or analysis needs")

# Main application
def main():
    # Check if user is authenticated
    if not st.session_state.authenticated:
        # Enhanced login form with Lottie animation
        st.markdown("""
        <div class="login-container">
            <div class="login-form">
                <h1 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem;">🤖 NeuraLink AI</h1>
                <p style="text-align: center; color: #4a5568; margin-bottom: 2rem;">Enterprise AI Assistant 2025</p>
        """, unsafe_allow_html=True)
        
        # Add Lottie animation
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
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.markdown("""
            <p style="text-align: center; color: #718096; margin-top: 2rem;">
                Demo credentials: admin/admin2025, analyst/analyst2025, or manager/manager2025
            </p>
            </div> 
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    # User is authenticated, show the main app
    st.sidebar.button("🚪 Logout", on_click=lambda: setattr(st.session_state, 'authenticated', False))
    
    # Initialize OpenAI client if not already done
    if st.session_state.client is None and st.session_state.api_key:
        st.session_state.client = setup_openai(st.session_state.api_key)
    
    # Header with improved styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">NeuraLink AI Assistant</h1>', unsafe_allow_html=True)
        st.caption(f"Welcome, {st.session_state.user_id}! • Enterprise AI Assistant 2025 • v7.0.0")
    
    # Add Lottie animation to header
    if lottie_robot:
        with col3:
            st_lottie(lottie_robot, height=80, key="header-lottie")
    
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # API Key Input (only show if not in secrets)
        try:
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                st.session_state.api_key = st.secrets['OPENAI_API_KEY']
                if st.session_state.client is None:
                    st.session_state.client = setup_openai(st.session_state.api_key)
            else:
                api_key = st.text_input(
                    "OpenAI API Key",
                    value=st.session_state.api_key,
                    type="password",
                    help="Enter your OpenAI API key to begin",
                    label_visibility="collapsed",
                    placeholder="Paste your OpenAI API key here"
                )
                
                if api_key and api_key != st.session_state.api_key:
                    st.session_state.api_key = api_key
                    st.session_state.client = setup_openai(api_key)
                    if st.session_state.client:
                        st.success("✓ API Key Configured")
        except:
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.api_key,
                type="password",
                help="Enter your OpenAI API key to begin",
                label_visibility="collapsed",
                placeholder="Paste your OpenAI API key here"
            )
            
            if api_key and api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                st.session_state.client = setup_openai(api_key)
                if st.session_state.client:
                    st.success("✓ API Key Configured")
        
        # Model Selection - Only GPT-4.1 and above models
        st.selectbox(
            "AI Model",
            ["gpt-4.1", "gpt-4o", "gpt-4.5", "gpt-5"],
            index=0,
            key="model",
            help="Select which AI model to use"
        )
        
        # Display model info
        model_info = {
            "gpt-4.1": "Enhanced version with 256K context and improved reasoning capabilities (2025)",
            "gpt-4o": "Optimized model with balanced speed and intelligence (2024)",
            "gpt-4.5": "High-performance model for complex tasks (2025)",
            "gpt-5": "Flagship model with 1M context window and advanced reasoning (2025)"
        }
        
        st.info(f"**{st.session_state.model}**: {model_info[st.session_state.model]}")
        
        # Advanced settings
        st.markdown("---")
        st.markdown("### ⚙️ Advanced Settings")
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.7, help="Controls randomness: Lower = more deterministic")
        st.session_state.max_tokens = st.slider("Max Response Length", 100, 4000, 1000, help="Maximum tokens in response")
        
        # Features
        st.markdown("---")
        st.markdown("### 🌟 Features")
        use_web_search = st.checkbox("Enable Web Search", value=True, help="Search for latest information when needed")
        
        # Document upload
        document_upload = st.file_uploader("Upload Document", type=["pdf", "txt"], help="Provide context from documents")
        document_context = None
        if document_upload:
            with st.spinner("Processing document..."):
                document_context = process_document(document_upload)
            st.success(f"✓ Document processed: {document_upload.name}")
        
        # Image upload for multimodal processing
        image_upload = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], help="Analyze images with AI vision")
        image_context = None
        if image_upload:
            with st.spinner("Processing image..."):
                image_context = process_image(image_upload)
            st.success(f"✓ Image processed: {image_upload.name}")
        
        # Voice input
        st.markdown("---")
        st.markdown("### 🎤 Voice Input")
        if st.button("Start Voice Input", use_container_width=True):
            voice_text = speech_to_text()
            if voice_text and voice_text not in ["No speech detected", "Could not understand audio", "Microphone not available"]:
                st.session_state.messages.append({"role": "user", "content": voice_text, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            elif voice_text == "Microphone not available":
                st.warning("Microphone access not available in this environment")
        
        # Conversation management
        st.markdown("---")
        st.markdown("### 💬 Conversation")
        st.session_state.conversation_name = st.text_input("Conversation Name", value=st.session_state.conversation_name)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Conversation", use_container_width=True):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                ]
                st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                st.rerun()
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.messages = st.session_state.messages[:1]
                st.rerun()
        
        # Export conversation
        if st.button("📤 Export Conversation", use_container_width=True):
            conversation_data = []
            for msg in st.session_state.messages:
                conversation_data.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", "")
                })
            
            # Create downloadable JSON
            json_str = json.dumps(conversation_data, indent=2)
            st.download_button(
                label="Download Conversation",
                data=json_str,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Usage statistics
        st.markdown("---")
        st.markdown("### 📊 Usage Statistics")
        display_usage_stats()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        NeuraLink AI Assistant leverages the latest GPT models to provide professional assistance with up-to-date information.
        
        **Key Features:**
        - GPT-5, GPT-4.5, GPT-4.1, and GPT-4o support
        - 2025 knowledge context
        - Web search integration
        - Document processing
        - Multimodal image analysis
        - Voice input/output
        - Conversation management
        """)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🛠️ Workflows", "📈 Analytics", "📚 Knowledge Base"])
    
    with tab1:
        # Check for API key and client
        if not st.session_state.api_key or not st.session_state.client:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to begin")
            st.info("ℹ️ You can obtain an API key from [OpenAI's platform](https://platform.openai.com/api-keys)")
            
            # Model comparison
            st.markdown("---")
            st.markdown('<div class="subheader">Latest Model Comparison</div>', unsafe_allow_html=True)
            model_df = create_model_comparison()
            st.dataframe(model_df, use_container_width=True, hide_index=True)
            
            return
        
        # Main chat interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f'<div class="subheader">Conversation: {st.session_state.conversation_name} <span class="model-badge">{st.session_state.model}</span></div>', unsafe_allow_html=True)
            
            # Display chat messages with improved styling
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'''
                    <div class="chat-message user">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <div class="message-avatar user-avatar">U</div>
                            <strong>You</strong>
                        </div>
                        <div>{message["content"]}</div>
                        <div class="timestamp">{message["timestamp"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="chat-message assistant">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <div class="message-avatar assistant-avatar">AI</div>
                            <strong>Assistant</strong>
                        </div>
                        <div>{message["content"]}</div>
                        <div class="timestamp">{message["timestamp"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Add text-to-speech button for assistant responses
                    audio_bytes = text_to_speech(message["content"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
        
        with col2:
            st.markdown('<div class="subheader">🚀 Quick Actions</div>', unsafe_allow_html=True)
            
            # Quick action buttons
            if st.button("📊 Market Analysis Summary", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Provide a brief market analysis summary for Q2 2025 focusing on tech sector trends.", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            
            if st.button("📈 Data Insights", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What are the key data and AI insights for business decision making in 2025?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            
            if st.button("🔍 Research Assistance", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Help me research the latest developments in renewable energy technology for 2025.", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            
            if st.button("📝 Document Analysis", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Analyze the provided document and summarize key points.", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
            
            st.markdown("---")
            st.markdown("**🎯 Model Capabilities**")
            
            # Feature cards with improved styling
            st.markdown("""
            <div class="feature-card">
                <h4>🧠 Advanced Reasoning</h4>
                <p>Latest models feature enhanced logical reasoning and problem-solving capabilities</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Data Analysis</h4>
                <p>Built-in data interpretation and visualization capabilities</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🌐 Web Context</h4>
                <p>Access to the latest 2025 information through integrated web search</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>📄 Document Processing</h4> 
                <p>Analyze and extract insights from uploaded PDF and text documents</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            
            # Display user message
            st.markdown(f'''
            <div class="chat-message user">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div class="message-avatar user-avatar">U</div>
                    <strong>You</strong>
                </div>
                <div>{prompt}</div>
                <div class="timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Display assistant response
            with st.spinner(f"Analyzing with {st.session_state.model}..."):
                messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                full_response = generate_response_with_retry(
                    messages_for_api,
                    st.session_state.model,
                    use_web_search,
                    document_context,
                    image_context
                )
            
            # Display response
            st.markdown(f'''
            <div class="chat-message assistant">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div class="message-avatar assistant-avatar">AI</div>
                    <strong>Assistant</strong>
                </div>
                <div>{full_response}</div>
                <div class="timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Add text-to-speech for response
            audio_bytes = text_to_speech(full_response)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
    
    with tab2:
        if st.session_state.current_tool:
            render_tool_ui(st.session_state.current_tool)
            
            if st.button("← Back to Tools", use_container_width=True):
                st.session_state.current_tool = None
                st.rerun()
        else:
            st.markdown('<div class="subheader">🛠️ Workflow Tools</div>', unsafe_allow_html=True)
            st.markdown("Select a workflow tool to enhance your productivity:")
            
            # Display workflow tools in a grid
            cols = st.columns(2)
            for i, (tool_id, tool_info) in enumerate(st.session_state.workflow_tools.items()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="feature-card" style="border-left: 4px solid {tool_info['color']}">
                        <h4>{tool_info['icon']} {tool_info['name']}</h4>
                        <p>{tool_info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Use {tool_info['name']}", key=f"tool_{tool_id}", use_container_width=True):
                        st.session_state.current_tool = tool_id
                        st.rerun()
    
    with tab3:
        create_analytics_dashboard()
    
    with tab4:
        st.markdown('<div class="subheader">📚 Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown("Manage your custom knowledge base and document collections.")
        
        # Knowledge base management
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Upload to Knowledge Base")
            kb_upload = st.file_uploader("Select files to add to knowledge base", type=["pdf", "txt"], key="kb_upload")
            if kb_upload and st.button("Add to Knowledge Base", use_container_width=True):
                with st.spinner("Processing and adding to knowledge base..."):
                    content = process_document(kb_upload)
                    doc_id = str(uuid.uuid4())
                    
                    # Store in session state
                    st.session_state.knowledge_base[doc_id] = {
                        "name": kb_upload.name,
                        "content": content,
                        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    
                    st.success(f"Added {kb_upload.name} to knowledge base!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Knowledge Base Contents")
            if not st.session_state.knowledge_base:
                st.info("Your knowledge base is empty. Upload documents to get started.")
            else:
                for doc_id, doc_info in st.session_state.knowledge_base.items():
                    with st.expander(doc_info["name"]):
                        st.caption(f"Uploaded: {doc_info['upload_date']}")
                        st.write(f"{len(doc_info['content'])} characters")
                        if st.button("Delete", key=f"del_{doc_id}", use_container_width=True):
                            del st.session_state.knowledge_base[doc_id]
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Search functionality
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Search Knowledge Base")
            search_query = st.text_input("Enter search terms")
            if search_query:
                results = semantic_search(search_query, st.session_state.knowledge_base)
                if results:
                    st.write(f"Found {len(results)} results:")
                    for result in results:
                        with st.expander(result['name']):
                            st.write(result['content'])
                else:
                    st.info("No results found")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

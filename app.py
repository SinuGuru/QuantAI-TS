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

# Set page configuration with professional theme
st.set_page_config(
    page_title="NeuraLink Assistant 2025",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f3d7a;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2c5aa0;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.3rem;
    }
    .chat-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #2c5aa0;
    }
    .stButton button {
        background-color: #2c5aa0;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #1f3d7a;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
    }
    .model-badge {
        background-color: #1f3d7a;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-left: 8px;
    }
    .feature-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border-left: 4px solid #2c5aa0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your NeuraLink Professional Assistant for 2025. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
        ]
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4-turbo"
    if "conversation_name" not in st.session_state:
        st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if "usage_stats" not in st.session_state:
        st.session_state.usage_stats = {"tokens": 0, "requests": 0, "cost": 0.0}
    if "client" not in st.session_state:
        st.session_state.client = None

# Initialize OpenAI client
def setup_openai():
    if st.session_state.api_key:
        st.session_state.client = OpenAI(api_key=st.session_state.api_key)
        return True
    return False

# Function to perform web search (simulated for 2025)
def web_search(query: str):
    """Simulate web search with latest 2025 data"""
    # In a real implementation, this would use Serper, SerpAPI or similar
    search_results = [
        {"title": "AI Trends 2025: GPT-4.5 Revolutionizes Enterprise Applications", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-4.5's enhanced reasoning and 128K context window are transforming business applications in 2025."},
        {"title": "OpenAI Releases GPT-4.1 and GPT-4.5: What's New", "url": "https://example.com/gpt45-update", "snippet": "GPT-4.5 features improved mathematical reasoning, better coding capabilities, and enhanced safety alignment."},
        {"title": "Multimodal AI Becomes Standard in 2025", "url": "https://example.com/multimodal-2025", "snippet": "Most enterprise AI systems now seamlessly process text, images, audio, and video in unified models."}
    ]
    return search_results

# Function to process uploaded documents
def process_document(uploaded_file):
    """Extract text from uploaded documents"""
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.type == "text/plain":
        text = str(uploaded_file.read(), "utf-8")
    return text

# Function to estimate token count
def estimate_tokens(text: str) -> int:
    """Estimate token count for a string (approx)"""
    return len(text.split()) * 1.33

# Function to calculate cost based on model and tokens
def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on OpenAI's 2025 pricing"""
    pricing = {
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
    }
    
    if model not in pricing:
        return 0.0
    
    cost = (prompt_tokens / 1000 * pricing[model]["input"] + 
            completion_tokens / 1000 * pricing[model]["output"])
    return cost

# Function to generate AI response with latest context
def generate_response(messages: List[Dict], model: str, use_web_search: bool = False, document_context: str = None) -> str:
    """Generate response from OpenAI API with enhanced 2025 context"""
    
    # Prepare system message with 2025 context
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_message = {
        "role": "system", 
        "content": f"""You are NeuraLink Assistant 2025, a professional AI assistant with knowledge up to December 2025.

Current date: {current_date}

Key 2025 Context:
- GPT-4.5 has been released with 128K context window and enhanced reasoning
- GPT-4.1 offers improved efficiency for enterprise applications
- AI regulations have evolved with the EU AI Act fully implemented
- Quantum computing is beginning to impact cryptography and optimization
- Climate tech solutions powered by AI are seeing widespread adoption

Always provide accurate, up-to-date information. If you're unsure, say so. 
Be professional, concise, and helpful.
"""
    }
    
    # Add document context if provided
    if document_context:
        system_message["content"] += f"\n\nDocument Context:\n{document_context}\n\nPlease reference this document when relevant to the user's query."
    
    # Add web search results if enabled
    web_context = ""
    if use_web_search and messages and messages[-1]["role"] == "user":
        search_query = messages[-1]["content"]
        search_results = web_search(search_query)
        if search_results:
            web_context = "\n\nCurrent Web Context:\n"
            for i, result in enumerate(search_results, 1):
                web_context += f"{i}. {result['title']}: {result['snippet']}\n"
    
    system_message["content"] += web_context
    
    try:
        # Create completion with the enhanced context using the new client API
        response = st.session_state.client.chat.completions.create(
            model=model,
            messages=[system_message] + messages,
            stream=True
        )
        
        # Collect streaming response
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                
        # Update usage statistics
        prompt_tokens = estimate_tokens(system_message["content"] + " ".join([m["content"] for m in messages]))
        completion_tokens = estimate_tokens(full_response)
        
        st.session_state.usage_stats["tokens"] += prompt_tokens + completion_tokens
        st.session_state.usage_stats["requests"] += 1
        st.session_state.usage_stats["cost"] += calculate_cost(model, prompt_tokens, completion_tokens)
                
        return full_response
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Function to create model comparison chart
def create_model_comparison():
    """Create a comparison chart of available models"""
    models_data = {
        "Model": ["GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo"],
        "Context Window": ["128K", "8K", "16K"],
        "Intelligence": [9.2, 9.0, 7.0],
        "Speed": [8, 7, 10],
        "Cost": [7.5, 9.0, 2.0]  # Relative cost (higher = more expensive)
    }
    
    df = pd.DataFrame(models_data)
    return df

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">NeuraLink Assistant 2025</h1>', unsafe_allow_html=True)
    st.caption("Professional AI Assistant with Latest GPT Models • v3.0.0")
    
    # Sidebar
    with st.sidebar:
        st.image("https://placehold.co/300x80/1f3d7a/white?text=NeuraLink+2025", use_container_width=True)
        st.markdown("---")
        
        # API Key Input
        st.subheader("Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key to begin"
        )
        
        if api_key:
            st.session_state.api_key = api_key
            if setup_openai():
                st.success("✓ API Key Configured")
        
        # Model Selection
        st.selectbox(
            "AI Model",
            ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            key="model",
            help="Select which AI model to use"
        )
        
        # Display model info
        model_info = {
            "gpt-4-turbo": "Latest model with 128K context and advanced reasoning",
            "gpt-4": "Powerful model with strong capabilities across domains",
            "gpt-3.5-turbo": "Fast and cost-effective for simpler tasks"
        }
        
        st.info(f"**{st.session_state.model}**: {model_info[st.session_state.model]
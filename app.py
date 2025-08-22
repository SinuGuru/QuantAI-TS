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
import chromadb
from chromadb.config import Settings
import hashlib
import tiktoken
from streamlit_authenticator import Authenticate
import secrets
import speech_recognition as sr
from gtts import gTTS
import io
import uuid

# Set page configuration with professional theme
st.set_page_config(
    page_title="Enterprise AI Assistant 2025",
    page_icon="🤖",
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
    .usage-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .tab-content {
        padding: 20px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
        ]
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o"
    if "conversation_name" not in st.session_state:
        st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if "usage_stats" not in st.session_state:
        st.session_state.usage_stats = {"tokens": 0, "requests": 0, "cost": 0.0}
    if "client" not in st.session_state:
        st.session_state.client = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1000
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = {}
    if "workflow_tools" not in st.session_state:
        st.session_state.workflow_tools = setup_workflow_tools()
    if "conversation_summaries" not in st.session_state:
        st.session_state.conversation_summaries = {}

# Initialize authentication
def setup_authentication():
    """Setup user authentication system"""
    # In a real application, you would use a proper database
    # For demo purposes, we'll use a simple dictionary
    credentials = {
        'usernames': {
            'jsmith': {
                'name': 'John Smith',
                'password': hashlib.sha256('password123'.encode()).hexdigest()
            },
            'rjohnson': {
                'name': 'Robert Johnson',
                'password': hashlib.sha256('password456'.encode()).hexdigest()
            }
        }
    }
    
    authenticator = Authenticate(
        credentials,
        'auth_cookie',
        'auth_key',
        cookie_expiry_days=30
    )
    
    return authenticator

# Initialize vector database
def setup_vector_db():
    """Initialize ChromaDB vector database"""
    try:
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection(name="documents")
        return collection
    except Exception as e:
        st.error(f"Error setting up vector database: {str(e)}")
        return None

# Initialize workflow tools
def setup_workflow_tools():
    """Add tools for specific professional workflows"""
    tools = {
        "code_review": {
            "name": "Code Review",
            "description": "Analyze and improve code snippets",
            "icon": "💻"
        },
        "document_generation": {
            "name": "Document Generation",
            "description": "Create professional documents",
            "icon": "📄"
        },
        "data_analysis": {
            "name": "Data Analysis",
            "description": "Analyze and visualize data",
            "icon": "📊"
        },
        "research_assistant": {
            "name": "Research Assistant",
            "description": "Help with academic research",
            "icon": "🔍"
        },
        "meeting_assistant": {
            "name": "Meeting Assistant",
            "description": "Summarize meetings and extract action items",
            "icon": "📅"
        }
    }
    return tools

# Initialize OpenAI client
def setup_openai():
    if st.session_state.api_key:
        try:
            st.session_state.client = OpenAI(api_key=st.session_state.api_key)
            # Test the client with a simple request
            st.session_state.client.models.list()
            return True
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            return False
    return False

# Function to perform web search (simulated for 2025)
def web_search(query: str):
    """Simulate web search with latest 2025 data"""
    search_results = [
        {"title": "AI Trends 2025: GPT-5 and Omni Models Transform Industries", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-5's enhanced reasoning and 1M token context window are revolutionizing enterprise applications in 2025."},
        {"title": "OpenAI Releases GPT-4.5 and GPT-5: What's New", "url": "https://example.com/gpt5-update", "snippet": "GPT-5 features improved mathematical reasoning, better coding capabilities, and enhanced multimodal understanding."},
        {"title": "Multimodal AI Becomes Standard in 2025", "url": "https://example.com/multimodal-2025", "snippet": "Most enterprise AI systems now seamlessly process text, images, audio, and video in unified models."},
        {"title": "OpenAI's o3 Series Sets New Standards", "url": "https://example.com/o3-series", "snippet": "The o3 model series offers unprecedented speed and accuracy for real-time applications."}
    ]
    return search_results

# Function to process uploaded documents
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

# Function to process images with multimodal capabilities
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
def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for a string"""
    try:
        # Try to get encoding for the specific model, fall back to cl100k_base
        try:
            encoding = tiktoken.encoding_for_model(model)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback estimation if tiktoken is not available
        return len(text.split()) * 1.33

# Function to calculate cost based on model and tokens
def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on OpenAI's 2025 pricing"""
    pricing = {
        "gpt-5": {"input": 0.015, "output": 0.06},
        "gpt-4.5": {"input": 0.012, "output": 0.045},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "o3-mini": {"input": 0.002, "output": 0.005},
        "o3-medium": {"input": 0.004, "output": 0.012},
        "o3-large": {"input": 0.008, "output": 0.024},
    }
    
    # Normalize model name
    model_key = model
    if "gpt-5" in model:
        model_key = "gpt-5"
    elif "gpt-4.5" in model:
        model_key = "gpt-4.5"
    elif "gpt-4o" in model:
        model_key = "gpt-4o"
    elif "gpt-4-turbo" in model:
        model_key = "gpt-4-turbo"
    elif "gpt-4" in model and "gpt-4-turbo" not in model:
        model_key = "gpt-4"
    elif "gpt-3.5" in model:
        model_key = "gpt-3.5-turbo"
    elif "o3" in model:
        if "mini" in model:
            model_key = "o3-mini"
        elif "medium" in model:
            model_key = "o3-medium"
        elif "large" in model:
            model_key = "o3-large"
    
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
def generate_response_with_retry(messages, model, max_retries=3):
    """Generate response with retry logic for rate limits"""
    for attempt in range(max_retries):
        try:
            return generate_response(messages, model)
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
def create_model_comparison():
    """Create a comparison chart of available models"""
    models_data = {
        "Model": ["GPT-5", "GPT-4.5", "GPT-4o", "o3-large", "o3-medium", "o3-mini"],
        "Context Window": ["1M", "128K", "128K", "128K", "128K", "128K"],
        "Intelligence": [10.0, 9.5, 9.2, 9.0, 8.5, 8.0],
        "Speed": [7, 8, 9, 9, 10, 10],
        "Cost per 1K tokens": ["$15/60", "$12/45", "$5/15", "$8/24", "$4/12", "$2/5"]
    }
    
    df = pd.DataFrame(models_data)
    return df

# Function to display usage statistics
def display_usage_stats():
    """Display usage statistics in a nice format"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tokens", f"{st.session_state.usage_stats['tokens']:,.0f}")
    with col2:
        st.metric("API Requests", st.session_state.usage_stats["requests"])
    with col3:
        st.metric("Estimated Cost", f"${st.session_state.usage_stats['cost']:.2f}")

# Function to create analytics dashboard
def create_analytics_dashboard():
    """Create a comprehensive analytics dashboard"""
    st.subheader("📈 Usage Analytics Dashboard")
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2025-01-01', end='2025-08-22', freq='D')
    usage_data = {
        'Date': dates,
        'Tokens': [max(1000, int(5000 * (0.5 + 0.5 * np.sin(i/10)))) for i in range(len(dates))],
        'Cost': [max(5, 25 * (0.5 + 0.5 * np.sin(i/7))) for i in range(len(dates))],
        'Requests': [max(10, int(50 * (0.5 + 0.5 * np.sin(i/5)))) for i in range(len(dates))]
    }
    
    df = pd.DataFrame(usage_data)
    
    # Token usage over time
    fig_tokens = px.line(df, x='Date', y='Tokens', title='Token Usage Over Time')
    st.plotly_chart(fig_tokens, use_container_width=True)
    
    # Cost analysis
    col1, col2 = st.columns(2)
    with col1:
        fig_cost = px.line(df, x='Date', y='Cost', title='Cost Over Time')
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        fig_requests = px.line(df, x='Date', y='Requests', title='API Requests Over Time')
        st.plotly_chart(fig_requests, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    model_df = create_model_comparison()
    st.dataframe(model_df, use_container_width=True, hide_index=True)

# Voice input functionality
def speech_to_text():
    """Convert speech to text using microphone input"""
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

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Setup authentication
    authenticator = setup_authentication()
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        # Login form
        st.title("Enterprise AI Assistant Login")
        
        try:
            name, authentication_status, username = authenticator.login('Login', 'main')
            
            if authentication_status:
                st.session_state.authenticated = True
                st.session_state.user_id = username
                st.rerun()
            elif authentication_status == False:
                st.error('Username/password is incorrect')
            elif authentication_status == None:
                st.warning('Please enter your username and password')
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
        
        return
    
    # User is authenticated, show the main app
    authenticator.logout('Logout', 'sidebar')
    
    # Initialize vector database
    if st.session_state.vector_db is None:
        st.session_state.vector_db = setup_vector_db()
    
    # Header
    st.markdown('<h1 class="main-header">Enterprise AI Assistant 2025</h1>', unsafe_allow_html=True)
    st.caption(f"Welcome, {authenticator.credentials['usernames'][st.session_state.user_id]['name']}! • v6.0.0")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # API Key Input
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key to begin",
            label_visibility="collapsed"
        )
        
        if api_key and api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if setup_openai():
                st.success("✓ API Key Configured")
        
        # Model Selection - Updated with latest 2025 models
        st.selectbox(
            "AI Model",
            ["gpt-5", "gpt-4.5", "gpt-4o", "o3-large", "o3-medium", "o3-mini"],
            index=2,  # Default to GPT-4o
            key="model",
            help="Select which AI model to use"
        )
        
        # Display model info
        model_info = {
            "gpt-5": "Latest flagship model with 1M context window and advanced reasoning (2025)",
            "gpt-4.5": "Enhanced version with 128K context and improved efficiency (2024)",
            "gpt-4o": "Optimized model with balanced speed and intelligence (2024)",
            "o3-large": "High-performance model in the o3 series for complex tasks",
            "o3-medium": "Balanced model in the o3 series for general use",
            "o3-mini": "Fast and efficient model in the o3 series for simple tasks"
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
        if st.button("Start Voice Input"):
            voice_text = speech_to_text()
            if voice_text and voice_text not in ["No speech detected", "Could not understand audio"]:
                st.session_state.messages.append({"role": "user", "content": voice_text, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.rerun()
        
        # Conversation management
        st.markdown("---")
        st.markdown("### 💬 Conversation")
        st.session_state.conversation_name = st.text_input("Conversation Name", value=st.session_state.conversation_name)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Conversation"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                ]
                st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                st.rerun()
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.messages = st.session_state.messages[:1]  # Keep only the first message
                st.rerun()
        
        # Export conversation
        if st.button("📤 Export Conversation"):
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
                mime="application/json"
            )
        
        # Usage statistics
        st.markdown("---")
        st.markdown("### 📊 Usage Statistics")
        display_usage_stats()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        Enterprise AI Assistant 2025 leverages the latest GPT models to provide professional assistance with up-to-date information.
        
        **Key Features:**
        - GPT-5, GPT-4.5, GPT-4o, and o3 series support
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
            st.subheader("Latest Model Comparison (2025)")
            model_df = create_model_comparison()
            st.dataframe(model_df, use_container_width=True, hide_index=True)
            
            return
        
        # Main chat interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f'<div class="subheader">Conversation: {st.session_state.conversation_name} <span class="model-badge">{st.session_state.model}</span></div>', unsafe_allow_html=True)
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.markdown(f'<div class="timestamp">{message["timestamp"]}</div>', unsafe_allow_html=True)
                    
                    # Add text-to-speech button for assistant responses
                    if message["role"] == "assistant":
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
            
            # Feature cards
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**🧠 Advanced Reasoning**")
            st.caption("Latest models feature enhanced logical reasoning and problem-solving capabilities")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**📊 Data Analysis**")
            st.caption("Built-in data interpretation and visualization capabilities")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**🌐 Web Context**")
            st.caption("Access to the latest 2025 information through integrated web search")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**📄 Document Processing**")
            st.caption("Analyze and extract insights from uploaded PDF and text documents")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
                st.markdown(f'<div class="timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
            
            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Generate response
                with st.spinner(f"Analyzing with {st.session_state.model}..."):
                    full_response = generate_response_with_retry(
                        [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        st.session_state.model,
                        use_web_search,
                        document_context,
                        image_context
                    )
                
                # Display response
                message_placeholder.markdown(full_response)
                st.markdown(f'<div class="timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
                
                # Add text-to-speech for response
                audio_bytes = text_to_speech(full_response)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
    
    with tab2:
        st.markdown('<div class="subheader">🛠️ Workflow Tools</div>', unsafe_allow_html=True)
        st.markdown("Select a workflow tool to enhance your productivity:")
        
        # Display workflow tools
        cols = st.columns(2)
        for i, (tool_id, tool_info) in enumerate(st.session_state.workflow_tools.items()):
            with cols[i % 2]:
                with st.expander(f"{tool_info['icon']} {tool_info['name']}"):
                    st.write(tool_info['description'])
                    if st.button(f"Use {tool_info['name']}", key=f"tool_{tool_id}"):
                        # Set up context for the selected tool
                        if tool_id == "code_review":
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": "I need help with code review. Please analyze this code for best practices, potential bugs, and optimization opportunities.",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                        elif tool_id == "document_generation":
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": "Help me create a professional document. I need a template for a project report with sections for executive summary, methodology, findings, and recommendations.",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                        elif tool_id == "data_analysis":
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": "I need assistance with data analysis. Can you help me understand how to approach analyzing this dataset and what visualizations would be most effective?",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                        elif tool_id == "research_assistant":
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": "I'm conducting research on AI ethics. Can you help me find relevant sources, identify key themes, and structure my literature review?",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                        elif tool_id == "meeting_assistant":
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": "I need help summarizing a meeting and extracting action items. Can you provide a template for meeting minutes and action item tracking?",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                        st.switch_page("💬 Chat")
    
    with tab3:
        create_analytics_dashboard()
    
    with tab4:
        st.markdown('<div class="subheader">📚 Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown("Manage your custom knowledge base and document collections.")
        
        # Knowledge base management
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload to Knowledge Base")
            kb_upload = st.file_uploader("Select files to add to knowledge base", type=["pdf", "txt"], key="kb_upload")
            if kb_upload and st.button("Add to Knowledge Base"):
                with st.spinner("Processing and adding to knowledge base..."):
                    content = process_document(kb_upload)
                    doc_id = str(uuid.uuid4())
                    
                    # Store in session state (in real app, would use proper database)
                    st.session_state.knowledge_base[doc_id] = {
                        "name": kb_upload.name,
                        "content": content,
                        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    
                    # Also add to vector database if available
                    if st.session_state.vector_db:
                        try:
                            st.session_state.vector_db.add(
                                documents=[content],
                                metadatas=[{"name": kb_upload.name, "type": "document"}],
                                ids=[doc_id]
                            )
                        except Exception as e:
                            st.error(f"Error adding to vector database: {str(e)}")
                    
                    st.success(f"Added {kb_upload.name} to knowledge base!")
        
        with col2:
            st.subheader("Knowledge Base Contents")
            if not st.session_state.knowledge_base:
                st.info("Your knowledge base is empty. Upload documents to get started.")
            else:
                for doc_id, doc_info in st.session_state.knowledge_base.items():
                    with st.expander(doc_info["name"]):
                        st.caption(f"Uploaded: {doc_info['upload_date']}")
                        st.write(f"{len(doc_info['content'])} characters")
                        if st.button("Delete", key=f"del_{doc_id}"):
                            del st.session_state.knowledge_base[doc_id]
                            if st.session_state.vector_db:
                                try:
                                    st.session_state.vector_db.delete(ids=[doc_id])
                                except:
                                    pass
                            st.rerun()

if __name__ == "__main__":
    # Add numpy import for analytics (not needed in production)
    try:
        import numpy as np
    except:
        pass
        
    main()
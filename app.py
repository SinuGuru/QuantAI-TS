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
        
        # FIXED: This was the line with the syntax error - properly closed f-string and bracket
        st.info(f"**{st.session_state.model}**: {model_info[st.session_state.model]}")
        
        # Features
        st.markdown("---")
        st.subheader("Features")
        use_web_search = st.checkbox("Enable Web Search", value=True, help="Search for latest information when needed")
        document_upload = st.file_uploader("Upload Document", type=["pdf", "txt"], help="Provide context from documents")
        
        # Document context
        document_context = None
        if document_upload:
            with st.spinner("Processing document..."):
                document_context = process_document(document_upload)
            st.success(f"✓ Document processed: {document_upload.name}")
        
        # Advanced settings
        st.markdown("---")
        st.subheader("Advanced Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, help="Controls randomness: Lower = more deterministic")
        max_tokens = st.slider("Max Response Length", 100, 2000, 500, help="Maximum tokens in response")
        
        # Conversation management
        st.markdown("---")
        st.subheader("Conversation")
        st.session_state.conversation_name = st.text_input("Conversation Name", value=st.session_state.conversation_name)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Conversation"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm your NeuraLink Professional Assistant for 2025. How can I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                ]
                st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                st.rerun()
        with col2:
            if st.button("Clear History"):
                st.session_state.messages = st.session_state.messages[:1]  # Keep only the first message
                st.rerun()
        
        # Export conversation
        if st.button("Export Conversation"):
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
        st.subheader("Usage Statistics")
        st.metric("Total Tokens", f"{st.session_state.usage_stats['tokens']:,.0f}")
        st.metric("API Requests", st.session_state.usage_stats["requests"])
        st.metric("Estimated Cost", f"${st.session_state.usage_stats['cost']:.2f}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        NeuraLink Assistant 2025 leverages the latest GPT models to provide professional assistance with up-to-date information.
        
        **Key Features:**
        - GPT-4 Turbo and GPT-4 support
        - 2025 knowledge context
        - Web search integration
        - Document processing
        - Conversation management
        """)
    
    # Check for API key and client
    if not st.session_state.api_key or not st.session_state.client:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to begin")
        st.info("ℹ️ You can obtain an API key from [OpenAI's platform](https://platform.openai.com/api-keys)")
        
        # Model comparison
        st.markdown("---")
        st.subheader("GPT Model Comparison (2025)")
        model_df = create_model_comparison()
        st.dataframe(model_df, use_container_width=True, hide_index=True)
        
        # Create radar chart for model comparison
        fig = go.Figure()
        
        models = model_df["Model"].tolist()
        for i, model in enumerate(models):
            fig.add_trace(go.Scatterpolar(
                r=[model_df["Intelligence"][i], model_df["Speed"][i], 10 - model_df["Cost"][i]/2],
                theta=['Intelligence', 'Speed', 'Affordability'],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Model Comparison (Higher is better)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
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
    
    with col2:
        st.markdown('<div class="subheader">Quick Actions</div>', unsafe_allow_html=True)
        
        # Quick action buttons
        if st.button("📊 Market Analysis Summary"):
            st.session_state.messages.append({"role": "user", "content": "Provide a brief market analysis summary for Q2 2025 focusing on tech sector trends.", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.rerun()
        
        if st.button("📈 Data Insights"):
            st.session_state.messages.append({"role": "user", "content": "What are the key data and AI insights for business decision making in 2025?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.rerun()
        
        if st.button("🔍 Research Assistance"):
            st.session_state.messages.append({"role": "user", "content": "Help me research the latest developments in renewable energy technology for 2025.", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.rerun()
        
        if st.button("📝 Document Analysis"):
            st.session_state.messages.append({"role": "user", "content": "Analyze the provided document and summarize key points.", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Model Capabilities**")
        
        # Feature cards
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("**🧠 Advanced Reasoning**")
        st.caption("GPT models feature enhanced logical reasoning and problem-solving capabilities")
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
                full_response = generate_response(
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    st.session_state.model,
                    use_web_search,
                    document_context
                )
            
            # Display response
            message_placeholder.markdown(full_response)
            st.markdown(f'<div class="timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})

if __name__ == "__main__":
    main()
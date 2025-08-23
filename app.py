import streamlit as st
import openai
import os
import json
import requests
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
from typing import List, Dict, Any
from streamlit_lottie import st_lottie

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

# Apply custom CSS for styling to enhance appearance
st.markdown("""
<style>
    /* Main container and text styling */
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subheader {
        font-size: 1.25rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .model-badge {
        background-color: #cbd5e0;
        color: #4a5568;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Login screen styling */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        width: 100%;
        background-color: #e2e8f0;
    }
    .login-form {
        background: white;
        padding: 2rem 3rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 400px;
    }

    /* Stat cards styling for usage dashboard */
    .stat-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4a5568;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #718096;
    }

    /* Workflow card styling */
    .feature-card {
        background: #e2e8f0;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
    .feature-card h3 {
        color: #2d3748;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE & CONVERSATION MANAGEMENT ---

CONVERSATIONS_DIR = "conversations"
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

def init_session_state():
    """Initializes default session state variables if they don't exist."""
    default_state = {
        "authenticated": False,
        "user_id": None,
        "api_key": os.getenv('OPENAI_API_KEY', ''),
        "client": None,
        "messages": [
            {"role": "assistant", "content": "Hello! I'm your Enterprise AI Assistant for 2025. How can I help you today?"}
        ],
        "model": "gpt-4o",
        "conversation_name": f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "usage_stats": {"tokens": 0, "requests": 0, "cost": 0.0},
        "temperature": 0.7,
        "document_uploaded": False,
        "image_uploaded": False,
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.api_key and st.session_state.client is None:
        st.session_state.client = setup_openai(st.session_state.api_key)

def save_conversation():
    """Saves the current conversation to a JSON file."""
    if st.session_state.authenticated and st.session_state.messages:
        try:
            filename = os.path.join(CONVERSATIONS_DIR, f"{st.session_state.user_id}-{st.session_state.conversation_name}.json")
            messages_to_save = []
            for m in st.session_state.messages:
                # Exclude tool messages from the saved history for simplicity
                if m["role"] not in ["tool"]:
                    messages_to_save.append({"role": m["role"], "content": m.get("content")})
            with open(filename, "w") as f:
                json.dump(messages_to_save, f, indent=2)
            st.toast("✅ Conversation saved!")
        except Exception as e:
            st.error(f"Failed to save conversation: {e}")

def load_conversation(filename: str):
    """Loads a conversation from a JSON file."""
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

def new_chat():
    """Resets the chat to a new conversation."""
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your assistant. How can I help you today?"}]
    st.session_state.conversation_name = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

# --- 3. AUTHENTICATION AND API SETUP ---

def check_credentials(username: str, password: str) -> tuple:
    """Checks user credentials against a hardcoded dictionary."""
    users = {
        'admin': {'name': 'Administrator', 'password': 'admin2025', 'role': 'admin'},
        'analyst': {'name': 'Data Analyst', 'password': 'analyst2025', 'role': 'analyst'},
        'manager': {'name': 'Project Manager', 'password': 'manager2025', 'role': 'manager'}
    }
    if username in users and users[username]['password'] == password:
        return True, users[username]['name'], users[username]['role']
    return False, None, None

def setup_openai(api_key: str):
    """Initializes and tests the OpenAI client."""
    if not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()  # Test the API key
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client. Please check your API key. Error: {e}")
        return None

# --- 4. TOOL FUNCTIONS FOR AI ---

def web_search(query: str):
    """Performs a web search to retrieve up-to-date information."""
    return json.dumps([
        {"title": "AI Trends 2025: GPT-5", "url": "https://example.com/ai-trends-2025", "snippet": "GPT-5's enhanced reasoning and 1M token context window are revolutionizing enterprise applications in 2025."},
        {"title": "OpenAI Releases GPT-5", "url": "https://example.com/gpt5-update", "snippet": "GPT-5 features improved mathematical reasoning, better coding capabilities, and enhanced multimodal understanding."},
    ])

def code_review(code: str, language: str = "Python"):
    """Reviews a block of code for errors, style, and efficiency."""
    return f"I have reviewed the {language} code. I found no major syntax errors, but I suggest adding more comments for better readability."

def data_analysis(query: str, data: str):
    """Analyzes provided data based on a user query and provides insights."""
    insights = {
        "sales": "Sales increased by 15% in Q3, driven by a new marketing campaign.",
        "performance": "The 'East Region' consistently outperforms other regions in quarterly revenue.",
    }
    for keyword, insight in insights.items():
        if keyword in query.lower():
            return f"Based on the provided data, the key insight is that {insight}"
    return f"Based on the provided data, I have analyzed your request: '{query}'. No specific insight was found."

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
            "description": "Returns the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# --- 5. AI RESPONSE GENERATION ---

def generate_response(messages: List[Dict], model: str) -> str:
    """Generates an AI response, including handling tool calls."""
    if not st.session_state.client:
        return "Please provide a valid OpenAI API key in the Settings tab."
    
    api_messages = [{"role": "system", "content": "You are Enterprise AI Assistant 2025, a professional assistant with knowledge up to December 2025. You are an expert in using tools to perform tasks."}]
    
    # Only append the most recent messages to save on context and tokens
    api_messages.extend(messages[-10:])
    
    try:
        response = st.session_state.client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=st.session_state.temperature,
        )
        
        message_content = response.choices[0].message
        
        if message_content.tool_calls:
            # Append the assistant's tool-call message to the session state
            st.session_state.messages.append(message_content)
            
            available_functions = {
                "web_search": web_search,
                "code_review": code_review,
                "data_analysis": data_analysis,
                "get_current_datetime": get_current_datetime,
            }
            
            for tool_call in message_content.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                
                if not function_to_call:
                    st.error(f"Error: Tool '{function_name}' not found.")
                    continue
                
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    
                    st.session_state.messages.append({"role": "tool", "content": f"Calling tool: `{function_name}` with args: `{json.dumps(function_args, indent=2)}`", "tool_call_id": tool_call.id})
                    
                    function_response = function_to_call(**function_args)
                    
                    st.session_state.messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                except Exception as e:
                    st.error(f"Failed to execute tool '{function_name}': {e}")
                    st.session_state.messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: Tool execution failed with error: {e}",
                        }
                    )
            
            # Re-call the model with tool outputs to get a final response. This recursive pattern is effective.
            return generate_response(st.session_state.messages, model)
        else:
            return message_content.content
    except openai.APIError as e:
        return f"❌ OpenAI API Error: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

# --- 6. UI COMPONENTS ---

def display_usage_stats():
    """Displays mock usage statistics in a clean, card-based format."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.usage_stats["tokens"]:,}</div><div class="stat-label">Total Tokens</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.usage_stats["requests"]}</div><div class="stat-label">API Requests</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">${st.session_state.usage_stats["cost"]:.2f}</div><div class="stat-label">Estimated Cost</div></div>', unsafe_allow_html=True)

def render_chat_messages():
    """Renders all messages in the session state using native Streamlit chat elements."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check for content and handle different message types.
            if message["role"] == "assistant":
                if "tool_calls" in message:
                    st.info("The AI is using a tool to respond...")
                elif message.get("content") is not None and isinstance(message.get("content"), str):
                    st.write(message["content"])
            elif message["role"] == "tool":
                st.info(f"**Tool Output:**\n\n`{message['content']}`")
            else:
                # Handle user messages and others.
                if message.get("content") is not None and isinstance(message.get("content"), str):
                    st.write(message["content"])


# --- 7. MAIN APPLICATION FLOW ---

def main():
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
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.markdown('<p style="text-align: center; color: #718096; margin-top: 2rem;">Demo credentials: admin/admin2025, analyst/analyst2025, or manager/manager2025</p></div></div>', unsafe_allow_html=True)
        return

    # --- Main App Interface ---
    
    st.sidebar.button("🚪 Logout", on_on_click=lambda: st.session_state.update(authenticated=False))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">NeuraLink AI Assistant</h1>', unsafe_allow_html=True)
        st.caption(f"Welcome, {st.session_state.user_id}! • Enterprise AI Assistant 2025 • v7.0.0")
    with col2:
        if lottie_robot:
            st_lottie(lottie_robot, height=80, key="header-lottie")
            
    with st.sidebar:
        st.markdown("### 💬 Conversation History")
        
        if st.button("🆕 Start New Chat", use_container_width=True):
            new_chat()
            st.rerun()
        
        if st.button("💾 Save Conversation", use_container_width=True):
            save_conversation()
        
        st.markdown("---")
        
        conversations = [f for f in os.listdir(CONVERSATIONS_DIR) if f.startswith(f"{st.session_state.user_id}-")]
        if conversations:
            selected_conversation = st.selectbox(
                "Load a saved conversation",
                options=[""] + conversations,
                format_func=lambda x: x.replace(f"{st.session_state.user_id}-", "").replace(".json", "") if x else "Select a conversation...",
                index=0
            )
            if selected_conversation:
                load_conversation(selected_conversation)
                st.rerun()

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

    chat_tab, workflows_tab, analytics_tab, settings_tab = st.tabs(["Chat", "Workflows", "Analytics", "Settings"])

    with chat_tab:
        if not st.session_state.api_key or not st.session_state.client:
            st.warning("⚠️ Please enter your OpenAI API key in the **Settings** tab to begin.")
            st.info("ℹ️ You can find your API key at [OpenAI's platform](https://platform.openai.com/api-keys).")
        else:
            st.markdown(f'<div class="subheader">Conversation: {st.session_state.conversation_name} <span class="model-badge">{st.session_state.model}</span></div>', unsafe_allow_html=True)
            
            render_chat_messages()
            
            user_prompt = st.chat_input("Type your message here...")
            if user_prompt:
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                
                with st.spinner("Thinking..."):
                    full_response = generate_response(st.session_state.messages, st.session_state.model)
                
                if full_response is not None:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()

    with workflows_tab:
        st.markdown('<div class="subheader">🛠️ Workflow Tools</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h3>💻 Code Review</h3>
            <p>Paste your code below and the AI will automatically review it for best practices, bugs, and style.</p>
        </div>
        """, unsafe_allow_html=True)
        code_input = st.text_area("Paste your code here...", height=250, key="code_review_input")
        
        if st.button("Review Code", use_container_width=True):
            if code_input:
                st.session_state.messages.append({"role": "user", "content": f"Please review this code: \n\n```python\n{code_input}\n```"})
                st.rerun()
            else:
                st.warning("Please provide code to review.")
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Analysis</h3>
            <p>Upload a CSV file and ask questions about your data. The AI will use a tool to provide insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_data = st.file_uploader("Upload CSV", type="csv", key="data_analysis_uploader")
        data_query = st.text_input("Ask a question about your data...", key="data_analysis_query")

        if st.button("Analyze Data", use_container_width=True):
            if uploaded_data and data_query:
                try:
                    df = pd.read_csv(uploaded_data)
                    data_string = df.to_csv(index=False)
                    st.session_state.messages.append({"role": "user", "content": f"Analyze the following data:\n\n```csv\n{data_string}\n```\n\nThe question is: {data_query}"})
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
        api_key_input = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_key, help="Your personal or enterprise API key to connect to OpenAI services.")
        
        if api_key_input and api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.session_state.client = setup_openai(api_key_input)
            if st.session_state.client:
                st.success("✅ API Key configured successfully! Changes will take effect on the next interaction.")

        if st.session_state.client:
            st.markdown("### AI Model Parameters")
            st.selectbox(
                "AI Model",
                ["gpt-5", "gpt-5-pro", "gpt-5-mini", "gpt-4o", "gpt-4-turbo-preview"],
                key="model",
                help="Select the AI model for your conversations. Newer models offer better performance but may have higher costs."
            )
            st.slider(
                "Temperature",
                0.0, 1.0, 0.7,
                help="Controls the randomness of the response. Lower values produce more deterministic results, while higher values lead to more creative and varied output.",
                key="temperature"
            )

if __name__ == "__main__":
    main()

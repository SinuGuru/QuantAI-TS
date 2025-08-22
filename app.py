import streamlit as st
import openai
import os
from typing import List, Dict

# Set page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
def initialize_openai_client():
    """Initialize the OpenAI client with API key"""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    
    # Check for environment variable (for deployment)
    if os.getenv('OPENAI_API_KEY'):
        st.session_state.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = st.session_state.api_key
        return True
    
    return False

# Initialize session state for messages
def initialize_chat():
    """Initialize session state for chat messages"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI assistant. How can I help you today?"}
        ]

# Function to generate AI response
def generate_response(messages: List[Dict]) -> str:
    """Generate response from OpenAI API"""
    try:
        # Create a completion with the messages
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True  # Enable streaming for better UX
        )
        
        # Collect streaming response
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.get("content"):
                full_response += chunk.choices[0].delta.content
                
        return full_response
    except openai.error.AuthenticationError:
        return "❌ Authentication error. Please check your API key."
    except openai.error.RateLimitError:
        return "❌ Rate limit exceeded. Please try again later."
    except openai.error.OpenAIError as e:
        return f"❌ OpenAI API error: {str(e)}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"

# Main app function
def main():
    # Initialize chat
    initialize_chat()
    
    # App title and description
    st.title("💬 AI Chat Assistant")
    st.caption("🚀 A ChatGPT-like chatbot powered by OpenAI and Streamlit")
    
    # Sidebar for API key input and settings
    with st.sidebar:
        st.header("Configuration")
        
        # API key input
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys"
        )
        
        if api_key_input:
            st.session_state.api_key = api_key_input
            openai.api_key = api_key_input
            st.success("API key set successfully!")
        
        # Model selection
        model_name = st.selectbox(
            "Select model",
            ("gpt-3.5-turbo", "gpt-4"),
            index=0,
            help="GPT-4 provides better responses but is more expensive and slower"
        )
        
        # Additional settings
        st.divider()
        st.subheader("Additional Settings")
        max_tokens = st.slider("Max response length", 50, 500, 250, help="Maximum number of tokens in the response")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, help="Higher values make output more random")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm your AI assistant. How can I help you today?"}
            ]
            st.rerun()
        
        # Display info
        st.divider()
        st.markdown("### About")
        st.markdown("This chatbot uses OpenAI's API to generate responses. Your conversations are not stored.")
    
    # Check if API key is available
    if not st.session_state.get('api_key') and not initialize_openai_client():
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start chatting!")
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Generate response
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Stream the response
                for chunk in response:
                    if chunk.choices[0].delta.get("content"):
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                full_response = error_message
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Run the app
if __name__ == "__main__":
    main()
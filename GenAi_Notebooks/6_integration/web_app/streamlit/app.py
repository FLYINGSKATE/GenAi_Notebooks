""
Streamlit Web Interface for LLM Integration

A user-friendly web interface for interacting with the fine-tuned LLM.
"""
import streamlit as st
import requests
import os
import time
from datetime import datetime

# Configure the app
st.set_page_config(
    page_title="LLM Chat Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        padding: 12px;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        font-weight: 600;
    }
    .message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        position: relative;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 10%;
    }
    .bot-message {
        background-color: #e3f2fd;
        margin-right: 10%;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #666;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🤖 LLM Chat Interface")
st.markdown("Interact with the fine-tuned language model through this web interface.")

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "your-api-key-here")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model parameters
    st.subheader("Generation Parameters")
    max_length = st.slider("Max Length", 50, 500, 150, 10)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 100, 50)
    
    # API Settings
    st.subheader("API Settings")
    api_url = st.text_input("API URL", API_URL)
    api_key = st.text_input("API Key", API_KEY, type="password")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
    
    # App info
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This interface connects to a fine-tuned LLM API.")

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
                <div class="message user-message">
                    <div><strong>You:</strong> {message["content"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message bot-message">
                    <div><strong>Assistant:</strong> {message["content"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.text_area("Your message:", key="user_input", height=100)

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Add user message to chat history
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Show typing indicator
        with st.spinner("Generating response..."):
            try:
                # Call the API
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "prompt": user_input,
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
                
                response = requests.post(
                    f"{api_url}/generate",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add bot response to chat history
                    timestamp = datetime.now().strftime("%H:%M")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["generated_text"],
                        "timestamp": timestamp
                    })
                    
                    # Rerun to update the chat display
                    st.experimental_rerun()
                    
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Auto-refresh the chat every 30 seconds
st.experimental_rerun()

if __name__ == "__main__":
    pass

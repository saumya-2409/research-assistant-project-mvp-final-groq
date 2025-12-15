import os
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

def get_api_key(key_name):
    """
    Tries to get API key from:
    1. Streamlit Secrets (Cloud)
    2. Environment Variables (Local/GitHub)
    """
    # Try getting from environment first (better for non-Streamlit contexts)
    key = os.getenv(key_name)
    if key:
        return key
    # 2. Try Streamlit Secrets (Cloud)
    try:
        import streamlit as st
        # Check if secrets are available and the key exists
        if hasattr(st, "secrets") and key_name in st.secrets:
            return st.secrets[key_name]
    except (ImportError, FileNotFoundError, AttributeError):
        pass

    return None

# --- Configuration Constants ---
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = get_api_key("SEMANTIC_SCHOLAR_API_KEY") 

# LLM Config
MODEL_NAME = "llama-3.1-8b-instant"
MAX_CONTEXT_CHARS = 15000  # Safety limit for Groq free tier

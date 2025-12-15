# File: src/config.py
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

def get_api_key(key_name):
    """
    Tries to get API key from:
    1. Streamlit Secrets (Cloud)
    2. Environment Variables (Local/GitHub)
    """
    try:
        return st.secrets[key_name]
    except (FileNotFoundError, AttributeError, KeyError):
        return os.getenv(key_name)

# --- Configuration Constants ---
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
# Semantic Scholar Key (Optional but recommended)
SEMANTIC_SCHOLAR_API_KEY = get_api_key("SEMANTIC_SCHOLAR_API_KEY") 

# LLM Config
MODEL_NAME = "llama-3.1-8b-instant"
MAX_CONTEXT_CHARS = 15000  # Safety limit for Groq free tier

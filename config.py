import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Provider Selection
    # Options: "gemini", "anthropic", "local", "mock"
    LLM_PROVIDER = os.getenv("AIOS_LLM_PROVIDER", "mock").lower()
    
    # Embedding Provider Selection
    # Options: "gemini", "local", "mock"
    # If not set, defaults to matching LLM_PROVIDER if possible, else "mock"
    EMBEDDING_PROVIDER = os.getenv("AIOS_EMBEDDING_PROVIDER", LLM_PROVIDER).lower()

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Local Model Configuration (Ollama)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Local Embedding Configuration
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not found. Falling back to mock.")
            cls.LLM_PROVIDER = "mock"
        
        if cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            print("WARNING: ANTHROPIC_API_KEY not found. Falling back to mock.")
            cls.LLM_PROVIDER = "mock"

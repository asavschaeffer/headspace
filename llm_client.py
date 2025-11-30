import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMClient(ABC):
    """Abstract base class for LLM interactions."""

    @abstractmethod
    def ask(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Sends a prompt to the LLM and returns the text response.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """
        Returns the embedding for the given text.
        """
        pass

class MockLLMClient(LLMClient):
    """Mock implementation for testing without API costs."""
    
    def ask(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        return f"[MOCK] Summary for prompt length {len(prompt)}"

    def get_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3] # Mock embedding

class GeminiLLMClient(LLMClient):
    """Implementation using Google's Generative AI."""
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_model = 'models/text-embedding-004'

    def ask(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        # Note: System instructions can be passed to GenerativeModel constructor or 
        # handled via prompt engineering if the specific model instance is reused.
        # For simplicity here, we just generate content.
        response = self.model.generate_content(prompt)
        return response.text

    def get_embedding(self, text: str) -> list[float]:
        import google.generativeai as genai
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document",
            title="File Content"
        )
        return result['embedding']

def get_llm_client() -> LLMClient:
    """Factory to get the appropriate LLM client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return GeminiLLMClient(api_key)
    print("No GEMINI_API_KEY found. Using MockLLMClient.")
    return MockLLMClient()

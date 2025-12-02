import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Type
from config import Config

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
    
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embedding_model = 'models/text-embedding-004'

    def ask(self, prompt: str, system_instruction: Optional[str] = None) -> str:
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

class AnthropicLLMClient(LLMClient):
    """Implementation using Anthropic's Claude."""
    
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model = "claude-3-haiku-20240307" # Cost-effective default

    def ask(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": messages
        }
        if system_instruction:
            kwargs["system"] = system_instruction
            
        message = self.client.messages.create(**kwargs)
        return message.content[0].text

    def get_embedding(self, text: str) -> list[float]:
        # Anthropic doesn't have a dedicated public embedding API in the same way yet,
        # or it's often better to use a different provider for embeddings.
        # For now, we'll return a mock or fallback to another provider if configured.
        # Ideally, we mix-and-match: Claude for text, Gemini/Local for embeddings.
        return [0.0] * 768 # Placeholder

class LocalLLMClient(LLMClient):
    """Implementation using Ollama for local inference."""
    
    def __init__(self):
        import ollama
        self.client = ollama.Client(host=Config.OLLAMA_BASE_URL)
        self.model = Config.OLLAMA_MODEL

    def ask(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})
            
        response = self.client.chat(model=self.model, messages=messages)
        return response['message']['content']

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings(model=self.model, prompt=text)
        return response['embedding']

class ProviderRegistry:
    _registry: Dict[str, Type[LLMClient]] = {
        "mock": MockLLMClient,
        "gemini": GeminiLLMClient,
        "anthropic": AnthropicLLMClient,
        "local": LocalLLMClient
    }

    @classmethod
    def get_client(cls, provider_name: str) -> LLMClient:
        provider_class = cls._registry.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(cls._registry.keys())}")
        
        try:
            return provider_class()
        except ImportError as e:
            print(f"Failed to initialize {provider_name} provider. Missing dependencies? Error: {e}")
            return MockLLMClient()
        except Exception as e:
            print(f"Error initializing {provider_name} provider: {e}")
            return MockLLMClient()

def get_llm_client() -> LLMClient:
    """Factory to get the appropriate LLM client based on config."""
    Config.validate()
    print(f"Initializing LLM provider: {Config.LLM_PROVIDER}")
    return ProviderRegistry.get_client(Config.LLM_PROVIDER)

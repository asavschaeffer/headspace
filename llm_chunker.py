#!/usr/bin/env python3
"""
LLM-based intelligent chunking with reasoning.
Uses centralized configuration with proper fallback hierarchy.
"""

import json
import requests
import re
from typing import List, Dict, Any, Optional
from config_manager import ConfigManager, ProviderType

class LLMChunker:
    """
    Chunks text into semantic units using an LLM, providing reasoning for each split.
    Uses centralized configuration with automatic fallback hierarchy.
    """
    def __init__(self, config_manager: ConfigManager, preferred_provider: Optional[str] = None):
        """
        Initializes the LLM chunker with centralized configuration.

        Args:
            config_manager: Centralized configuration manager
            preferred_provider: Optional preferred provider override
        """
        self.config_manager = config_manager
        self.model_config = config_manager.get_llm_config(preferred_provider)
        self.provider = self.model_config.provider
        self.model_name = self.model_config.name
        self.api_key = self.model_config.api_key
        self.api_url = self.model_config.url

        print(f"LLM Chunker initialized: {self.provider.value} - {self.model_name}")

    def chunk(self, text: str) -> List[Dict]:
        """
        Primary method to chunk text using the configured LLM provider.
        Automatically falls back through the provider hierarchy if needed.

        Args:
            text: The text to be chunked.

        Returns:
            A list of dictionaries, where each dictionary represents a chunk
            and contains 'text' and 'reasoning'.
        """
        prompt = self._build_prompt(text)

        try:
            if self.provider == ProviderType.OLLAMA:
                return self._chunk_with_ollama(prompt)
            elif self.provider == ProviderType.GEMINI:
                return self._chunk_with_gemini(prompt)
            elif self.provider == ProviderType.OPENAI:
                return self._chunk_with_openai(prompt)
            elif self.provider == ProviderType.MOCK:
                return self._chunk_with_mock(prompt)
            else:
                return self._fallback_chunking(text)
        except Exception as e:
            print(f"Error with {self.provider.value}: {e}")
            # Try fallback to next provider in chain
            return self._try_fallback_chunking(text)

    def _build_prompt(self, text: str) -> str:
        """Constructs the prompt for the LLM."""
        return f"""Break the following text into semantic chunks based on complete thoughts or ideas.

For each chunk, provide a brief reasoning for why you created that split.

Input text:
---
{text}
---

Respond ONLY with valid JSON in this exact format:
{{
  "chunks": [
    {{
      "text": "The first complete thought or idea as a string.",
      "reasoning": "A brief explanation of why this is a distinct chunk."
    }},
    {{
      "text": "The next semantic chunk.",
      "reasoning": "Reasoning for the split, e.g., 'Topic changed from X to Y'."
    }}
  ]
}}

JSON response:"""

    def _chunk_with_ollama(self, prompt: str) -> List[Dict]:
        """Chunks text using a local Ollama model."""
        print(f"Chunking with Ollama ({self.model_name})...")
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": prompt, "stream": False, "format": "json"},
                timeout=60
            )
            response.raise_for_status()
            response_text = response.json().get('response', '{}')
            data = json.loads(response_text)
            chunks = data.get('chunks', [])
            print(f"✓ LLM chunked into {len(chunks)} ideas.")
            return chunks
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Ollama error: {e}. Trying fallback.")
            raise e

    def _chunk_with_gemini(self, prompt: str) -> List[Dict]:
        """Chunks text using the Google Gemini API."""
        print(f"Chunking with Google Gemini ({self.model_name})...")

        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY":
            print(f"  Gemini API key not configured")
            raise ValueError("Gemini API key not configured")

        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "response_mime_type": "application/json"
                }
            }

            response = requests.post(api_url, json=payload, timeout=60)

            if response.status_code != 200:
                print(f"  Gemini API error: {response.status_code}")
                raise requests.RequestException(f"Gemini API error: {response.status_code}")

            result = response.json()

            # Extract text from Gemini response
            try:
                response_text = result['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError) as e:
                print(f"  Error parsing Gemini response: {e}")
                raise ValueError(f"Error parsing Gemini response: {e}")

            # Parse JSON response
            data = json.loads(response_text)
            chunks = data.get('chunks', [])
            print(f"✓ Gemini chunked into {len(chunks)} semantic units.")
            return chunks

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  Gemini error: {e}")
            raise e

    def _chunk_with_openai(self, prompt: str) -> List[Dict]:
        """Chunks text using the OpenAI API."""
        print(f"Chunking with OpenAI ({self.model_name})...")
        
        if not self.api_key or self.api_key == "YOUR_OPENAI_API_KEY":
            print(f"  OpenAI API key not configured")
            raise ValueError("OpenAI API key not configured")

        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60
            )

            if response.status_code != 200:
                print(f"  OpenAI API error: {response.status_code}")
                raise requests.RequestException(f"OpenAI API error: {response.status_code}")

            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            # Parse JSON response
            data = json.loads(response_text)
            chunks = data.get('chunks', [])
            print(f"✓ OpenAI chunked into {len(chunks)} semantic units.")
            return chunks

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  OpenAI error: {e}")
            raise e

    def _chunk_with_mock(self, prompt: str) -> List[Dict]:
        """Mock chunking for testing/fallback purposes."""
        print(f"Using mock chunking ({self.model_name})...")
        text = self._extract_text_from_prompt(prompt)
        
        # Simple mock chunking by sentences
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                chunks.append({
                    'text': sentence,
                    'reasoning': f'Mock chunking: Sentence {i+1}'
                })
        
        print(f"✓ Mock chunked into {len(chunks)} sentences.")
        return chunks

    def _try_fallback_chunking(self, text: str) -> List[Dict]:
        """Try fallback through the provider chain."""
        print("Trying fallback through provider chain...")
        
        # Get fallback chain from config
        fallback_chain = self.config_manager.config.get("llm", {}).get("fallback_chain", ["ollama", "gemini", "openai", "mock"])
        current_provider = self.provider.value
        
        # Find current provider in chain and try next ones
        try:
            current_index = fallback_chain.index(current_provider)
            for provider_name in fallback_chain[current_index + 1:]:
                if self.config_manager.is_provider_available(provider_name):
                    print(f"Trying fallback to {provider_name}...")
                    fallback_config = self.config_manager.get_llm_config(provider_name)
                    # Create temporary chunker with fallback config
                    temp_chunker = LLMChunker(self.config_manager, provider_name)
                    return temp_chunker.chunk(text)
        except (ValueError, IndexError):
            pass
        
        # If all else fails, use simple fallback
        return self._fallback_chunking(text)

    def _fallback_chunking(self, text: str) -> List[Dict]:
        """A simple fallback method that splits text by paragraphs."""
        print("Using simple paragraph chunking fallback.")
        paragraphs = text.split('\n\n')
        return [
            {'text': para.strip(), 'reasoning': 'Fallback: Paragraph split.'}
            for para in paragraphs if para.strip()
        ]
        
    def _extract_text_from_prompt(self, prompt: str) -> str:
        """Utility to get the original text back from a prompt for fallback."""
        match = re.search(r'Input text:\n---\n(.*?)\n---', prompt, re.DOTALL)
        return match.group(1) if match else ""

if __name__ == '__main__':
    # Example usage with centralized configuration
    try:
        config_manager = ConfigManager()
        config_manager.print_status()
        
        chunker = LLMChunker(config_manager)
        
        test_text = """The mapper evolves into a neural net using the physics of ideas. Individual retriever becomes a drafting buddy. It ripple-retrieves nearby documents to help you write."""
        
        chunks = chunker.chunk(test_text)
        
        print("\nChunks with reasoning:")
        for i, chunk in enumerate(chunks):
            print(f"\n{i+1}. {chunk['text']}")
            print(f"   → {chunk['reasoning']}")

    except Exception as e:
        print(f"An error occurred during the test: {e}")

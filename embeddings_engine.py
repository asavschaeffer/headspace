import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from config_manager import ConfigManager, ProviderType

class EmbeddingEngine:
    """Handles the generation of text embeddings using various models with centralized configuration."""

    def __init__(self, config_manager: ConfigManager, preferred_provider: Optional[str] = None):
        """
        Initializes the embedding engine with centralized configuration.

        Args:
            config_manager: Centralized configuration manager
            preferred_provider: Optional preferred provider override
        """
        self.config_manager = config_manager
        self.model_config = config_manager.get_embedding_config(preferred_provider)
        self.provider = self.model_config.provider
        self.model_name = self.model_config.name
        self.api_key = self.model_config.api_key
        self.api_url = self.model_config.url
        self.embedding_dim = 384

        # Initialize based on provider
        if self.provider == ProviderType.OLLAMA:
            self._init_ollama()
        elif self.provider == ProviderType.GEMINI:
            self._init_gemini()
        elif self.provider == ProviderType.SENTENCE_TRANSFORMERS:
            self._init_sentence_transformers()
        elif self.provider == ProviderType.MOCK:
            self._init_mock()
        else:
            print(f"⚠ Unknown provider: {self.provider}, defaulting to mock")
            self._init_mock()

        print(f"✅ EmbeddingEngine initialized: {self.provider.value} ({self.model_name})")

    def _init_ollama(self):
        """Initialize Ollama backend"""
        # Test connection
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                print(f"  ✓ Ollama connection successful")
                self.ollama_available = True
            else:
                print(f"  ⚠ Ollama unexpected response")
                self.ollama_available = False
        except:
            print(f"  ⚠ Ollama not reachable, will use fallback")
            self.ollama_available = False

    def _init_gemini(self):
        """Initialize Gemini backend"""
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY":
            print(f"  ⚠ Gemini API key not configured")

    def _init_sentence_transformers(self):
        """Initialize sentence-transformers backend"""
        print(f"  Loading SentenceTransformer: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"  ✓ Model loaded successfully. Dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"  ⚠ Error loading model: {e}")
            self.model = None

    def _init_mock(self):
        """Initialize mock backend for testing/fallback"""
        print(f"  Initializing mock embedding engine")
        self.embedding_dim = 384


    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of text strings using the configured provider.
        Automatically falls back through the provider hierarchy if needed.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A numpy array of embeddings.
        """
        if not texts:
            return np.array([])

        print(f"Generating embeddings for {len(texts)} text chunk(s) using {self.provider.value}...")

        try:
            if self.provider == ProviderType.OLLAMA:
                return self._embed_with_ollama(texts)
            elif self.provider == ProviderType.GEMINI:
                return self._embed_with_gemini(texts)
            elif self.provider == ProviderType.SENTENCE_TRANSFORMERS:
                return self._embed_with_sentence_transformers(texts)
            elif self.provider == ProviderType.MOCK:
                return self._embed_with_mock(texts)
            else:
                return self._embed_with_mock(texts)
        except Exception as e:
            print(f"Error with {self.provider.value}: {e}")
            # Try fallback to next provider in chain
            return self._try_fallback_embeddings(texts)

    def _embed_with_ollama(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Ollama"""
        try:
            all_embeddings = []

            # Ollama processes one text at a time
            for i, text in enumerate(texts):
                if i % 10 == 0 and i > 0:
                    print(f"    Progress: {i}/{len(texts)}")

                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "input": text
                    },
                    timeout=30
                )

                if response.status_code != 200:
                    print(f"  Ollama API error: {response.status_code}")
                    raise requests.RequestException(f"Ollama API error: {response.status_code}")

                result = response.json()

                # Ollama returns embeddings in different formats depending on version
                if 'embeddings' in result:
                    embedding = result['embeddings'][0] if isinstance(result['embeddings'], list) else result['embeddings']
                elif 'embedding' in result:
                    embedding = result['embedding']
                else:
                    print(f"  Unexpected Ollama response format")
                    raise ValueError("Unexpected Ollama response format")

                all_embeddings.append(embedding)

            print(f"  ✓ Generated {len(all_embeddings)} embeddings via Ollama")
            return np.array(all_embeddings)

        except Exception as e:
            print(f"  Ollama embedding error: {e}")
            raise e

    def _embed_with_gemini(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Gemini API"""
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY":
            print(f"  Gemini API key not configured")
            raise ValueError("Gemini API key not configured")

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents?key={self.api_key}"

            # Process in batches (Gemini limit is 100)
            all_embeddings = []
            batch_size = 100

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"    Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.model_name}",
                            "content": {"parts": [{"text": text}]}
                        }
                        for text in batch
                    ]
                }

                response = requests.post(url, json=payload, timeout=60)

                if response.status_code != 200:
                    print(f"  Gemini API error: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    raise requests.RequestException(f"Gemini API error: {response.status_code}")

                result = response.json()

                # Extract embeddings
                for embedding_obj in result.get('embeddings', []):
                    values = embedding_obj.get('values', [])
                    if values:
                        all_embeddings.append(values)
                    else:
                        print(f"  Missing values in Gemini response")
                        raise ValueError("Missing values in Gemini response")

            print(f"  ✓ Generated {len(all_embeddings)} embeddings via Gemini")
            return np.array(all_embeddings)

        except Exception as e:
            print(f"  Gemini embedding error: {e}")
            raise e

    def _embed_with_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local sentence-transformers"""
        try:
            # Try to import and load model if not already loaded
            if not hasattr(self, 'model') or self.model is None:
                from sentence_transformers import SentenceTransformer
                print(f"  Loading SentenceTransformer: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()

            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            print(f"  ✓ Generated {len(embeddings)} embeddings via sentence-transformers")
            return embeddings

        except ImportError:
            print(f"  sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers not installed")
        except Exception as e:
            print(f"  Sentence-transformers error: {e}")
            raise e

    def _embed_with_mock(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings for testing/fallback purposes"""
        print(f"  Using mock embeddings ({self.model_name})...")
        embeddings = []
        for text in texts:
            # Generate deterministic mock embedding based on text hash
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.embedding_dim) * 0.1

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

        print(f"  ✓ Generated {len(embeddings)} mock embeddings")
        return np.array(embeddings)

    def _try_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Try fallback through the provider chain"""
        print("Trying fallback through embedding provider chain...")
        
        # Get fallback chain from config
        fallback_chain = self.config_manager.config.get("embeddings", {}).get("fallback_chain", ["ollama", "gemini", "sentence-transformers", "mock"])
        current_provider = self.provider.value
        
        # Find current provider in chain and try next ones
        try:
            current_index = fallback_chain.index(current_provider)
            for provider_name in fallback_chain[current_index + 1:]:
                if self.config_manager.is_provider_available(provider_name):
                    print(f"Trying fallback to {provider_name}...")
                    # Create temporary embedding engine with fallback config
                    temp_engine = EmbeddingEngine(self.config_manager, provider_name)
                    return temp_engine.generate_embeddings(texts)
        except (ValueError, IndexError):
            pass
        
        # If all else fails, use mock embeddings
        return self._embed_with_mock(texts)

    def get_embedding_dim(self) -> int:
        """Returns the dimension of the embeddings."""
        return self.embedding_dim

if __name__ == '__main__':
    # Example usage and testing with centralized configuration
    try:
        config_manager = ConfigManager()
        config_manager.print_status()
        
        engine = EmbeddingEngine(config_manager)
        
        example_texts = [
            "This is a test sentence.",
            "Hello, world!",
            "Sentence transformers are great for semantic similarity."
        ]
        
        embeddings = engine.generate_embeddings(example_texts)
        
        print(f"\nGenerated {len(embeddings)} embeddings.")
        print(f"Shape of the first embedding: {embeddings[0].shape}")
        print(f"Embedding dimension from engine: {engine.get_embedding_dim()}")
        
        # Verify the shape
        assert embeddings.shape == (len(example_texts), engine.get_embedding_dim())
        
        print("\nExample embedding (first 10 dims):")
        print(embeddings[0][:10])
        
    except Exception as e:
        print(f"An error occurred during the test: {e}")

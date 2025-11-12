import json
import requests
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from config_manager import ConfigManager, ProviderType

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Handles the generation of text embeddings using various models with centralized configuration."""

    def __init__(self, config_manager: ConfigManager, preferred_provider: Optional[str] = None):
        """
        Initializes the embedding engine with centralized configuration.

        Args:
            config_manager: Centralized configuration manager
            preferred_provider: Optional preferred provider override
        """
        logger.info(f"🚀 Initializing EmbeddingEngine (preferred_provider={preferred_provider})")
        self.config_manager = config_manager
        self.model_config = config_manager.get_embedding_config(preferred_provider)
        self.provider = self.model_config.provider
        self.model_name = self.model_config.name
        self.api_key = self.model_config.api_key
        self.api_url = self.model_config.url
        self.embedding_dim = 384

        logger.debug(f"Provider: {self.provider.value}")
        logger.debug(f"Model: {self.model_name}")
        logger.debug(f"API URL: {self.api_url}")
        logger.debug(f"API Key present: {bool(self.api_key)}")

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
            logger.warning(f"Unknown provider: {self.provider}, defaulting to mock")
            self._init_mock()

        logger.info(f"✅ EmbeddingEngine ready: {self.provider.value} ({self.model_name}) [dim={self.embedding_dim}]")

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
            logger.error(f"❌ Gemini API key not configured")
            raise ValueError("Gemini API key is required but not set")
        logger.info(f"✓ Gemini initialized with API key (length={len(self.api_key)})")

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
            logger.warning("No texts provided for embedding")
            return np.array([])

        start_time = time.time()
        logger.info(f"📝 Generating embeddings for {len(texts)} chunk(s) using {self.provider.value}")
        logger.debug(f"Text samples: {[t[:50] + '...' if len(t) > 50 else t for t in texts[:2]]}")

        try:
            if self.provider == ProviderType.OLLAMA:
                result = self._embed_with_ollama(texts)
            elif self.provider == ProviderType.GEMINI:
                result = self._embed_with_gemini(texts)
            elif self.provider == ProviderType.SENTENCE_TRANSFORMERS:
                result = self._embed_with_sentence_transformers(texts)
            elif self.provider == ProviderType.MOCK:
                result = self._embed_with_mock(texts)
            else:
                result = self._embed_with_mock(texts)

            elapsed = time.time() - start_time
            logger.info(f"✅ Successfully generated {len(result)} embeddings in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Error with {self.provider.value} after {elapsed:.2f}s: {e}", exc_info=True)
            # Try fallback to next provider in chain
            logger.info(f"Attempting fallback...")
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
            logger.error(f"❌ Gemini API key not configured")
            raise ValueError("Gemini API key not configured")

        logger.debug(f"Starting Gemini embedding for {len(texts)} texts")

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents?key={self.api_key}"
            logger.debug(f"API URL: {url.replace(self.api_key, '***MASKED***')}")

            # Process in batches (Gemini limit is 100)
            all_embeddings = []
            batch_size = 100
            num_batches = (len(texts) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(texts))
                batch = texts[start_idx:end_idx]

                logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} texts)")

                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.model_name}",
                            "content": {"parts": [{"text": text}]}
                        }
                        for text in batch
                    ]
                }

                logger.debug(f"Sending request with {len(batch)} items")
                batch_start = time.time()
                response = requests.post(url, json=payload, timeout=60)
                batch_elapsed = time.time() - batch_start
                logger.debug(f"Response received in {batch_elapsed:.2f}s, status={response.status_code}")

                if response.status_code != 200:
                    logger.error(f"❌ Gemini API error: {response.status_code}")
                    logger.error(f"Response body: {response.text[:500]}")
                    raise requests.RequestException(f"Gemini API error: {response.status_code}")

                try:
                    result = response.json()
                    logger.debug(f"Response parsed successfully")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON response: {e}")
                    logger.error(f"Response text: {response.text[:500]}")
                    raise

                # Extract embeddings
                embeddings_in_response = result.get('embeddings', [])
                logger.debug(f"Found {len(embeddings_in_response)} embeddings in response")

                for idx, embedding_obj in enumerate(embeddings_in_response):
                    values = embedding_obj.get('values', [])
                    if values:
                        all_embeddings.append(values)
                        logger.debug(f"  Embedding {idx}: dimension={len(values)}")
                    else:
                        logger.error(f"❌ Missing values in embedding {idx}")
                        raise ValueError(f"Missing values in embedding {idx}")

            logger.info(f"✅ Generated {len(all_embeddings)} embeddings via Gemini")
            logger.debug(f"Final embeddings shape: {np.array(all_embeddings).shape}")
            return np.array(all_embeddings)

        except Exception as e:
            logger.error(f"❌ Gemini embedding error: {e}", exc_info=True)
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
        logger.warning("⚠️ Attempting fallback through embedding provider chain...")

        # Get fallback chain from config
        fallback_chain = self.config_manager.config.get("embeddings", {}).get("fallback_chain", ["gemini", "sentence-transformers", "ollama", "mock"])
        current_provider = self.provider.value
        logger.debug(f"Fallback chain: {fallback_chain}")
        logger.debug(f"Current provider: {current_provider}")

        # Find current provider in chain and try next ones
        try:
            current_index = fallback_chain.index(current_provider)
            for provider_name in fallback_chain[current_index + 1:]:
                logger.info(f"Trying fallback to {provider_name}...")
                try:
                    # Create temporary embedding engine with fallback config
                    temp_engine = EmbeddingEngine(self.config_manager, provider_name)
                    result = temp_engine.generate_embeddings(texts)
                    logger.info(f"✅ Fallback to {provider_name} succeeded")
                    return result
                except Exception as e:
                    logger.warning(f"Fallback to {provider_name} failed: {e}")
                    continue
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not find provider in fallback chain: {e}")

        # If all else fails, use mock embeddings
        logger.warning("🔄 All fallbacks failed, using mock embeddings")
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

# Semantic Embedding Service - Low Level Design
*Version: 1.0*  
*Date: 2025-07-13*  
*Status: Draft for Review*

## 1. Introduction

This document provides the detailed low-level design for Globule's Semantic Embedding Service, a foundational component that transforms human language into mathematical representations. The service captures the meaning, feeling, and relationships within text, enabling Globule to understand that "dog" and "puppy" are related concepts even when they share no common keywords. This semantic understanding powers the core magic of Globule - the ability to find connections between thoughts based on meaning rather than mere word matching.

### 1.1 Scope

This LLD covers:
- Core embedding generation architecture and model management
- Performance optimization strategies for sub-200ms latency
- Batching and caching mechanisms
- Integration with Ollama for local inference
- Fallback strategies and resilience patterns
- Content preprocessing and chunking
- Vector normalization and storage formats
- Quality monitoring and drift detection

### 1.2 Dependencies from HLD

From the High Level Design document:
- Dual Intelligence Services working in harmony (embedding + parsing)
- Local-first architecture with optional cloud capabilities
- Sub-500ms end-to-end processing requirement
- Integration with Orchestration Engine for collaborative processing
- Support for future multimodal content (images, audio)

## 2. Core Architecture

### 2.1 Technology Stack Decision

**Primary Technology**: Ollama with mxbai-embed-large model

**Rationale**:
- **Privacy-First**: All processing happens locally, no data leaves the user's machine
- **Cost-Effective**: No API fees for embedding generation
- **High Quality**: mxbai-embed-large achieves state-of-the-art performance on MTEB benchmarks
- **Flexible**: Supports quantization for resource-constrained environments
- **Future-Ready**: Ollama's architecture supports easy model swapping

**Architecture Pattern**: Service-oriented with provider abstraction

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

class EmbeddingProvider(ABC):
    """Abstract base for embedding providers"""
    
    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimensionality"""
        pass
```

### 2.2 Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Embedding Service API                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Request   │  │   Response   │  │  Health Check    │  │
│  │   Handler   │  │  Formatter   │  │    Endpoint      │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────────────┘  │
│         │                 │                                  │
├─────────┴─────────────────┴──────────────────────────────────┤
│                    Processing Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Content    │  │   Chunking   │  │   Embedding     │   │
│  │ Preprocessor │─→│   Strategy   │─→│   Generator     │   │
│  └──────────────┘  └──────────────┘  └────────┬────────┘   │
│                                                 │             │
├─────────────────────────────────────────────────┴─────────────┤
│                     Cache Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Memory     │  │  Persistent  │  │   Cache         │   │
│  │    Cache     │  │    Cache     │  │  Invalidator    │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                    Provider Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Ollama     │  │ HuggingFace  │  │ Sentence        │   │
│  │  Provider    │  │   Fallback   │  │ Transformers   │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

## 3. Model Management

### 3.1 Model Selection Strategy

The service implements a tiered model selection approach:

```python
class ModelRegistry:
    """Manages available embedding models and their characteristics"""
    
    MODELS = {
        'mxbai-embed-large': ModelConfig(
            provider='ollama',
            dimensions=1024,
            max_tokens=512,
            languages=['en'],
            memory_requirement_mb=700,
            performance_tier='high',
            quality_score=0.95
        ),
        'nomic-embed-text': ModelConfig(
            provider='ollama',
            dimensions=768,
            max_tokens=8192,
            languages=['en'],
            memory_requirement_mb=500,
            performance_tier='balanced',
            quality_score=0.90
        ),
        'bge-m3': ModelConfig(
            provider='ollama',
            dimensions=1024,
            max_tokens=8192,
            languages=['multilingual'],
            memory_requirement_mb=1200,
            performance_tier='high',
            quality_score=0.93
        ),
        'all-minilm': ModelConfig(
            provider='ollama',
            dimensions=384,
            max_tokens=256,
            languages=['en'],
            memory_requirement_mb=100,
            performance_tier='fast',
            quality_score=0.75
        )
    }
    
    def select_model(self, 
                    content_language: str = 'en',
                    content_length: int = 0,
                    performance_requirement: str = 'balanced') -> str:
        """Select optimal model based on requirements"""
        
        suitable_models = []
        
        for model_name, config in self.MODELS.items():
            # Check language support
            if content_language not in config.languages and 'multilingual' not in config.languages:
                continue
                
            # Check token limit
            estimated_tokens = content_length // 4  # Rough estimate
            if estimated_tokens > config.max_tokens:
                continue
                
            # Check performance tier
            if performance_requirement == 'fast' and config.performance_tier == 'high':
                continue
                
            suitable_models.append((model_name, config))
        
        # Sort by quality score descending
        suitable_models.sort(key=lambda x: x[1].quality_score, reverse=True)
        
        return suitable_models[0][0] if suitable_models else 'mxbai-embed-large'
```

### 3.2 Model Lifecycle Management

```python
class ModelManager:
    """Handles model loading, unloading, and resource management"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
        self.loaded_models = {}
        self.model_usage = {}  # Track usage for intelligent unloading
        self._lock = asyncio.Lock()
        
    async def ensure_model_loaded(self, model_name: str) -> None:
        """Ensure model is loaded in Ollama, downloading if necessary"""
        
        async with self._lock:
            if model_name in self.loaded_models:
                self.model_usage[model_name] = time.time()
                return
                
            # Check if model exists
            try:
                await self.ollama.show(model_name)
                self.loaded_models[model_name] = True
            except ModelNotFoundError:
                # Pull model
                logger.info(f"Downloading model {model_name}...")
                await self.ollama.pull(model_name)
                self.loaded_models[model_name] = True
                
            self.model_usage[model_name] = time.time()
            
            # Unload least recently used if memory pressure
            await self._manage_memory_pressure()
    
    async def _manage_memory_pressure(self):
        """Unload models if system memory is constrained"""
        
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        if available_memory < 2.0 and len(self.loaded_models) > 1:
            # Find LRU model
            lru_model = min(self.model_usage.items(), key=lambda x: x[1])[0]
            
            # Keep at least one model loaded
            if lru_model != 'mxbai-embed-large':
                await self.ollama.unload(lru_model)
                del self.loaded_models[lru_model]
                del self.model_usage[lru_model]
                logger.info(f"Unloaded model {lru_model} due to memory pressure")
```

## 4. Content Preprocessing

### 4.1 Text Normalization Pipeline

Content must be prepared carefully to ensure consistent, high-quality embeddings:

```python
class ContentPreprocessor:
    """Prepares content for embedding generation"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.unicode_normalizer = unicodedata.normalize
        
    def preprocess(self, text: str) -> str:
        """Apply preprocessing pipeline"""
        
        # Step 1: Unicode normalization (NFC for consistency)
        text = self.unicode_normalizer('NFC', text)
        
        # Step 2: Preserve but normalize URLs
        text = self.url_pattern.sub('[URL]', text)
        
        # Step 3: Preserve but normalize emails  
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Step 4: Normalize whitespace
        text = ' '.join(text.split())
        
        # Step 5: Remove zero-width characters
        text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cf')
        
        # Step 6: Truncate if necessary (preserve whole words)
        max_length = 8000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
            
        return text
```

### 4.2 Intelligent Chunking Strategy

For longer documents, we need smart chunking that preserves semantic coherence:

```python
class ChunkingStrategy:
    """Splits long content into semantically coherent chunks"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 respect_boundaries: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_boundaries = respect_boundaries
        
    def chunk_text(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """Split text into overlapping chunks"""
        
        # Estimate tokens (rough approximation)
        estimated_tokens = len(text.split()) * 1.3
        
        if estimated_tokens <= self.chunk_size:
            # No chunking needed
            return [TextChunk(
                content=text,
                start_idx=0,
                end_idx=len(text),
                metadata=metadata
            )]
            
        chunks = []
        
        if self.respect_boundaries:
            # Try to split on natural boundaries
            sentences = self._split_sentences(text)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split()) * 1.3
                
                if current_length + sentence_length > self.chunk_size:
                    # Finalize current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(TextChunk(
                        content=chunk_text,
                        start_idx=text.find(current_chunk[0]),
                        end_idx=text.find(current_chunk[-1]) + len(current_chunk[-1]),
                        metadata=metadata
                    ))
                    
                    # Start new chunk with overlap
                    overlap_sentences = self._calculate_overlap(current_chunk)
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s.split()) * 1.3 for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                    
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_text,
                    start_idx=text.find(current_chunk[0]),
                    end_idx=len(text),
                    metadata=metadata
                ))
                
        else:
            # Simple sliding window
            chunks = self._sliding_window_chunk(text, metadata)
            
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics"""
        # This is simplified - in production, use NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
```

## 5. Embedding Generation

### 5.1 Core Generation Logic

The heart of the service - actually generating embeddings:

```python
class EmbeddingGenerator:
    """Core embedding generation with optimization strategies"""
    
    def __init__(self, 
                 provider: EmbeddingProvider,
                 cache: EmbeddingCache,
                 config: EmbeddingConfig):
        self.provider = provider
        self.cache = cache
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
    async def generate(self, text: str, bypass_cache: bool = False) -> np.ndarray:
        """Generate embedding for single text"""
        
        # Check cache first
        if not bypass_cache:
            cache_key = self._compute_cache_key(text)
            if cached := await self.cache.get(cache_key):
                return cached
                
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        # Generate with concurrency control
        async with self.semaphore:
            start_time = time.time()
            
            try:
                embedding = await self.provider.embed(processed_text)
                
                # Normalize if configured
                if self.config.normalize_embeddings:
                    embedding = self._normalize_embedding(embedding)
                    
                # Cache the result
                await self.cache.set(cache_key, embedding)
                
                # Log performance
                latency = (time.time() - start_time) * 1000
                self.metrics.record_latency(latency)
                
                if latency > 200:
                    logger.warning(f"Embedding generation exceeded target: {latency:.1f}ms")
                    
                return embedding
                
            except Exception as e:
                self.metrics.record_error(str(e))
                raise EmbeddingGenerationError(f"Failed to generate embedding: {e}")
    
    async def generate_batch(self, 
                            texts: List[str], 
                            bypass_cache: bool = False) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently"""
        
        # Separate cached and uncached
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        if not bypass_cache:
            for i, text in enumerate(texts):
                cache_key = self._compute_cache_key(text)
                if cached := await self.cache.get(cache_key):
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
            
        # Process uncached in batches
        if uncached_texts:
            batch_size = self.config.optimal_batch_size
            
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_embeddings = await self._generate_batch_with_retry(batch)
                
                # Place results in correct positions
                for j, embedding in enumerate(batch_embeddings):
                    original_index = uncached_indices[i + j]
                    results[original_index] = embedding
                    
                    # Cache individual results
                    cache_key = self._compute_cache_key(texts[original_index])
                    await self.cache.set(cache_key, embedding)
                    
        return results
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalization for cosine similarity optimization"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
```

### 5.2 Ollama Provider Implementation

```python
class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama-specific embedding provider"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "mxbai-embed-large",
                 timeout: int = 30):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.session = None
        self._dimension = None
        
    async def initialize(self):
        """Initialize HTTP session and validate model"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # Validate model and get dimension
        test_response = await self.embed("test")
        self._dimension = len(test_response)
        
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding via Ollama API"""
        
        payload = {
            "model": self.model,
            "input": text,
            "truncate": True  # Handle long inputs gracefully
        }
        
        async with self.session.post(
            f"{self.base_url}/api/embed",
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise OllamaError(f"Ollama API error: {response.status} - {error_text}")
                
            data = await response.json()
            
            # Ollama returns nested structure
            if "embeddings" in data and len(data["embeddings"]) > 0:
                return np.array(data["embeddings"][0], dtype=np.float32)
            else:
                raise OllamaError("Invalid response format from Ollama")
                
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embedding via Ollama API"""
        
        # Ollama supports batch natively
        payload = {
            "model": self.model,
            "input": texts,
            "truncate": True
        }
        
        async with self.session.post(
            f"{self.base_url}/api/embed",
            json=payload
        ) as response:
            if response.status != 200:
                # Fallback to sequential on batch failure
                logger.warning("Batch embedding failed, falling back to sequential")
                return await self._sequential_fallback(texts)
                
            data = await response.json()
            
            if "embeddings" in data:
                return [np.array(emb, dtype=np.float32) for emb in data["embeddings"]]
            else:
                raise OllamaError("Invalid batch response format")
```

## 6. Caching Architecture

### 6.1 Multi-Level Cache Design

A sophisticated caching system is crucial for achieving sub-200ms performance:

```python
class MultiLevelEmbeddingCache:
    """Two-tier cache: memory (L1) and persistent (L2)"""
    
    def __init__(self,
                 memory_size_mb: int = 500,
                 disk_cache_path: Path = None,
                 ttl_seconds: int = 3600):
        
        # L1: In-memory LRU cache
        self.memory_cache = LRUCache(
            max_size=self._calculate_max_entries(memory_size_mb),
            ttl=ttl_seconds
        )
        
        # L2: Persistent disk cache (SQLite)
        self.disk_cache = DiskCache(disk_cache_path) if disk_cache_path else None
        
        # Metrics
        self.hits = 0
        self.misses = 0
        
    async def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve from cache with fallthrough"""
        
        # Check L1
        if embedding := self.memory_cache.get(key):
            self.hits += 1
            return embedding
            
        # Check L2
        if self.disk_cache and (embedding := await self.disk_cache.get(key)):
            # Promote to L1
            self.memory_cache.set(key, embedding)
            self.hits += 1
            return embedding
            
        self.misses += 1
        return None
        
    async def set(self, key: str, embedding: np.ndarray):
        """Store in both cache levels"""
        
        # Always store in L1
        self.memory_cache.set(key, embedding)
        
        # Async store in L2
        if self.disk_cache:
            asyncio.create_task(self.disk_cache.set(key, embedding))
            
    def _calculate_max_entries(self, memory_mb: int) -> int:
        """Estimate max cache entries based on memory and embedding size"""
        
        # Assume 1024-dim float32 embeddings
        bytes_per_embedding = 1024 * 4  # 4KB
        overhead_factor = 1.5  # Python object overhead
        
        max_entries = int((memory_mb * 1024 * 1024) / (bytes_per_embedding * overhead_factor))
        return max(100, max_entries)  # At least 100 entries
```

### 6.2 Cache Key Strategy

```python
class CacheKeyGenerator:
    """Generate stable, collision-resistant cache keys"""
    
    def __init__(self, include_model: bool = True):
        self.include_model = include_model
        
    def generate_key(self, 
                    text: str, 
                    model: str = None,
                    version: str = None) -> str:
        """Generate cache key from text and metadata"""
        
        # Normalize text for consistent hashing
        normalized = text.strip().lower()
        
        # Create composite key
        key_parts = [normalized]
        
        if self.include_model and model:
            key_parts.append(f"model:{model}")
            
        if version:
            key_parts.append(f"v:{version}")
            
        # Use SHA256 for consistent length and low collision
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()[:16]
```

## 7. Performance Optimization

### 7.1 Hardware Optimization

```python
class HardwareOptimizer:
    """Optimize for available hardware"""
    
    def __init__(self):
        self.has_cuda = torch.cuda.is_available() if 'torch' in sys.modules else False
        self.cpu_count = os.cpu_count()
        self.available_memory = psutil.virtual_memory().available
        
    def get_optimal_settings(self) -> dict:
        """Determine optimal settings for current hardware"""
        
        settings = {
            'device': 'cuda' if self.has_cuda else 'cpu',
            'num_threads': self.cpu_count,
            'batch_size': 1,
            'use_fp16': False,
            'quantization': None
        }
        
        if self.has_cuda:
            # GPU optimizations
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
            if gpu_memory > 8 * 1024**3:  # 8GB+
                settings['batch_size'] = 32
                settings['use_fp16'] = False
            elif gpu_memory > 4 * 1024**3:  # 4GB+
                settings['batch_size'] = 16
                settings['use_fp16'] = True
            else:
                settings['batch_size'] = 8
                settings['use_fp16'] = True
                settings['quantization'] = 'int8'
        else:
            # CPU optimizations
            if self.available_memory > 16 * 1024**3:  # 16GB+ RAM
                settings['batch_size'] = 8
            elif self.available_memory > 8 * 1024**3:  # 8GB+ RAM
                settings['batch_size'] = 4
                settings['quantization'] = 'int8'
            else:
                settings['batch_size'] = 1
                settings['quantization'] = 'q4_0'
                
        return settings
```

### 7.2 Request Batching and Queuing

```python
class BatchQueue:
    """Intelligent request batching for throughput optimization"""
    
    def __init__(self, 
                 batch_size: int = 16,
                 max_wait_ms: int = 50):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        self.processing = False
        self._lock = asyncio.Lock()
        
    async def add_request(self, text: str) -> np.ndarray:
        """Add request to queue and wait for result"""
        
        future = asyncio.Future()
        
        async with self._lock:
            self.pending_requests.append((text, future))
            
            # Process immediately if batch is full
            if len(self.pending_requests) >= self.batch_size:
                asyncio.create_task(self._process_batch())
            # Or schedule processing after timeout
            elif len(self.pending_requests) == 1:
                asyncio.create_task(self._wait_and_process())
                
        return await future
        
    async def _wait_and_process(self):
        """Wait for more requests or timeout"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        await self._process_batch()
        
    async def _process_batch(self):
        """Process all pending requests as a batch"""
        
        async with self._lock:
            if not self.pending_requests or self.processing:
                return
                
            self.processing = True
            batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
            
        try:
            texts = [text for text, _ in batch]
            embeddings = await self.embedding_generator.generate_batch(texts)
            
            # Resolve futures
            for (_, future), embedding in zip(batch, embeddings):
                future.set_result(embedding)
                
        except Exception as e:
            # Reject all futures in batch
            for _, future in batch:
                future.set_exception(e)
                
        finally:
            self.processing = False
```

## 8. Fallback and Resilience

### 8.1 Provider Fallback Chain

```python
class FallbackEmbeddingService:
    """Resilient embedding service with multiple fallback providers"""
    
    def __init__(self, providers: List[EmbeddingProvider]):
        self.providers = providers  # Ordered by preference
        self.circuit_breakers = {
            provider: CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30,
                expected_exception=EmbeddingProviderError
            ) for provider in providers
        }
        
    async def generate_with_fallback(self, text: str) -> tuple[np.ndarray, str]:
        """Try each provider until success"""
        
        last_error = None
        
        for provider in self.providers:
            breaker = self.circuit_breakers[provider]
            
            if breaker.state == CircuitBreakerState.OPEN:
                continue  # Skip failed providers
                
            try:
                async with breaker:
                    embedding = await provider.embed(text)
                    return embedding, provider.__class__.__name__
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
                continue
                
        # All providers failed
        raise AllProvidersFailed(f"All embedding providers failed. Last error: {last_error}")
```

### 8.2 Graceful Degradation

```python
class DegradationStrategy:
    """Strategies for degraded operation"""
    
    async def degrade_gracefully(self, text: str, failure_reason: str) -> np.ndarray:
        """Provide degraded but functional embedding"""
        
        if failure_reason == "model_unavailable":
            # Use simpler model
            return await self.use_fallback_model(text)
            
        elif failure_reason == "timeout":
            # Use cached similar content
            return await self.find_similar_cached(text)
            
        elif failure_reason == "resource_exhaustion":
            # Use hash-based pseudo-embedding
            return self.generate_hash_embedding(text)
            
        else:
            # Last resort: random embedding (maintains system operation)
            logger.error(f"Using random embedding due to: {failure_reason}")
            return np.random.randn(self.embedding_dimension)
            
    def generate_hash_embedding(self, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding from hash"""
        
        # Create multiple hashes for higher dimension
        hashes = []
        for i in range(self.embedding_dimension // 64):
            hasher = hashlib.sha256(f"{text}:{i}".encode())
            hash_int = int(hasher.hexdigest(), 16)
            hashes.append(hash_int)
            
        # Convert to normalized float array
        embedding = np.array(hashes, dtype=np.float32)
        embedding = (embedding / (2**256 - 1)) * 2 - 1  # Normalize to [-1, 1]
        
        return embedding[:self.embedding_dimension]
```

## 9. Quality Monitoring

### 9.1 Embedding Quality Metrics

```python
class QualityMonitor:
    """Monitor embedding quality and detect issues"""
    
    def __init__(self, reference_pairs: List[tuple[str, str, float]]):
        self.reference_pairs = reference_pairs  # (text1, text2, expected_similarity)
        self.baseline_scores = {}
        self.drift_threshold = 0.1
        
    async def establish_baseline(self, embedding_service: EmbeddingService):
        """Establish quality baseline with reference pairs"""
        
        for text1, text2, expected_sim in self.reference_pairs:
            emb1 = await embedding_service.generate(text1)
            emb2 = await embedding_service.generate(text2)
            
            actual_sim = self._cosine_similarity(emb1, emb2)
            self.baseline_scores[(text1, text2)] = actual_sim
            
            if abs(actual_sim - expected_sim) > 0.2:
                logger.warning(
                    f"Large deviation from expected similarity: "
                    f"{actual_sim:.3f} vs {expected_sim:.3f} for '{text1}' - '{text2}'"
                )
                
    async def check_quality(self, embedding_service: EmbeddingService) -> QualityReport:
        """Periodic quality check"""
        
        report = QualityReport()
        deviations = []
        
        for (text1, text2), baseline_sim in self.baseline_scores.items():
            emb1 = await embedding_service.generate(text1)
            emb2 = await embedding_service.generate(text2)
            
            current_sim = self._cosine_similarity(emb1, emb2)
            deviation = abs(current_sim - baseline_sim)
            
            if deviation > self.drift_threshold:
                deviations.append({
                    'pair': (text1, text2),
                    'baseline': baseline_sim,
                    'current': current_sim,
                    'deviation': deviation
                })
                
        report.drift_detected = len(deviations) > len(self.reference_pairs) * 0.2
        report.max_deviation = max(d['deviation'] for d in deviations) if deviations else 0
        report.affected_pairs = deviations
        
        return report
```

### 9.2 Performance Monitoring

```python
class PerformanceMonitor:
    """Track performance metrics and alert on degradation"""
    
    def __init__(self, 
                 target_p50_ms: float = 100,
                 target_p99_ms: float = 200):
        self.latencies = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.target_p50 = target_p50_ms
        self.target_p99 = target_p99_ms
        
    def record_request(self, 
                      latency_ms: float, 
                      success: bool,
                      error_type: str = None):
        """Record request metrics"""
        
        self.latencies.append(latency_ms)
        
        if not success:
            self.error_counts[error_type or 'unknown'] += 1
            
    def get_metrics(self) -> dict:
        """Calculate current performance metrics"""
        
        if not self.latencies:
            return {}
            
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        metrics = {
            'count': n,
            'p50_ms': sorted_latencies[n // 2],
            'p90_ms': sorted_latencies[int(n * 0.9)],
            'p99_ms': sorted_latencies[int(n * 0.99)],
            'mean_ms': sum(sorted_latencies) / n,
            'slo_violations': {
                'p50': sorted_latencies[n // 2] > self.target_p50,
                'p99': sorted_latencies[int(n * 0.99)] > self.target_p99
            },
            'error_rate': sum(self.error_counts.values()) / (n + sum(self.error_counts.values()))
        }
        
        return metrics
```

## 10. Testing Strategy

### 10.1 Unit Tests

```python
class TestEmbeddingService:
    """Comprehensive unit tests for embedding service"""
    
    async def test_single_embedding_generation(self):
        """Test basic embedding generation"""
        service = EmbeddingService(provider=MockProvider())
        embedding = await service.generate("test text")
        
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        assert np.allclose(np.linalg.norm(embedding), 1.0)  # Normalized
        
    async def test_batch_generation_efficiency(self):
        """Test that batching improves throughput"""
        service = EmbeddingService(provider=MockProvider())
        
        # Time individual requests
        start = time.time()
        for text in ["text1", "text2", "text3", "text4"]:
            await service.generate(text)
        individual_time = time.time() - start
        
        # Time batch request
        start = time.time()
        await service.generate_batch(["text1", "text2", "text3", "text4"])
        batch_time = time.time() - start
        
        assert batch_time < individual_time * 0.5  # At least 2x faster
        
    async def test_cache_effectiveness(self):
        """Test cache hit rates"""
        cache = MockCache()
        service = EmbeddingService(provider=MockProvider(), cache=cache)
        
        # First request - cache miss
        emb1 = await service.generate("test")
        assert cache.hits == 0
        assert cache.misses == 1
        
        # Second request - cache hit
        emb2 = await service.generate("test")
        assert cache.hits == 1
        assert cache.misses == 1
        assert np.array_equal(emb1, emb2)
        
    async def test_fallback_on_provider_failure(self):
        """Test fallback behavior"""
        providers = [FailingProvider(), WorkingProvider()]
        service = FallbackEmbeddingService(providers)
        
        embedding, provider_name = await service.generate_with_fallback("test")
        assert provider_name == "WorkingProvider"
        assert embedding is not None
```

### 10.2 Integration Tests

```python
class TestEmbeddingIntegration:
    """Integration tests with real Ollama"""
    
    @pytest.mark.integration
    async def test_ollama_connection(self):
        """Test real Ollama connection"""
        provider = OllamaEmbeddingProvider()
        await provider.initialize()
        
        embedding = await provider.embed("integration test")
        assert embedding.shape[0] == 1024
        
    @pytest.mark.integration
    async def test_model_switching(self):
        """Test switching between models"""
        service = EmbeddingService()
        
        # Generate with default model
        emb1 = await service.generate("test", model="mxbai-embed-large")
        assert emb1.shape[0] == 1024
        
        # Switch to smaller model
        emb2 = await service.generate("test", model="all-minilm")
        assert emb2.shape[0] == 384
```

### 10.3 Performance Tests

```python
class TestEmbeddingPerformance:
    """Performance benchmarks"""
    
    @pytest.mark.benchmark
    async def test_latency_targets(self):
        """Verify latency meets targets"""
        service = EmbeddingService()
        latencies = []
        
        for _ in range(100):
            start = time.time()
            await service.generate("performance test text")
            latencies.append((time.time() - start) * 1000)
            
        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)
        
        assert p50 < 100, f"P50 latency {p50:.1f}ms exceeds target"
        assert p99 < 200, f"P99 latency {p99:.1f}ms exceeds target"
```

## 11. Security Considerations

### 11.1 Input Validation

```python
class SecurityValidator:
    """Validate inputs for security concerns"""
    
    def __init__(self):
        self.max_input_size = 1_000_000  # 1MB
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=1000,
            max_requests_per_hour=10000
        )
        
    async def validate_input(self, text: str, user_id: str = None) -> ValidationResult:
        """Comprehensive input validation"""
        
        # Size check
        if len(text.encode('utf-8')) > self.max_input_size:
            return ValidationResult(
                valid=False,
                reason="Input exceeds maximum size limit"
            )
            
        # Rate limiting
        if user_id and not self.rate_limiter.check_allowed(user_id):
            return ValidationResult(
                valid=False,
                reason="Rate limit exceeded"
            )
            
        # Content validation (no PII in embeddings)
        if self._contains_sensitive_data(text):
            return ValidationResult(
                valid=False,
                reason="Input contains sensitive data patterns"
            )
            
        return ValidationResult(valid=True)
        
    def _contains_sensitive_data(self, text: str) -> bool:
        """Check for patterns that look like sensitive data"""
        
        # Credit card pattern
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
            return True
            
        # SSN pattern
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            return True
            
        # Multiple email addresses might indicate data dump
        email_count = len(re.findall(r'\S+@\S+\.\S+', text))
        if email_count > 10:
            return True
            
        return False
```

### 11.2 Local-First Security

```python
class LocalSecurityManager:
    """Ensure data never leaves the local machine unless explicitly configured"""
    
    def __init__(self, allow_remote: bool = False):
        self.allow_remote = allow_remote
        self.allowed_endpoints = {
            'localhost',
            '127.0.0.1',
            '::1'
        }
        
    def validate_endpoint(self, url: str) -> bool:
        """Ensure endpoint is local unless explicitly allowed"""
        
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        if hostname in self.allowed_endpoints:
            return True
            
        if self.allow_remote and self._is_trusted_remote(hostname):
            return True
            
        raise SecurityError(
            f"Remote endpoint {hostname} not allowed in local-first mode. "
            f"Set allow_remote=True to enable remote endpoints."
        )
```

## 12. Configuration

### 12.1 Service Configuration Schema

```yaml
# Embedding service configuration
embedding_service:
  # Model configuration
  model:
    default: "mxbai-embed-large"
    alternatives:
      - "nomic-embed-text"
      - "all-minilm"
    auto_select: true  # Choose based on content
    
  # Performance settings
  performance:
    target_latency_ms: 200
    max_concurrent_requests: 10
    batch_size: 16
    batch_timeout_ms: 50
    
  # Hardware optimization
  hardware:
    device: "auto"  # auto, cpu, cuda
    num_threads: 8
    use_fp16: false
    quantization: null  # null, int8, q4_0
    
  # Caching
  cache:
    enabled: true
    memory_size_mb: 500
    disk_cache_path: "~/.globule/cache/embeddings"
    ttl_seconds: 3600
    
  # Ollama settings
  ollama:
    base_url: "http://localhost:11434"
    timeout_seconds: 30
    keep_alive: "5m"
    
  # Fallback providers
  fallback:
    providers:
      - type: "huggingface"
        enabled: false
        api_key: null
      - type: "sentence_transformers"
        enabled: true
        model_path: "~/.globule/models/all-MiniLM-L6-v2"
        
  # Monitoring
  monitoring:
    quality_checks: true
    check_interval_minutes: 60
    reference_pairs:
      - ["dog", "puppy", 0.8]
      - ["car", "automobile", 0.9]
      - ["happy", "sad", 0.2]
```

## 13. API Specification

### 13.1 Internal API

```python
class EmbeddingServiceAPI:
    """Internal API for other Globule components"""
    
    async def generate_embedding(self, 
                                text: str,
                                options: EmbeddingOptions = None) -> EmbeddingResult:
        """Generate embedding for single text
        
        Args:
            text: Input text to embed
            options: Optional configuration overrides
            
        Returns:
            EmbeddingResult with embedding vector and metadata
            
        Raises:
            EmbeddingGenerationError: If generation fails
            ValidationError: If input is invalid
        """
        
    async def generate_embeddings_batch(self,
                                       texts: List[str],
                                       options: EmbeddingOptions = None) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            options: Optional configuration overrides
            
        Returns:
            List of EmbeddingResults in same order as input
            
        Raises:
            EmbeddingGenerationError: If generation fails
            ValidationError: If any input is invalid
        """
        
    async def calculate_similarity(self,
                                  embedding1: np.ndarray,
                                  embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        
    async def find_similar(self,
                          query_embedding: np.ndarray,
                          candidate_embeddings: List[np.ndarray],
                          top_k: int = 10) -> List[SimilarityResult]:
        """Find most similar embeddings from candidates
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: List of candidate vectors
            top_k: Number of results to return
            
        Returns:
            List of SimilarityResults with indices and scores
        """
```

### 13.2 Data Models

```python
@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding: np.ndarray
    model: str
    dimension: int
    generation_time_ms: float
    cached: bool
    metadata: dict = field(default_factory=dict)
    
@dataclass
class EmbeddingOptions:
    """Options for embedding generation"""
    model: Optional[str] = None
    bypass_cache: bool = False
    normalize: bool = True
    timeout_seconds: Optional[int] = None
    
@dataclass
class SimilarityResult:
    """Result of similarity search"""
    index: int
    score: float
    metadata: Optional[dict] = None
```

## 14. Future Enhancements

### 14.1 Planned Improvements

1. **Multimodal Embeddings**
   - Image embeddings via CLIP
   - Audio embeddings via speech models
   - Unified embedding space for all content types

2. **Advanced Chunking**
   - Semantic chunking using sentence embeddings
   - Hierarchical chunking for documents
   - Context-aware overlap strategies

3. **Model Fine-tuning**
   - Domain-specific fine-tuning on user data
   - Personalized embeddings based on usage patterns
   - Active learning from user feedback

4. **Performance Enhancements**
   - GPU clustering for large-scale processing
   - Distributed caching with Redis
   - Streaming embedding generation

5. **Quality Improvements**
   - Automated A/B testing of models
   - Continuous quality monitoring
   - Adaptive model selection

## 15. Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Ollama as primary provider | Local-first, privacy, cost-effective | 2025-07-13 |
| mxbai-embed-large as default | Best quality/performance balance | 2025-07-13 |
| 1024 dimensions | Good balance of quality and storage | 2025-07-13 |
| L2 normalization by default | Optimizes cosine similarity calc | 2025-07-13 |
| Two-tier caching | Balance memory usage and hit rate | 2025-07-13 |
| SHA256 for cache keys | Low collision, consistent length | 2025-07-13 |
| 200ms latency target | Responsive UX while achievable | 2025-07-13 |

---

*This LLD provides the complete blueprint for implementing a production-ready Semantic Embedding Service that balances performance, quality, and reliability while maintaining Globule's privacy-first principles.*
# Orchestration Engine - Low Level Design

## 1. Component Overview

The Orchestration Engine coordinates the dual-track processing pipeline by managing parallel execution of the Semantic Embedding Service and Structural Parsing Service, intelligently combining their outputs, and making content-aware decisions about processing strategies. It acts as the central intelligence coordinator that ensures these services work in harmony rather than isolation.

### 1.1 Core Responsibilities

The Orchestration Engine is responsible for:
- Analyzing input content to determine optimal processing weights between semantic and structural analysis
- Coordinating parallel or sequential execution of AI services based on content characteristics
- Detecting and preserving disagreements between services (such as sarcasm or metaphor)
- Managing processing timeouts and handling service failures gracefully
- Generating file path recommendations by combining semantic and structural insights
- Tracking processing metrics for performance optimization

### 1.2 Component Boundaries

The Orchestration Engine operates within these constraints:
- Receives EnrichedInput from the Adaptive Input Module
- Outputs ProcessedGlobule to the Intelligent Storage Manager
- Calls the Semantic Embedding Service for vector generation
- Calls the Structural Parsing Service for entity and metadata extraction
- Queries the Configuration System for runtime settings
- Does not directly access the database or filesystem

## 2. Data Structures and Contracts

### 2.1 Input Contract

```python
@dataclass
class EnrichedInput:
    """Input received from Adaptive Input Module"""
    original_text: str                    # Raw user input
    enriched_text: str                    # Text after schema processing
    detected_schema_id: Optional[str]     # e.g., "link_curation", "task_entry"
    schema_config: Optional[Dict[str, Any]]  # Schema-specific settings
    additional_context: Dict[str, Any]    # User corrections, clarifications
    source: str                          # "cli", "api", "tui"
    timestamp: datetime
    session_id: Optional[str]            # For context tracking
    verbosity: str = "concise"           # "silent", "concise", "verbose"
```

### 2.2 Output Contract

```python
@dataclass
class ProcessedGlobule:
    """Output sent to Intelligent Storage Manager"""
    # Core content
    text: str                            # Original text
    embedding: np.ndarray                # Final embedding vector (1024-d)
    embedding_confidence: float          # 0.0-1.0
    
    # Structured data from parsing
    parsed_data: Dict[str, Any]          # Entities, categories, metadata
    parsing_confidence: float            # 0.0-1.0
    
    # File organization
    file_decision: FileDecision          # Suggested path and metadata
    
    # Processing metadata
    processing_time_ms: Dict[str, float] # Breakdown by stage
    orchestration_strategy: str          # "parallel", "sequential", "iterative"
    confidence_scores: Dict[str, float]  # Per-component confidence
    
    # Disagreement handling
    interpretations: List[Interpretation] # Multiple possible interpretations
    has_nuance: bool                    # Sarcasm, metaphor detected
    
    # Context
    semantic_neighbors: List[str]        # UUIDs of related content
    processing_notes: List[str]          # Warnings, info for debugging

@dataclass
class FileDecision:
    """File organization recommendation"""
    semantic_path: Path                  # e.g., /writing/fantasy/dragons/
    filename: str                        # e.g., dragon-lore-fire-breathing.md
    metadata: Dict[str, Any]             # Additional file metadata
    confidence: float                    # 0.0-1.0
    alternative_paths: List[Path]        # Other considered paths

@dataclass
class Interpretation:
    """Represents one possible interpretation of content"""
    type: str                           # "literal", "semantic", "contextual"
    confidence: float
    data: Dict[str, Any]
    source: str                         # Which service generated this
```

### 2.3 Internal Data Structures

```python
@dataclass
class ContentProfile:
    """Content characteristics for strategy selection"""
    structure_score: float              # 0.0-1.0 (code, lists, tables)
    creativity_score: float             # 0.0-1.0 (prose, poetry)
    technical_score: float              # 0.0-1.0 (jargon, formulas)
    length: int                         # Character count
    estimated_tokens: int               # For LLM context planning
    detected_languages: List[str]       # Programming and natural languages
    has_urls: bool
    has_code_blocks: bool
    entity_density: float               # Entities per 100 words
    processing_complexity: float        # Estimated processing time multiplier

@dataclass
class ProcessingContext:
    """Runtime context for orchestration decisions"""
    session_context: List[ProcessedGlobule]  # Recent processing history
    available_memory_mb: int
    gpu_available: bool
    service_health: Dict[str, ServiceHealth]
    current_load: float                      # 0.0-1.0 system load

class ServiceHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
```

## 3. Core Algorithms and Processing Logic

### 3.1 Content Profiling Algorithm

The content profiler analyzes input text to determine optimal processing strategies within a 10ms budget.

```python
def profile_content(text: str, schema_id: Optional[str]) -> ContentProfile:
    """
    Analyze content characteristics for strategy selection.
    Must complete within 10ms budget.
    """
    profile = ContentProfile()
    
    # Quick length analysis (< 1ms)
    profile.length = len(text)
    profile.estimated_tokens = len(text.split()) * 1.3
    
    # Parallel regex scanning (< 3ms)
    # Use compiled regex patterns for performance
    profile.has_urls = bool(URL_PATTERN.search(text))
    profile.has_code_blocks = bool(CODE_BLOCK_PATTERN.search(text))
    
    # Structure detection (< 3ms)
    lines = text.split('\n')
    list_lines = sum(1 for line in lines if LIST_PATTERN.match(line))
    code_lines = sum(1 for line in lines if looks_like_code(line))
    total_lines = len(lines)
    
    profile.structure_score = min(1.0, (list_lines + code_lines) / max(total_lines, 1))
    
    # Language detection (< 2ms)
    # Sample first 500 chars for speed
    sample = text[:500]
    profile.detected_languages = detect_languages_fast(sample)
    
    if any(lang in PROGRAMMING_LANGUAGES for lang in profile.detected_languages):
        profile.technical_score = 0.8
    
    # Entity density estimation (< 1ms)
    # Use simple heuristics instead of full NLP
    capitalized_words = len(CAPITALIZED_PATTERN.findall(text))
    total_words = len(text.split())
    profile.entity_density = (capitalized_words / max(total_words, 1)) * 100
    
    # Creativity score based on vocabulary diversity
    unique_words = len(set(text.lower().split()))
    profile.creativity_score = min(1.0, unique_words / max(total_words, 1) * 2)
    
    # Processing complexity estimation
    profile.processing_complexity = calculate_complexity(profile)
    
    return profile
```

### 3.2 Strategy Selection Logic

Based on content profile and system state, select the optimal orchestration strategy.

```python
def select_orchestration_strategy(
    profile: ContentProfile,
    schema_config: Optional[Dict],
    context: ProcessingContext
) -> OrchestrationStrategy:
    """
    Determine optimal processing strategy based on content and context.
    Returns configured strategy instance.
    """
    # Schema override takes precedence
    if schema_config and 'orchestration_mode' in schema_config:
        mode = schema_config['orchestration_mode']
        if mode == 'parallel':
            return ParallelStrategy()
        elif mode == 'sequential':
            return SequentialStrategy(order=schema_config.get('order', 'embed_first'))
        elif mode == 'iterative':
            return IterativeStrategy(max_iterations=schema_config.get('max_iterations', 3))
    
    # High structure content (code, data) benefits from sequential parse-first
    if profile.structure_score > 0.7 and profile.technical_score > 0.6:
        return SequentialStrategy(order='parse_first')
    
    # Creative content benefits from iterative refinement if we have time
    if profile.creativity_score > 0.7 and profile.processing_complexity < 0.8:
        # Only use iterative if system load is low
        if context.current_load < 0.5:
            return IterativeStrategy(max_iterations=2)
    
    # Short content can afford parallel processing
    if profile.estimated_tokens < 100:
        return ParallelStrategy()
    
    # Default to parallel for balance of speed and quality
    return ParallelStrategy()
```

### 3.3 Parallel Processing Strategy

Execute embedding and parsing concurrently, combining results.

```python
async def execute_parallel_strategy(
    text: str,
    profile: ContentProfile,
    context: ProcessingContext
) -> ProcessingResult:
    """
    Execute embedding and parsing in parallel.
    Total timeout: 400ms (leaving 100ms for overhead).
    """
    # Calculate processing weights
    weights = calculate_processing_weights(profile)
    
    # Prepare service calls with timeouts
    embed_timeout = min(200, 150 + profile.processing_complexity * 50)
    parse_timeout = min(200, 150 + profile.processing_complexity * 50)
    
    # Launch parallel tasks
    embed_task = asyncio.create_task(
        call_embedding_service(text, timeout=embed_timeout)
    )
    parse_task = asyncio.create_task(
        call_parsing_service(text, profile, timeout=parse_timeout)
    )
    
    # Wait for both with individual error handling
    results = await asyncio.gather(
        embed_task,
        parse_task,
        return_exceptions=True
    )
    
    embedding_result, parsing_result = results
    
    # Handle partial failures
    if isinstance(embedding_result, Exception):
        embedding = generate_fallback_embedding(text)
        embed_confidence = 0.3
    else:
        embedding = embedding_result.vector
        embed_confidence = embedding_result.confidence
    
    if isinstance(parsing_result, Exception):
        parsed_data = generate_fallback_parsing(text)
        parse_confidence = 0.3
    else:
        parsed_data = parsing_result.data
        parse_confidence = parsing_result.confidence
    
    # Combine results with weights
    return ProcessingResult(
        embedding=embedding,
        parsed_data=parsed_data,
        weights=weights,
        confidences={
            'embedding': embed_confidence,
            'parsing': parse_confidence
        }
    )
```

### 3.4 Sequential Processing Strategy

Execute services in order, using output from one to enhance the other.

```python
async def execute_sequential_strategy(
    text: str,
    profile: ContentProfile,
    context: ProcessingContext,
    order: str = 'embed_first'
) -> ProcessingResult:
    """
    Execute services sequentially with context passing.
    Order can be 'embed_first' or 'parse_first'.
    """
    if order == 'embed_first':
        # Generate initial embedding
        embedding_result = await call_embedding_service(text, timeout=150)
        
        # Find semantic neighbors for context
        neighbors = await find_semantic_neighbors(
            embedding_result.vector,
            limit=5,
            timeout=50
        )
        
        # Parse with semantic context
        parsing_context = {
            'semantic_neighbors': neighbors,
            'embedding_confidence': embedding_result.confidence
        }
        parsing_result = await call_parsing_service(
            text,
            profile,
            context=parsing_context,
            timeout=200
        )
        
        return ProcessingResult(
            embedding=embedding_result.vector,
            parsed_data=parsing_result.data,
            semantic_neighbors=neighbors,
            confidences={
                'embedding': embedding_result.confidence,
                'parsing': parsing_result.confidence
            }
        )
    
    else:  # parse_first
        # Parse to extract structure
        parsing_result = await call_parsing_service(text, profile, timeout=150)
        
        # Enhance text with parsed entities for better embedding
        enhanced_text = enhance_text_with_entities(text, parsing_result.data)
        
        # Generate embedding with enhanced context
        embedding_result = await call_embedding_service(
            enhanced_text,
            timeout=200
        )
        
        return ProcessingResult(
            embedding=embedding_result.vector,
            parsed_data=parsing_result.data,
            enhanced_text=enhanced_text,
            confidences={
                'embedding': embedding_result.confidence,
                'parsing': parsing_result.confidence
            }
        )
```

### 3.5 Iterative Processing Strategy

Refine understanding through multiple passes, used for complex content when time permits.

```python
async def execute_iterative_strategy(
    text: str,
    profile: ContentProfile,
    context: ProcessingContext,
    max_iterations: int = 3
) -> ProcessingResult:
    """
    Iteratively refine understanding through multiple passes.
    Each iteration uses previous results to improve accuracy.
    """
    current_embedding = None
    current_parsing = None
    iteration_history = []
    
    for iteration in range(max_iterations):
        # Check if we have time for another iteration
        elapsed = sum(h['time_ms'] for h in iteration_history)
        if elapsed > 350:  # Leave buffer for final processing
            break
        
        # Build context from previous iteration
        iteration_context = {
            'iteration': iteration,
            'previous_embedding': current_embedding,
            'previous_parsing': current_parsing,
            'convergence_history': iteration_history
        }
        
        # Parallel execution with context
        embed_task = asyncio.create_task(
            call_embedding_service(
                text,
                context=iteration_context,
                timeout=100
            )
        )
        parse_task = asyncio.create_task(
            call_parsing_service(
                text,
                profile,
                context=iteration_context,
                timeout=100
            )
        )
        
        results = await asyncio.gather(embed_task, parse_task)
        new_embedding = results[0].vector
        new_parsing = results[1].data
        
        # Check for convergence
        if iteration > 0:
            embedding_change = cosine_distance(current_embedding, new_embedding)
            parsing_change = calculate_parsing_diff(current_parsing, new_parsing)
            
            iteration_history.append({
                'iteration': iteration,
                'embedding_change': embedding_change,
                'parsing_change': parsing_change,
                'time_ms': results[0].processing_time + results[1].processing_time
            })
            
            # Early exit if converged
            if embedding_change < 0.05 and parsing_change < 0.1:
                break
        
        current_embedding = new_embedding
        current_parsing = new_parsing
    
    return ProcessingResult(
        embedding=current_embedding,
        parsed_data=current_parsing,
        iteration_count=len(iteration_history) + 1,
        convergence_history=iteration_history
    )
```

### 3.6 Disagreement Detection Algorithm

Identify when semantic and structural analyses disagree, particularly for nuanced content.

```python
def detect_disagreements(
    embedding_result: EmbeddingResult,
    parsing_result: ParsingResult,
    text: str
) -> List[Disagreement]:
    """
    Detect semantic-structural disagreements using multiple techniques.
    Returns list of detected disagreements with confidence scores.
    """
    disagreements = []
    
    # 1. Sentiment disagreement detection
    parsed_sentiment = parsing_result.data.get('sentiment', 'neutral')
    
    # Get embedding's sentiment by comparing to reference vectors
    semantic_sentiment = classify_embedding_sentiment(embedding_result.vector)
    
    if parsed_sentiment != semantic_sentiment:
        # Calculate disagreement severity
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        severity = abs(
            sentiment_map.get(parsed_sentiment, 0) - 
            sentiment_map.get(semantic_sentiment, 0)
        )
        
        disagreements.append(Disagreement(
            type='sentiment',
            literal_interpretation=parsed_sentiment,
            semantic_interpretation=semantic_sentiment,
            confidence=0.8 * severity,
            explanation=f"Text says '{parsed_sentiment}' but means '{semantic_sentiment}'"
        ))
    
    # 2. Sarcasm detection via contradiction
    if contains_positive_words(text) and semantic_sentiment == 'negative':
        # Apply SparseCL-inspired algorithm
        sparse_score = calculate_sparse_contrast(
            parsing_result.data.get('word_sentiments', {}),
            embedding_result.vector
        )
        
        if sparse_score > 0.7:
            disagreements.append(Disagreement(
                type='sarcasm',
                literal_interpretation='positive',
                semantic_interpretation='negative',
                confidence=sparse_score,
                explanation="Possible sarcasm detected"
            ))
    
    # 3. Entity type disagreement
    parsed_entities = parsing_result.data.get('entities', [])
    semantic_topics = get_embedding_topics(embedding_result.vector)
    
    for entity in parsed_entities:
        if entity['type'] == 'person' and 'technical' in semantic_topics:
            # Possible metaphorical use
            disagreements.append(Disagreement(
                type='metaphor',
                literal_interpretation=f"{entity['text']} as person",
                semantic_interpretation=f"{entity['text']} as concept",
                confidence=0.6,
                explanation="Possible personification or metaphor"
            ))
    
    return disagreements

def calculate_sparse_contrast(word_sentiments: Dict, embedding: np.ndarray) -> float:
    """
    SparseCL-inspired contradiction detection.
    Combines cosine similarity with Hoyer sparsity measure.
    """
    # Create word sentiment vector
    word_vector = np.zeros(len(embedding))
    for word, sentiment in word_sentiments.items():
        # Simple hash-based positioning
        idx = hash(word) % len(word_vector)
        word_vector[idx] = sentiment
    
    # Cosine similarity
    cos_sim = np.dot(embedding, word_vector) / (
        np.linalg.norm(embedding) * np.linalg.norm(word_vector)
    )
    
    # Hoyer sparsity of difference
    diff = embedding - word_vector
    l1_norm = np.sum(np.abs(diff))
    l2_norm = np.sqrt(np.sum(diff**2))
    
    if l2_norm == 0:
        return 0.0
    
    d = len(diff)
    hoyer = (np.sqrt(d) - (l1_norm / l2_norm)) / (np.sqrt(d) - 1)
    
    # Combine measures (high score = high contradiction)
    sparse_score = (1 - cos_sim) * 0.7 + hoyer * 0.3
    
    return min(1.0, max(0.0, sparse_score))
```

### 3.7 File Path Generation Algorithm

Combine semantic and structural insights to generate intuitive file paths.

```python
def generate_file_decision(
    embedding: np.ndarray,
    parsed_data: Dict[str, Any],
    weights: Dict[str, float],
    existing_structure: FileSystemStructure
) -> FileDecision:
    """
    Generate semantic file path using weighted combination of insights.
    """
    path_components = []
    
    # 1. Determine primary category
    if weights['parsing'] > 0.7:
        # Parsing-dominant: use extracted categories
        primary_category = parsed_data.get('category', 'general')
        subcategory = parsed_data.get('subcategory', '')
    else:
        # Embedding-dominant: find semantic cluster
        cluster = find_best_cluster(embedding, existing_structure.clusters)
        primary_category = cluster.name
        subcategory = cluster.subcategory
    
    path_components.append(sanitize_path_component(primary_category))
    if subcategory:
        path_components.append(sanitize_path_component(subcategory))
    
    # 2. Add distinguishing elements based on entities
    entities = parsed_data.get('entities', [])
    key_entities = filter_key_entities(entities)
    
    for entity in key_entities[:2]:  # Max 2 entity-based subdirs
        if len(path_components) < 4:  # Max depth constraint
            path_components.append(sanitize_path_component(entity['text']))
    
    # 3. Generate filename
    title = parsed_data.get('title', '')
    if not title:
        # Extract from first line or entities
        title = extract_title(parsed_data, text[:100])
    
    filename_base = sanitize_filename(title)
    
    # 4. Handle collisions
    semantic_path = Path(*path_components)
    full_path = semantic_path / f"{filename_base}.md"
    
    collision_count = count_similar_files(full_path, existing_structure)
    if collision_count > 0:
        # Add distinguisher
        distinguisher = extract_distinguisher(parsed_data, embedding)
        filename = f"{filename_base}_{distinguisher}.md"
    else:
        filename = f"{filename_base}.md"
    
    # 5. Calculate alternatives
    alternatives = []
    
    # Pure semantic path
    semantic_cluster = find_nearest_files(embedding, existing_structure, limit=1)
    if semantic_cluster:
        alternatives.append(semantic_cluster[0].parent)
    
    # Pure structural path
    if parsed_data.get('type') == 'task':
        alternatives.append(Path('tasks') / parsed_data.get('project', 'general'))
    
    return FileDecision(
        semantic_path=semantic_path,
        filename=filename,
        metadata={
            'primary_category': primary_category,
            'entities': key_entities,
            'semantic_cluster_id': cluster.id if 'cluster' in locals() else None
        },
        confidence=calculate_path_confidence(path_components, existing_structure),
        alternative_paths=alternatives[:3]
    )
```

## 4. Service Integration

### 4.1 Embedding Service Interface

```python
class EmbeddingServiceClient:
    """Client for interacting with Semantic Embedding Service"""
    
    def __init__(self, base_url: str, timeout: float = 200):
        self.base_url = base_url
        self.timeout = timeout
        self.session = aiohttp.ClientSession()
    
    async def embed(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for text.
        
        Args:
            text: Input text to embed
            context: Optional context from previous iterations
            timeout: Override default timeout (ms)
        
        Returns:
            EmbeddingResult with vector and metadata
            
        Raises:
            EmbeddingServiceError: Service unavailable or error
            TimeoutError: Request exceeded timeout
        """
        payload = {
            'text': text,
            'context': context or {},
            'model_hint': 'mxbai-embed-large'
        }
        
        try:
            async with asyncio.timeout(timeout or self.timeout / 1000):
                async with self.session.post(
                    f"{self.base_url}/embed",
                    json=payload
                ) as response:
                    if response.status != 200:
                        raise EmbeddingServiceError(
                            f"Embedding service returned {response.status}"
                        )
                    
                    data = await response.json()
                    return EmbeddingResult(
                        vector=np.array(data['embedding']),
                        confidence=data.get('confidence', 1.0),
                        model_used=data.get('model', 'unknown'),
                        processing_time=data.get('processing_time_ms', 0)
                    )
                    
        except asyncio.TimeoutError:
            raise TimeoutError(f"Embedding service timeout after {timeout}ms")
```

### 4.2 Parsing Service Interface

```python
class ParsingServiceClient:
    """Client for interacting with Structural Parsing Service"""
    
    async def parse(
        self,
        text: str,
        profile: ContentProfile,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 200
    ) -> ParsingResult:
        """
        Parse text to extract structured data.
        
        Args:
            text: Input text to parse
            profile: Content profile for optimization
            context: Optional context (semantic neighbors, etc.)
            timeout: Request timeout in ms
            
        Returns:
            ParsingResult with extracted data
        """
        # Build parsing hints from profile
        hints = {
            'expected_entities': [],
            'expected_structure': 'prose'
        }
        
        if profile.technical_score > 0.7:
            hints['expected_entities'].extend(['function', 'class', 'variable'])
            hints['expected_structure'] = 'code'
        elif profile.structure_score > 0.7:
            hints['expected_structure'] = 'structured'
        
        payload = {
            'text': text,
            'hints': hints,
            'context': context or {},
            'model_hint': 'llama3.2:3b'
        }
        
        # Similar HTTP handling as embedding service
        # Returns: ParsingResult with entities, categories, sentiment, etc.
```

### 4.3 Storage Manager Interface

```python
class StorageQueryClient:
    """Client for querying existing content for context"""
    
    async def find_semantic_neighbors(
        self,
        embedding: np.ndarray,
        limit: int = 5,
        timeout: float = 50
    ) -> List[NeighborInfo]:
        """
        Find semantically similar content.
        
        Returns:
            List of neighboring content with similarity scores
        """
        # Query vector similarity search
        # Returns: List of UUIDs with similarity scores
```

## 5. Configuration Parameters

### 5.1 Configuration Schema

```yaml
orchestration:
  # Processing strategies
  default_strategy: "parallel"  # parallel, sequential, iterative
  enable_iterative: true       # Allow iterative for complex content
  
  # Performance tuning
  max_processing_time_ms: 400  # Leave 100ms buffer for 500ms target
  content_profiling_timeout_ms: 10
  embedding_timeout_ms: 200
  parsing_timeout_ms: 200
  
  # Parallel processing
  parallel:
    enable_result_correlation: true
    correlation_timeout_ms: 50
  
  # Sequential processing  
  sequential:
    default_order: "embed_first"  # embed_first, parse_first
    enable_context_passing: true
    max_neighbors_for_context: 5
  
  # Iterative processing
  iterative:
    max_iterations: 3
    convergence_threshold: 0.05
    min_iteration_time_ms: 100
  
  # Content profiling
  profiling:
    enable_advanced_analysis: false  # Trade accuracy for speed
    cache_profiles: true
    cache_ttl_seconds: 300
  
  # Disagreement detection
  disagreement:
    enable_sarcasm_detection: true
    sentiment_threshold: 0.7
    sparsecl_weight: 0.3
    preserve_all_interpretations: true
  
  # Service configuration
  services:
    embedding:
      base_url: "http://localhost:8001"
      default_timeout_ms: 200
      retry_count: 1
      circuit_breaker:
        failure_threshold: 5
        recovery_timeout_seconds: 30
    
    parsing:
      base_url: "http://localhost:8002"
      default_timeout_ms: 200
      retry_count: 1
      circuit_breaker:
        failure_threshold: 5
        recovery_timeout_seconds: 30
  
  # File path generation
  file_generation:
    max_path_depth: 4
    max_filename_length: 100
    prefer_semantic_paths: true
    entity_weight: 0.3
    cluster_weight: 0.7
```

### 5.2 Runtime Tuning

```python
class OrchestrationConfig:
    """Runtime configuration with hot-reload support"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = self._load_config()
        self._last_modified = config_path.stat().st_mtime
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation support"""
        # Supports: config.get('orchestration.parallel.enable_result_correlation')
        
    def reload_if_changed(self) -> bool:
        """Check and reload configuration if file changed"""
        current_mtime = self.config_path.stat().st_mtime
        if current_mtime > self._last_modified:
            self._config = self._load_config()
            self._last_modified = current_mtime
            return True
        return False
```

## 6. Error Handling and Edge Cases

### 6.1 Service Failure Handling

```python
class ServiceFailureHandler:
    """Handles failures with graceful degradation"""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.circuit_breakers = {}
    
    async def handle_embedding_failure(
        self,
        text: str,
        error: Exception
    ) -> EmbeddingResult:
        """Generate fallback embedding on service failure"""
        
        if isinstance(error, TimeoutError):
            # Try with shorter timeout
            return await self.quick_embed_fallback(text, timeout=50)
        
        elif isinstance(error, EmbeddingServiceError):
            # Service down - use local fallback
            logger.warning(f"Embedding service failed: {error}")
            
            # Simple hash-based embedding
            embedding = self.generate_hash_embedding(text)
            return EmbeddingResult(
                vector=embedding,
                confidence=0.3,
                model_used='hash_fallback',
                processing_time=1
            )
    
    def generate_hash_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic pseudo-embedding from text"""
        # Use multiple hash functions for different dimensions
        embedding = np.zeros(1024)
        
        # Generate multiple hashes
        for i in range(16):
            hasher = hashlib.sha256(f"{text}:{i}".encode())
            hash_bytes = hasher.digest()
            
            # Convert to floats
            for j in range(64):
                byte_idx = j % len(hash_bytes)
                embedding[i * 64 + j] = (hash_bytes[byte_idx] - 128) / 128.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
```

### 6.2 Edge Cases

```python
class EdgeCaseHandler:
    """Handles various edge cases in processing"""
    
    async def process_edge_cases(self, text: str, profile: ContentProfile):
        # Empty input
        if not text or text.isspace():
            return ProcessedGlobule(
                text=text,
                embedding=np.zeros(1024),
                embedding_confidence=0.0,
                parsed_data={'type': 'empty'},
                parsing_confidence=1.0,
                file_decision=FileDecision(
                    semantic_path=Path('misc'),
                    filename='empty_note.md',
                    confidence=1.0
                ),
                processing_notes=['Empty input received']
            )
        
        # Extremely long input
        if profile.length > 50000:
            # Truncate with warning
            text = text[:50000] + '\n\n[Content truncated...]'
            profile.processing_complexity = 1.0
            
        # Binary data detection
        if contains_binary_data(text):
            # Switch to simplified processing
            return await self.process_binary_content(text)
        
        # Malformed unicode
        try:
            text.encode('utf-8')
        except UnicodeError:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
        return None  # Continue normal processing
```

### 6.3 Circuit Breaker Implementation

```python
class CircuitBreaker:
    """Prevents cascade failures by tracking service health"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.half_open_count = 0
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_count = 0
            else:
                raise CircuitOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - update state
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_count += 1
                if self.half_open_count >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                
            raise e
```

## 7. Performance Monitoring

### 7.1 Metrics Collection

```python
@dataclass
class OrchestrationMetrics:
    """Performance metrics for monitoring and optimization"""
    
    # Timing breakdowns
    total_time_ms: float
    profiling_time_ms: float
    embedding_time_ms: float
    parsing_time_ms: float
    coordination_time_ms: float
    file_generation_time_ms: float
    
    # Strategy metrics
    strategy_used: str
    iteration_count: int = 1
    
    # Quality metrics
    embedding_confidence: float
    parsing_confidence: float
    disagreement_count: int
    
    # Resource usage
    memory_used_mb: float
    services_called: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)
```

### 7.2 Performance Tracking

```python
class PerformanceTracker:
    """Track and analyze orchestration performance"""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.aggregated_stats = {}
    
    def record_metrics(self, metrics: OrchestrationMetrics):
        """Record metrics for analysis"""
        self.metrics_buffer.append(metrics)
        
        # Update running statistics
        strategy = metrics.strategy_used
        if strategy not in self.aggregated_stats:
            self.aggregated_stats[strategy] = {
                'count': 0,
                'total_time': 0,
                'max_time': 0,
                'timeouts': 0
            }
        
        stats = self.aggregated_stats[strategy]
        stats['count'] += 1
        stats['total_time'] += metrics.total_time_ms
        stats['max_time'] = max(stats['max_time'], metrics.total_time_ms)
        
        if metrics.total_time_ms > 450:  # Near timeout
            stats['timeouts'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance statistics summary"""
        return {
            'strategies': self.aggregated_stats,
            'avg_total_time': np.mean([m.total_time_ms for m in self.metrics_buffer]),
            'p95_total_time': np.percentile([m.total_time_ms for m in self.metrics_buffer], 95),
            'avg_confidence': np.mean([
                (m.embedding_confidence + m.parsing_confidence) / 2 
                for m in self.metrics_buffer
            ])
        }
```

## 8. Testing Hooks and Observability

### 8.1 Test Mode Configuration

```python
class TestableOrchestrationEngine:
    """Orchestration engine with testing hooks"""
    
    def __init__(self, config: OrchestrationConfig, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.test_hooks = {}
        
    def register_test_hook(self, name: str, callback: Callable):
        """Register callback for testing specific behavior"""
        if self.test_mode:
            self.test_hooks[name] = callback
    
    async def process_globule(self, enriched_input: EnrichedInput) -> ProcessedGlobule:
        """Main processing with test hooks"""
        
        # Test hook: before processing
        if self.test_mode and 'before_process' in self.test_hooks:
            self.test_hooks['before_process'](enriched_input)
        
        # ... normal processing ...
        
        # Test hook: mock service responses
        if self.test_mode and 'mock_embedding' in self.test_hooks:
            embedding_result = self.test_hooks['mock_embedding'](enriched_input.text)
        else:
            embedding_result = await self.embedding_client.embed(enriched_input.text)
        
        # ... continue processing ...
```

### 8.2 Debug Logging

```python
class OrchestrationLogger:
    """Structured logging for debugging and monitoring"""
    
    def log_processing_decision(
        self,
        input_text: str,
        profile: ContentProfile,
        strategy: str,
        reason: str
    ):
        """Log strategy selection decision"""
        logger.debug(
            "Orchestration decision",
            extra={
                'text_preview': input_text[:100],
                'profile': {
                    'structure_score': profile.structure_score,
                    'creativity_score': profile.creativity_score,
                    'length': profile.length
                },
                'strategy_selected': strategy,
                'selection_reason': reason
            }
        )
    
    def log_disagreement(self, disagreement: Disagreement):
        """Log detected disagreements for analysis"""
        logger.info(
            f"Disagreement detected: {disagreement.type}",
            extra={
                'type': disagreement.type,
                'literal': disagreement.literal_interpretation,
                'semantic': disagreement.semantic_interpretation,
                'confidence': disagreement.confidence
            }
        )
```

## 9. Module Dependencies

The Orchestration Engine depends on:
- **Semantic Embedding Service**: For vector generation
- **Structural Parsing Service**: For entity extraction
- **Configuration System**: For runtime settings
- **Intelligent Storage Manager**: For semantic neighbor queries (optional)

External libraries:
- `numpy`: Vector operations
- `aiohttp`: Async HTTP client
- `asyncio`: Concurrent execution
- `hashlib`: Fallback embedding generation

## 10. Resource Constraints

### 10.1 Memory Usage
- Base memory: ~50MB for engine and caches
- Per-request memory: ~5MB (vectors, intermediate results)
- Maximum concurrent requests: 10 (configurable)

### 10.2 CPU Usage
- Content profiling: Single-threaded, <10ms
- Service coordination: Async I/O bound
- Disagreement detection: Single-threaded, <50ms

### 10.3 Network Usage
- Embedding service: ~5KB request, ~5KB response
- Parsing service: ~5KB request, ~10KB response
- Semantic neighbor query: ~5KB request, ~20KB response
# Structural Parsing Research

## Introduction
The Structural Parsing Service is a cornerstone of Globule, a local-first, privacy-focused knowledge management system designed to transform unstructured notes into a structured, searchable knowledge base. This service leverages Large Language Models (LLMs) to extract metadata—such as domains, entities, categories, and timestamps—from text, enabling semantic path generation (e.g., `/valet/2025-07/task_123.json`) and hybrid retrieval combining keyword and semantic searches. This comprehensive technical analysis and low-level design (LLD) integrates insights from multiple perspectives, detailing metadata extraction, caching, asynchronous processing, provider abstraction, and quality assurance. Key findings include the efficacy of a provider-agnostic architecture, the necessity of robust caching and async task queuing for CLI responsiveness, and the importance of schema-driven prompt engineering for consistent JSON outputs, all tailored to Globule’s goals of performance, extensibility, and user empowerment as of July 13, 2025.

---

## Metadata Generation for Semantic Paths and Keyword Search

The Structural Parsing Service must extract metadata to support the Intelligent Storage Manager’s semantic path generation and the Query Engine’s keyword-based search capabilities.

## Metadata Extraction for Semantic Paths
To enable intuitive file organization, the service should extract a core set of metadata fields:

- **Domain**: The primary topic (e.g., "valet", "research"), forming the top-level directory.
- **Timestamp**: An ISO 8601 string (e.g., "2025-07-13T10:30:00Z"), used for chronological subdirectories (e.g., "2025-07").
- **Category**: A granular classification (e.g., "task", "note"), often part of the filename or subdirectory.
- **Title**: An LLM-generated summary (e.g., "Project Alpha Planning"), enhancing filename readability.
- **Entities**: Named entities (e.g., `[{"type": "person", "value": "Alice"}]`) for rich search and filename specificity.
- **Task ID**: A unique identifier (e.g., "123") for precise file naming.
- **Keywords**: Descriptive tags (e.g., ["planning", "goals"]) for search enhancement.
- **Sentiment**: Emotional tone (e.g., "neutral"), optional for analysis.
- **Summary**: A brief content overview, aiding previews.

These fields prioritize path generation as `/domain/YYYY-MM/category_title_taskid.json`. For multi-domain content (e.g., "research" and "valet"), the parser assigns a primary domain using LLM classification, supplemented by entity weighting and keyword analysis, storing secondary domains in metadata for search flexibility.

**Example Output**:
```json
{
  "domain": "valet",
  "timestamp": "2025-07-13T09:03:00Z",
  "category": "task",
  "title": "Tesla Maintenance",
  "task_id": "123",
  "entities": [{"type": "vehicle", "value": "Tesla Model 3"}],
  "keywords": ["maintenance", "car"],
  "sentiment": "neutral",
  "summary": "Notes on Tesla Model 3 maintenance."
}
```

**Trade-offs**: Extracting extensive metadata enriches organization but risks complexity; insufficient extraction leads to generic paths. Pydantic schemas ensure consistency, balancing breadth and precision.

## Metadata Formatting for Keyword Search
For Query Engine compatibility, metadata should be stored in SQLite with a hybrid approach:
- **Dedicated Columns**: `domain`, `category`, `task_id`, and `timestamp` for fast, indexed lookups.
- **JSONB Column**: Full nested metadata for flexibility.
- **FTS5 Table**: Concatenated text (e.g., "Tesla Maintenance car") for efficient keyword searches.

**Schema Example**:
```sql
CREATE TABLE globules (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL,
    category TEXT,
    task_id TEXT,
    timestamp TEXT,
    metadata JSONB,
    content TEXT
);
CREATE VIRTUAL TABLE globule_fts USING fts5(domain, category, content);
```

**Trade-offs**: Dedicated columns optimize speed but limit flexibility; JSONB offers adaptability at a query performance cost. FTS5 enhances keyword search efficiency with moderate storage overhead.

## Handling Multi-Domain Globules
For notes spanning domains, the parser:
1. Uses LLM classification to select a primary domain (e.g., "valet").
2. Stores secondary domains (e.g., "research") in metadata.
3. Optionally creates symbolic links (e.g., `/research/2025-07/task_123.json`) for retrieval flexibility.

The Adaptive Input Module allows user overrides, ensuring alignment with intent. **Example**: A note on "researching car models" might reside at `/valet/2025-07/car_models.json` with metadata indexing "research."

---

# Caching and Background Processing

To ensure performance and CLI responsiveness, the service employs caching and asynchronous processing.

## CacheManager Storage and Retrieval for `globule add`
The CacheManager uses a cache-aside strategy with a multi-tier approach:
- **Key**: SHA-256 hash of `input_text:model_name:schema_version:prompt_version`.
- **L1 (In-memory)**: Redis or LRU cache for hot data, offering sub-millisecond access with a 24-hour TTL.
- **L2 (SQLite)**: Persistent storage for all parsed outputs, with no strict TTL.

**Workflow**:
1. Compute key from `globule add` input.
2. Check L1; return if hit.
3. Check L2; return and warm L1 if hit.
4. Parse via LLM, store in both tiers if miss.

**Implementation**:
```python
class CacheManager:
    def __init__(self, db_path="globule_cache.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.memory_cache = {}

    async def get(self, content: str) -> dict:
        key = hashlib.sha256(content.encode()).hexdigest()
        if key in self.memory_cache:
            return self.memory_cache[key]
        result = self.conn.execute("SELECT parsed_json FROM cache WHERE key=?", (key,)).fetchone()
        if result:
            data = json.loads(result[0])
            self.memory_cache[key] = data
            return data
        return None

    async def set(self, content: str, parsed: dict):
        key = hashlib.sha256(content.encode()).hexdigest()
        self.memory_cache[key] = parsed
        self.conn.execute("INSERT OR REPLACE INTO cache (key, parsed_json) VALUES (?, ?)",
                          (key, json.dumps(parsed)))
```

**Trade-offs**: In-memory caching is fast but volatile; SQLite ensures durability with higher latency. Invalidation on schema/model changes maintains freshness.

## Async Task Queuing for CLI Responsiveness
The parser offloads processing using `asyncio.Queue` or `AsyncBatcher`:
- **Immediate Response**: `globule add` queues tasks and returns "queued" instantly.
- **Batching**: Groups 5-10 inputs every 1-2 seconds for efficient LLM calls.

**Implementation**:
```python
class AsyncParsingQueue:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def submit_task(self, text: str) -> str:
        task_id = str(uuid.uuid4())
        await self.queue.put((task_id, text))
        return task_id

    async def process(self):
        while True:
            text = await self.queue.get()
            result = await parse_text(text)  # LLM call
            # Store result
```

**Trade-offs**: Async improves responsiveness but adds complexity; batching optimizes throughput at the cost of slight delays.

## Metadata Caching for File-Based Retrieval
A separate in-memory cache maps `globule_id` to `{path, domain, category}`, synced with Storage Manager updates. This reduces disk I/O for `globule draft`, achieving <10ms lookups.

**Example**: `/valet/2025-07/task_123.json` cached as `{id: "123", path: "...", domain: "valet"}`.

---

# Prompt Engineering and Schema Enforcement

## Consistent JSON Output
Prompt strategies ensure reliable JSON:
- **Explicit Instructions**: "Respond ONLY in valid JSON per schema."
- **Schema Inclusion**: Embed Pydantic-generated JSON schema.
- **Few-Shot Examples**: Provide input-output pairs.
- **Provider Features**: Use OpenAI’s JSON Mode, Gemini’s responseSchema, or Ollama’s GBNF.

**Prompt Example**:
```
Extract metadata in JSON:
{
  "domain": "string",
  "title": "string",
  "timestamp": "ISO 8601"
}
Text: "Alice parked her Tesla at 9am."
```

## Schema Definition Engine Integration
The parser retrieves Pydantic schemas asynchronously, generates prompts, and validates outputs. Schema version locking prevents mid-task inconsistencies.

**Implementation**:
```python
class SchemaManager:
    async def get_schema(self, version: str) -> dict:
        # Fetch schema
        return {"domain": "string", "title": "string"}
```

## Handling Parsing Failures
- **Retries**: 3 attempts with exponential backoff (1s, 2s, 4s).
- **Fallback**: Regex for critical fields (e.g., domain).
- **Logging**: Store errors in SQLite for review.

**Example**:
```python
async def parse_with_retry(text: str) -> dict:
    for attempt in range(3):
        try:
            return await llm.parse(text)
        except Exception:
            await asyncio.sleep(2 ** attempt)
    # Fallback
    return {"domain": re.search(r"valet|research", text).group(0)}
```

---

# Integration with Hybrid Retrieval

## Enhancing File-Based Retrieval
Metadata drives `pathlib.Path.glob` patterns (e.g., `/valet/2025-07/*.json`), with SQLite pre-filtering for efficiency.

## Supporting Hybrid Search
- **Keyword Search**: FTS5 on `domain`, `title`, etc.
- **Semantic Search**: Async handoff to Embedding Service, prioritizing entities.

## Conflict Resolution
- **Ranking**: Reciprocal Rank Fusion (RRF) combines file and semantic scores.
- **User Feedback**: TUI prompts resolve ambiguities (e.g., "valet vs. research").

**Implementation**:
```python
def hybrid_search(file_results: list, semantic_results: list) -> list:
    scores = {}
    for path in set(file_results + semantic_results):
        scores[path] = (1 / (60 + file_results.index(path))) + (1 / (60 + semantic_results.index(path)))
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

---

# Quality and Evaluation

## Quality Metrics
- **Schema Compliance**: >95% valid JSON outputs.
- **Domain Accuracy**: F1-score >85%.
- **Entity F1-score**: >80%.
- **Latency**: Avg <2s (local), <5s (cloud).

## User Feedback Loops
Async SQLite table (`feedback_corrections`) logs corrections, refining prompts and schemas.

## Metadata Validation
Pydantic and custom rules ensure `domain`, `timestamp`, etc., are valid and file-system safe.

---

# Provider Abstraction and Async Processing

## Architectural Pattern
The Strategy Pattern with an ABC (`BaseParser`) supports async, provider-agnostic parsing.

**Implementation**:
```python
class BaseParser(ABC):
    @abstractmethod
    async def parse(self, text: str) -> dict:
        pass

class OpenAIParser(BaseParser):
    async def parse(self, text: str) -> dict:
        # OpenAI call
        return {"domain": "valet"}
```

## Configuration Management
A Pydantic `Settings` class manages provider settings, stored encrypted in SQLite.

**Example**:
```python
class Settings(BaseSettings):
    ollama: dict = {"model": "llama3.2:3b"}
    openai: dict = {"api_key": "sk-..."}
```

## Provider-Specific Metadata
A canonical `GlobuleMetadata` schema normalizes outputs, preserving extras in `provider_metadata`.

**Example**:
```json
{
  "domain": "valet",
  "provider_metadata": {"usage": {"tokens": 50}}
}
```

---

# Special Concerns for Globule

## Automatic Domain Detection
LLM classification, entity analysis, and keyword scoring detect domains, with user overrides via TUI.

## Handling Nuanced Content
Sentiment analysis and confidence scores flag sarcasm or ambiguity for TUI review.

**Example**:
```json
{
  "domain": "valet",
  "sentiment": "negative",
  "is_ambiguous": true,
  "confidence": 0.6
}
```

---

# Conclusion
This merged design ensures the Structural Parsing Service delivers a robust, scalable solution for Globule, integrating metadata extraction, caching, async processing, and quality assurance into a cohesive framework that supports its local-first, user-centric vision.
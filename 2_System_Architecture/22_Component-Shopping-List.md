# Globule Component Shopping List
*Version: 1.0*  
*Date: 2025-07-10*  
*Purpose: Technical component breakdown for MVP implementation*

## Overview

This document provides a structured breakdown of the components required to build the Globule MVP ("Ollie"). Each component is defined with clear boundaries, interfaces, and dependencies to enable parallel development and systematic integration.

## Component Dependency Graph

This table of contents lists the 8 core components, ordered by their architectural layer. This ordering represents the dependency flow of the system, from foundational services to user-facing applications.

<pre>
<a href="../3_Core_Components/32_Adaptive_Input_Module/30_LLD_Adaptive_Input_Module.md">Adaptive Input Module</a>
├─ <a href="../3_Core_Components/35_Orchestration_Engine/30_LLD_Orchestration_Engine.md">Orchestration Engine</a>
│  ├─ <a href="../3_Core_Components/33_Semantic_Embedding_Service/30_LLD_Semantic_Embedding_Service.md">Semantic Embedding Service</a>
│  ├─ <a href="../3_Core_Components/34_Structural_Parsing_Service/30_LLD_Structural_Parsing_Service.md">Structural Parsing Service</a>
│  └─ <a href="../3_Core_Components/36_Intelligent_Storage_Manager/30_LLD_Intelligent_Storage_Manager.md">Intelligent Storage Manager</a>
├─ <a href="../3_Core_Components/31_Schema_Engine/30_LLD_Schema-Engine.md">Schema Definition Engine</a>
│  └─ <a href="../3_Core_Components/30_Configuration_System/30_LLD_Configuration-System.md">Configuration System</a>
└─ <a href="../3_Core_Components/37_Interactive_Synthesis_Engine/30_LLD_Interactive_Synthesis_Engine.md">Interactive Synthesis Engine</a>
</pre>


### 1. Orchestration Engine
**Module:** `orchestration.py`  
**Purpose:** Coordinates all AI services to process input collaboratively rather than competitively

**Interfaces:**
- Input: Raw text + enriched context from Input Module
- Output: `ProcessedGlobule` object with embedding, parsed data, and file decision
- Dependencies: Embedding Service, Parsing Service, Storage Manager

**Key Methods:**
```python
async def process_globule(text: str, context: dict) -> ProcessedGlobule
async def determine_processing_weights(content_profile: ContentProfile) -> dict
async def handle_service_disagreement(embedding_result, parsing_result) -> Resolution
```

**MVP Requirements:**
- Dual-track processing coordination
- Content-type aware weight determination
- Disagreement preservation (e.g., sarcasm detection)
- File path generation using both semantic and structural insights

**Success Criteria:**
- Processes input in <500ms for typical text
- Correctly identifies and preserves nuanced content
- Generates human-navigable file paths

---

### 2. Adaptive Input Module
**Module:** `input_adapter.py`  
**Purpose:** Conversational gateway that validates input and applies schemas

**Interfaces:**
- Input: Raw user text from CLI
- Output: Enriched text with schema context
- Dependencies: Schema Engine, Configuration System

**Key Methods:**
```python
async def process_input(text: str) -> EnrichedInput
async def detect_schema(text: str) -> Optional[Schema]
async def gather_additional_context(text: str, schema: Schema) -> dict
def get_confirmation_prompt(detected_type: str) -> str
```

**MVP Requirements:**
- 3-second auto-confirmation with manual override
- Basic schema detection (URLs, prompts, structured data)
- Configurable verbosity levels
- Context gathering for special input types

**Success Criteria:**
- <100ms response time for user feedback
- 90%+ accuracy in schema detection
- Smooth UX for both automatic and manual modes

---

### 3. Semantic Embedding Service
**Module:** `embedding_service.py`  
**Purpose:** Captures semantic meaning and relationships through vector representations

**Interfaces:**
- Input: Text (raw or enriched)
- Output: High-dimensional vector embedding
- Dependencies: Ollama or HuggingFace API

**Key Methods:**
```python
async def embed(text: str) -> np.ndarray
async def batch_embed(texts: List[str]) -> List[np.ndarray]
def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float
```

**MVP Requirements:**
- Local embedding using mxbai-embed-large via Ollama
- Fallback to sentence-transformers if Ollama unavailable
- Consistent vector dimensions (1024-d)
- Batch processing support

**Success Criteria:**
- <200ms embedding generation
- Semantic similarity that matches human intuition
- Stable embeddings across sessions

---

### 4. Structural Parsing Service
**Module:** `parsing_service.py`  
**Purpose:** Extracts entities, structure, and metadata from text

**Interfaces:**
- Input: Text + optional semantic context
- Output: Structured JSON with entities, categories, sentiment
- Dependencies: Ollama or HuggingFace API

**Key Methods:**
```python
async def parse(text: str, context: Optional[dict] = None) -> ParsedData
def build_context_aware_prompt(text: str, semantic_neighbors: List[str]) -> str
```

**MVP Requirements:**
- Local parsing using llama3.2:3b via Ollama
- JSON schema enforcement
- Entity extraction (people, places, concepts)
- Category and sentiment detection

**Success Criteria:**
- <300ms parsing time
- Structured output that validates against schema
- Meaningful category assignments

---

### 5. Intelligent Storage Manager
**Module:** `storage_manager.py`  
**Purpose:** Creates semantic filesystem structure and manages all data persistence

**Interfaces:**
- Input: ProcessedGlobule with file decision
- Output: Stored file with metadata
- Dependencies: SQLite (via aiosqlite)

**Key Methods:**
```python
async def store_globule(globule: ProcessedGlobule) -> str
async def search_temporal(timeframe: str) -> List[Globule]
async def search_semantic(embedding: np.ndarray, limit: int) -> List[Globule]
def generate_semantic_path(globule: ProcessedGlobule) -> Path
```

**MVP Requirements:**
- SQLite database with JSON and BLOB support
- Semantic directory structure generation
- Metadata in companion .globule files
- Cross-platform compatibility

**Database Schema:**
```sql
CREATE TABLE globules (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB,
    parsed_data JSON,
    file_path TEXT,
    metadata JSON
);
```

**Success Criteria:**
- Human-navigable directory structure
- <50ms for temporal queries
- <500ms for semantic search (up to 10k globules)

---

### 6. Interactive Synthesis Engine
**Module:** `synthesis_engine.py`  
**Purpose:** Powers the two-pane TUI for drafting documents

**Interfaces:**
- Input: Query parameters (timeframe, topic, etc.)
- Output: Interactive TUI application
- Dependencies: Textual framework, Storage Manager, Parsing Service

**Key Components:**
```python
class PalettePane:  # Left side - organized thoughts
    async def load_initial_view(query: str) -> List[GlobuleCluster]
    async def switch_view(view_type: ViewType) -> None
    async def explore_semantic(selected: Globule) -> List[Globule]

class CanvasPane:  # Right side - document editor
    def generate_starter_content(clusters: List[GlobuleCluster]) -> str
    async def ai_assist(selected_text: str, action: AIAction) -> str
```

**MVP Requirements:**
- Textual-based TUI with two panes
- Multiple Palette views (clustered, chronological)
- Build mode (Enter) vs Explore mode (Tab)
- Basic AI actions (expand, summarize, rephrase)
- Smart starter content generation

**Success Criteria:**
- Responsive UI (<100ms for all interactions)
- Intuitive keyboard navigation
- Successful synthesis of 10+ notes in <15 minutes

---

### 7. Configuration System
**Module:** `config_manager.py`  
**Purpose:** Three-tier configuration cascade for user empowerment

**Interfaces:**
- Input: YAML configuration files
- Output: Configuration objects for all modules
- Dependencies: PyYAML

**Key Methods:**
```python
def load_cascade() -> ConfigCascade
def get_setting(key: str, context: Optional[str] = None) -> Any
def update_user_preference(key: str, value: Any) -> None
```

**MVP Requirements:**
- System defaults → User preferences → Context overrides
- YAML-based configuration
- Runtime configuration updates
- Sensible defaults that work without configuration

**Success Criteria:**
- Zero-config works for new users
- Power users can customize everything
- Context switching is seamless

---

### 8. Schema Definition Engine
**Module:** `schema_engine.py`  
**Purpose:** Allows users to define custom workflows as schemas

**Interfaces:**
- Input: YAML schema definitions
- Output: Schema objects used by Input Module
- Dependencies: Configuration System

**Key Methods:**
```python
def load_schema(name: str) -> Schema
def validate_schema(schema_dict: dict) -> bool
def apply_schema(text: str, schema: Schema) -> EnrichedInput
```

**MVP Requirements:**
- YAML-based schema definitions
- Basic built-in schemas (links, tasks, notes)
- Schema validation
- User-defined schemas support

**Example Schema:**
```yaml
schemas:
  url_capture:
    triggers: ["http://", "https://"]
    actions:
      - fetch_title
      - extract_description
    prompt_context: "Why save this link?"
    output_template: "[{title}]({url})\n{context}"
```

**Success Criteria:**
- Users can create custom schemas without code
- Schemas are shareable as YAML files
- Built-in schemas cover common use cases

---

## Implementation Order

### Phase 1: Foundation (Week 1-2)
1. **Configuration System** - Needed by all other components
2. **Schema Definition Engine** - Defines data structures
3. **Storage Manager** (basic version) - SQLite setup and basic operations

### Phase 2: Intelligence (Week 3-4)
4. **Embedding Service** - Core semantic understanding
5. **Parsing Service** - Structural analysis
6. **Orchestration Engine** - Brings intelligence together

### Phase 3: User Experience (Week 5-6)
7. **Adaptive Input Module** - Entry point for users
8. **Interactive Synthesis Engine** - The killer feature

### Phase 4: Integration & Polish
- End-to-end testing
- Performance optimization
- Documentation and examples

---

## Module Interfaces Specification

Each module communicates through well-defined Pydantic models:

```python
# Shared data models (models.py)
class Globule(BaseModel):
    id: str
    content: str
    embedding: Optional[List[float]]
    parsed_data: Optional[Dict]
    created_at: datetime
    file_path: Optional[str]
    metadata: Dict

class ProcessedGlobule(Globule):
    confidence_scores: Dict[str, float]
    processing_time: float
    schema_used: Optional[str]

class EnrichedInput(BaseModel):
    original_text: str
    enriched_text: str
    detected_schema: Optional[str]
    additional_context: Dict
```

---

## Testing Strategy

Each component must include:
- Unit tests for all public methods
- Integration tests with mock dependencies
- Performance benchmarks
- Example usage in docstrings

---

## Future Extensibility Considerations

While building for the MVP, each component should consider:
- Plugin interfaces for future extensions
- Async-first design for scalability
- Clean separation of concerns
- Well-documented extension points

This shopping list provides the blueprint for transforming the Globule vision into reality, one component at a time.
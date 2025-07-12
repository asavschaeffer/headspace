# Globule Technical Architecture

*How the pieces fit together (and stay flexible)*

## Core Architecture Principles

1. **Separation of Concerns**: Each component does ONE thing well
2. **Message-Oriented**: Components communicate through well-defined interfaces
3. **Plugin-Ready**: Even if not exposed in MVP, the architecture supports extensions
4. **Storage Agnostic**: Can swap backends without changing business logic
5. **Progressive Enhancement**: Each layer adds capability without modifying lower layers

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │   CLI   │ │   TUI   │ │  Voice  │ │   API   │          │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │
│       └───────────┴───────────┴───────────┘                 │
│                          │                                   │
│                    Input Router                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                   Processing Pipeline                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Input Validator                      │   │
│  │  • Type detection (text/url/image/voice)            │   │
│  │  • Basic sanitization                               │   │
│  │  • Rate limiting                                    │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────┴──────────────────────────────┐   │
│  │              Parallel Processors                     │   │
│  │  ┌─────────────────┐ ┌─────────────────┐           │   │
│  │  │ Embedding Engine│ │  Parser Engine  │           │   │
│  │  │                 │ │                 │           │   │
│  │  │ • Sentence      │ │ • Domain detect │           │   │
│  │  │   transformer   │ │ • LLM parsing  │           │   │
│  │  │ • Vector output │ │ • Entity extract│           │   │
│  │  └─────────────────┘ └─────────────────┘           │   │
│  │         │                     │                      │   │
│  │         └──────────┬─────────┘                      │   │
│  │                    │                                 │   │
│  │            Cross-Validation                          │   │
│  │      (Embeddings ↔ Parsed Entities)                 │   │
│  └────────────────────┴─────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                    Storage Layer                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Storage Manager                       │   │
│  │          (Abstract interface - swappable)            │   │
│  └──────────────────────┬──────────────────────────────┘   │
│           ┌─────────────┴─────────────┐                     │
│     ┌─────┴─────┐            ┌────────┴────────┐           │
│     │  SQLite   │            │   Vector Store  │           │
│     │           │            │   (ChromaDB)    │           │
│     │ • Globules│            │ • Embeddings    │           │
│     │ • Metadata│            │ • Similarity    │           │
│     │ • JSON    │            │                 │           │
│     └───────────┘            └─────────────────┘           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                   Retrieval Layer                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Query Engine                          │   │
│  │  • Natural language → structured query              │   │
│  │  • Semantic search via embeddings                   │   │
│  │  • Temporal/entity/pattern queries                  │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────┴──────────────────────────────┐   │
│  │              Synthesis Engine                        │   │
│  │  • Combine related globules                        │   │
│  │  • Generate narratives                             │   │
│  │  • Apply output templates                          │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### Globule (Base Unit)
```python
@dataclass
class Globule:
    # Immutable core
    id: UUID
    content: str
    created_at: datetime
    
    # Processing results
    embedding: Optional[Vector]
    parsed_data: Optional[Dict[str, Any]]
    entities: List[str]
    
    # Metadata
    type: GlobuleType  # text, url, image, voice
    source: InputSource  # cli, api, voice
    version: int
    
    # Relationships (for graph features)
    links_to: List[UUID]
    linked_from: List[UUID]
```

### Domain Schema (Pluggable)
```python
@dataclass 
class DomainSchema:
    name: str  # "valet", "research", "generic"
    
    # What to extract
    fields: Dict[str, FieldType]
    
    # How to extract
    few_shot_examples: List[Example]
    parser_prompt_template: str
    
    # How to validate
    validators: List[Callable]
    
    # How to output
    report_templates: Dict[str, Template]
```

## Processing Pipeline Details

### 1. Input Router
```python
class InputRouter:
    def route(self, raw_input: Any) -> ProcessableInput:
        # Detect type
        input_type = self.detect_type(raw_input)
        
        # Select processor
        processor = self.processors[input_type]
        
        # Return wrapped input
        return ProcessableInput(
            raw=raw_input,
            type=input_type,
            processor=processor
        )
```

### 2. Parallel Processing
```python
async def process_globule(input: ProcessableInput) -> Globule:
    # Create base globule
    globule = Globule(
        id=uuid4(),
        content=input.raw,
        created_at=now()
    )
    
    # Process in parallel
    embedding_task = asyncio.create_task(
        embed_content(globule.content)
    )
    parsing_task = asyncio.create_task(
        parse_content(globule.content, domain=detect_domain(globule))
    )
    
    # Wait for both
    globule.embedding = await embedding_task
    globule.parsed_data = await parsing_task
    
    # Cross-validate
    globule = validate_extraction(globule)
    
    return globule
```

### 3. Storage Abstraction
```python
class StorageManager(ABC):
    @abstractmethod
    async def store(self, globule: Globule) -> None:
        pass
        
    @abstractmethod
    async def search_semantic(self, query: str, limit: int) -> List[Globule]:
        pass
        
    @abstractmethod  
    async def search_structured(self, filters: Dict) -> List[Globule]:
        pass

class SQLiteStorage(StorageManager):
    # Concrete implementation
    
class GraphStorage(StorageManager):
    # Future implementation
```

## Query & Synthesis

### Query Engine
```python
class QueryEngine:
    async def query(self, natural_language: str) -> QueryResult:
        # Parse intent
        intent = await self.llm.parse_query(natural_language)
        
        # Route to appropriate search
        if intent.type == "semantic":
            results = await self.storage.search_semantic(
                intent.embedding_query
            )
        elif intent.type == "structured":
            results = await self.storage.search_structured(
                intent.filters
            )
        elif intent.type == "temporal":
            results = await self.storage.search_timerange(
                intent.start, 
                intent.end
            )
            
        # Post-process
        return self.rank_and_filter(results, intent)
```

### Synthesis Engine
```python
class SynthesisEngine:
    def generate_report(
        self, 
        globules: List[Globule],
        template: str,
        context: Dict
    ) -> str:
        # Group related globules
        clusters = self.cluster_by_similarity(globules)
        
        # Extract key information
        summary_data = self.extract_summary_data(clusters)
        
        # Generate narrative sections
        narratives = {}
        for section, cluster in clusters.items():
            narratives[section] = self.llm.generate_narrative(
                cluster, 
                style=template.style
            )
            
        # Render template
        return self.template_engine.render(
            template,
            data=summary_data,
            narratives=narratives,
            context=context
        )
```

## Extension Points (Plugin Architecture)

### Input Processors
```python
class ProcessorPlugin(ABC):
    @abstractmethod
    def can_handle(self, input_type: str) -> bool:
        pass
        
    @abstractmethod
    async def process(self, input: Any) -> ProcessedData:
        pass

# Example: URL processor plugin
class URLProcessor(ProcessorPlugin):
    def can_handle(self, input_type: str) -> bool:
        return input_type == "url"
        
    async def process(self, url: str) -> ProcessedData:
        # Crawl URL
        content = await self.crawl(url)
        # Extract text
        text = self.extract_text(content)
        # Get metadata
        metadata = self.extract_metadata(content)
        
        return ProcessedData(text=text, metadata=metadata)
```

### Domain Schemas
```python
class DomainPlugin(ABC):
    @property
    @abstractmethod
    def schema(self) -> DomainSchema:
        pass
        
    @abstractmethod
    def detect_domain(self, content: str) -> float:
        """Return confidence 0-1 that this domain applies"""
        pass

# Domains are auto-discovered and registered
domain_registry.register(ValetDomain())
domain_registry.register(ResearchDomain())
```

### Output Formatters
```python
class OutputPlugin(ABC):
    @abstractmethod
    def format(self, data: SynthesizedData) -> str:
        pass

# Example: Markdown blog formatter
class BlogFormatter(OutputPlugin):
    def format(self, data: SynthesizedData) -> str:
        return f"""
# {data.title}

{data.introduction}

## Key Points
{self.format_bullets(data.key_points)}

## Conclusion
{data.conclusion}
        """
```

## Performance Considerations

### Targets
- Input capture: <50ms
- Embedding generation: <100ms  
- LLM parsing: <500ms (local), <2s (cloud)
- Semantic search: <100ms for 10k globules
- Report generation: <5s

### Optimization Strategies
1. **Async everything**: Never block the UI
2. **Batch processing**: Group embeddings
3. **Caching**: Reuse embeddings for similar content
4. **Indexes**: Temporal, entity, and type indexes
5. **Progressive loading**: Stream results as found

## Security & Privacy

### Data Flow
- All data encrypted at rest (SQLite encryption)
- No data leaves device without explicit permission
- Cloud features use end-to-end encryption
- API requires authentication tokens

### Permissions
- File system access: Read-only by default
- Network access: Only for opted-in features
- No background services without consent

## Migration Path

### From MVP to Graph Database
```python
# Current: SQLite JSON
globule = {
    "id": "uuid",
    "links_to": ["uuid1", "uuid2"]
}

# Future: Neo4j
CREATE (g:Globule {id: $id})
CREATE (g)-[:LINKS_TO]->(g2)
```

### From Local to Distributed
- Add sync adapter layer
- Implement conflict resolution
- Use CRDTs for distributed state

## Testing Strategy

### Unit Tests
- Each processor independently
- Storage adapters with mocks
- Query parsing accuracy

### Integration Tests  
- Full pipeline with sample data
- Performance benchmarks
- Domain detection accuracy

### End-to-End Tests
- User scenarios (valet shift, research session)
- Report quality assessment
- Search relevance metrics
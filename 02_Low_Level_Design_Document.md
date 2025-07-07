<html><body>
<!--StartFragment--><html><head></head><body><h1>Low-Level Design Document: Globule</h1>
<blockquote>
<p><strong>Document Suite Status</strong>: ~~[[Project-Vision]]~~ | ~~[[Elevator-Pitch]]~~ | ~~[[Requirements]]~~ | ~~[[Non-Functional-Requirements]]~~ | ~~[[User-Flows-and-Mockups]]~~ | ~~[[High-Level-Design]]~~ | <strong>LOW-LEVEL DESIGN</strong> | [[API-Specifications]] | [[Business-Strategy]] | [[Operational-Planning]] | [[Glossary]]</p>
</blockquote>
<h2>Cross-References &amp; Dependencies</h2>
<ul>
<li><strong>[[High-Level-Design]]</strong>: System architecture and component breakdown that this design implements</li>
<li><strong>[[Requirements]]</strong>: Functional requirements driving these implementation decisions</li>
<li><strong>[[Non-Functional-Requirements]]</strong>: Performance and scalability constraints for implementation</li>
<li><strong>[[API-Specifications]]</strong>: External interface contracts that these components must fulfill</li>
<li><strong>[[Database-Schema]]</strong>: Detailed data model specifications (will expand on entities outlined here)</li>
<li><strong>[[Security-Implementation]]</strong>: Detailed security protocols and implementation patterns</li>
</ul>
<h2>Implementation Overview</h2>
<p>This document details the concrete implementation of Globule's semantic thought processing system, focusing on:</p>
<ul>
<li>Asynchronous service architecture with hybrid concurrency model for AI processing</li>
<li>Plugin-based processing pipeline supporting domain-specific extensions</li>
<li>Dual-track semantic understanding (embeddings + parsing) with cross-validation</li>
<li>Abstract storage layer enabling future migration from SQLite to graph databases</li>
<li>Event-driven UI with background AI processing that never blocks user interaction</li>
</ul>
<h2>System Architecture Implementation</h2>
<h3>Service Layer Design</h3>
<p><strong>Pattern</strong>: Plugin-Based Pipeline Architecture with Dependency Injection</p>
<h4>Core Services</h4>

Service Name | Responsibilities | Dependencies | Key Methods
-- | -- | -- | --
InputRouter | Detect input type and route to processors | ProcessorRegistry, DomainDetector | route(input: str) -> ProcessableInput
EmbeddingEngine | Generate semantic vectors | SentenceTransformer, CacheManager | async embed(text: str) -> np.ndarray
ParserEngine | Extract structured data via LLM | LLMClient, DomainSchemaRegistry | async parse(text: str, domain: Domain) -> Dict
StorageManager | Abstract storage operations | SQLiteAdapter, ChromaAdapter | async store(globule: Globule) -> str
QueryEngine | Natural language to structured queries | EmbeddingEngine, LLMClient | async query(nl_query: str) -> QueryResult
SynthesisEngine | Generate reports and narratives | TemplateEngine, LLMClient | async synthesize(globules: List[Globule], template: str) -> str


<h3>Hybrid Concurrency Implementation</h3>
<pre><code class="language-python">import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable
import multiprocessing as mp

class HybridProcessor:
    """Manages CPU-bound tasks without blocking the async event loop"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self._executor = None
        self._loop = None
    
    async def initialize(self):
        """Initialize the process pool and event loop reference"""
        self._loop = asyncio.get_running_loop()
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
    
    async def run_cpu_bound(self, func: Callable, *args) -&gt; Any:
        """Execute CPU-bound function in process pool"""
        if not self._executor:
            await self.initialize()
        
        # Run in executor to avoid blocking the event loop
        return await self._loop.run_in_executor(
            self._executor, 
            func, 
            *args
        )
    
    async def cleanup(self):
        """Gracefully shutdown the process pool"""
        if self._executor:
            self._executor.shutdown(wait=True)
</code></pre>
<h3>Plugin Registry Pattern</h3>
<pre><code class="language-python">from abc import ABC, abstractmethod
from typing import Dict, Type, Optional

class ProcessorPlugin(ABC):
    """Base class for all input processors"""
    
    @abstractmethod
    def can_handle(self, input_type: str) -&gt; bool:
        """Check if this processor can handle the input type"""
        pass
    
    @abstractmethod
    async def process(self, input_data: Any) -&gt; ProcessedData:
        """Process the input and return structured data"""
        pass

class ProcessorRegistry:
    """Manages and routes to input processors"""
    
    def __init__(self):
        self._processors: Dict[str, ProcessorPlugin] = {}
        self._type_detectors = []
    
    def register(self, name: str, processor: ProcessorPlugin):
        """Register a new processor"""
        self._processors[name] = processor
    
    def detect_type(self, input_data: str) -&gt; str:
        """Detect input type using registered patterns"""
        # URL detection
        if input_data.startswith(('http://', 'https://')):
            return 'url'
        # Voice marker detection
        elif input_data.startswith('[voice]'):
            return 'voice'
        # Default to text
        return 'text'
    
    async def process(self, input_data: str, input_type: str = None) -&gt; ProcessedData:
        """Route to appropriate processor"""
        if not input_type:
            input_type = self.detect_type(input_data)
        
        for processor in self._processors.values():
            if processor.can_handle(input_type):
                return await processor.process(input_data)
        
        raise ValueError(f"No processor found for type: {input_type}")
</code></pre>
<h3>Data Access Layer</h3>
<p><strong>Pattern</strong>: Repository Pattern with Abstract Storage Adapters</p>
<h4>Repository Implementations</h4>
<pre><code class="language-python">from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import aiosqlite
import chromadb

@dataclass
class Globule:
    """Core data unit"""
    id: str
    content: str
    created_at: datetime
    type: str = 'text'
    source: str = 'cli'
    embedding: Optional[List[float]] = None
    parsed_data: Optional[Dict[str, Any]] = None
    entities: List[str] = None
    version: int = 1
    
    def to_dict(self) -&gt; Dict:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'type': self.type,
            'source': self.source,
            'parsed_data': self.parsed_data,
            'entities': self.entities or [],
            'version': self.version
        }

class StorageAdapter(ABC):
    """Abstract interface for storage backends"""
    
    @abstractmethod
    async def store_globule(self, globule: Globule) -&gt; None:
        pass
    
    @abstractmethod
    async def get_globule(self, globule_id: str) -&gt; Optional[Globule]:
        pass
    
    @abstractmethod
    async def search_by_time(self, start: datetime, end: datetime) -&gt; List[Globule]:
        pass

class SQLiteAdapter(StorageAdapter):
    """SQLite implementation of storage adapter"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialized = False
    
    async def initialize(self):
        """Create tables if not exists"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS globules (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata JSON,
                    version INTEGER DEFAULT 1
                )
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON globules(created_at)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_type 
                ON globules(type)
            ''')
            await db.commit()
        self._initialized = True
    
    async def store_globule(self, globule: Globule) -&gt; None:
        if not self._initialized:
            await self.initialize()
        
        metadata = {
            'parsed_data': globule.parsed_data,
            'entities': globule.entities
        }
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO globules 
                (id, content, created_at, type, source, metadata, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                globule.id,
                globule.content,
                globule.created_at.isoformat(),
                globule.type,
                globule.source,
                json.dumps(metadata),
                globule.version
            ))
            await db.commit()

class ChromaAdapter:
    """ChromaDB adapter for vector storage"""
    
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
    
    async def initialize(self):
        """Create or get collection"""
        self.collection = self.client.get_or_create_collection(
            name="globules",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def store_embedding(self, globule_id: str, embedding: List[float], 
                            metadata: Dict[str, Any] = None):
        """Store embedding with metadata"""
        if not self.collection:
            await self.initialize()
        
        self.collection.add(
            embeddings=[embedding],
            ids=[globule_id],
            metadatas=[metadata or {}]
        )
    
    async def search_similar(self, query_embedding: List[float], 
                           n_results: int = 10, 
                           where: Dict = None) -&gt; List[str]:
        """Search for similar embeddings"""
        if not self.collection:
            await self.initialize()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results['ids'][0] if results['ids'] else []
</code></pre>
<h2>Data Model Implementation</h2>
<h3>Entity Relationships</h3>
<p>The core Globule entity serves as the atomic unit with relationships managed through:</p>
<ul>
<li><strong>Internal Links</strong>: Direct references between globules via ID</li>
<li><strong>Semantic Clusters</strong>: Implicit relationships via embedding similarity</li>
<li><strong>Entity Co-occurrence</strong>: Shared entities create implicit connections</li>
<li><strong>Temporal Proximity</strong>: Time-based relationships for session reconstruction</li>
</ul>
<h3>Database Schema Design</h3>
<h4>Core Entities</h4>
<pre><code class="language-sql">-- SQLite schema with JSON support for flexibility
CREATE TABLE globules (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL CHECK(type IN ('text', 'voice', 'url', 'image', 'code')),
    source TEXT NOT NULL CHECK(source IN ('cli', 'tui', 'api', 'voice_recorder')),
    status TEXT DEFAULT 'raw' CHECK(status IN ('raw', 'processing', 'processed', 'error')),
    metadata JSON,
    version INTEGER DEFAULT 1,
    parent_id TEXT REFERENCES globules(id),
    
    -- Full-text search
    content_fts TEXT GENERATED ALWAYS AS (content) STORED
);

-- Indexes for common query patterns
CREATE INDEX idx_globules_created_at ON globules(created_at);
CREATE INDEX idx_globules_type ON globules(type);
CREATE INDEX idx_globules_status ON globules(status);
CREATE INDEX idx_globules_parent ON globules(parent_id) WHERE parent_id IS NOT NULL;

-- Full-text search index
CREATE VIRTUAL TABLE globules_fts USING fts5(
    content, 
    tokenize='porter unicode61'
);

-- Entity extraction results
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    globule_id TEXT NOT NULL REFERENCES globules(id),
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(globule_id, entity_type, entity_value)
);

CREATE INDEX idx_entities_globule ON entities(globule_id);
CREATE INDEX idx_entities_type_value ON entities(entity_type, entity_value);

-- Future migration path to graph relationships
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES globules(id),
    target_id TEXT NOT NULL REFERENCES globules(id),
    relationship_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    metadata JSON,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_id, target_id, relationship_type)
);

CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_relationships_target ON relationships(target_id);
</code></pre>
<h3>Data Validation Rules</h3>
<pre><code class="language-python">from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class GlobuleType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    URL = "url"
    IMAGE = "image"
    CODE = "code"

class GlobuleStatus(str, Enum):
    RAW = "raw"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"

class GlobuleInput(BaseModel):
    """Validation for incoming globule data"""
    content: str = Field(..., min_length=1, max_length=50000)
    type: GlobuleType = GlobuleType.TEXT
    source: str = Field(..., regex="^(cli|tui|api|voice_recorder)$")
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('content')
    def sanitize_content(cls, v):
        """Remove potentially harmful content"""
        # Basic sanitization - extend based on security requirements
        return v.strip()
    
    @validator('metadata')
    def validate_metadata_size(cls, v):
        """Ensure metadata doesn't exceed reasonable size"""
        if v and len(json.dumps(v)) &gt; 10000:
            raise ValueError("Metadata too large")
        return v

class ParsedData(BaseModel):
    """Validation for LLM-parsed structured data"""
    entities: List[Dict[str, Any]] = []
    intent: Optional[str] = None
    sentiment: Optional[str] = Field(None, regex="^(positive|negative|neutral|mixed)$")
    domain: Optional[str] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    
    @validator('entities')
    def validate_entities(cls, v):
        """Ensure entities have required fields"""
        for entity in v:
            if 'type' not in entity or 'value' not in entity:
                raise ValueError("Entity must have 'type' and 'value' fields")
        return v
</code></pre>
<h2>Business Logic Implementation</h2>
<h3>Core Algorithms</h3>
<h4>Dual-Track Processing Algorithm</h4>
<pre><code class="language-python">import asyncio
from typing import Tuple, Dict, Any

class DualTrackProcessor:
    """Implements parallel embedding and parsing with cross-validation"""
    
    def __init__(self, embedding_engine, parser_engine, validator):
        self.embedding_engine = embedding_engine
        self.parser_engine = parser_engine
        self.validator = validator
    
    async def process(self, globule: Globule) -&gt; Tuple[List[float], Dict[str, Any]]:
        """
        1. Detect domain from content
        2. Run embedding and parsing in parallel
        3. Cross-validate results
        4. Return validated results
        """
        # Step 1: Domain detection
        domain = await self._detect_domain(globule.content)
        
        # Step 2: Parallel processing
        embedding_task = asyncio.create_task(
            self.embedding_engine.embed(globule.content)
        )
        parsing_task = asyncio.create_task(
            self.parser_engine.parse(globule.content, domain)
        )
        
        try:
            embedding, parsed_data = await asyncio.gather(
                embedding_task, 
                parsing_task,
                return_exceptions=True
            )
            
            # Step 3: Handle errors
            if isinstance(embedding, Exception):
                raise ProcessingError(f"Embedding failed: {embedding}")
            if isinstance(parsed_data, Exception):
                raise ProcessingError(f"Parsing failed: {parsed_data}")
            
            # Step 4: Cross-validation
            validated_data = await self.validator.validate(
                globule.content,
                embedding,
                parsed_data
            )
            
            return embedding, validated_data
            
        except Exception as e:
            # Cleanup on error
            embedding_task.cancel()
            parsing_task.cancel()
            raise
    
    async def _detect_domain(self, content: str) -&gt; str:
        """Detect domain using keyword matching (MVP) or embeddings (future)"""
        content_lower = content.lower()
        
        # Domain detection rules
        valet_keywords = ['parked', 'tips', 'late', 'damage', 'customer', 'car']
        research_keywords = ['research', 'paper', 'study', 'hypothesis', 'analysis']
        code_keywords = ['function', 'class', 'debug', 'error', 'implement']
        
        valet_score = sum(1 for kw in valet_keywords if kw in content_lower)
        research_score = sum(1 for kw in research_keywords if kw in content_lower)
        code_score = sum(1 for kw in code_keywords if kw in content_lower)
        
        scores = {
            'valet': valet_score,
            'research': research_score,
            'code': code_score
        }
        
        # Return domain with highest score, default to generic
        domain = max(scores, key=scores.get)
        return domain if scores[domain] &gt; 0 else 'generic'
</code></pre>
<h4>Semantic Clustering Algorithm</h4>
<pre><code class="language-python">import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple

class SemanticClusterer:
    """Clusters globules by semantic similarity for virtual folders"""
    
    def __init__(self, min_samples: int = 3, eps: float = 0.3):
        self.min_samples = min_samples
        self.eps = eps
    
    async def cluster_embeddings(
        self, 
        embeddings: List[Tuple[str, np.ndarray]]
    ) -&gt; Dict[int, List[str]]:
        """
        1. Prepare embedding matrix
        2. Apply DBSCAN clustering
        3. Group globule IDs by cluster
        4. Return cluster assignments
        """
        if len(embeddings) &lt; self.min_samples:
            # Not enough data for clustering
            return {0: [gid for gid, _ in embeddings]}
        
        # Step 1: Extract embeddings and IDs
        globule_ids = [gid for gid, _ in embeddings]
        embedding_matrix = np.array([emb for _, emb in embeddings])
        
        # Step 2: Normalize embeddings for cosine similarity
        normalized = embedding_matrix / np.linalg.norm(
            embedding_matrix, axis=1, keepdims=True
        )
        
        # Step 3: Apply DBSCAN
        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='cosine'
        )
        cluster_labels = clusterer.fit_predict(normalized)
        
        # Step 4: Group by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(globule_ids[idx])
        
        return clusters
    
    async def label_cluster(
        self, 
        cluster_globules: List[Globule],
        llm_client
    ) -&gt; str:
        """Generate descriptive label for cluster"""
        # Take up to 5 representative globules
        sample_size = min(5, len(cluster_globules))
        samples = cluster_globules[:sample_size]
        
        # Create prompt for LLM
        content_samples = '\n'.join([
            f"- {g.content[:200]}..." for g in samples
        ])
        
        prompt = f"""Based on these related notes, generate a concise 2-4 word label:
        
{content_samples}

Label:"""
        
        label = await llm_client.generate(prompt, max_tokens=20)
        return label.strip()
</code></pre>
<h3>Design Patterns Implementation</h3>
<h4>Domain Plugin Pattern</h4>
<pre><code class="language-python">from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class FieldSpec:
    """Specification for a parseable field"""
    name: str
    type: str  # 'string', 'number', 'enum', 'list'
    required: bool = False
    validation: str = None  # Regex or validation rule

class DomainPlugin(ABC):
    """Base class for domain-specific parsing plugins"""
    
    @property
    @abstractmethod
    def name(self) -&gt; str:
        """Domain identifier"""
        pass
    
    @property
    @abstractmethod
    def schema(self) -&gt; Dict[str, FieldSpec]:
        """Fields this domain can extract"""
        pass
    
    @property
    @abstractmethod
    def few_shot_examples(self) -&gt; List[Dict[str, Any]]:
        """Examples for LLM prompting"""
        pass
    
    @abstractmethod
    def detect_confidence(self, content: str) -&gt; float:
        """Return confidence (0-1) that this domain applies"""
        pass
    
    def build_prompt(self, content: str) -&gt; str:
        """Build extraction prompt for LLM"""
        examples = '\n'.join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in self.few_shot_examples
        ])
        
        schema_desc = '\n'.join([
            f"- {name}: {spec.type} {'(required)' if spec.required else ''}"
            for name, spec in self.schema.items()
        ])
        
        return f"""Extract structured data from the input according to this schema:

{schema_desc}

Examples:
{examples}

Input: {content}
Output:"""

# Concrete implementation for valet domain
class ValetDomainPlugin(DomainPlugin):
    @property
    def name(self) -&gt; str:
        return "valet"
    
    @property
    def schema(self) -&gt; Dict[str, FieldSpec]:
        return {
            "event_type": FieldSpec(
                name="event_type",
                type="enum[arrival,departure,damage,tips,tardiness,incident]",
                required=True
            ),
            "employee": FieldSpec(
                name="employee",
                type="string",
                required=False
            ),
            "customer": FieldSpec(
                name="customer",
                type="string",
                required=False
            ),
            "amount": FieldSpec(
                name="amount",
                type="number",
                required=False
            ),
            "location": FieldSpec(
                name="location",
                type="string",
                required=False,
                validation=r"^[A-Z]\d+$"  # e.g., "A5", "B12"
            )
        }
    
    @property
    def few_shot_examples(self) -&gt; List[Dict[str, Any]]:
        return [
            {
                "input": "Timmy was 20 minutes late",
                "output": {"event_type": "tardiness", "employee": "timmy", "duration": 20}
            },
            {
                "input": "Mr Jones complained about damage but it was already there",
                "output": {"event_type": "damage", "customer": "mr_jones", "pre_existing": true}
            },
            {
                "input": "Split tips $60 between timmy barbara and me",
                "output": {"event_type": "tips", "amount": 60, "employees": ["timmy", "barbara", "self"]}
            }
        ]
    
    def detect_confidence(self, content: str) -&gt; float:
        """Calculate domain relevance score"""
        keywords = ['parked', 'car', 'tips', 'late', 'damage', 'customer', 'valet']
        content_lower = content.lower()
        matches = sum(1 for kw in keywords if kw in content_lower)
        return min(matches / 3.0, 1.0)  # Normalize to 0-1
</code></pre>
<h2>Security Implementation Details</h2>
<h3>Authentication Flow</h3>
<pre><code class="language-python"># Local-first design - no authentication needed for MVP
# Cloud features (Stage 2) will implement JWT-based auth

from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext

class AuthService:
    """Handles authentication for cloud features"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.token_expiry = timedelta(hours=24)
    
    def create_access_token(self, user_id: str) -&gt; str:
        """Generate JWT token for authenticated user"""
        expire = datetime.utcnow() + self.token_expiry
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -&gt; Optional[str]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload.get("sub")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
</code></pre>
<h3>Data Protection Measures</h3>
<pre><code class="language-python">import os
from cryptography.fernet import Fernet
from typing import Optional

class EncryptionService:
    """Handles encryption for sensitive data"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key:
            self.fernet = Fernet(key)
        else:
            # Generate new key for first run
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            # Store key securely (OS keyring in production)
            self._store_key(key)
    
    def encrypt_content(self, content: str) -&gt; str:
        """Encrypt globule content"""
        return self.fernet.encrypt(content.encode()).decode()
    
    def decrypt_content(self, encrypted: str) -&gt; str:
        """Decrypt globule content"""
        return self.fernet.decrypt(encrypted.encode()).decode()
    
    def _store_key(self, key: bytes):
        """Store encryption key securely"""
        # MVP: Store in local config file
        # Production: Use OS keyring or HSM
        key_path = os.path.expanduser("~/.globule/encryption.key")
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, 'wb') as f:
            f.write(key)
        os.chmod(key_path, 0o600)  # Read/write for owner only
</code></pre>
<h2>Performance Optimization Strategies</h2>
<h3>Caching Implementation</h3>
<pre><code class="language-python">from functools import lru_cache
from typing import Dict, Any, Optional
import hashlib
import asyncio
from datetime import datetime, timedelta

class CacheManager:
    """Manages various caching strategies"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.embedding_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self.llm_cache: Dict[str, Tuple[str, datetime]] = {}
        self._cleanup_task = None
    
    async def start(self):
        """Start cache cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop cache cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    def _content_hash(self, content: str) -&gt; str:
        """Generate stable hash for content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_embedding(self, content: str) -&gt; Optional[List[float]]:
        """Retrieve cached embedding if available"""
        key = self._content_hash(content)
        if key in self.embedding_cache:
            embedding, timestamp = self.embedding_cache[key]
            if datetime.now() - timestamp &lt; self.ttl:
                return embedding
            else:
                del self.embedding_cache[key]
        return None
    
    async def set_embedding(self, content: str, embedding: List[float]):
        """Cache embedding with TTL"""
        key = self._content_hash(content)
        self.embedding_cache[key] = (embedding, datetime.now())
    
    @lru_cache(maxsize=1000)
    def get_domain_schema(self, domain: str) -&gt; Dict[str, Any]:
        """Cache domain schemas (immutable during runtime)"""
        # This would load from domain registry
        return {}
    
    async def _cleanup_loop(self):
        """Periodically clean expired cache entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                now = datetime.now()
                
                # Clean embedding cache
                expired_keys = [
                    k for k, (_, ts) in self.embedding_cache.items()
                    if now - ts &gt; self.ttl
                ]
                for key in expired_keys:
                    del self.embedding_cache[key]
                    
            except asyncio.CancelledError:
                break
</code></pre>
<h3>Database Query Optimization</h3>
<pre><code class="language-python">class QueryOptimizer:
    """Optimizes database queries for performance"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._prepared_statements = {}
    
    async def prepare_statements(self):
        """Prepare commonly used SQL statements"""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable query optimizer
            await db.execute("PRAGMA optimize")
            
            # Prepare common queries
            self._prepared_statements['get_by_id'] = """
                SELECT g.*, 
                       GROUP_CONCAT(e.entity_type || ':' || e.entity_value) as entities
                FROM globules g
                LEFT JOIN entities e ON g.id = e.globule_id
                WHERE g.id = ?
                GROUP BY g.id
            """
            
            self._prepared_statements['search_by_time'] = """
                SELECT * FROM globules
                WHERE created_at BETWEEN ? AND ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            
            self._prepared_statements['search_by_entity'] = """
                SELECT DISTINCT g.*
                FROM globules g
                JOIN entities e ON g.id = e.globule_id
                WHERE e.entity_value = ? AND e.entity_type = ?
                ORDER BY g.created_at DESC
            """
    
    async def analyze_query_performance(self):
        """Analyze and optimize query performance"""
        async with aiosqlite.connect(self.db_path) as db:
            # Update statistics
            await db.execute("ANALYZE")
            
            # Check query plans for slow queries
            explain_query = """
                EXPLAIN QUERY PLAN
                SELECT * FROM globules
                WHERE json_extract(metadata, '$.domain') = 'valet'
                AND created_at &gt; date('now', '-7 days')
            """
            
            cursor = await db.execute(explain_query)
            plan = await cursor.fetchall()
            
            # Log any table scans for optimization
            for step in plan:
                if 'SCAN TABLE' in str(step):
                    # This indicates a potential performance issue
                    # Consider adding appropriate index
                    pass
</code></pre>
<h3>Async Processing Design</h3>
<pre><code class="language-python">import asyncio
from typing import List, Callable, Any
from dataclasses import dataclass
import time

@dataclass
class BatchProcessor:
    """Processes items in batches for efficiency"""
    
    batch_size: int = 10
    max_wait_time: float = 1.0  # seconds
    
    def __init__(self):
        self.pending_items = []
        self.pending_callbacks = []
        self._process_task = None
        self._last_process_time = time.time()
    
    async def add_item(self, item: Any, callback: Callable):
        """Add item for batch processing"""
        self.pending_items.append(item)
        self.pending_callbacks.append(callback)
        
        # Start processing if not running
        if not self._process_task or self._process_task.done():
            self._process_task = asyncio.create_task(self._process_batch())
    
    async def _process_batch(self):
        """Process accumulated items"""
        while True:
            # Wait for batch to fill or timeout
            elapsed = time.time() - self._last_process_time
            if len(self.pending_items) &lt; self.batch_size and elapsed &lt; self.max_wait_time:
                await asyncio.sleep(0.1)
                continue
            
            if not self.pending_items:
                break
            
            # Process batch
            batch_items = self.pending_items[:self.batch_size]
            batch_callbacks = self.pending_callbacks[:self.batch_size]
            
            self.pending_items = self.pending_items[self.batch_size:]
            self.pending_callbacks = self.pending_callbacks[self.batch_size:]
            
            # Process in parallel
            results = await self._process_items(batch_items)
            
            # Execute callbacks
            for callback, result in zip(batch_callbacks, results):
                await callback(result)
            
            self._last_process_time = time.time()
    
    async def _process_items(self, items: List[Any]) -&gt; List[Any]:
        """Override in subclass for specific processing logic"""
        raise NotImplementedError
</code></pre>
<h2>Error Handling &amp; Resilience</h2>
<h3>Exception Hierarchy</h3>
<pre><code class="language-python">class GlobuleError(Exception):
    """Base exception for all Globule errors"""
    pass

class ValidationError(GlobuleError):
    """Input validation failed"""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error on {field}: {message}")

class ProcessingError(GlobuleError):
    """Processing pipeline error"""
    def __init__(self, stage: str, original_error: Exception):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Processing failed at {stage}: {str(original_error)}")

class StorageError(GlobuleError):
    """Storage operation failed"""
    pass

class EmbeddingError(ProcessingError):
    """Embedding generation failed"""
    pass

class ParsingError(ProcessingError):
    """LLM parsing failed"""
    pass

class DomainDetectionError(ProcessingError):
    """Could not detect appropriate domain"""
    pass

class QuotaExceededError(GlobuleError):
    """API quota exceeded"""
    def __init__(self, service: str, reset_time: datetime):
        self.service = service
        self.reset_time = reset_time
        super().__init__(
            f"{service} quota exceeded. Resets at {reset_time.isoformat()}"
        )
</code></pre>
<h3>Circuit Breaker Implementation</h3>
<pre><code class="language-python">from enum import Enum
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Prevents cascading failures in external service calls"""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Retry after "
                    f"{self.last_failure_time + timedelta(seconds=self.recovery_timeout)}"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -&gt; bool:
        """Check if enough time has passed to try again"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time &gt;= timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Reset failure count on successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Increment failure count and possibly open circuit"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count &gt;= self.failure_threshold:
            self.state = CircuitState.OPEN
</code></pre>
<h3>Retry Mechanisms</h3>
<pre><code class="language-python">import asyncio
import random
from typing import TypeVar, Callable, Optional

T = TypeVar('T')

class RetryPolicy:
    """Configurable retry policy with exponential backoff"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -&gt; float:
        """Calculate delay for given attempt number"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random())
        
        return delay

async def retry_async(
    func: Callable[..., T],
    policy: RetryPolicy,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
) -&gt; T:
    """Execute async function with retry logic"""
    last_exception = None
    
    for attempt in range(policy.max_attempts):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt == policy.max_attempts - 1:
                # Last attempt failed
                raise
            
            delay = policy.calculate_delay(attempt)
            
            if on_retry:
                await on_retry(attempt, delay, e)
            
            await asyncio.sleep(delay)
    
    raise last_exception
</code></pre>
<h2>File Structure &amp; Organization</h2>
<h3>Project Structure</h3>
<pre><code>src/
├── core/                    # Core domain logic
│   ├── __init__.py
│   ├── models.py           # Globule and domain models
│   ├── processors.py       # Processing pipeline
│   └── validators.py       # Input validation
│
├── services/               # Business logic layer
│   ├── __init__.py
│   ├── input_router.py     # Route inputs to processors
│   ├── embedding_engine.py # Semantic embedding service
│   ├── parser_engine.py    # LLM parsing service
│   ├── query_engine.py     # Natural language queries
│   └── synthesis_engine.py # Report generation
│
├── repositories/           # Data access layer
│   ├── __init__.py
│   ├── storage_manager.py  # Abstract storage interface
│   ├── sqlite_adapter.py   # SQLite implementation
│   └── chroma_adapter.py   # ChromaDB implementation
│
├── plugins/                # Domain and processor plugins
│   ├── __init__.py
│   ├── base.py            # Plugin base classes
│   ├── domains/           # Domain-specific plugins
│   │   ├── valet.py
│   │   ├── research.py
│   │   └── generic.py
│   └── processors/        # Input processors
│       ├── text.py
│       ├── url.py
│       └── voice.py
│
├── interfaces/            # User interfaces
│   ├── __init__.py
│   ├── cli.py            # Command line interface
│   ├── tui/              # Textual UI
│   │   ├── app.py
│   │   ├── widgets.py
│   │   └── styles.css
│   └── api/              # REST API (future)
│       ├── app.py
│       └── routes.py
│
├── utils/                 # Shared utilities
│   ├── __init__.py
│   ├── cache.py          # Caching utilities
│   ├── crypto.py         # Encryption utilities
│   ├── retry.py          # Retry/circuit breaker
│   └── performance.py    # Performance monitoring
│
├── config/               # Configuration management
│   ├── __init__.py
│   ├── settings.py       # App settings
│   └── logging.yaml      # Logging configuration
│
└── tests/                # Test suite
    ├── unit/
    ├── integration/
    └── fixtures/
</code></pre>
<h3>Module Dependencies</h3>
<pre><code class="language-python"># Dependency injection container
from typing import Dict, Any, Type
from dataclasses import dataclass

@dataclass
class ServiceConfig:
    """Configuration for service initialization"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "local"  # or "openai", "anthropic"
    storage_path: str = "./data"
    cache_ttl: int = 3600

class ServiceContainer:
    """Dependency injection container"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register_factory(self, name: str, factory: Callable):
        """Register a service factory"""
        self._factories[name] = factory
    
    def get(self, service_type: Type[T]) -&gt; T:
        """Get or create a service instance"""
        name = service_type.__name__
        
        if name not in self._services:
            if name in self._factories:
                self._services[name] = self._factories[name](self)
            else:
                raise ValueError(f"No factory registered for {name}")
        
        return self._services[name]
    
    async def initialize(self):
        """Initialize all async services"""
        for service in self._services.values():
            if hasattr(service, 'initialize'):
                await service.initialize()
    
    async def cleanup(self):
        """Cleanup all services"""
        for service in self._services.values():
            if hasattr(service, 'cleanup'):
                await service.cleanup()

# Wire up dependencies
def configure_services(config: ServiceConfig) -&gt; ServiceContainer:
    container = ServiceContainer(config)
    
    # Register factories
    container.register_factory(
        'StorageManager',
        lambda c: StorageManager(
            sqlite_adapter=SQLiteAdapter(f"{c.config.storage_path}/globules.db"),
            chroma_adapter=ChromaAdapter(f"{c.config.storage_path}/vectors")
        )
    )
    
    container.register_factory(
        'EmbeddingEngine',
        lambda c: EmbeddingEngine(
            model_name=c.config.embedding_model,
            cache_manager=c.get(CacheManager)
        )
    )
    
    container.register_factory(
        'ParserEngine',
        lambda c: ParserEngine(
            llm_provider=c.config.llm_provider,
            domain_registry=c.get(DomainRegistry)
        )
    )
    
    return container
</code></pre>
<h2>Testing Strategy Implementation</h2>
<h3>Unit Test Patterns</h3>
<pre><code class="language-python">import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

class TestEmbeddingEngine:
    """Unit tests for embedding engine"""
    
    @pytest.fixture
    def mock_model(self):
        """Mock sentence transformer model"""
        model = Mock()
        model.encode.return_value = [0.1] * 384  # Mock embedding
        return model
    
    @pytest.fixture
    def cache_manager(self):
        """Mock cache manager"""
        return AsyncMock(spec=CacheManager)
    
    @pytest.fixture
    async def engine(self, mock_model, cache_manager):
        """Create engine with mocked dependencies"""
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            engine = EmbeddingEngine(
                model_name="test-model",
                cache_manager=cache_manager
            )
            await engine.initialize()
            yield engine
            await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_embed_with_cache_hit(self, engine, cache_manager):
        """Test embedding retrieval from cache"""
        # Arrange
        content = "test content"
        cached_embedding = [0.2] * 384
        cache_manager.get_embedding.return_value = cached_embedding
        
        # Act
        result = await engine.embed(content)
        
        # Assert
        assert result == cached_embedding
        cache_manager.get_embedding.assert_called_once_with(content)
        engine.model.encode.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_embed_with_cache_miss(self, engine, cache_manager):
        """Test embedding generation on cache miss"""
        # Arrange
        content = "test content"
        cache_manager.get_embedding.return_value = None
        
        # Act
        result = await engine.embed(content)
        
        # Assert
        assert len(result) == 384
        cache_manager.get_embedding.assert_called_once_with(content)
        cache_manager.set_embedding.assert_called_once()
        engine.model.encode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embed_handles_long_content(self, engine):
        """Test handling of content exceeding model limits"""
        # Arrange
        long_content = "x" * 10000  # Very long content
        
        # Act
        result = await engine.embed(long_content)
        
        # Assert
        assert len(result) == 384
        # Verify content was truncated before encoding
        called_content = engine.model.encode.call_args[0][0]
        assert len(called_content) &lt;= engine.max_content_length
</code></pre>
<h3>Integration Test Patterns</h3>
<pre><code class="language-python">import tempfile
import shutil
from pathlib import Path

class TestGlobuleIntegration:
    """Integration tests for full pipeline"""
    
    @pytest.fixture
    async def test_environment(self):
        """Create isolated test environment"""
        temp_dir = tempfile.mkdtemp()
        
        config = ServiceConfig(
            storage_path=temp_dir,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        container = configure_services(config)
        await container.initialize()
        
        yield container
        
        await container.cleanup()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_globule_lifecycle(self, test_environment):
        """Test complete globule creation and retrieval flow"""
        # Arrange
        input_router = test_environment.get(InputRouter)
        query_engine = test_environment.get(QueryEngine)
        
        test_input = "Mr. Jones arrived with damaged fender at 2:30 PM"
        
        # Act - Create globule
        globule = await input_router.process(test_input)
        
        # Assert - Verify storage
        assert globule.id is not None
        assert globule.embedding is not None
        assert globule.parsed_data is not None
        assert "mr_jones" in [e['value'] for e in globule.parsed_data.get('entities', [])]
        
        # Act - Query globule
        results = await query_engine.query("damage incidents today")
        
        # Assert - Verify retrieval
        assert len(results) &gt;= 1
        assert any(g.id == globule.id for g in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, test_environment):
        """Test system handles concurrent inputs"""
        # Arrange
        input_router = test_environment.get(InputRouter)
        inputs = [
            f"Test input {i}" for i in range(10)
        ]
        
        # Act - Process concurrently
        tasks = [
            input_router.process(input_text) 
            for input_text in inputs
        ]
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 10
        assert all(g.id is not None for g in results)
        assert len(set(g.id for g in results)) == 10  # All unique IDs
</code></pre>
<h2>Implementation Decision Log</h2>
<h3>⚠️ IMPLEMENTATION NEEDED</h3>
<ul>
<li><strong>Vector Index Selection</strong>: Choose between FAISS, Annoy, or ChromaDB's built-in indexing for production scale</li>
<li><strong>LLM Provider Strategy</strong>: Implement provider abstraction for OpenAI, Anthropic, and local models</li>
<li><strong>Plugin Sandboxing</strong>: Security model for untrusted third-party plugins</li>
</ul>
<h3>🔄 OPTIMIZATION PENDING</h3>
<ul>
<li><strong>Batch Embedding Generation</strong>: Process multiple globules in single model call (10x throughput improvement expected)</li>
<li><strong>Query Result Caching</strong>: Cache recent query results with semantic similarity keys (reduce latency by 50%)</li>
<li><strong>Database Connection Pooling</strong>: Implement connection pool for concurrent SQLite access</li>
</ul>
<h3>📝 CODE SPECIFICATION REQUIRED</h3>
<ul>
<li><strong>Domain Schema Versioning</strong>: Migration strategy when domain schemas evolve</li>
<li><strong>Federated Learning Protocol</strong>: Specification for privacy-preserving model sharing</li>
<li><strong>Real-time Collaboration</strong>: WebSocket protocol for multi-user features</li>
</ul>
<h2>Performance Targets &amp; Monitoring</h2>
<h3>Response Time Targets</h3>
<p>Based on our conversation requirements:</p>
<ul>
<li><strong>Input Capture</strong>: &lt;50ms (local storage write)</li>
<li><strong>Embedding Generation</strong>: &lt;100ms for text up to 1000 tokens</li>
<li><strong>LLM Parsing</strong>: &lt;500ms local, &lt;2s cloud</li>
<li><strong>Semantic Search</strong>: &lt;100ms for corpus up to 100k globules</li>
<li><strong>Report Generation</strong>: &lt;5s for daily summary with 100 globules</li>
</ul>
<h3>Monitoring Implementation</h3>
<pre><code class="language-python">import time
from contextlib import asynccontextmanager
from typing import Dict
import prometheus_client as prom

class PerformanceMonitor:
    """Tracks operation performance metrics"""
    
    def __init__(self):
        self.operation_histogram = prom.Histogram(
            'globule_operation_duration_seconds',
            'Duration of operations',
            ['operation', 'status']
        )
        
        self.active_operations = prom.Gauge(
            'globule_active_operations',
            'Number of active operations',
            ['operation']
        )
        
        self.error_counter = prom.Counter(
            'globule_operation_errors_total',
            'Total operation errors',
            ['operation', 'error_type']
        )
    
    @asynccontextmanager
    async def measure_operation(self, operation_name: str):
        """Context manager for measuring operation duration"""
        start_time = time.time()
        self.active_operations.labels(operation=operation_name).inc()
        
        try:
            yield
            # Success
            duration = time.time() - start_time
            self.operation_histogram.labels(
                operation=operation_name,
                status='success'
            ).observe(duration)
            
        except Exception as e:
            # Failure
            duration = time.time() - start_time
            self.operation_histogram.labels(
                operation=operation_name,
                status='error'
            ).observe(duration)
            
            self.error_counter.labels(
                operation=operation_name,
                error_type=type(e).__name__
            ).inc()
            
            raise
        
        finally:
            self.active_operations.labels(operation=operation_name).dec()
</code></pre>
<h2>Deployment Implementation Details</h2>
<h3>Database Migration Strategy</h3>
<pre><code class="language-python">import aiosqlite
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Migration:
    """Database migration definition"""
    version: int
    name: str
    up_sql: str
    down_sql: str
    
class MigrationManager:
    """Manages database schema migrations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations: List[Migration] = []
    
    def register_migration(self, migration: Migration):
        """Register a new migration"""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    async def initialize(self):
        """Create migration tracking table"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
            ''')
            await db.commit()
    
    async def migrate(self):
        """Apply pending migrations"""
        async with aiosqlite.connect(self.db_path) as db:
            # Get current version
            cursor = await db.execute(
                "SELECT MAX(version) FROM schema_migrations"
            )
            row = await cursor.fetchone()
            current_version = row[0] if row[0] else 0
            
            # Apply pending migrations
            for migration in self.migrations:
                if migration.version &gt; current_version:
                    await self._apply_migration(db, migration)
    
    async def _apply_migration(self, db, migration: Migration):
        """Apply a single migration with transaction"""
        try:
            await db.execute("BEGIN")
            
            # Apply migration
            await db.executescript(migration.up_sql)
            
            # Record migration
            await db.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, datetime.now().isoformat())
            )
            
            await db.commit()
            
        except Exception as e:
            await db.rollback()
            raise MigrationError(f"Failed to apply migration {migration.name}: {e}")
</code></pre>
<h3>Configuration Management</h3>
<pre><code class="language-python">from pydantic import BaseSettings, Field
from typing import Optional
import os

class ApplicationConfig(BaseSettings):
    """Application configuration with environment variable support"""
    
    # Storage settings
    data_path: str = Field(
        default="./data",
        env="GLOBULE_DATA_PATH"
    )
    
    # Model settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="GLOBULE_EMBEDDING_MODEL"
    )
    
    llm_provider: str = Field(
        default="local",
        env="GLOBULE_LLM_PROVIDER"
    )
    
    llm_model: Optional[str] = Field(
        default=None,
        env="GLOBULE_LLM_MODEL"
    )
    
    # Performance settings
    max_concurrent_processors: int = Field(
        default=4,
        env="GLOBULE_MAX_PROCESSORS"
    )
    
    cache_ttl_seconds: int = Field(
        default=3600,
        env="GLOBULE_CACHE_TTL"
    )
    
    # Security settings
    enable_encryption: bool = Field(
        default=False,
        env="GLOBULE_ENABLE_ENCRYPTION"
    )
    
    encryption_key_path: Optional[str] = Field(
        default=None,
        env="GLOBULE_ENCRYPTION_KEY_PATH"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_db_path(self) -&gt; str:
        """Get full database path"""
        return os.path.join(self.data_path, "globules.db")
    
    def get_vector_path(self) -&gt; str:
        """Get vector storage path"""
        return os.path.join(self.data_path, "vectors")
</code></pre>
<h3>Health Check Implementation</h3>
<pre><code class="language-python">from enum import Enum
from typing import Dict, List
from dataclasses import dataclass

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentHealth:
    """Health status of a single component"""
    name: str
    status: HealthStatus
    message: str = ""
    metadata: Dict = None

class HealthCheckService:
    """Monitors system health"""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.checks = []
    
    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks.append((name, check_func))
    
    async def check_system_health(self) -&gt; Dict:
        """Check health of all components"""
        components = []
        overall_status = HealthStatus.HEALTHY
        
        # Check each component
        for name, check_func in self.checks:
            try:
                health = await check_func()
                components.append(health)
                
                if health.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif health.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                components.append(
                    ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=str(e)
                    )
                )
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "metadata": c.metadata
                }
                for c in components
            ]
        }
    
    async def check_database(self) -&gt; ComponentHealth:
        """Check database connectivity"""
        try:
            storage = self.container.get(StorageManager)
            # Attempt a simple query
            await storage.get_globule("health-check-test")
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )
    
    async def check_embeddings(self) -&gt; ComponentHealth:
        """Check embedding engine"""
        try:
            engine = self.container.get(EmbeddingEngine)
            # Test embedding generation
            test_embedding = await engine.embed("health check")
            
            if len(test_embedding) == 384:
                return ComponentHealth(
                    name="embeddings",
                    status=HealthStatus.HEALTHY
                )
            else:
                return ComponentHealth(
                    name="embeddings",
                    status=HealthStatus.DEGRADED,
                    message="Unexpected embedding dimension"
                )
                
        except Exception as e:
            return ComponentHealth(
                name="embeddings",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )
</code></pre></body></html><!--EndFragment-->
</body>
</html>
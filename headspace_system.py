#!/usr/bin/env python3
"""
Headspace System - Complete backend for cosmic document visualization
Handles document storage, chunking, embeddings, and serving data to frontend
"""

import os
import json
import sqlite3
import hashlib
import numpy as np
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from data_models import Document, Chunk, ChunkConnection
import asyncio
from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from llm_chunker import LLMChunker
from tag_engine import TagEngine
from embeddings_engine import EmbeddingEngine
from config_manager import ConfigManager
from model_monitor import ModelMonitor, ModelType, ModelStatus
import uvicorn

# Configuration
DATABASE_PATH = "headspace.db"
DOCUMENTS_FOLDER = "documents"
STATIC_FOLDER = "static"

# Ensure folders exist
Path(DOCUMENTS_FOLDER).mkdir(exist_ok=True)
Path(STATIC_FOLDER).mkdir(exist_ok=True)

# Initialize model monitor for comprehensive tracking
monitor = ModelMonitor(log_level="DEBUG")

# ==============================================================================
# DATABASE MANAGER
# ==============================================================================

class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_type TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata TEXT
            )
        """)

        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                chunk_type TEXT,
                embedding BLOB,
                position_3d TEXT,
                color TEXT,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Connections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                from_chunk_id TEXT NOT NULL,
                to_chunk_id TEXT NOT NULL,
                connection_type TEXT,
                strength REAL,
                PRIMARY KEY (from_chunk_id, to_chunk_id),
                FOREIGN KEY (from_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
                FOREIGN KEY (to_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
        """)

        # Attachments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attachments (
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                attachment_type TEXT DEFAULT 'document',
                created_at TIMESTAMP,
                metadata TEXT,
                PRIMARY KEY (chunk_id, document_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_connections_from ON connections(from_chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_connections_to ON connections(to_chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attachments_chunk ON attachments(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attachments_document ON attachments(document_id)")

        conn.commit()
        conn.close()

    def save_document(self, document: Document) -> str:
        """Save a document to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (id, title, content, doc_type, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            document.id,
            document.title,
            document.content,
            document.doc_type,
            document.created_at,
            document.updated_at,
            json.dumps(document.metadata)
        ))

        conn.commit()
        conn.close()
        return document.id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return Document(
                id=row[0],
                title=row[1],
                content=row[2],
                doc_type=row[3],
                created_at=row[4],
                updated_at=row[5],
                metadata=json.loads(row[6] or '{}')
            )
        return None

    def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents ORDER BY updated_at DESC")
        rows = cursor.fetchall()

        conn.close()

        documents = []
        for row in rows:
            documents.append(Document(
                id=row[0],
                title=row[1],
                content=row[2],
                doc_type=row[3],
                created_at=row[4],
                updated_at=row[5],
                metadata=json.loads(row[6] or '{}')
            ))

        return documents

    def save_chunk(self, chunk: Chunk) -> str:
        """Save a chunk to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert embedding to bytes
        embedding_bytes = np.array(chunk.embedding, dtype=np.float32).tobytes() if chunk.embedding else None

        # Serialize metadata, including new tagging and reasoning info
        metadata_to_save = chunk.metadata.copy()
        metadata_to_save['tags'] = chunk.tags
        metadata_to_save['tag_confidence'] = chunk.tag_confidence
        metadata_to_save['reasoning'] = chunk.reasoning

        cursor.execute("""
            INSERT OR REPLACE INTO chunks
            (id, document_id, chunk_index, content, chunk_type, embedding, position_3d, color, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.id,
            chunk.document_id,
            chunk.chunk_index,
            chunk.content,
            chunk.chunk_type,
            embedding_bytes,
            json.dumps(chunk.position_3d),
            chunk.color,
            json.dumps(metadata_to_save)
        ))

        conn.commit()
        conn.close()
        return chunk.id

    def get_chunks_by_document(self, doc_id: str) -> List[Chunk]:
        """Get all chunks for a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,))
        rows = cursor.fetchall()

        chunks = []
        for row in rows:
            # Convert embedding bytes back to list
            embedding = None
            if row[5]:
                embedding = np.frombuffer(row[5], dtype=np.float32).tolist()

            chunk_id = row[0]
            metadata = json.loads(row[8] or '{}')

            # Get attachments for this chunk
            cursor.execute("""
                SELECT document_id FROM attachments
                WHERE chunk_id = ?
                ORDER BY created_at DESC
            """, (chunk_id,))
            attachment_rows = cursor.fetchall()
            attachments = [att_row[0] for att_row in attachment_rows]

            chunks.append(Chunk(
                id=chunk_id,
                document_id=row[1],
                chunk_index=row[2],
                content=row[3],
                chunk_type=row[4],
                embedding=embedding or [],
                position_3d=json.loads(row[6] or '[]'),
                color=row[7],
                metadata=metadata,
                attachments=attachments,
                tags=metadata.get('tags', []),
                tag_confidence=metadata.get('tag_confidence', {}),
                reasoning=metadata.get('reasoning', '')
            ))

        conn.close()
        return chunks

    def save_connection(self, connection: ChunkConnection):
        """Save a connection between chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO connections
            (from_chunk_id, to_chunk_id, connection_type, strength)
            VALUES (?, ?, ?, ?)
        """, (
            connection.from_chunk_id,
            connection.to_chunk_id,
            connection.connection_type,
            connection.strength
        ))

        conn.commit()
        conn.close()

    def get_connections(self, chunk_ids: List[str] = None) -> List[ChunkConnection]:
        """Get connections for specific chunks or all connections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if chunk_ids:
            placeholders = ','.join(['?' for _ in chunk_ids])
            cursor.execute(f"""
                SELECT * FROM connections
                WHERE from_chunk_id IN ({placeholders})
                OR to_chunk_id IN ({placeholders})
            """, chunk_ids + chunk_ids)
        else:
            cursor.execute("SELECT * FROM connections")

        rows = cursor.fetchall()
        conn.close()

        connections = []
        for row in rows:
            connections.append(ChunkConnection(
                from_chunk_id=row[0],
                to_chunk_id=row[1],
                connection_type=row[2],
                strength=row[3]
            ))

        return connections

    def delete_document(self, doc_id: str):
        """Delete a document and all its chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete document (cascades to chunks and connections)
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        conn.commit()
        conn.close()

    def add_attachment(self, chunk_id: str, document_id: str, attachment_type: str = "document"):
        """Attach a document to a chunk"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO attachments
            (chunk_id, document_id, attachment_type, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (chunk_id, document_id, attachment_type, datetime.now(), json.dumps({})))

        conn.commit()
        conn.close()

    def remove_attachment(self, chunk_id: str, document_id: str):
        """Remove an attachment from a chunk"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM attachments
            WHERE chunk_id = ? AND document_id = ?
        """, (chunk_id, document_id))

        conn.commit()
        conn.close()

    def get_chunk_attachments(self, chunk_id: str) -> List[Document]:
        """Get all documents attached to a chunk"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT d.* FROM documents d
            JOIN attachments a ON d.id = a.document_id
            WHERE a.chunk_id = ?
            ORDER BY a.created_at DESC
        """, (chunk_id,))

        rows = cursor.fetchall()
        conn.close()

        documents = []
        for row in rows:
            documents.append(Document(
                id=row[0],
                title=row[1],
                content=row[2],
                doc_type=row[3],
                created_at=row[4],
                updated_at=row[5],
                metadata=json.loads(row[6] or '{}')
            ))

        return documents


# ==============================================================================
# DOCUMENT PROCESSOR
# ==============================================================================

class DocumentProcessor:
    def __init__(self, db: DatabaseManager, embedder: EmbeddingEngine, tag_engine: TagEngine, llm_chunker: LLMChunker, config_manager: ConfigManager):
        self.db = db
        self.embedder = embedder
        self.tag_engine = tag_engine
        self.llm_chunker = llm_chunker
        self.config_manager = config_manager

    def process_document(self, title: str, content: str, doc_type: str = "text") -> str:
        """Process a document: chunk it, generate embeddings, calculate positions with comprehensive monitoring"""
        doc_id = hashlib.md5(f"{title}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        document = Document(
            id=doc_id, title=title, content=content, doc_type=doc_type,
            created_at=datetime.now(), updated_at=datetime.now(),
            metadata={"word_count": len(content.split())}
        )
        self.db.save_document(document)
        monitor.logger.info(f"📄 Processing document: {title} (ID: {doc_id}, {len(content)} chars)")

        # Decide chunking strategy
        strategy = self.config_manager.config.get("chunking_strategy", {}).get("preferred_chunker", "llm")
        chunks_data = None
        chunking_start = time.time()

        if strategy == "llm":
            try:
                monitor.logger.debug(f"Attempting LLM chunking for document {doc_id}")
                chunks_data = self.llm_chunker.chunk(content)
                # Add chunk_type for compatibility
                for chunk in chunks_data:
                    chunk['type'] = 'llm'
                chunking_time = (time.time() - chunking_start) * 1000
                monitor.record_model_usage("chunker-main", True, chunking_time)
                monitor.logger.info(f"✓ LLM chunking successful: {len(chunks_data)} chunks ({chunking_time:.2f}ms)")
            except Exception as e:
                chunking_time = (time.time() - chunking_start) * 1000
                monitor.record_model_usage("chunker-main", False, chunking_time, str(e))
                monitor.logger.warning(f"✗ LLM chunking failed, using structural fallback: {e}")
                chunks_data = self._chunk_structural(content, doc_type)
        else:
            monitor.logger.debug(f"Using structural chunking strategy for {doc_id}")
            chunks_data = self._chunk_structural(content, doc_type)

        chunk_texts = [chunk['text'] for chunk in chunks_data]
        if not chunk_texts:
            monitor.logger.warning(f"No chunks extracted from document {doc_id}")
            return doc_id # No content to process

        # Generate embeddings with monitoring
        embedding_start = time.time()
        try:
            monitor.logger.debug(f"Generating embeddings for {len(chunk_texts)} chunks")
            embeddings = self.embedder.generate_embeddings(chunk_texts)
            embedding_time = (time.time() - embedding_start) * 1000
            monitor.record_model_usage("embedder-main", True, embedding_time)
            monitor.logger.info(f"✓ Generated {len(embeddings)} embeddings ({embedding_time:.2f}ms)")
        except Exception as e:
            embedding_time = (time.time() - embedding_start) * 1000
            monitor.record_model_usage("embedder-main", False, embedding_time, str(e))
            monitor.logger.error(f"✗ Embedding generation failed: {e}")
            raise Exception(f"Failed to generate embeddings: {e}")
        positions_3d = self._calculate_3d_positions(embeddings)

        saved_chunks = []
        for i, chunk_data in enumerate(chunks_data):
            # Generate tags with monitoring
            tagging_start = time.time()
            try:
                tag_results = self.tag_engine.generate_tags(chunk_data['text'])
                tagging_time = (time.time() - tagging_start) * 1000
                monitor.record_model_usage("tagger-main", True, tagging_time)
            except Exception as e:
                tagging_time = (time.time() - tagging_start) * 1000
                monitor.record_model_usage("tagger-main", False, tagging_time, str(e))
                monitor.logger.debug(f"Tag generation failed for chunk {i}: {e}")
                tag_results = {}
            chunk_obj = Chunk(
                id=f"{doc_id}_chunk_{i}",
                document_id=doc_id,
                chunk_index=i,
                content=chunk_data['text'],
                chunk_type=chunk_data.get('type', 'paragraph'),
                embedding=embeddings[i].tolist(),
                position_3d=positions_3d[i].tolist(),
                color=self._get_color_from_embedding(embeddings[i]),
                metadata=chunk_data.get('metadata', {}),
                embedding_model=self.embedder.model_name,
                tags=list(tag_results.keys()),
                tag_confidence=tag_results,
                reasoning=chunk_data.get('reasoning', '')
            )
            self.db.save_chunk(chunk_obj)
            saved_chunks.append(chunk_obj)

        self._create_connections(saved_chunks, embeddings)
        return doc_id

    def _chunk_structural(self, content: str, doc_type: str) -> List[Dict]:
        """Wrapper for old structural chunking methods."""
        if doc_type == "code":
            return self._chunk_code(content)
        return self._chunk_text(content)

    def _create_connections(self, saved_chunks: List[Chunk], embeddings: np.ndarray):
        """Create sequential and semantic connections between chunks."""
        # Sequential connections
        for i in range(len(saved_chunks) - 1):
            connection = ChunkConnection(
                from_chunk_id=saved_chunks[i].id, to_chunk_id=saved_chunks[i+1].id,
                connection_type="sequential", strength=1.0
            )
            self.db.save_connection(connection)

        # Semantic connections
        similarities = self._calculate_similarities(embeddings)
        for i in range(len(saved_chunks)):
            for j in range(i + 1, len(saved_chunks)):
                if similarities[i][j] > 0.8 and abs(i - j) > 1:
                    connection = ChunkConnection(
                        from_chunk_id=saved_chunks[i].id, to_chunk_id=saved_chunks[j].id,
                        connection_type="semantic", strength=float(similarities[i][j])
                    )
                    self.db.save_connection(connection)

    def _chunk_text(self, content: str) -> List[Dict]:
        """Chunk text content by paragraphs and simple rules."""
        chunks = []
        for i, para in enumerate(content.split('\n\n')):
            if not para.strip(): continue
            chunk_type = 'paragraph'
            if para.startswith('#'):
                level = len(para.split()[0])
                chunk_type = f'heading_{level}'
                para = para.lstrip('#').strip()
            elif para.startswith('```'):
                chunk_type = 'code'
                para = para.strip('`').strip()
            elif para.startswith(('- ', '* ', '1. ')):
                chunk_type = 'list'
            chunks.append({'text': para, 'type': chunk_type, 'metadata': {'index': i}})
        return chunks

    def _chunk_code(self, content: str) -> List[Dict]:
        """Chunk code content by functions or line count."""
        chunks, current_chunk, current_type = [], [], 'code'
        for line in content.split('\n'):
            if any(keyword in line for keyword in ['def ', 'function ', 'fn ', 'func ']):
                if current_chunk:
                    chunks.append({'text': '\n'.join(current_chunk), 'type': current_type, 'metadata': {}})
                    current_chunk = []
                current_type = 'function'
            current_chunk.append(line)
            if len(current_chunk) >= 20:
                chunks.append({'text': '\n'.join(current_chunk), 'type': current_type, 'metadata': {}})
                current_chunk, current_type = [], 'code'
        if current_chunk:
            chunks.append({'text': '\n'.join(current_chunk), 'type': current_type, 'metadata': {}})
        return chunks

    def _calculate_3d_positions(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate 3D positions from embeddings using a spiral layout."""
        n = len(embeddings)
        positions = np.zeros((n, 3))
        for i in range(n):
            angle, radius, height = i * 0.5, 20 + i * 0.5, (i - n/2) * 2
            positions[i] = [radius * np.cos(angle), height, radius * np.sin(angle)]
            if embeddings.shape[1] >= 3:
                positions[i] += embeddings[i, :3] * 10
        return positions

    def _calculate_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between embeddings."""
        n = len(embeddings)
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarities[i][j] = 1.0
                else:
                    dot = np.dot(embeddings[i], embeddings[j])
                    norm1, norm2 = np.linalg.norm(embeddings[i]), np.linalg.norm(embeddings[j])
                    if norm1 > 0 and norm2 > 0:
                        sim = dot / (norm1 * norm2)
                        similarities[i][j] = similarities[j][i] = sim
        return similarities

    def _get_color_from_embedding(self, embedding: np.ndarray) -> str:
        """Generate a hex color from an embedding vector."""
        if len(embedding) >= 3:
            norm_emb = embedding[:3] / np.linalg.norm(embedding[:3]) if np.linalg.norm(embedding[:3]) > 0 else embedding[:3]
            r, g, b = [int((val * 0.5 + 0.5) * 255) for val in norm_emb]
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#667eea"

    def _get_shape_from_tags(self, tags: List[str]) -> str:
        """Determines the 3D shape for a chunk based on its tags."""
        if "code" in tags:
            return "cube"
        if "philosophy" in tags:
            return "icosahedron"
        if "visualization" in tags:
            return "torus"
        return "sphere"


# ==============================================================================
# API MODELS
# ==============================================================================


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    chunk_type: str
    position_3d: List[float]
    color: str
    tags: List[str]
    reasoning: str
    shape_3d: str
    metadata: Dict = {}

class DocumentResponse(BaseModel):
    id: str
    title: str
    doc_type: str
    created_at: str
    chunk_count: int = 0

class VisualizationData(BaseModel):
    documents: List[DocumentResponse]
    chunks: List[ChunkResponse]
    connections: List[Dict]

# Input validation models
class DocumentCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    content: str = Field(..., min_length=1, max_length=1000000, description="Document content")
    doc_type: str = Field(default="text", pattern="^(text|markdown|code|json)$", description="Document type")
    
    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()

class ChunkAttachmentRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=100, description="Document ID to attach")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        if not v.strip():
            raise ValueError('Document ID cannot be empty')
        return v.strip()


# ==============================================================================
# API SERVER
# ==============================================================================

app = FastAPI(title="Headspace API", version="1.0.0")

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # Same origin
        "http://127.0.0.1:8000",  # Localhost alternative
        "http://127.0.0.1:3000",  # React dev alternative
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Initialize services with centralized configuration
print("\n" + "="*80)
print("🔧 INITIALIZING HEADSPACE SYSTEM SERVICES")
print("="*80 + "\n")

# Initialize configuration
config_manager = ConfigManager()
config_manager.print_status()

# Run comprehensive service checks before initialization
print("\n" + "="*80)
print("🔍 PERFORMING PRE-INITIALIZATION SERVICE CHECKS")
print("="*80)
service_check_results = monitor.check_all_services({
    'gemini_api_key': config_manager.config.get('api_keys', {}).get('gemini'),
    'ollama_url': os.environ.get('OLLAMA_URL', 'http://localhost:11434')
})

# Initialize database
print("\n📊 Initializing Database...")
db = DatabaseManager()
print("✅ Database initialized")

# Initialize embedding engine with status monitoring
print("\n🧠 Initializing Embedding Engine...")
try:
    embedder = EmbeddingEngine(config_manager)
    embedder_config = config_manager.get_embedding_config()
    monitor.register_model(
        model_id="embedder-main",
        model_type=ModelType.EMBEDDING,
        provider=embedder_config.provider.value,
        model_name=embedder_config.name
    )

    # Update status based on service check
    provider_key = embedder_config.provider.value.replace('-', '_')
    if provider_key in service_check_results:
        status = service_check_results[provider_key].get('status', ModelStatus.UNKNOWN)
        # Convert status to string for serialization
        details = service_check_results[provider_key].copy()
        if 'status' in details and isinstance(details['status'], ModelStatus):
            details['status'] = details['status'].value
        monitor.update_model_status("embedder-main", status, details)

    print(f"✅ Embedding Engine initialized: {embedder_config.provider.value} ({embedder_config.name})")
except Exception as e:
    print(f"❌ Failed to initialize Embedding Engine: {e}")
    monitor.update_model_status("embedder-main", ModelStatus.FAILED, {"error": str(e)})
    embedder = None

# Initialize tag engine
print("\n🏷️ Initializing Tag Engine...")
try:
    tag_engine = TagEngine()
    monitor.register_model(
        model_id="tagger-main",
        model_type=ModelType.TAGGER,
        provider="ollama",
        model_name="llama2"
    )

    # Check if Ollama is available for tagging
    if 'ollama' in service_check_results:
        status = service_check_results['ollama'].get('status', ModelStatus.UNKNOWN)
        # Convert status to string for serialization
        details = service_check_results['ollama'].copy()
        if 'status' in details and isinstance(details['status'], ModelStatus):
            details['status'] = details['status'].value
        monitor.update_model_status("tagger-main", status, details)

    print("✅ Tag Engine initialized")
except Exception as e:
    print(f"❌ Failed to initialize Tag Engine: {e}")
    tag_engine = None

# Initialize LLM chunker with status monitoring
print("\n📝 Initializing LLM Chunker...")
try:
    llm_chunker = LLMChunker(config_manager)
    chunker_config = config_manager.get_llm_config()
    monitor.register_model(
        model_id="chunker-main",
        model_type=ModelType.CHUNKER,
        provider=chunker_config.provider.value,
        model_name=chunker_config.name
    )

    # Update status based on service check
    provider_key = chunker_config.provider.value.replace('-', '_')
    if provider_key in service_check_results:
        status = service_check_results[provider_key].get('status', ModelStatus.UNKNOWN)
        # Convert status to string for serialization
        details = service_check_results[provider_key].copy()
        if 'status' in details and isinstance(details['status'], ModelStatus):
            details['status'] = details['status'].value
        monitor.update_model_status("chunker-main", status, details)
    else:
        # Check if it's using fallback method
        if chunker_config.provider.value == "fallback":
            monitor.update_model_status("chunker-main", ModelStatus.HEALTHY, {"info": "Using fallback chunking method"})

    print(f"✅ LLM Chunker initialized: {chunker_config.provider.value} ({chunker_config.name})")
except Exception as e:
    print(f"❌ Failed to initialize LLM Chunker: {e}")
    monitor.update_model_status("chunker-main", ModelStatus.FAILED, {"error": str(e)})
    llm_chunker = None

# Initialize document processor
print("\n📄 Initializing Document Processor...")
processor = DocumentProcessor(db, embedder, tag_engine, llm_chunker, config_manager)
print("✅ Document Processor initialized")

# Print initialization summary
print("\n" + "="*80)
print("📋 INITIALIZATION SUMMARY")
print("="*80)
monitor.print_status_dashboard()


@app.get("/")
async def root():
    """Serve the main application"""
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/health/models")
async def model_health():
    """Comprehensive model status endpoint"""
    try:
        # Get full status report from monitor
        status_report = monitor.get_full_status_report()

        # Add current service availability
        services = {
            "ollama": False,
            "sentence_transformers": False,
            "gemini": False
        }

        # Quick check for Ollama
        try:
            ollama_url = config_manager.config.get("llm", {}).get("providers", {}).get("ollama", {}).get("url", "http://localhost:11434")
            response = requests.get(f'{ollama_url}/api/tags', timeout=1)
            services["ollama"] = response.status_code == 200
        except:
            pass

        # Check if sentence-transformers is available
        try:
            import sentence_transformers
            services["sentence_transformers"] = True
        except:
            pass

        # Check Gemini API key
        gemini_key = config_manager.config.get('gemini', {}).get('api_key')
        services["gemini"] = bool(gemini_key and gemini_key not in ["YOUR_GEMINI_API_KEY", ""])

        return {
            "status": "operational",
            "services": services,
            "models": status_report["models"],
            "timestamp": status_report["timestamp"],
            "uptime_seconds": status_report["uptime_seconds"]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/health/detailed")
async def detailed_health():
    """Run comprehensive service checks and return detailed status"""
    try:
        # Run full service check
        results = monitor.check_all_services({
            'gemini_api_key': config_manager.config.get('gemini', {}).get('api_key')
        })

        # Convert ModelStatus enums to strings for JSON serialization
        for service, details in results.items():
            if 'status' in details and hasattr(details['status'], 'value'):
                details['status'] = details['status'].value

        return {
            "status": "check_complete",
            "services": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents")
async def create_document(doc: DocumentCreateRequest):
    """Create a new document with validation"""
    try:
        doc_id = processor.process_document(doc.title, doc.content, doc.doc_type)
        return {"id": doc_id, "message": "Document processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get all documents"""
    documents = db.get_all_documents()
    response = []
    for doc in documents:
        chunks = db.get_chunks_by_document(doc.id)
        response.append(DocumentResponse(
            id=doc.id,
            title=doc.title,
            doc_type=doc.doc_type,
            created_at=doc.created_at.isoformat() if isinstance(doc.created_at, datetime) else str(doc.created_at),
            chunk_count=len(chunks)
        ))
    return response


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document with chunks and validation"""
    try:
        # Validate document ID
        if not doc_id or not doc_id.strip():
            raise HTTPException(status_code=400, detail="Invalid document ID")
        
        # Get document
        doc = db.get_document(doc_id.strip())
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks and connections
        chunks = db.get_chunks_by_document(doc_id.strip())
        chunk_ids = [c.id for c in chunks]
        connections = db.get_connections(chunk_ids)

        return {
            "document": doc.model_dump(),
            "chunks": [c.model_dump() for c in chunks],
            "connections": [c.model_dump() for c in connections]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


@app.get("/api/visualization")
async def get_visualization_data():
    """Get all data for 3D visualization"""
    documents = db.get_all_documents()
    all_chunks = []
    all_connections = []

    doc_responses = []
    for doc in documents:
        chunks = db.get_chunks_by_document(doc.id)
        chunk_ids = [c.id for c in chunks]
        connections = db.get_connections(chunk_ids)

        doc_responses.append(DocumentResponse(
            id=doc.id,
            title=doc.title,
            doc_type=doc.doc_type,
            created_at=doc.created_at.isoformat(),
            chunk_count=len(chunks)
        ))

        for chunk in chunks:
            all_chunks.append(ChunkResponse(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content[:200],  # Truncate for performance
                chunk_type=chunk.chunk_type,
                position_3d=chunk.position_3d,
                color=chunk.color,
                tags=chunk.tags,
                reasoning=chunk.reasoning,
                shape_3d=processor._get_shape_from_tags(chunk.tags),
                metadata=chunk.metadata
            ))

        all_connections.extend([c.model_dump() for c in connections])

    return VisualizationData(
        documents=doc_responses,
        chunks=all_chunks,
        connections=all_connections
    )


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document with validation"""
    try:
        # Validate document ID
        if not doc_id or not doc_id.strip():
            raise HTTPException(status_code=400, detail="Invalid document ID")
        
        # Check if document exists
        try:
            doc = db.get_document(doc_id.strip())
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
        except Exception:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete the document
        db.delete_document(doc_id.strip())
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing with security validation"""
    try:
        # Security validations
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (10MB limit)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        
        # Check file extension
        allowed_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.log'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Decode content safely
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File contains invalid UTF-8 characters")

        # Determine file type based on extension
        doc_type = "text"
        if file.filename.endswith(('.py', '.js', '.rs', '.cpp', '.java')):
            doc_type = "code"

        doc_id = processor.process_document(
            title=file.filename,
            content=content_str,
            doc_type=doc_type
        )

        return {"id": doc_id, "message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chunks/{chunk_id}/attach")
async def attach_document_to_chunk(chunk_id: str, request: ChunkAttachmentRequest):
    """Attach a document to a chunk with validation"""
    try:
        # Validate chunk_id
        if not chunk_id or not chunk_id.strip():
            raise HTTPException(status_code=400, detail="Invalid chunk ID")
        
        # Verify chunk exists (basic validation)
        if len(chunk_id.strip()) < 5:  # Basic length check
            raise HTTPException(status_code=400, detail="Invalid chunk ID format")
        
        db.add_attachment(chunk_id.strip(), request.document_id)
        return {"message": "Document attached successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to attach document: {str(e)}")


@app.delete("/api/chunks/{chunk_id}/attach/{document_id}")
async def remove_attachment_from_chunk(chunk_id: str, document_id: str):
    """Remove a document attachment from a chunk"""
    try:
        db.remove_attachment(chunk_id, document_id)
        return {"message": "Attachment removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chunks/{chunk_id}/attachments")
async def get_chunk_attachments(chunk_id: str):
    """Get all documents attached to a chunk"""
    try:
        attachments = db.get_chunk_attachments(chunk_id)
        return [doc.model_dump() for doc in attachments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo for now, can add real-time features
            await websocket.send_text(f"Echo: {data}")
    except:
        pass


# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==============================================================================
# INITIALIZATION
# ==============================================================================

def load_all_documents():
    """Load ALL documents from documents folder into the system"""
    docs_dir = Path(DOCUMENTS_FOLDER)

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True)
        print(f"📁 Created documents folder: {docs_dir}")
        return

    # Get all text files
    file_patterns = ['*.txt', '*.md', '*.py', '*.js', '*.rs', '*.java', '*.cpp']
    all_files = []
    for pattern in file_patterns:
        all_files.extend(docs_dir.glob(pattern))

    if not all_files:
        print("📄 No documents found in documents folder")
        return

    print(f"\n📚 Found {len(all_files)} document(s) in {DOCUMENTS_FOLDER}/")
    print("=" * 60)

    # Get existing documents to avoid reprocessing
    existing_docs = db.get_all_documents()
    existing_titles = {doc.title for doc in existing_docs}

    processed_count = 0
    skipped_count = 0

    for filepath in sorted(all_files):
        # Create title from filename
        title = filepath.stem.replace('_', ' ').replace('-', ' ').title()

        # Check if already processed
        if title in existing_titles:
            skipped_count += 1
            print(f"  ⏭️  Skipped (already loaded): {filepath.name}")
            continue

        try:
            # Read content
            content = filepath.read_text(encoding='utf-8', errors='ignore')

            # Determine document type
            if filepath.suffix in ['.py', '.js', '.rs', '.java', '.cpp', '.c', '.ts']:
                doc_type = "code"
            else:
                doc_type = "text"

            # Process with full pipeline (embeddings, tags, etc.)
            print(f"  🔄 Processing: {filepath.name} ({len(content)} chars)")
            doc_id = processor.process_document(title, content, doc_type)
            print(f"     ✅ Created: {doc_id}")

            processed_count += 1

        except Exception as e:
            print(f"     ❌ Error processing {filepath.name}: {e}")

    print("=" * 60)
    print(f"📊 Summary: {processed_count} processed, {skipped_count} skipped")
    print()


if __name__ == "__main__":
    print("\n🌌 Headspace System Starting...")
    print("📂 Documents folder: ./documents/")
    print("🗄️ Database: headspace.db")
    print("🌐 Server: http://localhost:8000")
    print("\nAPI Endpoints:")
    print("  POST   /api/documents        - Create document")
    print("  GET    /api/documents        - List documents")
    print("  GET    /api/documents/:id    - Get document")
    print("  DELETE /api/documents/:id    - Delete document")
    print("  GET    /api/visualization    - Get viz data")
    print("  POST   /api/upload           - Upload file")
    print("\n🏥 Health Check Endpoints:")
    print("  GET    /api/health           - Basic health check")
    print("  GET    /api/health/models    - Model status overview")
    print("  GET    /api/health/detailed  - Comprehensive service check")

    # Load all documents from documents folder
    load_all_documents()

    print("🚀 Starting API server...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
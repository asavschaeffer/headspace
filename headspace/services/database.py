"""
Database Manager for Headspace System
Handles all database operations for documents, chunks, and connections
"""

import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from data_models import Document, Chunk, ChunkConnection


class DatabaseManager:
    """Manages all database operations for the Headspace system"""

    def __init__(self, db_path: str = "headspace.db"):
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
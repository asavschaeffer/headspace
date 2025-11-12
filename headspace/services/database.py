"""
Database Manager for Headspace System
Handles all database operations for documents, chunks, and connections
"""

import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Optional
CLUSTER_COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#5254a3",
    "#8ca252",
    "#bd9e39",
    "#ad494a",
    "#a55194",
]


def _cluster_color_from_id(cluster_id: Optional[int]) -> Optional[str]:
    if cluster_id is None:
        return None
    return CLUSTER_COLOR_PALETTE[cluster_id % len(CLUSTER_COLOR_PALETTE)]
from data_models import Document, Chunk, ChunkConnection


class DatabaseManager:
    """Manages all database operations for the Headspace system"""

    def __init__(self, db_path: str = "headspace.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
        self._ensure_schema_updates(conn)
        conn.close()

    def _ensure_schema_updates(self, conn: sqlite3.Connection):
        """Ensure newer schema fields and tables exist."""
        cursor = conn.cursor()

        # Ensure additional columns on chunks table
        cursor.execute("PRAGMA table_info(chunks)")
        existing_columns = {row["name"] for row in cursor.fetchall()}

        required_columns = {
            "cluster_id": "INTEGER",
            "cluster_confidence": "REAL",
            "cluster_label": "TEXT",
            "umap_x": "REAL",
            "umap_y": "REAL",
            "umap_z": "REAL",
            "nearest_chunk_ids": "TEXT"
        }

        for column, col_type in required_columns.items():
            if column not in existing_columns:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {column} {col_type}")

        # Ensure cluster metadata table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_metadata (
                cluster_id INTEGER PRIMARY KEY,
                label TEXT,
                size INTEGER,
                color TEXT,
                updated_at TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_cluster ON chunks(cluster_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_label ON cluster_metadata(label)")

        conn.commit()

    def save_document(self, document: Document) -> str:
        """Save a document to the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
        conn.row_factory = sqlite3.Row
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
        conn.row_factory = sqlite3.Row
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
        conn.row_factory = sqlite3.Row
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
            (id, document_id, chunk_index, content, chunk_type, embedding, position_3d, color, metadata,
             cluster_id, cluster_confidence, cluster_label, umap_x, umap_y, umap_z, nearest_chunk_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.id,
            chunk.document_id,
            chunk.chunk_index,
            chunk.content,
            chunk.chunk_type,
            embedding_bytes,
            json.dumps(chunk.position_3d),
            chunk.color,
            json.dumps(metadata_to_save),
            chunk.cluster_id,
            chunk.cluster_confidence,
            chunk.cluster_label,
            chunk.umap_coordinates[0] if len(chunk.umap_coordinates) > 0 else None,
            chunk.umap_coordinates[1] if len(chunk.umap_coordinates) > 1 else None,
            chunk.umap_coordinates[2] if len(chunk.umap_coordinates) > 2 else None,
            json.dumps(chunk.nearest_chunk_ids)
        ))

        conn.commit()
        conn.close()
        return chunk.id

    def get_chunks_by_document(self, doc_id: str) -> List[Chunk]:
        """Get all chunks for a document"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,))
        rows = cursor.fetchall()

        chunks = []
        for row in rows:
            # Convert embedding bytes back to list
            embedding = None
            if row["embedding"]:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()

            chunk_id = row["id"]
            metadata = json.loads(row["metadata"] or '{}')

            # Get attachments for this chunk
            cursor.execute("""
                SELECT document_id FROM attachments
                WHERE chunk_id = ?
                ORDER BY created_at DESC
            """, (chunk_id,))
            attachment_rows = cursor.fetchall()
            attachments = [att_row[0] for att_row in attachment_rows]

            umap_coordinates = []
            if row["umap_x"] is not None and row["umap_y"] is not None and row["umap_z"] is not None:
                umap_coordinates = [row["umap_x"], row["umap_y"], row["umap_z"]]

            nearest_ids = []
            if row["nearest_chunk_ids"]:
                try:
                    nearest_ids = json.loads(row["nearest_chunk_ids"])
                except json.JSONDecodeError:
                    nearest_ids = []

            chunks.append(Chunk(
                id=chunk_id,
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                chunk_type=row["chunk_type"],
                embedding=embedding or [],
                position_3d=json.loads(row["position_3d"] or '[]'),
                color=row["color"],
                metadata=metadata,
                attachments=attachments,
                tags=metadata.get('tags', []),
                tag_confidence=metadata.get('tag_confidence', {}),
                reasoning=metadata.get('reasoning', ''),
                cluster_id=row["cluster_id"],
                cluster_confidence=row["cluster_confidence"],
                cluster_label=row["cluster_label"],
                umap_coordinates=umap_coordinates,
                nearest_chunk_ids=nearest_ids
            ))

        conn.close()
        return chunks

    def save_connection(self, connection: ChunkConnection):
        """Save a connection between chunks"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
        conn.row_factory = sqlite3.Row
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
                from_chunk_id=row["from_chunk_id"],
                to_chunk_id=row["to_chunk_id"],
                connection_type=row["connection_type"],
                strength=row["strength"]
            ))

        return connections

    def delete_document(self, doc_id: str):
        """Delete a document and all its chunks"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Delete document (cascades to chunks and connections)
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        conn.commit()
        conn.close()

    def add_attachment(self, chunk_id: str, document_id: str, attachment_type: str = "document"):
        """Attach a document to a chunk"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
        conn.row_factory = sqlite3.Row
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
        conn.row_factory = sqlite3.Row
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
                id=row["id"],
                title=row["title"],
                content=row["content"],
                doc_type=row["doc_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"] or '{}')
            ))

        return documents

    def update_chunk_cluster_info(
        self,
        chunk_id: str,
        cluster_id: Optional[int],
        cluster_confidence: Optional[float],
        cluster_label: Optional[str],
        coordinates: Optional[List[float]] = None,
        nearest_ids: Optional[List[str]] = None,
        color: Optional[str] = None,
    ):
        """Update clustering metadata for a specific chunk."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        coord_x = coord_y = coord_z = None
        if coordinates and len(coordinates) >= 3:
            coord_x, coord_y, coord_z = coordinates[:3]

        nearest_json = json.dumps(nearest_ids) if nearest_ids is not None else None

        set_color_sql = ", color = ?" if color is not None else ""
        params = [
            cluster_id,
            cluster_confidence,
            cluster_label,
            coord_x,
            coord_y,
            coord_z,
            nearest_json,
            json.dumps(coordinates or []),
        ]
        if color is not None:
            params.append(color)
        params.append(chunk_id)

        cursor.execute(
            f"""
            UPDATE chunks
            SET cluster_id = ?,
                cluster_confidence = ?,
                cluster_label = ?,
                umap_x = ?,
                umap_y = ?,
                umap_z = ?,
                nearest_chunk_ids = ?,
                position_3d = ?
                {set_color_sql}
            WHERE id = ?
            """,
            params,
        )

        conn.commit()
        conn.close()

    def clear_cluster_metadata(self):
        """Remove all cluster metadata entries."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cluster_metadata")
        conn.commit()
        conn.close()

    def upsert_cluster_metadata(self, cluster_id: int, label: str | None, size: int, color: str | None):
        """Upsert clustering metadata for label & sizing."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO cluster_metadata (cluster_id, label, size, color, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(cluster_id) DO UPDATE SET
                label=excluded.label,
                size=excluded.size,
                color=excluded.color,
                updated_at=CURRENT_TIMESTAMP
            """,
            (cluster_id, label, size, color),
        )
        conn.commit()
        conn.close()

    def get_cluster_metadata(self) -> List[dict]:
        """Return cluster metadata joined with chunk counts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                cm.cluster_id,
                cm.label,
                cm.size,
                cm.color,
                cm.updated_at,
                (
                    SELECT COUNT(*) FROM chunks c WHERE c.cluster_id = cm.cluster_id
                ) as chunk_count
            FROM cluster_metadata cm
            ORDER BY chunk_count DESC
            """
        )
        rows = cursor.fetchall()

        if rows:
            conn.close()
            return [dict(row) for row in rows]

        # Fallback: derive rollup directly from chunks when metadata table is empty
        cursor.execute(
            """
            SELECT
                cluster_id,
                cluster_label,
                COUNT(*) as chunk_count
            FROM chunks
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id, cluster_label
            ORDER BY chunk_count DESC
            """
        )
        fallback_rows = cursor.fetchall()
        conn.close()

        return [
            {
                "cluster_id": row["cluster_id"],
                "label": row["cluster_label"],
                "size": row["chunk_count"],
                "color": _cluster_color_from_id(row["cluster_id"]),
                "updated_at": None,
                "chunk_count": row["chunk_count"],
            }
            for row in fallback_rows
        ]

    def get_all_chunk_embeddings(self) -> List[dict]:
        """Fetch all chunk embeddings and metadata for offline processing."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding, c.cluster_id,
                   c.cluster_label, c.color, c.metadata, d.title as document_title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.document_id, c.chunk_index
            """
        )
        rows = cursor.fetchall()
        conn.close()

        payload = []
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist() if row["embedding"] else []
            metadata = json.loads(row["metadata"] or "{}")
            payload.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "document_title": row["document_title"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "embedding": embedding,
                    "cluster_id": row["cluster_id"],
                    "cluster_label": row["cluster_label"],
                    "color": row["color"],
                    "metadata": metadata,
                    "tags": metadata.get("tags", []),
                }
            )
        return payload
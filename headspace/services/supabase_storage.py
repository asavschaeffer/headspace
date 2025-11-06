"""
Supabase Storage Backend
Implements cloud storage for Headspace using Supabase
"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import create_client, Client
from data_models import Document, Chunk, ChunkConnection


class SupabaseStorage:
    """Supabase-backed storage for Headspace documents and chunks"""
    
    def __init__(self, supabase_url: str, supabase_key: str, user_id: Optional[str] = None):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.user_id = user_id or "anonymous"
        
    def _serialize_document(self, doc: Document) -> Dict[str, Any]:
        """Serialize document for Supabase"""
        return {
            "id": doc.id,
            "user_id": self.user_id,
            "title": doc.title,
            "content": doc.content,
            "doc_type": doc.doc_type,
            "created_at": doc.created_at.isoformat() if isinstance(doc.created_at, datetime) else str(doc.created_at),
            "updated_at": doc.updated_at.isoformat() if isinstance(doc.updated_at, datetime) else str(doc.updated_at),
            "metadata": json.dumps(doc.metadata)
        }
    
    def _deserialize_document(self, row: Dict) -> Document:
        """Deserialize document from Supabase"""
        return Document(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            doc_type=row["doc_type"],
            created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            updated_at=datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        )
    
    def save_document(self, document: Document) -> str:
        """Save document to Supabase"""
        data = self._serialize_document(document)
        self.client.table("documents").upsert(data).execute()
        return document.id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document from Supabase"""
        result = self.client.table("documents").select("*").eq("id", doc_id).eq("user_id", self.user_id).execute()
        if result.data:
            return self._deserialize_document(result.data[0])
        return None
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents for user"""
        result = self.client.table("documents").select("*").eq("user_id", self.user_id).order("updated_at", desc=True).execute()
        return [self._deserialize_document(row) for row in result.data]
    
    def delete_document(self, doc_id: str):
        """Delete document from Supabase"""
        self.client.table("documents").delete().eq("id", doc_id).eq("user_id", self.user_id).execute()
        # Cascade delete chunks
        self.client.table("chunks").delete().eq("document_id", doc_id).execute()
    
    def save_chunk(self, chunk: Chunk) -> str:
        """Save chunk to Supabase"""
        embedding_blob = json.dumps(chunk.embedding) if chunk.embedding else None
        data = {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "user_id": self.user_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            "embedding": embedding_blob,
            "position_3d": json.dumps(chunk.position_3d),
            "color": chunk.color,
            "metadata": json.dumps(chunk.metadata),
            "tags": json.dumps(chunk.tags),
            "tag_confidence": json.dumps(chunk.tag_confidence),
            "reasoning": chunk.reasoning or "",
            "shape_3d": getattr(chunk, "shape_3d", "sphere"),
            "texture": getattr(chunk, "texture", "smooth")
        }
        self.client.table("chunks").upsert(data).execute()
        return chunk.id
    
    def get_chunks_by_document(self, doc_id: str) -> List[Chunk]:
        """Get chunks for a document"""
        result = self.client.table("chunks").select("*").eq("document_id", doc_id).eq("user_id", self.user_id).order("chunk_index").execute()
        chunks = []
        for row in result.data:
            chunk = Chunk(
                id=row["id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                chunk_type=row["chunk_type"],
                embedding=json.loads(row["embedding"]) if row["embedding"] else [],
                position_3d=json.loads(row["position_3d"]) if row["position_3d"] else [],
                color=row["color"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                tags=json.loads(row["tags"]) if row["tags"] else [],
                tag_confidence=json.loads(row["tag_confidence"]) if row["tag_confidence"] else {},
                reasoning=row.get("reasoning", "")
            )
            chunks.append(chunk)
        return chunks
    
    def save_connection(self, connection: ChunkConnection):
        """Save connection to Supabase"""
        data = {
            "from_chunk_id": connection.from_chunk_id,
            "to_chunk_id": connection.to_chunk_id,
            "user_id": self.user_id,
            "connection_type": connection.connection_type,
            "strength": connection.strength
        }
        self.client.table("connections").upsert(data).execute()
    
    def get_connections(self, chunk_ids: Optional[List[str]] = None) -> List[ChunkConnection]:
        """Get connections for chunks"""
        query = self.client.table("connections").select("*").eq("user_id", self.user_id)
        if chunk_ids:
            query = query.in_("from_chunk_id", chunk_ids).or_(f"to_chunk_id.in.({','.join(chunk_ids)})")
        result = query.execute()
        return [
            ChunkConnection(
                from_chunk_id=row["from_chunk_id"],
                to_chunk_id=row["to_chunk_id"],
                connection_type=row["connection_type"],
                strength=row["strength"]
            )
            for row in result.data
        ]


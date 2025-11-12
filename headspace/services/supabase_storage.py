"""
Supabase Storage Backend
Implements cloud storage for Headspace using Supabase
"""

import json
import numpy as np
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
        try:
            data = self._serialize_document(document)
            self.client.table("documents").upsert(data).execute()
            print(f"✅ Document {document.id} saved to Supabase")
            return document.id
        except Exception as e:
            print(f"❌ Error saving document to Supabase: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
    
    def _serialize_embedding(self, embedding) -> Optional[str]:
        """Serialize embedding to JSON string, handling numpy arrays and lists"""
        if not embedding:
            return None
        
        try:
            # Convert numpy array to list if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                # Try to convert to list
                embedding = list(embedding)
            
            # Ensure all values are Python native types (not numpy types)
            embedding = [float(x) for x in embedding]
            
            # Return as JSON string - Supabase JSONB will parse this correctly
            return json.dumps(embedding)
        except (TypeError, ValueError) as e:
            print(f"Warning: Failed to serialize embedding: {e}")
            return None
    
    def _deserialize_embedding(self, embedding_data) -> List[float]:
        """Deserialize embedding from Supabase JSONB"""
        if not embedding_data:
            return []
        
        try:
            # If it's already a list (Supabase might return it parsed)
            if isinstance(embedding_data, list):
                return [float(x) for x in embedding_data]
            
            # If it's a string, parse it
            if isinstance(embedding_data, str):
                parsed = json.loads(embedding_data)
                return [float(x) for x in parsed] if isinstance(parsed, list) else []
            
            # If it's already a dict/object, try to extract values
            return []
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Warning: Failed to deserialize embedding: {e}")
            return []
    
    def save_chunk(self, chunk: Chunk) -> str:
        """Save chunk to Supabase"""
        try:
            # Serialize embedding properly
            embedding_json = self._serialize_embedding(chunk.embedding)
            
            # Debug logging
            if embedding_json:
                print(f"💾 Saving chunk {chunk.id} with embedding ({len(chunk.embedding) if chunk.embedding else 0} dims)")
            else:
                print(f"⚠️  Saving chunk {chunk.id} WITHOUT embedding")
            
            data = {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "user_id": self.user_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "embedding": embedding_json,  # JSONB will parse the JSON string
                "position_3d": json.dumps(chunk.position_3d) if chunk.position_3d else None,
                "color": chunk.color,
                "metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}",
                "tags": json.dumps(chunk.tags) if chunk.tags else "[]",
                "tag_confidence": json.dumps(chunk.tag_confidence) if chunk.tag_confidence else "{}",
                "reasoning": chunk.reasoning or "",
                "shape_3d": getattr(chunk, "shape_3d", "sphere"),
                "texture": getattr(chunk, "texture", "smooth"),
                "embedding_model": getattr(chunk, "embedding_model", "") or ""
            }
            
            result = self.client.table("chunks").upsert(data).execute()
            print(f"✅ Chunk {chunk.id} saved to Supabase")
            return chunk.id
        except Exception as e:
            print(f"❌ Error saving chunk to Supabase: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_chunks_by_document(self, doc_id: str) -> List[Chunk]:
        """Get chunks for a document"""
        try:
            result = self.client.table("chunks").select("*").eq("document_id", doc_id).eq("user_id", self.user_id).order("chunk_index").execute()
            chunks = []
            for row in result.data:
                # Deserialize embedding properly
                embedding = self._deserialize_embedding(row.get("embedding"))
                
                # Debug logging
                if embedding:
                    print(f"📖 Retrieved chunk {row.get('id')} with embedding ({len(embedding)} dims)")
                else:
                    print(f"⚠️  Retrieved chunk {row.get('id')} WITHOUT embedding")
                
                # Handle position_3d
                position_3d = []
                if row.get("position_3d"):
                    try:
                        if isinstance(row["position_3d"], str):
                            position_3d = json.loads(row["position_3d"])
                        elif isinstance(row["position_3d"], list):
                            position_3d = row["position_3d"]
                    except (json.JSONDecodeError, TypeError):
                        position_3d = []
                
                # Handle metadata
                metadata = {}
                if row.get("metadata"):
                    try:
                        if isinstance(row["metadata"], str):
                            metadata = json.loads(row["metadata"])
                        elif isinstance(row["metadata"], dict):
                            metadata = row["metadata"]
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                # Handle tags
                tags = []
                if row.get("tags"):
                    try:
                        if isinstance(row["tags"], str):
                            tags = json.loads(row["tags"])
                        elif isinstance(row["tags"], list):
                            tags = row["tags"]
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                
                # Handle tag_confidence
                tag_confidence = {}
                if row.get("tag_confidence"):
                    try:
                        if isinstance(row["tag_confidence"], str):
                            tag_confidence = json.loads(row["tag_confidence"])
                        elif isinstance(row["tag_confidence"], dict):
                            tag_confidence = row["tag_confidence"]
                    except (json.JSONDecodeError, TypeError):
                        tag_confidence = {}
                
                chunk = Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    chunk_type=row.get("chunk_type"),
                    embedding=embedding,
                    position_3d=position_3d,
                    color=row.get("color", ""),
                    metadata=metadata,
                    tags=tags,
                    tag_confidence=tag_confidence,
                    reasoning=row.get("reasoning", ""),
                    shape_3d=row.get("shape_3d", "sphere"),
                    texture=row.get("texture", "smooth"),
                    embedding_model=row.get("embedding_model", "")
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            print(f"Error retrieving chunks from Supabase: {e}")
            raise
    
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
    
    def get_all_chunks(self) -> List[Chunk]:
        """Get all chunks for the current user"""
        try:
            result = self.client.table("chunks").select("*").eq("user_id", self.user_id).order("chunk_index").execute()
            chunks = []
            for row in result.data:
                embedding = self._deserialize_embedding(row.get("embedding"))
                
                position_3d = []
                if row.get("position_3d"):
                    try:
                        if isinstance(row["position_3d"], str):
                            position_3d = json.loads(row["position_3d"])
                        elif isinstance(row["position_3d"], list):
                            position_3d = row["position_3d"]
                    except (json.JSONDecodeError, TypeError):
                        position_3d = []
                
                metadata = {}
                if row.get("metadata"):
                    try:
                        if isinstance(row["metadata"], str):
                            metadata = json.loads(row["metadata"])
                        elif isinstance(row["metadata"], dict):
                            metadata = row["metadata"]
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                tags = []
                if row.get("tags"):
                    try:
                        if isinstance(row["tags"], str):
                            tags = json.loads(row["tags"])
                        elif isinstance(row["tags"], list):
                            tags = row["tags"]
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                
                tag_confidence = {}
                if row.get("tag_confidence"):
                    try:
                        if isinstance(row["tag_confidence"], str):
                            tag_confidence = json.loads(row["tag_confidence"])
                        elif isinstance(row["tag_confidence"], dict):
                            tag_confidence = row["tag_confidence"]
                    except (json.JSONDecodeError, TypeError):
                        tag_confidence = {}
                
                chunk = Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    chunk_type=row.get("chunk_type"),
                    embedding=embedding,
                    position_3d=position_3d,
                    color=row.get("color", ""),
                    metadata=metadata,
                    tags=tags,
                    tag_confidence=tag_confidence,
                    reasoning=row.get("reasoning", ""),
                    shape_3d=row.get("shape_3d", "sphere"),
                    texture=row.get("texture", "smooth"),
                    embedding_model=row.get("embedding_model", "")
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            print(f"Error retrieving all chunks from Supabase: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test the Supabase connection"""
        try:
            # Try to query a simple table to verify connection
            result = self.client.table("documents").select("id").limit(1).execute()
            return True
        except Exception as e:
            print(f"Supabase connection test failed: {e}")
            return False
    
    def get_connections(self, chunk_ids: Optional[List[str]] = None) -> List[ChunkConnection]:
        """Get connections for chunks"""
        try:
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
        except Exception as e:
            print(f"Error retrieving connections from Supabase: {e}")
            raise
    
    def add_attachment(self, chunk_id: str, document_id: str, attachment_type: str = "document"):
        """Attach a document to a chunk"""
        try:
            data = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "user_id": self.user_id,
                "attachment_type": attachment_type,
                "created_at": datetime.now().isoformat(),
                "metadata": "{}"
            }
            self.client.table("attachments").upsert(data).execute()
        except Exception as e:
            print(f"Error adding attachment to Supabase: {e}")
            raise
    
    def remove_attachment(self, chunk_id: str, document_id: str):
        """Remove an attachment from a chunk"""
        try:
            self.client.table("attachments").delete()\
                .eq("chunk_id", chunk_id)\
                .eq("document_id", document_id)\
                .eq("user_id", self.user_id)\
                .execute()
        except Exception as e:
            print(f"Error removing attachment from Supabase: {e}")
            raise
    
    def get_chunk_attachments(self, chunk_id: str) -> List[Document]:
        """Get all documents attached to a chunk"""
        try:
            # Get attachment records
            result = self.client.table("attachments")\
                .select("document_id")\
                .eq("chunk_id", chunk_id)\
                .eq("user_id", self.user_id)\
                .order("created_at", desc=True)\
                .execute()
            
            if not result.data:
                return []
            
            # Get the actual documents
            doc_ids = [row["document_id"] for row in result.data]
            documents = []
            for doc_id in doc_ids:
                doc = self.get_document(doc_id)
                if doc:
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error retrieving chunk attachments from Supabase: {e}")
            raise


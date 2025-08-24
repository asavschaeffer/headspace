"""
SQLite-based storage manager for Globule.

Implements the core storage interface using SQLite with basic schema.
Vector search capabilities will be added in Phase 2.
"""

import json
import sqlite3
import asyncio
import aiosqlite
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
from uuid import UUID
import numpy as np

from globule.core.interfaces import IStorageManager
from globule.core.models import ProcessedGlobuleV1, FileDecisionV1
from globule.core.errors import StorageError
from globule.config.settings import get_config


class SQLiteStorageManager(IStorageManager):
    """SQLite implementation of StorageManager"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        if db_path is None:
            storage_dir = self.config.get_storage_dir()
            if str(storage_dir) == ':memory:':
                # Use in-memory database for SQLite
                db_path = Path(':memory:')
            else:
                db_path = storage_dir / "globules.db"
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        
        # FileManager is a private, internal component
        from globule.storage.file_manager import FileManager
        self._file_manager = FileManager()
    
    async def initialize(self, auto_reconcile: bool = False) -> None:
        """
        Initialize database schema and optionally perform file reconciliation.
        
        Args:
            auto_reconcile: If True, automatically reconcile files with database on startup
        """
        db = await self._get_connection()
        await self._create_schema(db)
        
        # Run migrations to ensure schema is up to date
        await self._run_migrations(db)
            
        # Optional automatic reconciliation on startup
        if auto_reconcile:
            await self._perform_startup_reconciliation()
    
    async def _create_schema(self, db: aiosqlite.Connection) -> None:
        """Create database tables"""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS globules (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB,
                embedding_confidence REAL DEFAULT 0.0,
                parsed_data TEXT,  -- JSON
                parsing_confidence REAL DEFAULT 0.0,
                file_path TEXT,
                original_file_path TEXT,
                orchestration_strategy TEXT DEFAULT 'parallel',
                confidence_scores TEXT,  -- JSON
                processing_time_ms TEXT,  -- JSON
                semantic_neighbors TEXT,  -- JSON array of IDs
                processing_notes TEXT,   -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create vector search virtual table using sqlite-vec
        await db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vss_globules USING vec0(
                embedding FLOAT[1024]
            )
        """)
        
        # Create indexes for performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_globules_created_at 
            ON globules(created_at DESC)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_globules_text 
            ON globules(text)
        """)
        
        # Add indexes for Index-First architecture
        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_globules_original_file_path 
            ON globules(original_file_path) WHERE original_file_path IS NOT NULL
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_globules_file_path 
            ON globules(file_path) WHERE file_path IS NOT NULL
        """)
        
        await db.commit()
    
    async def _run_migrations(self, db: aiosqlite.Connection) -> None:
        """Run database migrations to ensure schema is up to date."""
        try:
            from .migrations import MigrationManager
            migration_manager = MigrationManager(self.db_path)
            await migration_manager.migrate_to_index_first_schema(db)
        except Exception as e:
            # Don't fail initialization due to migration errors for now
            # In production, you might want to fail or handle this more gracefully
            print(f"Warning: Migration failed: {e}")
    
    async def _perform_startup_reconciliation(self) -> None:
        """
        Perform automatic file reconciliation on startup.
        
        This ensures the database reflects the actual state of files on disk,
        handling cases where users have moved, renamed, or organized files.
        """
        try:
            from globule.storage.file_manager import FileManager
            
            file_manager = FileManager()
            print("STARTUP: Performing automatic file reconciliation...")
            
            stats = await file_manager.reconcile_files_with_database(self)
            
            if stats['database_records_updated'] > 0:
                print(f"RECONCILIATION: Updated {stats['database_records_updated']} database records to match file locations")
            
            if stats['files_orphaned'] > 0:
                print(f"RECONCILIATION: Found {stats['files_orphaned']} orphaned files without UUIDs")
                
            print("STARTUP: File reconciliation complete")
            
        except Exception as e:
            print(f"STARTUP WARNING: File reconciliation failed: {e}")
            # Don't fail initialization due to reconciliation errors
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(str(self.db_path))
            await self._connection.enable_load_extension(True)
            
            # Load sqlite-vec extension
            try:
                import sqlite_vec
                await self._connection.execute("SELECT load_extension(?)", (sqlite_vec.loadable_path(),))
            except ImportError:
                # Fallback to old vec0 name for compatibility
                await self._connection.load_extension("vec0")
            # Enable foreign keys and set performance optimizations
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA journal_mode = WAL")
            await self._connection.execute("PRAGMA synchronous = NORMAL")
        return self._connection
    
    async def store_globule(self, globule: ProcessedGlobuleV1) -> str:
        """
        Store a processed globule using the transactional Outbox Pattern.
        
        This implementation ensures true atomicity:
        1. Determine final file path before any operations
        2. Create file in temporary location
        3. Execute database transaction with final file path
        4. Commit file to final location only after DB success
        5. Clean up temp file on any failure
        
        Args:
            globule: The processed globule to store
            
        Returns:
            The globule ID
            
        Raises:
            Exception: If any part of the atomic operation fails
        """
        if globule.id is None:
            globule.id = str(uuid.uuid4())
        
        # OUTBOX PATTERN STEP 1: Determine final file path before any operations
        final_file_path = self._file_manager.determine_path(globule)
        
        # Update globule's file_decision to reflect the determined path
        relative_path = final_file_path.relative_to(self._file_manager.base_path)
        globule.file_decision = FileDecisionV1(
            semantic_path=str(relative_path.parent),
            filename=relative_path.name,
            metadata={"outbox_pattern": True, "atomic_storage": True},
            confidence=1.0,  # High confidence as we determined the path
            alternative_paths=[]
        )
        
        # OUTBOX PATTERN STEP 2: Create file in temporary location
        temp_file_path = self._file_manager.save_to_temp(globule)
        
        try:
            # OUTBOX PATTERN STEP 3: Database transaction with final path
            # Serialize complex fields to JSON
            embedding_blob = None
            if globule.embedding is not None:
                # Convert to numpy array if it's a list
                embedding_array = np.array(globule.embedding, dtype=np.float32) if isinstance(globule.embedding, list) else globule.embedding.astype(np.float32)
                # Normalize the embedding for consistent similarity calculations
                normalized_embedding = self._normalize_vector(embedding_array)
                embedding_blob = normalized_embedding.tobytes()
            
            parsed_data_json = json.dumps(globule.parsed_data)
            confidence_scores_json = json.dumps(globule.confidence_scores)
            processing_time_json = json.dumps(globule.processing_time_ms)
            semantic_neighbors_json = json.dumps(globule.semantic_neighbors)
            processing_notes_json = json.dumps(globule.processing_notes)
            
            # Use the determined file path for database storage
            from pathlib import Path
            file_path = str(Path(globule.file_decision.semantic_path) / globule.file_decision.filename)
            
            db = await self._get_connection()
            
            # This transaction block guarantees all-or-nothing database operations
            try:
                await db.execute("BEGIN TRANSACTION")
                
                # Insert into the main table
                cursor = await db.execute("""
                    INSERT OR REPLACE INTO globules (
                        id, text, embedding, embedding_confidence, parsed_data,
                        parsing_confidence, file_path, original_file_path, orchestration_strategy,
                        confidence_scores, processing_time_ms, semantic_neighbors,
                        processing_notes, created_at, modified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    globule.id,
                    globule.text,
                    embedding_blob,
                    globule.embedding_confidence,
                    parsed_data_json,
                    globule.parsing_confidence,
                    file_path,
                    file_path,  # For managed globules, original_file_path = file_path
                    globule.orchestration_strategy,
                    confidence_scores_json,
                    processing_time_json,
                    semantic_neighbors_json,
                    processing_notes_json,
                    globule.created_at.isoformat(),
                    globule.modified_at.isoformat()
                ))
                
                globule_rowid = cursor.lastrowid
                
                # Insert into the vector search index
                if embedding_blob is not None:
                    await db.execute("""
                        INSERT OR REPLACE INTO vss_globules (rowid, embedding)
                        VALUES (?, ?)
                    """, (globule_rowid, embedding_blob))
                
                await db.commit()
            except Exception as db_error:
                await db.rollback()
                raise db_error
            
            # OUTBOX PATTERN STEP 4: Database transaction succeeded, commit file
            self._file_manager.commit_file(temp_file_path, final_file_path)
            
            return globule.id
            
        except Exception as e:
            # OUTBOX PATTERN STEP 5: Any failure - clean up temp file
            self._file_manager.cleanup_temp(temp_file_path)
            raise Exception(f"Atomic storage operation failed: {e}")
    
    async def store_globule_indexed(self, globule: ProcessedGlobuleV1, original_file_path: str) -> str:
        """
        Store a globule from indexing operation (read-only, no file creation).
        
        This method stores globules that come from indexing existing files.
        It sets original_file_path and leaves file_path as NULL to mark them as unmanaged.
        
        Args:
            globule: The processed globule to store
            original_file_path: Absolute path to the original source file
            
        Returns:
            The globule ID
            
        Raises:
            Exception: If storage fails or if original_file_path already indexed
        """
        if globule.id is None:
            globule.id = str(uuid.uuid4())
        
        # Serialize complex fields to JSON
        embedding_blob = None
        if globule.embedding is not None:
            # Convert to numpy array if it's a list
            embedding_array = np.array(globule.embedding, dtype=np.float32) if isinstance(globule.embedding, list) else globule.embedding.astype(np.float32)
            # Normalize the embedding for consistent similarity calculations
            normalized_embedding = self._normalize_vector(embedding_array)
            embedding_blob = normalized_embedding.tobytes()
        
        parsed_data_json = json.dumps(globule.parsed_data)
        confidence_scores_json = json.dumps(globule.confidence_scores)
        processing_time_json = json.dumps(globule.processing_time_ms)
        semantic_neighbors_json = json.dumps(globule.semantic_neighbors)
        processing_notes_json = json.dumps(globule.processing_notes)
        
        db = await self._get_connection()
        
        try:
            await db.execute("BEGIN TRANSACTION")
            
            # Insert into the main table with original_file_path but NULL file_path
            cursor = await db.execute("""
                INSERT OR REPLACE INTO globules (
                    id, text, embedding, embedding_confidence, parsed_data,
                    parsing_confidence, file_path, original_file_path, orchestration_strategy,
                    confidence_scores, processing_time_ms, semantic_neighbors,
                    processing_notes, created_at, modified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                globule.id,
                globule.text,
                embedding_blob,
                globule.embedding_confidence,
                parsed_data_json,
                globule.parsing_confidence,
                None,  # file_path is NULL for indexed/unmanaged globules
                original_file_path,  # original_file_path tracks source file
                globule.orchestration_strategy,
                confidence_scores_json,
                processing_time_json,
                semantic_neighbors_json,
                processing_notes_json,
                globule.created_at.isoformat(),
                globule.modified_at.isoformat()
            ))
            
            globule_rowid = cursor.lastrowid
            
            # Insert into the vector search index
            if embedding_blob is not None:
                await db.execute("""
                    INSERT OR REPLACE INTO vss_globules (rowid, embedding)
                    VALUES (?, ?)
                """, (globule_rowid, embedding_blob))
            
            await db.commit()
            return globule.id
            
        except Exception as db_error:
            await db.rollback()
            raise Exception(f"Failed to store indexed globule: {db_error}")
    
    async def update_globule(self, globule: ProcessedGlobuleV1) -> bool:
        """
        Update an existing globule atomically.
        
        Args:
            globule: ProcessedGlobule with existing ID to update
            
        Returns:
            True if update succeeded, False if globule doesn't exist
        """
        if globule.id is None:
            raise ValueError("Cannot update globule without an ID")
        
        # Serialize complex fields to JSON
        embedding_blob = None
        if globule.embedding is not None:
            embedding_array = np.array(globule.embedding, dtype=np.float32) if isinstance(globule.embedding, list) else globule.embedding.astype(np.float32)
            embedding_blob = embedding_array.tobytes()
        
        parsed_data_json = json.dumps(globule.parsed_data)
        confidence_scores_json = json.dumps(globule.confidence_scores)
        processing_time_json = json.dumps(globule.processing_time_ms)
        semantic_neighbors_json = json.dumps(globule.semantic_neighbors)
        processing_notes_json = json.dumps(globule.processing_notes)
        
        # Store file path from file decision
        file_path = None
        if globule.file_decision:
            from pathlib import Path
            file_path = str(Path(globule.file_decision.semantic_path) / globule.file_decision.filename)
        
        db = await self._get_connection()
        
        # This transaction block guarantees all-or-nothing update.
        async with db.transaction():
            # Update the main table
            cursor = await db.execute("""
                UPDATE globules SET
                    text = ?, embedding = ?, embedding_confidence = ?, parsed_data = ?,
                    parsing_confidence = ?, file_path = ?, orchestration_strategy = ?,
                    confidence_scores = ?, processing_time_ms = ?, semantic_neighbors = ?,
                    processing_notes = ?, modified_at = ?
                WHERE id = ?
            """, (
                globule.text,
                embedding_blob,
                globule.embedding_confidence,
                parsed_data_json,
                globule.parsing_confidence,
                file_path,
                globule.orchestration_strategy,
                confidence_scores_json,
                processing_time_json,
                semantic_neighbors_json,
                processing_notes_json,
                globule.modified_at.isoformat(),
                globule.id
            ))
            
            # Check if the update affected any rows
            if cursor.rowcount == 0:
                return False
            
            # Get the rowid for the vector table update
            async with db.execute("SELECT rowid FROM globules WHERE id = ?", (globule.id,)) as rowid_cursor:
                row = await rowid_cursor.fetchone()
                if not row:
                    return False
                globule_rowid = row[0]
            
            # Update the vector search index
            if embedding_blob is not None:
                await db.execute("""
                    INSERT OR REPLACE INTO vss_globules (rowid, embedding)
                    VALUES (?, ?)
                """, (globule_rowid, embedding_blob))
            else:
                # Remove from vector search if no embedding
                await db.execute("DELETE FROM vss_globules WHERE rowid = ?", (globule_rowid,))
        
        return True
    
    async def delete_globule(self, globule_id: str) -> bool:
        """
        Delete a globule and its vector embedding atomically.
        
        Args:
            globule_id: The ID of the globule to delete
            
        Returns:
            True if globule was deleted, False if it didn't exist
        """
        db = await self._get_connection()
        
        # This transaction block guarantees all-or-nothing deletion.
        async with db.transaction():
            # First get the rowid before deletion
            async with db.execute("SELECT rowid FROM globules WHERE id = ?", (globule_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return False
                globule_rowid = row[0]
            
            # Delete from vector search table first
            await db.execute("DELETE FROM vss_globules WHERE rowid = ?", (globule_rowid,))
            
            # Delete from main table
            cursor = await db.execute("DELETE FROM globules WHERE id = ?", (globule_id,))
            
            # Check if the deletion affected any rows
            return cursor.rowcount > 0
    
    async def get_globule(self, globule_id: str) -> Optional[ProcessedGlobuleV1]:
        """Retrieve a globule by ID"""
        db = await self._get_connection()
        async with db.execute(
            "SELECT * FROM globules WHERE id = ?", (globule_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_globule(row)
    
    async def get_recent_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]:
        """Get recent globules ordered by creation time"""
        db = await self._get_connection()
        async with db.execute(
            "SELECT * FROM globules ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_globule(row) for row in rows]
    
    async def search_by_embedding(
        self, 
        query_vector: np.ndarray, 
        limit: int = 50,
        similarity_threshold: float = 0.1,
        min_embedding_confidence: Optional[float] = None
        ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Finds semantically similar globules using a single, efficient query.
        This is the correct, non-looping implementation.
        
        Args:
            query_vector: The embedding vector to search for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            min_embedding_confidence: Optional minimum embedding confidence filter
            
        Returns:
            List of (ProcessedGlobule, similarity_score) tuples, sorted by similarity
        """
        if query_vector is None:
            return []
        
        # Normalize the query vector to match stored embeddings
        normalized_query = self._normalize_vector(query_vector.astype(np.float32))
            
        db = await self._get_connection()
        
        # Step 1: Get the rowids of the nearest neighbors from the vector index.
        # This is a fast, native C operation.
        
        # First, check if we have any vectors in the table
        async with db.execute("SELECT COUNT(*) FROM vss_globules") as cursor:
            count_result = await cursor.fetchone()
            if count_result[0] == 0:
                return []  # No vectors in database
        
        async with db.execute("""
            SELECT rowid, distance
            FROM vss_globules
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (normalized_query.tobytes(), limit)) as cursor:
            rows = await cursor.fetchall()
            if not rows:
                return []
        
        neighbor_ids = [row[0] for row in rows]
        distances = {row[0]: row[1] for row in rows}
        
        # Step 2: Fetch all the corresponding globules in a SINGLE query.
        # We build a query with the correct number of placeholders.
        placeholders = ','.join('?' for _ in neighbor_ids)
        
        if min_embedding_confidence is not None:
            sql = f"SELECT rowid, * FROM globules WHERE rowid IN ({placeholders}) AND embedding_confidence >= ?"
            params = neighbor_ids + [min_embedding_confidence]
        else:
            sql = f"SELECT rowid, * FROM globules WHERE rowid IN ({placeholders})"
            params = neighbor_ids
        
        async with db.execute(sql, params) as cursor:
            globule_rows = await cursor.fetchall()
        
        # The database does not guarantee the order of IN clauses,
        # so we re-order the results in Python to match the similarity ranking.
        # row[0] is rowid, row[1:] contains the globule data
        globule_map = {row[0]: self._row_to_globule(row[1:]) for row in globule_rows}
        
        results = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in globule_map:
                globule = globule_map[neighbor_id]
                distance = distances[neighbor_id]
                # Convert distance to similarity score (0-1 range)
                # For cosine distance, similarity = 1 - distance/2 (since cosine distance is in range [0,2])
                # For euclidean distance, we use a different formula
                # Since we don't know the exact distance metric, use a robust conversion
                similarity = 1.0 / (1.0 + distance)  # This works for any positive distance
                
                if similarity >= similarity_threshold:
                    results.append((globule, similarity))
        
        return results


    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector for consistent similarity calculations.
        
        Phase 2: Proper L2 normalization for accurate cosine similarity.
        """
        if vector is None:
            return None
            
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    
    def _row_to_globule(self, row: sqlite3.Row) -> ProcessedGlobuleV1:
        """Convert database row to ProcessedGlobule"""
        # Deserialize embedding
        embedding = None
        if row[2] is not None:  # embedding blob
            embedding = np.frombuffer(row[2], dtype=np.float32)
        
        # Deserialize JSON fields  
        parsed_data = json.loads(row[4]) if row[4] else {}
        confidence_scores = json.loads(row[9]) if row[9] else {}  # Updated index
        processing_time_ms = json.loads(row[10]) if row[10] else {}  # Updated index
        semantic_neighbors = json.loads(row[11]) if row[11] else []  # Updated index
        processing_notes = json.loads(row[12]) if row[12] else []  # Updated index
        
        # Create file decision if file path exists
        file_decision = None
        if row[6]:  # file_path
            file_path = Path(row[6])
            file_decision = FileDecisionV1(
                semantic_path=str(file_path.parent),
                filename=file_path.name,
                metadata={},
                confidence=0.8,  # Default confidence
                alternative_paths=[]
            )
        
        # Create the original globule
        from globule.core.models import GlobuleV1
        original_globule = GlobuleV1(
            globule_id=UUID(row[0]),
            raw_text=row[1],
            source="database",  # We don't store original source separately
            creation_timestamp=datetime.fromisoformat(row[13])  # Updated index
        )
        
        # Create the processed globule with proper field names
        return ProcessedGlobuleV1(
            globule_id=UUID(row[0]),
            processed_timestamp=datetime.fromisoformat(row[14]),  # Updated index
            original_globule=original_globule,
            embedding=embedding,
            parsed_data=parsed_data,
            file_decision=file_decision,
            processing_time_ms=processing_time_ms,
            provider_metadata={
                'embedding_confidence': row[3],
                'parsing_confidence': row[5],
                'orchestration_strategy': row[8],  # Updated index
                'confidence_scores': confidence_scores,
                'semantic_neighbors': semantic_neighbors,
                'processing_notes': processing_notes,
                'original_file_path': row[7]  # Add original_file_path to metadata
            }
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def get_unmanaged_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]:
        """
        Retrieve globules that are indexed but not yet organized (unmanaged).
        
        Returns globules where file_path IS NULL and original_file_path IS NOT NULL.
        
        Args:
            limit: Maximum number of unmanaged globules to retrieve
            
        Returns:
            List of unmanaged ProcessedGlobuleV1 objects
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT id, text, embedding, embedding_confidence, parsed_data,
                   parsing_confidence, file_path, original_file_path, orchestration_strategy,
                   confidence_scores, processing_time_ms, semantic_neighbors,
                   processing_notes, created_at, modified_at
            FROM globules
            WHERE file_path IS NULL AND original_file_path IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = await cursor.fetchall()
        await cursor.close()
        
        return [self._row_to_globule(row) for row in rows]
    
    async def get_managed_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]:
        """
        Retrieve globules that are organized/managed by Globule.
        
        Returns globules where file_path IS NOT NULL.
        
        Args:
            limit: Maximum number of managed globules to retrieve
            
        Returns:
            List of managed ProcessedGlobuleV1 objects
        """
        db = await self._get_connection()
        cursor = await db.execute("""
            SELECT id, text, embedding, embedding_confidence, parsed_data,
                   parsing_confidence, file_path, original_file_path, orchestration_strategy,
                   confidence_scores, processing_time_ms, semantic_neighbors,
                   processing_notes, created_at, modified_at
            FROM globules
            WHERE file_path IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = await cursor.fetchall()
        await cursor.close()
        
        return [self._row_to_globule(row) for row in rows]
    
    async def promote_to_managed(self, globule_id: str, managed_file_path: str) -> bool:
        """
        Promote an unmanaged globule to managed status by setting its file_path.
        
        Args:
            globule_id: ID of the globule to promote
            managed_file_path: Path to the newly created managed file
            
        Returns:
            True if promotion succeeded, False if globule not found or already managed
        """
        db = await self._get_connection()
        
        try:
            cursor = await db.execute("""
                UPDATE globules 
                SET file_path = ?, modified_at = ?
                WHERE id = ? AND file_path IS NULL
            """, (managed_file_path, datetime.now().isoformat(), globule_id))
            
            await db.commit()
            rows_affected = cursor.rowcount
            await cursor.close()
            
            return rows_affected > 0
            
        except Exception as e:
            await db.rollback()
            raise Exception(f"Failed to promote globule to managed: {e}")
    
    async def close(self) -> None:
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        limit: int = 20,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Hybrid search combining text and embedding similarity.
        
        Args:
            query_text: Text query for keyword matching
            query_embedding: Embedding vector for semantic search
            limit: Maximum results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (ProcessedGlobule, combined_score) tuples
        """
        # Get semantic results
        semantic_results = await self.search_by_embedding(
            query_embedding, limit=limit * 2, similarity_threshold=similarity_threshold
        )
        
        # Get text results
        text_results = await self._search_by_text_keywords(query_text, limit=limit * 2)
        
        # Fuse results
        fused_results = self._fuse_search_results(semantic_results, text_results)
        
        return fused_results[:limit]

    async def _search_by_text_keywords(
        self,
        query: str,
        limit: int = 20
    ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Search for globules containing specific keywords.
        
        Args:
            query: Text query with keywords
            limit: Maximum results to return
            
        Returns:
            List of (ProcessedGlobule, relevance_score) tuples
        """
        db = await self._get_connection()
        
        # Simple keyword search using LIKE operator
        keywords = query.lower().split()
        where_clauses = []
        params = []
        
        for keyword in keywords:
            where_clauses.append("LOWER(text) LIKE ?")
            params.append(f"%{keyword}%")
        
        where_sql = " OR ".join(where_clauses)
        
        async with db.execute(f"""
            SELECT id, text, embedding, embedding_confidence, parsed_data,
                   parsing_confidence, file_path, orchestration_strategy,
                   confidence_scores, processing_time_ms, semantic_neighbors,
                   processing_notes, created_at, modified_at
            FROM globules
            WHERE {where_sql}
            ORDER BY embedding_confidence DESC
            LIMIT ?
        """, params + [limit]) as cursor:
            rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            globule = self._row_to_globule(row)
            
            # Calculate simple relevance score based on keyword matches
            text_lower = globule.text.lower()
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            relevance = min(1.0, matches / len(keywords))
            
            results.append((globule, relevance))
        
        return results

    def _fuse_search_results(
        self,
        semantic_results: List[Tuple[ProcessedGlobuleV1, float]],
        text_results: List[Tuple[ProcessedGlobuleV1, float]]
    ) -> List[Tuple[ProcessedGlobuleV1, float]]:
        """
        Fuse semantic and text search results with intelligent scoring.
        
        Args:
            semantic_results: Results from embedding search
            text_results: Results from text keyword search
            
        Returns:
            Combined and deduplicated results with fused scores
        """
        # Create lookup for efficient merging
        semantic_scores = {globule.id: score for globule, score in semantic_results}
        text_scores = {globule.id: score for globule, score in text_results}
        
        # Collect all unique globules
        all_globules = {}
        for globule, _ in semantic_results:
            all_globules[globule.id] = globule
        for globule, _ in text_results:
            all_globules[globule.id] = globule
        
        # Calculate combined scores
        fused_results = []
        for globule_id, globule in all_globules.items():
            semantic_score = semantic_scores.get(globule_id, 0.0)
            text_score = text_scores.get(globule_id, 0.0)
            
            # Weighted combination: 70% semantic, 30% text
            combined_score = 0.7 * semantic_score + 0.3 * text_score
            
            # Boost if found in both searches
            if semantic_score > 0 and text_score > 0:
                combined_score *= 1.2  # 20% boost for multi-match
            
            combined_score = min(1.0, combined_score)  # Cap at 1.0
            fused_results.append((globule, combined_score))
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results

    # Implementation of abstract methods from IStorageManager
    
    def save(self, globule: ProcessedGlobuleV1) -> None:
        """
        Synchronous wrapper for store_globule.
        This is required by the IStorageManager interface.
        """
        # Run the async store_globule in a new event loop
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we can't use run()
                # This shouldn't happen in normal usage as the orchestrator is async
                raise StorageError("Cannot save globule synchronously from within an async context")
            else:
                loop.run_until_complete(self.store_globule(globule))
        except RuntimeError:
            # No event loop exists, create one
            asyncio.run(self.store_globule(globule))
    
    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        """
        Synchronous wrapper for get_globule.
        This is required by the IStorageManager interface.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise StorageError("Cannot get globule synchronously from within an async context")
            else:
                result = loop.run_until_complete(self.get_globule(str(globule_id)))
        except RuntimeError:
            result = asyncio.run(self.get_globule(str(globule_id)))
        
        if result is None:
            raise StorageError(f"Globule {globule_id} not found")
        return result

    async def search(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        Search for globules using natural language query.
        
        This method implements the search functionality that was previously
        embedded in the orchestrator, properly isolating the SQL logic.
        """
        try:
            db = await self._get_connection()
            
            # Simple LIKE search for now - this is where we can implement
            # more sophisticated search logic later
            async with db.execute("""
                SELECT id, text, embedding, embedding_confidence, parsed_data,
                       parsing_confidence, file_path, orchestration_strategy,
                       confidence_scores, processing_time_ms, semantic_neighbors,
                       processing_notes, created_at, modified_at
                FROM globules 
                WHERE text LIKE ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (f"%{query}%", limit)) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    globule = self._row_to_globule(row)
                    results.append(globule)
                
                return results
                
        except Exception as e:
            raise StorageError(f"Search failed: {e}")

    async def execute_sql(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """
        Execute SQL query against the database.
        
        This method implements the SQL execution functionality that was previously
        embedded in the orchestrator, with proper safety checks and error handling.
        """
        try:
            db = await self._get_connection()
            
            # Validate SQL safety (basic check)
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER']
            if any(keyword in query.upper() for keyword in dangerous_keywords):
                raise StorageError("Potentially dangerous SQL detected")
            
            async with db.execute(query) as cursor:
                results = await cursor.fetchall()
                
                # Convert to list of dicts
                results_list = [dict(row) for row in results] if results else []
                headers = [desc[0] for desc in cursor.description] if cursor.description else []
                
                return {
                    "type": "sql_results",
                    "query": query,
                    "query_name": query_name,
                    "results": results_list,
                    "headers": headers,
                    "count": len(results_list)
                }
                
        except StorageError:
            # Re-raise storage errors as-is
            raise
        except Exception as e:
            raise StorageError(f"SQL query execution failed: {e}")

    async def get_table_schema(self, table_name: str) -> str:
        """Returns the CREATE TABLE statement for a given table."""
        db = await self._get_connection()
        async with db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            else:
                raise StorageError(f"Table {table_name} not found.")

    async def execute_raw_query(self, sql: str) -> List[Dict[str, Any]]:
        """Executes a raw SQL query and returns the results as a list of dicts."""
        db = await self._get_connection()
        try:
            async with db.execute(sql) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            raise StorageError(f"Raw SQL query failed: {e}")

    async def query_structured(self, query) -> List[ProcessedGlobuleV1]:
        """
        Execute structured query for high-performance domain-specific searches.
        
        This method provides fast queries for specific domains (e.g., valet workflow)
        by querying indexed fields directly, bypassing vector/full-text search.
        
        Args:
            query: StructuredQuery object containing domain and filter parameters
            
        Returns:
            List of ProcessedGlobuleV1 objects matching the query criteria
            
        Raises:
            StorageError: If query execution fails
        """
        from globule.core.models import StructuredQuery
        
        try:
            db = await self._get_connection()
            
            # Build SQL query based on domain and filters
            where_clauses = []
            params = []
            
            # Add domain-specific filters
            if query.domain and query.domain != "all":
                # For now, map domain to a field in parsed_data
                where_clauses.append("JSON_EXTRACT(parsed_data, '$.domain') = ?")
                params.append(query.domain)
            
            # Add general filters
            for field, value in query.filters.items():
                if field == "text":
                    where_clauses.append("text LIKE ?")
                    params.append(f"%{value}%")
                elif field == "created_after":
                    where_clauses.append("created_at > ?")
                    params.append(value)
                elif field == "created_before":
                    where_clauses.append("created_at < ?")
                    params.append(value)
                elif field == "confidence_threshold":
                    where_clauses.append("embedding_confidence >= ?")
                    params.append(value)
                else:
                    # Try to match in parsed_data JSON
                    where_clauses.append(f"JSON_EXTRACT(parsed_data, '$.{field}') = ?")
                    params.append(value)
            
            # Build WHERE clause
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Build ORDER BY clause
            order_by = "created_at"
            if query.sort_by:
                # Sanitize sort field
                allowed_sort_fields = ["created_at", "modified_at", "embedding_confidence", "parsing_confidence"]
                if query.sort_by in allowed_sort_fields:
                    order_by = query.sort_by
            
            order_direction = "DESC" if query.sort_desc else "ASC"
            
            # Execute query
            sql = f"""
                SELECT id, text, embedding, embedding_confidence, parsed_data,
                       parsing_confidence, file_path, orchestration_strategy,
                       confidence_scores, processing_time_ms, semantic_neighbors,
                       processing_notes, created_at, modified_at
                FROM globules
                WHERE {where_sql}
                ORDER BY {order_by} {order_direction}
                LIMIT ?
            """
            
            params.append(query.limit)
            
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    globule = self._row_to_globule(row)
                    results.append(globule)
                
                return results
                
        except Exception as e:
            raise StorageError(f"Structured query failed: {e}")
"""
Database migration utilities for Globule.

Handles schema changes while maintaining backward compatibility.
"""

import logging
import aiosqlite
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    async def get_schema_version(self, db: aiosqlite.Connection) -> int:
        """Get current schema version from database."""
        try:
            # Create migrations table if it doesn't exist
            await db.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            await db.commit()
            
            # Get current version
            cursor = await db.execute("SELECT MAX(version) FROM schema_migrations")
            result = await cursor.fetchone()
            await cursor.close()
            
            return result[0] if result[0] is not None else 0
            
        except Exception as e:
            logger.warning(f"Failed to get schema version: {e}")
            return 0
    
    async def apply_migration(self, db: aiosqlite.Connection, version: int, description: str, sql_commands: list) -> None:
        """Apply a single migration."""
        logger.info(f"Applying migration {version}: {description}")
        
        try:
            # Execute all SQL commands in the migration
            for command in sql_commands:
                await db.execute(command)
            
            # Record the migration
            await db.execute("""
                INSERT INTO schema_migrations (version, description) 
                VALUES (?, ?)
            """, (version, description))
            
            await db.commit()
            logger.info(f"Migration {version} applied successfully")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to apply migration {version}: {e}")
            raise
    
    async def migrate_to_index_first_schema(self, db: aiosqlite.Connection) -> None:
        """Apply the Index-First architectural schema changes."""
        current_version = await self.get_schema_version(db)
        
        # Migration 1: Add original_file_path column for index-first architecture
        if current_version < 1:
            migration_sql = [
                # Add the new column
                "ALTER TABLE globules ADD COLUMN original_file_path TEXT",
                
                # Create unique index on original_file_path to prevent duplicate indexing
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_globules_original_file_path ON globules(original_file_path) WHERE original_file_path IS NOT NULL",
                
                # Create index on file_path for managed files
                "CREATE INDEX IF NOT EXISTS idx_globules_file_path ON globules(file_path) WHERE file_path IS NOT NULL",
                
                # Update existing records to mark them as managed (they have file_path but no original_file_path)
                """UPDATE globules 
                   SET original_file_path = file_path 
                   WHERE file_path IS NOT NULL AND original_file_path IS NULL"""
            ]
            
            await self.apply_migration(
                db, 1, 
                "Add original_file_path column for index-first architecture",
                migration_sql
            )


async def run_migrations(db_path: Path) -> Dict[str, Any]:
    """
    Run all pending migrations on the database.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Dictionary with migration results and stats
    """
    stats = {
        "migrations_run": 0,
        "current_version": 0,
        "success": True,
        "errors": []
    }
    
    try:
        migration_manager = MigrationManager(db_path)
        
        async with aiosqlite.connect(str(db_path)) as db:
            # Enable foreign keys and WAL mode
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA journal_mode = WAL")
            
            initial_version = await migration_manager.get_schema_version(db)
            logger.info(f"Current schema version: {initial_version}")
            
            # Apply Index-First schema migration
            await migration_manager.migrate_to_index_first_schema(db)
            
            final_version = await migration_manager.get_schema_version(db)
            stats["migrations_run"] = final_version - initial_version
            stats["current_version"] = final_version
            
            logger.info(f"Migrations complete. Schema version: {final_version}")
            
    except Exception as e:
        stats["success"] = False
        stats["errors"].append(str(e))
        logger.error(f"Migration failed: {e}")
        
    return stats


async def check_schema_compatibility(db_path: Path) -> Dict[str, Any]:
    """
    Check if database schema is compatible with Index-First architecture.
    
    Returns:
        Dictionary with compatibility status and required actions
    """
    compatibility = {
        "compatible": False,
        "schema_version": 0,
        "requires_migration": False,
        "has_original_file_path_column": False,
        "existing_records": 0
    }
    
    try:
        async with aiosqlite.connect(str(db_path)) as db:
            migration_manager = MigrationManager(db_path)
            compatibility["schema_version"] = await migration_manager.get_schema_version(db)
            
            # Check if original_file_path column exists
            cursor = await db.execute("PRAGMA table_info(globules)")
            columns = await cursor.fetchall()
            await cursor.close()
            
            column_names = [col[1] for col in columns]
            compatibility["has_original_file_path_column"] = "original_file_path" in column_names
            
            # Count existing records
            cursor = await db.execute("SELECT COUNT(*) FROM globules")
            result = await cursor.fetchone()
            await cursor.close()
            compatibility["existing_records"] = result[0] if result else 0
            
            # Determine compatibility
            compatibility["compatible"] = compatibility["has_original_file_path_column"]
            compatibility["requires_migration"] = not compatibility["compatible"]
            
    except Exception as e:
        logger.error(f"Schema compatibility check failed: {e}")
        compatibility["requires_migration"] = True
        
    return compatibility
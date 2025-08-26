"""
File Management for Globule.

Handles saving and loading globules as markdown files with YAML frontmatter,
implementing the UUID-based canonical ID system as specified in the refactoring plan.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from globule.core.models import ProcessedGlobuleV1, FileDecisionV1
from globule.config.settings import get_config


class FileManager:
    """
    Manages file operations for globules with UUID-based canonical IDs.
    
    Files are saved with human-readable names for user convenience,
    but the UUID is embedded in YAML frontmatter as the canonical identifier.
    """
    
    def __init__(self):
        self.config = get_config()
        storage_dir = self.config.get_storage_dir()
        
        # Handle in-memory database case
        if str(storage_dir) == ':memory:':
            # Use a temporary directory for files when using in-memory database
            import tempfile
            self.base_path = Path(tempfile.gettempdir()) / "globule_temp_files"
        else:
            self.base_path = storage_dir / "files"
        
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_globule_to_file(self, globule: ProcessedGlobuleV1) -> Path:
        """
        Save a globule as a markdown file with YAML frontmatter.
        
        Args:
            globule: The processed globule to save
            
        Returns:
            Path to the saved file
        """
        # Generate human-readable filename
        filename = self._generate_filename(globule)
        
        # Determine file path
        if globule.file_decision and globule.file_decision.semantic_path:
            file_dir = self.base_path / globule.file_decision.semantic_path
            file_path = file_dir / filename
        else:
            # Default to organized by date and category
            category = globule.parsed_data.get('category', 'notes')
            date_dir = globule.created_at.strftime('%Y/%m')
            file_dir = self.base_path / category / date_dir
            file_path = file_dir / filename
        
        # Ensure directory exists
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate YAML frontmatter
        frontmatter = self._generate_frontmatter(globule)
        
        # Create markdown content
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{globule.text}\n"
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    def determine_path(self, globule: ProcessedGlobuleV1) -> Path:
        """
        Determine the final file path BEFORE any transaction begins.
        
        This method calculates where the file will be stored but doesn't create it yet.
        This supports the Outbox Pattern by determining the path before the transaction.
        
        Args:
            globule: The processed globule to determine path for
            
        Returns:
            Absolute path where the file will be stored
        """
        # Generate human-readable filename
        filename = self._generate_filename(globule)
        
        # Determine file directory based on file decision or defaults
        if globule.file_decision and globule.file_decision.semantic_path:
            file_dir = self.base_path / globule.file_decision.semantic_path
        else:
            # Default to organized by date and category
            category = globule.parsed_data.get('category', 'notes')
            date_dir = globule.created_at.strftime('%Y/%m')
            file_dir = self.base_path / category / date_dir
        
        return file_dir / filename
    
    def save_to_temp(self, globule: ProcessedGlobuleV1) -> Path:
        """
        Save globule to a temporary file location.
        
        This is step 1 of the Outbox Pattern - create the file content in a temporary
        location that can be safely cleaned up if the database transaction fails.
        
        Args:
            globule: The processed globule to save
            
        Returns:
            Path to the temporary file
        """
        import tempfile
        
        # Generate YAML frontmatter
        frontmatter = self._generate_frontmatter(globule)
        
        # Create markdown content
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{globule.text}\n"
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.md', prefix='globule_')
        temp_path_obj = Path(temp_path)
        
        try:
            # Write content to temporary file
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return temp_path_obj
            
        except Exception as e:
            # Clean up file descriptor if writing fails
            try:
                os.close(temp_fd)
            except:
                pass
            # Clean up temp file if it was created
            try:
                temp_path_obj.unlink()
            except:
                pass
            raise Exception(f"Failed to create temporary file: {e}")
    
    def commit_file(self, temp_path: Path, final_path: Path) -> None:
        """
        Move file from temporary location to final location.
        
        This is step 3 of the Outbox Pattern - only called after the database
        transaction has successfully committed. This makes the file visible at
        its final location.
        
        Args:
            temp_path: Path to the temporary file
            final_path: Path where the file should be moved to
        """
        try:
            # Ensure the destination directory exists
            final_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file from temp to final location
            # On Windows, rename fails if destination exists, so remove it first
            if final_path.exists():
                final_path.unlink()
            temp_path.rename(final_path)
            
        except Exception as e:
            # If the move fails, we have a problem - the database transaction
            # has already committed, but the file isn't in the right place
            raise Exception(f"CRITICAL: Failed to move file from {temp_path} to {final_path}: {e}")
    
    def cleanup_temp(self, temp_path: Path) -> None:
        """
        Clean up a temporary file after a failed transaction.
        
        This is called when the database transaction fails and we need to
        remove the temporary file that was created.
        
        Args:
            temp_path: Path to the temporary file to remove
        """
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            # Log the error but don't fail - cleanup is best effort
            print(f"Warning: Failed to clean up temporary file {temp_path}: {e}")
    
    def load_globule_from_file(self, file_path: Path) -> Optional[ProcessedGlobuleV1]:
        """
        Load a globule from a markdown file using the UUID from frontmatter.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            ProcessedGlobuleV1 if successful, None if file is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse YAML frontmatter and content
            frontmatter, text = self._parse_frontmatter(content)
            
            if not frontmatter or 'uuid' not in frontmatter:
                return None
            
            # Create ProcessedGlobuleV1 from frontmatter and content
            globule = ProcessedGlobuleV1(
                id=frontmatter['uuid'],
                text=text.strip(),
                parsed_data=frontmatter.get('parsed_data', {}),
                parsing_confidence=frontmatter.get('parsing_confidence', 0.0),
                embedding_confidence=frontmatter.get('embedding_confidence', 0.0),
                orchestration_strategy=frontmatter.get('orchestration_strategy', 'parallel'),
                confidence_scores=frontmatter.get('confidence_scores', {}),
                processing_time_ms=frontmatter.get('processing_time_ms', {}),
                semantic_neighbors=frontmatter.get('semantic_neighbors', []),
                processing_notes=frontmatter.get('processing_notes', []),
                created_at=datetime.fromisoformat(frontmatter.get('created_at', datetime.now().isoformat())),
                modified_at=datetime.fromisoformat(frontmatter.get('modified_at', datetime.now().isoformat()))
            )
            
            # Create file decision if path info exists
            if frontmatter.get('file_decision'):
                fd_data = frontmatter['file_decision']
                globule.file_decision = FileDecisionV1(
                    semantic_path=str(fd_data.get('semantic_path', '')),
                    filename=fd_data.get('filename', file_path.name),
                    metadata=fd_data.get('metadata', {}),
                    confidence=fd_data.get('confidence', 0.8),
                    alternative_paths=[Path(p) for p in fd_data.get('alternative_paths', [])]
                )
            
            return globule
            
        except Exception as e:
            print(f"Error loading globule from {file_path}: {e}")
            return None
    
    def find_globule_by_uuid(self, uuid: str) -> Optional[Path]:
        """
        Find a file by its UUID, searching through all markdown files.
        
        Args:
            uuid: The canonical UUID to search for
            
        Returns:
            Path to the file if found, None otherwise
        """
        # Search through all .md files in the base directory
        for md_file in self.base_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, _ = self._parse_frontmatter(content)
                if frontmatter and frontmatter.get('uuid') == uuid:
                    return md_file
                    
            except Exception:
                continue  # Skip files that can't be read
        
        return None
    
    def update_file_from_globule(self, globule: ProcessedGlobuleV1) -> Optional[Path]:
        """
        Update an existing file or create a new one based on the globule's UUID.
        
        Args:
            globule: The processed globule to save
            
        Returns:
            Path to the updated/created file
        """
        if not globule.id:
            # Generate new UUID if none exists
            import uuid
            globule.id = str(uuid.uuid4())
        
        # Try to find existing file
        existing_file = self.find_globule_by_uuid(globule.id)
        
        if existing_file:
            # Update existing file
            return self._update_existing_file(existing_file, globule)
        else:
            # Create new file
            return self.save_globule_to_file(globule)
    
    def _generate_filename(self, globule: ProcessedGlobuleV1) -> str:
        """Generate a human-readable filename for the globule."""
        # Use title from parsed data or create from text
        if globule.parsed_data and 'title' in globule.parsed_data:
            title = globule.parsed_data['title']
        else:
            # Create title from first 50 characters
            title = globule.text[:50].strip()
        
        # Clean title for filename
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'[-\s]+', '-', clean_title).strip('-')
        
        # Limit length and add .md extension
        filename = clean_title[:50].lower() + '.md'
        
        return filename
    
    def _generate_frontmatter(self, globule: ProcessedGlobuleV1) -> Dict[str, Any]:
        """Generate YAML frontmatter with UUID and metadata."""
        frontmatter = {
            'uuid': globule.id,
            'created_at': globule.created_at.isoformat(),
            'modified_at': globule.modified_at.isoformat(),
            'parsed_data': globule.parsed_data,
            'parsing_confidence': globule.parsing_confidence,
            'embedding_confidence': globule.embedding_confidence,
            'orchestration_strategy': globule.orchestration_strategy,
            'confidence_scores': globule.confidence_scores,
            'processing_time_ms': globule.processing_time_ms,
            'semantic_neighbors': globule.semantic_neighbors,
            'processing_notes': globule.processing_notes
        }
        
        # Add file decision if present
        if globule.file_decision:
            frontmatter['file_decision'] = {
                'semantic_path': str(globule.file_decision.semantic_path),
                'filename': globule.file_decision.filename,
                'metadata': globule.file_decision.metadata,
                'confidence': globule.file_decision.confidence,
                'alternative_paths': [str(p) for p in globule.file_decision.alternative_paths]
            }
        
        return frontmatter
    
    def _parse_frontmatter(self, content: str) -> tuple[Optional[Dict[str, Any]], str]:
        """
        Parse YAML frontmatter from markdown content.
        
        Returns:
            (frontmatter_dict, content_text)
        """
        if not content.startswith('---'):
            return None, content
        
        try:
            # Find the closing ---
            end_marker = content.find('---', 3)
            if end_marker == -1:
                return None, content
            
            # Extract and parse YAML
            yaml_content = content[3:end_marker].strip()
            frontmatter = yaml.safe_load(yaml_content)
            
            # Extract remaining content
            text_content = content[end_marker + 3:].strip()
            
            return frontmatter, text_content
            
        except Exception:
            return None, content
    
    def _update_existing_file(self, file_path: Path, globule: ProcessedGlobuleV1) -> Path:
        """Update an existing file with new globule data."""
        # Update the modified_at timestamp
        globule.modified_at = datetime.now()
        
        # Generate new frontmatter and content
        frontmatter = self._generate_frontmatter(globule)
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{globule.text}\n"
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    async def reconcile_files_with_database(self, storage_manager) -> Dict[str, Any]:
        """
        Reconcile files on disk with database records using UUID as canonical link.
        
        This is the core of Priority 4: ensuring the database reflects the actual
        state of files on disk, regardless of how users have moved or renamed them.
        
        Args:
            storage_manager: SQLiteStorageManager instance for database operations
            
        Returns:
            Dict with reconciliation statistics and actions taken
        """
        stats = {
            "files_scanned": 0,
            "files_reconciled": 0,
            "files_orphaned": 0,
            "database_records_updated": 0,
            "errors": []
        }
        
        # Step 1: Build a map of all files on disk with their UUIDs
        disk_files = {}  # uuid -> file_path
        orphaned_files = []  # files without valid UUIDs
        
        print(f"RECONCILIATION: Scanning files in {self.base_path}")
        
        for md_file in self.base_path.rglob("*.md"):
            stats["files_scanned"] += 1
            
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, _ = self._parse_frontmatter(content)
                
                if frontmatter and 'uuid' in frontmatter:
                    uuid = frontmatter['uuid']
                    if uuid:  # Ensure UUID is not None or empty
                        disk_files[uuid] = md_file
                    else:
                        orphaned_files.append(md_file)
                        stats["files_orphaned"] += 1
                else:
                    orphaned_files.append(md_file)
                    stats["files_orphaned"] += 1
                    
            except Exception as e:
                stats["errors"].append(f"Error reading {md_file}: {e}")
                continue
        
        print(f"RECONCILIATION: Found {len(disk_files)} files with UUIDs, {len(orphaned_files)} orphaned files")
        
        # Step 2: Get all database records and check against disk files
        database_records = await storage_manager.get_recent_globules(limit=10000)  # Get all records
        
        for globule in database_records:
            if not globule.id:
                continue  # Skip records without UUIDs
                
            current_db_path = None
            if globule.file_decision and globule.file_decision.semantic_path:
                current_db_path = globule.file_decision.semantic_path / globule.file_decision.filename
            
            # Check if this UUID exists on disk
            if globule.id in disk_files:
                actual_file_path = disk_files[globule.id]
                relative_path = actual_file_path.relative_to(self.base_path)
                
                # Check if database path matches actual file location
                if current_db_path != relative_path:
                    # File has been moved/renamed - update database
                    await self._update_globule_file_path(storage_manager, globule, actual_file_path)
                    stats["database_records_updated"] += 1
                    stats["files_reconciled"] += 1
                    
                    print(f"RECONCILED: {globule.id[:8]}... moved from {current_db_path} to {relative_path}")
                else:
                    # File is in expected location
                    stats["files_reconciled"] += 1
            else:
                # Database record exists but file is missing
                print(f"WARNING: Database record {globule.id[:8]}... points to missing file: {current_db_path}")
                # Note: We don't delete database records as the file might be temporarily unavailable
        
        # Step 3: Report orphaned files that could be imported
        if orphaned_files:
            print(f"ORPHANED FILES: Found {len(orphaned_files)} files without UUIDs:")
            for orphan in orphaned_files[:5]:  # Show first 5
                relative_path = orphan.relative_to(self.base_path)
                print(f"  - {relative_path}")
            if len(orphaned_files) > 5:
                print(f"  ... and {len(orphaned_files) - 5} more")
        
        return stats
    
    async def _update_globule_file_path(self, storage_manager, globule: ProcessedGlobuleV1, actual_file_path: Path):
        """
        Update a globule's file_decision to reflect its actual location on disk.
        
        Args:
            storage_manager: SQLiteStorageManager instance
            globule: The globule to update
            actual_file_path: The actual path where the file exists
        """
        from globule.core.models import FileDecisionV1
        
        # Calculate relative path from base_path
        relative_path = actual_file_path.relative_to(self.base_path)
        
        # Create new FileDecisionV1 reflecting actual location
        globule.file_decision = FileDecisionV1(
            semantic_path=str(relative_path.parent),
            filename=relative_path.name,
            metadata={"reconciled": True, "original_path": str(globule.file_decision.semantic_path / globule.file_decision.filename) if globule.file_decision else None},
            confidence=0.9,  # High confidence as we found the actual file
            alternative_paths=[]
        )
        
        # Update the globule in the database
        globule.modified_at = datetime.now()
        success = await storage_manager.update_globule(globule)
        
        if not success:
            raise Exception(f"Failed to update globule {globule.id} with new file path")
    
    def scan_for_orphaned_files(self) -> List[Path]:
        """
        Find markdown files that don't have UUIDs in their frontmatter.
        
        These files could potentially be imported into the system.
        
        Returns:
            List of file paths that lack UUID frontmatter
        """
        orphaned_files = []
        
        for md_file in self.base_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, _ = self._parse_frontmatter(content)
                
                if not frontmatter or not frontmatter.get('uuid'):
                    orphaned_files.append(md_file)
                    
            except Exception:
                # If we can't read the file, consider it orphaned
                orphaned_files.append(md_file)
                continue
        
        return orphaned_files
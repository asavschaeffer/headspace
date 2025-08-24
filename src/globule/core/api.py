"""
GlobuleAPI: A clean, UI-agnostic interface to the core application logic.

This class provides a stable, high-level API for any frontend (TUI, Web, etc.)
to interact with Globule's features without needing to know about the
underlying orchestration, storage, or service providers.
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from pathlib import Path

from .interfaces import IStorageManager
from .models import ProcessedGlobuleV1, StructuredQuery
from ..orchestration.engine import GlobuleOrchestrator
from ..storage.file_manager import FileManager
from ..storage.sqlite_manager import SQLiteStorageManager

logger = logging.getLogger(__name__)


class GlobuleAPI:
    """
    The single point of entry for all frontend interactions.
    """

    def __init__(self, storage: IStorageManager, orchestrator: GlobuleOrchestrator):
        self.storage = storage
        self.orchestrator = orchestrator
        self.file_manager = FileManager()

    async def add_thought(self, text: str, source: str = "api") -> ProcessedGlobuleV1:
        """
        Adds and processes a new thought.

        Args:
            text: The raw text of the thought.
            source: The source of the input (e.g., 'tui', 'web').

        Returns:
            The processed globule object.
        """
        from .models import GlobuleV1
        from ..core.models import EnrichedInput
        from datetime import datetime
        
        # Create enriched input for orchestrator
        enriched_input = EnrichedInput(
            original_text=text,
            enriched_text=text,
            detected_schema_id=None,
            schema_config=None,
            additional_context={},
            source=source,
            timestamp=datetime.now(),
            verbosity="concise"
        )
        
        # Process through orchestrator
        processed_globule = await self.orchestrator.process_globule(enriched_input)
        
        # Store the result
        globule_id = await self.storage.store_globule(processed_globule)
        
        return processed_globule

    async def add_from_input_message(self, input_message) -> List[ProcessedGlobuleV1]:
        """
        Process an InputMessage from external sources (WhatsApp, email, etc.) and store all resulting globules.
        
        This method handles bundled content - text and attachments from a single message
        are processed together and linked with shared metadata.
        
        Args:
            input_message: InputMessage object from the inputs module
            
        Returns:
            List of ProcessedGlobuleV1 objects that were created and stored
        """
        # Import here to avoid circular dependency
        from globule.inputs.models import InputMessage
        
        if not isinstance(input_message, InputMessage):
            raise ValueError(f"Expected InputMessage, got {type(input_message)}")
        
        # Process through orchestrator (handles both text and attachments)
        processed_globules = await self.orchestrator.process_input_message(input_message)
        
        # Store all resulting globules
        stored_globules = []
        for globule in processed_globules:
            try:
                globule_id = await self.storage.store_globule(globule)
                stored_globules.append(globule)
            except Exception as e:
                logger.error(f"Failed to store globule from {input_message.source}: {e}")
                # Continue storing other globules even if one fails
                continue
        
        logger.info(f"Stored {len(stored_globules)} globules from {input_message.source} message {input_message.message_id}")
        return stored_globules

    async def search_semantic(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        Performs a semantic vector search for globules.

        Args:
            query: The natural language query.
            limit: The maximum number of results to return.

        Returns:
            A list of processed globules that are semantically similar to the query.
        """
        # Generate embedding for the query
        embedding_result = await self.orchestrator.embedding_provider.embed_single(query)
        
        # Convert embedding to numpy array if needed
        import numpy as np
        embedding_array = np.array(embedding_result.embedding, dtype=np.float32)
        
        # Search storage by embedding
        results = await self.storage.search_by_embedding(embedding_array, limit)
        
        return [result[0] for result in results]  # Extract globules from (globule, similarity) tuples

    async def search_structured(self, query: StructuredQuery) -> List[ProcessedGlobuleV1]:
        """
        Performs a structured search based on metadata filters.

        Args:
            query: A StructuredQuery object with filters.

        Returns:
            A list of globules matching the structured query.
        """
        return await self.storage.query_structured(query)

    async def get_globule_by_id(self, globule_id: UUID) -> Optional[ProcessedGlobuleV1]:
        """
        Retrieves a single globule by its unique ID.

        Args:
            globule_id: The UUID of the globule.

        Returns:
            The processed globule object, or None if not found.
        """
        return await self.storage.get(globule_id)

    async def get_all_globules(self, limit: int = 100) -> List[ProcessedGlobuleV1]:
        """
        Retrieves all globules from storage, up to a limit.

        Args:
            limit: The maximum number of globules to retrieve.

        Returns:
            A list of all processed globules.
        """
        return await self.storage.get_recent_globules(limit)

    async def get_summary_for_text(self, text: str) -> str:
        """
        Generates a summary for a given piece of text.

        Args:
            text: The text to summarize.

        Returns:
            The generated summary.
        """
        try:
            # Use parser to generate summary
            schema_param = {"name": "default"}  # Use default schema for summary
            parsed_data = await self.orchestrator.parser_provider.parse(text, schema_param)
            
            # Try to extract summary from various possible fields
            if isinstance(parsed_data, dict):
                return (parsed_data.get("summary") or 
                       parsed_data.get("title") or 
                       text[:200] + "..." if len(text) > 200 else text)
            
            return text[:200] + "..." if len(text) > 200 else text
            
        except ParserError as e:
            logger.warning(f"Summarization failed due to parser error: {e}. Falling back to truncation.")
            # Fallback to simple truncation
            return text[:200] + "..." if len(text) > 200 else text

    async def reconcile_files(self) -> Dict[str, Any]:
        """
        Reconciles the file system with the database.

        Returns:
            Statistics about the reconciliation process.
        """
        if isinstance(self.storage, SQLiteStorageManager):
            return await self.file_manager.reconcile_files_with_database(self.storage)
        return {"error": "Reconciliation only supported for SQLiteStorageManager"}

    async def export_draft(self, draft_content: str, file_path: str) -> bool:
        """
        Exports draft content to a file.

        Args:
            draft_content: The content of the draft to export.
            file_path: The path to save the file to.

        Returns:
            True if the export was successful, False otherwise.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(draft_content)
            return True
        except IOError:
            return False

    async def get_clusters(self) -> Any:
        """
        Analyzes all globules and groups them into semantic clusters.

        Returns:
            A ClusteringAnalysis object containing the discovered clusters.
        """
        from ..services.clustering.semantic_clustering import SemanticClusteringEngine

        all_globules = await self.get_all_globules(limit=1000)

        # Filter for globules that are suitable for clustering
        clusterable_globules = []
        for globule in all_globules:
            # This logic is moved from the engine to the API layer
            if (hasattr(globule, 'embedding') and globule.embedding and
                hasattr(globule, 'embedding_confidence') and globule.embedding_confidence > 0.5 and
                len(globule.original_globule.raw_text.strip()) > 10):
                clusterable_globules.append(globule)

        clustering_engine = SemanticClusteringEngine()
        analysis = await clustering_engine.analyze_semantic_clusters(
            globules=clusterable_globules,
            min_globules=5
        )
        return analysis

    async def natural_language_query(self, question: str) -> List[Dict[str, Any]]:
        """
        Takes a natural language question, converts it to SQL, executes it,
        and returns the result.

        Args:
            question: The user's natural language question.

        Returns:
            The result of the SQL query as a list of dictionaries.
        """
        # 1. Get the database schema to provide context to the LLM.
        db_schema = await self.storage.get_table_schema('globules')

        # 2. Use the parser to convert the question to a SQL query.
        sql_query = await self.orchestrator.parser_provider.text_to_sql(question, db_schema)

        # 3. Execute the generated SQL query.
        result = await self.storage.execute_raw_query(sql_query)

        return result

    # === Skeleton Management ===
    # TODO: Initialize self.layout_engine in __init__ when layout engine is implemented

    def list_skeletons(self) -> List[Dict[str, Any]]:
        """Lists all available canvas skeleton templates."""
        skeletons = self.layout_engine.list_skeletons()
        return [s.to_dict() for s in skeletons] # Assuming CanvasSkeleton has a to_dict method

    def apply_skeleton(self, name: str, query_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Applies a skeleton template to create a new canvas layout."""
        # A default query_data if none is provided
        if query_data is None:
            from datetime import datetime
            query_data = {'query': f'Applied template {name}', 'timestamp': datetime.now().isoformat()}
        
        modules = self.layout_engine.apply_skeleton_to_canvas(name, query_data)
        # Assuming CanvasModule has a to_dict method or can be represented as a dict
        return [module.__dict__ for module in modules]

    def get_skeleton_stats(self) -> Dict[str, Any]:
        """Gets statistics about skeleton templates."""
        return self.layout_engine.get_skeleton_stats()

    def create_default_skeletons(self) -> List[str]:
        """Creates default skeleton templates for common use cases."""
        return self.layout_engine.create_default_skeletons()
    
    # === Index-First Architecture Methods ===
    
    async def index_path(self, path: str, include_patterns: Optional[List[str]] = None, exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Index all processable files in a directory path for read-only analysis.
        
        This is Phase A of the Index-First architecture - it processes files
        and stores them in the database without creating or modifying any files.
        
        Uses the orchestration engine's processor router to intelligently determine
        what content can be processed (text, images, PDFs, etc.) rather than
        hardcoding file types.
        
        Args:
            path: Absolute path to directory to index
            include_patterns: Optional list of glob patterns to include (e.g., ['*.md', '*.jpg'])
            exclude_patterns: Optional list of glob patterns to exclude (e.g., ['*.tmp', '*.log'])
            
        Returns:
            Dictionary with indexing statistics and results
        """
        from pathlib import Path
        import os
        import fnmatch
        
        source_path = Path(path)
        if not source_path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if not source_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        # Default exclusion patterns for common non-content files
        default_excludes = ['*.tmp', '*.log', '*.cache', '.DS_Store', 'thumbs.db', '*.lock']
        exclude_patterns = (exclude_patterns or []) + default_excludes
        
        stats = {
            "files_processed": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "files_unsupported": 0,
            "total_size_bytes": 0,
            "content_types": {},
            "errors": []
        }
        
        # Process all files recursively
        for root, dirs, files in os.walk(source_path):
            # Skip hidden directories and common non-content directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', '.git']]
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                # Apply include/exclude patterns
                if include_patterns and not any(fnmatch.fnmatch(file, pattern) for pattern in include_patterns):
                    continue
                    
                if any(fnmatch.fnmatch(file, pattern) for pattern in exclude_patterns):
                    stats["files_skipped"] += 1
                    continue
                
                stats["files_processed"] += 1
                
                try:
                    # Let the orchestrator determine if this file can be processed
                    processed_globule = await self._process_file_for_indexing(
                        str(file_path.absolute()),
                        file_path.name
                    )
                    
                    if processed_globule is None:
                        stats["files_unsupported"] += 1
                        continue
                    
                    # Track content types for statistics
                    content_type = processed_globule.provider_metadata.get('content_type', 'unknown')
                    stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
                    
                    # Store as indexed/unmanaged globule
                    if hasattr(self.storage, 'store_globule_indexed'):
                        await self.storage.store_globule_indexed(processed_globule, str(file_path.absolute()))
                    else:
                        # Fallback for older storage managers
                        await self.storage.store_globule(processed_globule)
                    
                    stats["files_indexed"] += 1
                    
                    # Update size stats (handle different content types)
                    if processed_globule.original_globule.raw_text:
                        stats["total_size_bytes"] += len(processed_globule.original_globule.raw_text.encode('utf-8'))
                    
                except Exception as e:
                    stats["files_failed"] += 1
                    stats["errors"].append(f"{file_path}: {str(e)}")
                    logger.error(f"Failed to index file {file_path}: {e}")
        
        logger.info(f"Indexing complete: {stats['files_indexed']} files indexed across {len(stats['content_types'])} content types")
        return stats
    
    async def _process_file_for_indexing(self, file_path: str, filename: str) -> Optional[ProcessedGlobuleV1]:
        """
        Process a file for indexing using the intelligent processor router.
        
        This method leverages Globule's existing processor architecture to determine
        what content can be processed and how, supporting multi-modal content.
        
        Args:
            file_path: Absolute path to the file to process
            filename: Name of the file
            
        Returns:
            ProcessedGlobuleV1 ready for indexed storage, or None if unsupported
        """
        from .models import EnrichedInput, GlobuleV1
        from datetime import datetime
        from pathlib import Path
        
        file_path_obj = Path(file_path)
        
        try:
            # Create a globule representing the file
            # For non-text files, we'll let the processor determine how to handle content
            if self._is_likely_text_file(file_path_obj):
                # For text files, read content directly
                content = file_path_obj.read_text(encoding='utf-8', errors='ignore')
                if len(content.strip()) < 10:
                    return None  # Skip empty files
            else:
                # For non-text files (images, PDFs, etc.), use file path as content
                # The processor router will determine how to handle it
                content = str(file_path_obj)
            
            # Create a globule for the processor router to evaluate
            globule = GlobuleV1(
                raw_text=content,
                source="index",
                initial_context={
                    "filename": filename,
                    "file_path": file_path,
                    "file_extension": file_path_obj.suffix.lower(),
                    "is_file_path": not self._is_likely_text_file(file_path_obj)
                }
            )
            
            # Let the orchestrator process it with the mode flag for indexing
            enriched_input = EnrichedInput(
                original_text=content,
                enriched_text=content,
                detected_schema_id=None,
                schema_config=None,
                additional_context=globule.initial_context,
                source="index",
                timestamp=datetime.now(),
                verbosity="concise"
            )
            
            processed_globule = await self.orchestrator.process_globule(
                enriched_input,
                skip_file_operations=True  # This is the key difference for indexing
            )
            
            return processed_globule
            
        except Exception as e:
            logger.warning(f"Could not process file {file_path}: {e}")
            return None
    
    def _is_likely_text_file(self, file_path: Path) -> bool:
        """
        Determine if a file is likely to be a text file that can be read directly.
        
        This is a heuristic - the processor router will make the final determination.
        """
        text_extensions = {
            '.md', '.txt', '.py', '.js', '.ts', '.json', '.yaml', '.yml', 
            '.rst', '.org', '.html', '.htm', '.xml', '.css', '.sql', '.sh',
            '.bat', '.ps1', '.rb', '.go', '.rs', '.java', '.cpp', '.c', '.h',
            '.php', '.pl', '.r', '.scala', '.swift', '.kt', '.dart', '.vue',
            '.svelte', '.jsx', '.tsx', '.less', '.scss', '.sass', '.ini',
            '.cfg', '.conf', '.toml', '.properties', '.env', '.gitignore',
            '.dockerfile', '.makefile', '.cmake', '.gradle'
        }
        
        return file_path.suffix.lower() in text_extensions
    
    async def organize_repository(self, output_dir: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Organize unmanaged globules into a structured directory layout.
        
        This is Phase B of the Index-First architecture - it takes indexed globules
        and optionally creates an organized file structure based on AI clustering.
        
        Args:
            output_dir: Directory where organized files should be created (default: current config)
            dry_run: If True, show what would be organized without creating files
            
        Returns:
            Dictionary with organization results and proposed structure
        """
        # Get all unmanaged globules
        if hasattr(self.storage, 'get_unmanaged_globules'):
            unmanaged_globules = await self.storage.get_unmanaged_globules(limit=1000)
        else:
            # Fallback: get all globules and filter
            all_globules = await self.get_all_globules(limit=1000)
            unmanaged_globules = [g for g in all_globules if not hasattr(g, 'file_decision') or not g.file_decision]
        
        if not unmanaged_globules:
            return {"status": "no_unmanaged_globules", "count": 0}
        
        # Perform clustering analysis to determine organization structure
        cluster_analysis = await self._analyze_unmanaged_globules(unmanaged_globules)
        
        # Generate proposed directory structure
        proposed_structure = await self._generate_directory_structure(cluster_analysis)
        
        organization_stats = {
            "unmanaged_count": len(unmanaged_globules),
            "clusters_found": len(cluster_analysis.clusters) if cluster_analysis.clusters else 0,
            "proposed_structure": proposed_structure,
            "dry_run": dry_run,
            "files_organized": 0,
            "directories_created": 0
        }
        
        if dry_run:
            return organization_stats
        
        # TODO: Implement interactive approval and file creation
        # This would include:
        # 1. Present structure to user for approval/modification
        # 2. Create directory structure
        # 3. Create organized files with YAML frontmatter
        # 4. Update database records to mark as managed
        
        logger.info("Organization feature not yet implemented beyond dry-run analysis")
        return organization_stats
    
    async def _analyze_unmanaged_globules(self, globules: List[ProcessedGlobuleV1]) -> Any:
        """Analyze unmanaged globules to determine clustering structure."""
        from ..services.clustering.semantic_clustering import SemanticClusteringEngine
        
        # Filter globules suitable for clustering
        clusterable_globules = []
        for globule in globules:
            if (hasattr(globule, 'embedding') and globule.embedding and
                len(globule.original_globule.raw_text.strip()) > 10):
                clusterable_globules.append(globule)
        
        if not clusterable_globules:
            return None
        
        clustering_engine = SemanticClusteringEngine()
        analysis = await clustering_engine.analyze_semantic_clusters(
            globules=clusterable_globules,
            min_globules=3  # Lower threshold for organizing
        )
        return analysis
    
    async def _generate_directory_structure(self, cluster_analysis: Any) -> Dict[str, Any]:
        """Generate proposed directory structure from cluster analysis."""
        if not cluster_analysis or not cluster_analysis.clusters:
            return {"type": "flat", "directories": ["unsorted"]}
        
        structure = {
            "type": "clustered",
            "directories": {}
        }
        
        for i, cluster_obj in enumerate(cluster_analysis.clusters):
            # Use cluster label as directory name, sanitized
            dir_name = self._sanitize_directory_name(cluster_obj.label)
            structure["directories"][dir_name] = {
                "cluster_id": i,
                "description": cluster_obj.description,
                "keywords": cluster_obj.keywords,
                "globule_count": cluster_obj.size,
                "confidence": cluster_obj.confidence_score
            }
        
        return structure
    
    def _sanitize_directory_name(self, name: str) -> str:
        """Sanitize a cluster label into a valid directory name."""
        import re
        # Remove special characters and convert to lowercase
        sanitized = re.sub(r'[^\w\s-]', '', name.lower())
        # Replace spaces with hyphens and limit length
        sanitized = re.sub(r'[-\s]+', '-', sanitized)
        return sanitized.strip('-')[:50]

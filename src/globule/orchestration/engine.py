"""
GlobuleOrchestrator: The core business logic engine for Globule.

This orchestrator implements the IOrchestrationEngine interface and coordinates
all business logic operations while remaining UI-agnostic. It serves as the
bridge between the UI layer and the various service providers.
"""

import asyncio
import time
import logging
import os
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from globule.core.interfaces import IOrchestrationEngine, IEmbeddingAdapter, IParserProvider, IStorageManager
from globule.core.models import GlobuleV1, ProcessedGlobuleV1, FileDecisionV1, NuanceMetaDataV1, ProcessedContent, StructuredQuery, EnrichedInput
from globule.core.errors import ParserError, EmbeddingError, StorageError
from pathlib import Path

logger = logging.getLogger(__name__)


class GlobuleOrchestrator(IOrchestrationEngine):
    """
    Main orchestration engine for processing globules and coordinating business logic.
    
    This class implements the IOrchestrationEngine interface and contains all the 
    business logic previously embedded in the TUI, making it UI-agnostic and reusable.
    """
    
    def __init__(self, 
                 parser_provider: IParserProvider,
                 embedding_provider: IEmbeddingAdapter,
                 storage_manager: IStorageManager,
                 processor_router=None):
        self.parser_provider = parser_provider
        self.embedding_provider = embedding_provider  
        self.storage_manager = storage_manager
        # Processor router for Phase 4 multi-modal support (optional for backward compatibility)
        self.processor_router = processor_router
        
    async def process(self, globule: GlobuleV1, skip_file_operations: bool = False) -> ProcessedGlobuleV1:
        """
        Process a raw globule into a processed globule.
        
        Implements the IOrchestrationEngine interface requirement.
        Coordinates parsing and embedding in parallel, then constructs the final result.
        
        Args:
            globule: The raw globule to process
            skip_file_operations: If True, skip file decision generation (for indexing)
        """
        start_time = time.time()
        
        logger.debug(f"Processing globule: {globule.raw_text[:50]}...")

        # Launch embedding, parsing, and processor tasks concurrently
        embedding_task = asyncio.create_task(self._generate_embedding(globule.raw_text))
        parsing_task = asyncio.create_task(self._parse_content(globule.raw_text))
        
        # Phase 4: Add processor routing if available
        tasks = [embedding_task, parsing_task]
        if self.processor_router:
            processor_task = asyncio.create_task(self._process_with_router(globule))
            tasks.append(processor_task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle embedding result
        embedding_result = results[0]
        if isinstance(embedding_result, Exception):
            logger.error(f"Embedding failed: {embedding_result}")
            embedding = []
        else:
            embedding, _ = embedding_result
        
        # Handle parsing result  
        parsing_result = results[1]
        if isinstance(parsing_result, Exception):
            logger.error(f"Parsing failed: {parsing_result}")
            parsed_data = {"error": str(parsing_result)}
        else:
            parsed_data, _ = parsing_result
            
        # Handle processor result (Phase 4)
        processor_result = None
        if self.processor_router and len(results) > 2:
            if isinstance(results[2], Exception):
                logger.warning(f"Processor routing failed: {results[2]}")
            else:
                processor_result = results[2]
        
        # Generate file decision from parsed data or processor result (skip for indexing)
        file_decision = None
        primary_data = parsed_data
        if processor_result and processor_result.confidence > 0.5:
            # Use processor result if it has high confidence
            primary_data = processor_result.structured_data
        
        if not skip_file_operations:
            file_decision = self._generate_file_decision(globule.raw_text, primary_data)
        
        # Calculate total processing time
        total_time = (time.time() - start_time) * 1000
        
        # Build provider metadata with processor info
        provider_metadata = {
            "parser": self.parser_provider.__class__.__name__,
            "embedder": self.embedding_provider.__class__.__name__,
            "storage": self.storage_manager.__class__.__name__
        }
        
        if processor_result:
            provider_metadata.update({
                "processor_type": processor_result.processor_type,
                "processor_confidence": processor_result.confidence,
                "processor_time_ms": processor_result.processing_time_ms
            })
        
        # Create processed globule
        processed_globule = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=embedding,
            parsed_data=primary_data,
            file_decision=file_decision,
            processing_time_ms=total_time,
            provider_metadata=provider_metadata
        )
        
        logger.debug(f"Globule processed in {total_time:.1f}ms")
        return processed_globule
    
    async def process_globule(self, enriched_input, skip_file_operations: bool = False) -> ProcessedGlobuleV1:
        """
        Process an enriched input into a processed globule.
        
        This method takes an EnrichedInput and converts it to a GlobuleV1 
        before processing, maintaining compatibility with the existing API.
        
        Args:
            enriched_input: The enriched input to process
            skip_file_operations: If True, skip file decision generation (for indexing)
        """
        from globule.core.models import GlobuleV1
        
        # Convert EnrichedInput to GlobuleV1
        globule = GlobuleV1(
            raw_text=enriched_input.original_text,
            source=enriched_input.source,
            initial_context=enriched_input.additional_context
        )
        
        # Use the existing process method with optional file operations skip
        return await self.process(globule, skip_file_operations=skip_file_operations)
    
    # Business Logic Methods (extracted from TUI)
    
    async def capture_thought(self, raw_text: str, source: str = "tui", 
                             context: Dict[str, Any] = None) -> ProcessedGlobuleV1:
        """
        Capture and process a thought/globule.
        
        This method handles the complete workflow of taking raw text input,
        creating a globule, processing it, and storing it.
        """
        if context is None:
            context = {}
            
        # Create raw globule
        globule = GlobuleV1(
            raw_text=raw_text,
            source=source,
            initial_context=context
        )
        
        # Process the globule
        processed_globule = await self.process(globule)
        
        # Store the processed globule
        self.storage_manager.save(processed_globule)
        
        logger.info(f"Captured and stored globule: {globule.globule_id}")
        return processed_globule
    
    async def search_globules(self, query: str, limit: int = 10) -> List[ProcessedGlobuleV1]:
        """
        Search for globules using natural language query.
        
        Delegates to the storage manager's search implementation.
        The orchestrator doesn't know HOW the search is done, only that it CAN be done.
        """
        try:
            return await self.storage_manager.search(query, limit)
        except StorageError as e:
            logger.error(f"Search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            return []
    
    async def get_globule(self, globule_id: UUID) -> Optional[ProcessedGlobuleV1]:
        """Retrieve a specific globule by ID."""
        try:
            return self.storage_manager.get(globule_id)
        except StorageError as e:
            logger.error(f"Failed to get globule {globule_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting globule {globule_id}: {e}")
            return None
    
    def save_draft(self, content: str, topic: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Save content as a draft file.
        
        This implements the actual draft-saving logic from the TUI.
        """
        try:
            # Generate filename - this is the real logic from TUI
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_part = topic.replace(" ", "_") if topic else "draft"
            filename = f"globule_{topic_part}_{timestamp}.md"
            
            # Save to drafts directory
            drafts_dir = "drafts"
            os.makedirs(drafts_dir, exist_ok=True)
            filepath = os.path.join(drafts_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add metadata frontmatter if available
                if metadata or topic:
                    f.write("---\
")
                    if metadata:
                        for key, value in metadata.items():
                            f.write(f"{key}: {value}\
")
                    if topic:
                        f.write(f"topic: {topic}\
")
                    f.write(f"generated: {datetime.now().isoformat()}\
")
                    f.write("---\
\n")
                f.write(content)
            
            logger.info(f"Draft saved to {filepath}")
            return filepath
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to save draft: {e}")
            raise
    
    async def execute_sql_query(self, query: str, query_name: str = "Query") -> Dict[str, Any]:
        """
        Execute SQL query against the database.
        
        Delegates the entire execution, including safety checks, to the storage manager.
        The orchestrator is free of infrastructure details and only deals with abstractions.
        """
        try:
            return await self.storage_manager.execute_sql(query, query_name)
        except StorageError as e:
            logger.error(f"SQL query execution failed: {e}")
            return {
                "type": "error",
                "query": query,
                "query_name": query_name,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected SQL execution error: {e}")
            return {
                "type": "error",
                "query": query,
                "query_name": query_name,
                "error": str(e)
            }
    
    async def execute_query(self, query: str, query_type: str = "natural") -> Dict[str, Any]:
        """
        Execute a query based on its type.
        
        This coordinates different query execution strategies.
        """
        try:
            if query_type == "sql":
                return await self.execute_sql_query(query)
            elif query_type == "natural":
                # For natural language, search globules
                results = await self.search_globules(query)
                return {
                    "type": "search_results",
                    "query": query,
                    "results": results,
                    "count": len(results)
                }
            else:
                return {"type": "unknown", "query": query, "error": f"Unknown query type: {query_type}"}
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"type": "error", "query": query, "error": str(e)}
    
    async def query_structured(self, query: StructuredQuery) -> List[ProcessedGlobuleV1]:
        """
        Execute structured query for high-performance domain-specific searches.
        
        This method provides Phase 4 support for querying processed content
        by domain, processor type, and other structured fields.
        
        Args:
            query: The structured query with domain, filters, and options.
            
        Returns:
            List of ProcessedGlobules matching the query criteria.
        """
        try:
            return await self.storage_manager.query_structured(query)
        except StorageError as e:
            logger.error(f"Structured query failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected structured query error: {e}")
            return []
    
    async def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get processor routing statistics and capabilities.
        
        Returns:
            Dictionary with processor statistics and routing information.
        """
        if not self.processor_router:
            return {"processors_enabled": False}
        
        return {
            "processors_enabled": True,
            "routing_stats": self.processor_router.get_routing_stats(),
            "capabilities": self.processor_router.get_processor_capabilities()
        }
    
    async def process_input_message(self, input_message) -> List[ProcessedGlobuleV1]:
        """
        Process an InputMessage from external sources (WhatsApp, email, etc.) into processed globules.
        
        This method handles bundled content - it can process both text content and attachments
        from a single message, linking them together with shared metadata.
        
        Args:
            input_message: InputMessage object from the inputs module
            
        Returns:
            List of ProcessedGlobuleV1 objects (one for text, one per attachment)
        """
        # Import here to avoid circular dependency
        from globule.inputs.models import InputMessage
        
        if not isinstance(input_message, InputMessage):
            raise ValueError(f"Expected InputMessage, got {type(input_message)}")
        
        if not input_message.has_content:
            logger.warning(f"InputMessage from {input_message.source} has no processable content")
            return []
        
        processed_globules = []
        shared_metadata = input_message.to_globule_source_metadata()
        
        # Process text content if present
        if input_message.content:
            try:
                # Create EnrichedInput for text content
                enriched_input = EnrichedInput(
                    original_text=input_message.content,
                    enriched_text=input_message.content,
                    detected_schema_id=None,
                    schema_config=None,
                    additional_context=shared_metadata,
                    source=input_message.source,
                    timestamp=input_message.timestamp,
                    verbosity="concise"
                )
                
                # Process through existing pipeline
                text_globule = await self.process_globule(enriched_input)
                processed_globules.append(text_globule)
                
                logger.info(f"Processed text content from {input_message.source} message {input_message.message_id}")
                
            except Exception as e:
                logger.error(f"Failed to process text content from {input_message.source}: {e}")
                # Continue processing attachments even if text fails
        
        # Process attachments if present
        for i, attachment in enumerate(input_message.attachments):
            try:
                # Create a GlobuleV1 for the attachment
                attachment_text = f"Attachment: {attachment.filename or f'file_{i+1}'}"
                if attachment.mime_type:
                    attachment_text += f" (type: {attachment.mime_type})"
                
                # Add attachment metadata to shared context
                attachment_metadata = {
                    **shared_metadata,
                    "attachment_index": i,
                    "attachment_filename": attachment.filename,
                    "attachment_type": attachment.attachment_type.value,
                    "attachment_mime_type": attachment.mime_type,
                    "attachment_size_bytes": attachment.size_bytes,
                    "is_attachment": True
                }
                
                # Create EnrichedInput for attachment
                enriched_input = EnrichedInput(
                    original_text=attachment_text,
                    enriched_text=attachment_text,
                    detected_schema_id=None,
                    schema_config=None,
                    additional_context=attachment_metadata,
                    source=f"{input_message.source}_attachment",
                    timestamp=input_message.timestamp,
                    verbosity="concise"
                )
                
                # Process through existing pipeline
                attachment_globule = await self.process_globule(enriched_input)
                
                # Store the actual attachment content in provider_metadata
                # This preserves the binary content for future processing
                if attachment_globule.provider_metadata is None:
                    attachment_globule.provider_metadata = {}
                attachment_globule.provider_metadata["attachment_content"] = attachment.content
                
                processed_globules.append(attachment_globule)
                
                logger.info(f"Processed {attachment.attachment_type.value} attachment from {input_message.source} message {input_message.message_id}")
                
            except Exception as e:
                logger.error(f"Failed to process attachment {i} from {input_message.source}: {e}")
                # Continue processing other attachments
                continue
        
        # Log final results
        if processed_globules:
            logger.info(f"Successfully processed {len(processed_globules)} globules from {input_message.source} message: "
                       f"{len([g for g in processed_globules if not g.provider_metadata.get('is_attachment')])} text, "
                       f"{len([g for g in processed_globules if g.provider_metadata.get('is_attachment')])} attachments")
        else:
            logger.warning(f"No globules could be processed from {input_message.source} message {input_message.message_id}")
        
        return processed_globules

    # Internal helper methods
    
    async def _process_with_router(self, globule: GlobuleV1) -> ProcessedContent:
        """
        Process globule using processor router.
        
        Args:
            globule: The globule to process.
            
        Returns:
            ProcessedContent from the appropriate processor.
        """
        return await self.processor_router.route_and_process(globule)
    
    async def _generate_embedding(self, text: str) -> tuple:
        """Generate embedding and return (embedding, time_ms)"""
        try:
            embedding_result = await self.embedding_provider.embed_single(text)
            return embedding_result.embedding, embedding_result.processing_time_ms
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    async def _parse_content(self, text: str) -> tuple:
        """Parse content and return (parsed_data, time_ms)"""
        start_time = time.time()
        try:
            parsed_data = await self.parser_provider.parse(text)
            processing_time = (time.time() - start_time) * 1000
            return parsed_data, processing_time
        except Exception as e:
            logger.error(f"Content parsing failed: {e}")
            raise ParserError(f"Failed to parse content: {e}")
    
    def _generate_file_decision(self, text: str, parsed_data: Dict[str, Any]) -> FileDecisionV1:
        """Generate file decision based on parsed data"""
        # Use parsed data if available, otherwise create simple path
        domain = parsed_data.get("domain", "general")
        category = parsed_data.get("category", "note")
        title = parsed_data.get("title", text[:30].replace(" ", "-").lower())
        
        # Clean title for filename
        clean_title = "".join(c for c in title if c.isalnum() or c in "-_").strip("-_")
        if not clean_title:
            clean_title = "untitled"
        
        # Create semantic path: domain/category/
        semantic_path = Path(domain) / category
        filename = f"{clean_title}.md"
        
        return FileDecisionV1(
            semantic_path=str(semantic_path),
            filename=filename,
            confidence=0.8  # Default confidence
        )
    

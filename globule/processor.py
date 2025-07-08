"""Main processing pipeline for Globule - orchestrates the entire thought processing flow."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .config import Config
from .embedding_engine import create_embedder
from .parser_engine import create_parser, detect_domain
from .storage import Globule, SQLiteStorage, generate_id


class ThoughtProcessor:
    """Main processor that orchestrates the entire thought processing pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.storage = SQLiteStorage(config.db_path)
        self.embedder = None
        self.parser = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the processor components."""
        self.embedder = await create_embedder(use_ollama=self.config.embedding_provider == "local")
        self.parser = await create_parser(use_ollama=self.config.llm_provider == "local")
        self.logger.info("ThoughtProcessor initialized")
    
    async def process_input(self, text: str) -> str:
        """Process a new thought input through the complete pipeline."""
        # Step 1: Generate unique ID and cache input immediately
        globule_id = generate_id()
        timestamp = datetime.now()
        
        # Step 2: Store raw input immediately for responsiveness
        basic_globule = Globule(
            id=globule_id,
            content=text,
            created_at=timestamp
        )
        
        try:
            await self.storage.store_globule(basic_globule)
            self.logger.info(f"Cached input for globule {globule_id}")
            
            # Step 3: Process in background
            asyncio.create_task(self._process_background(globule_id, text))
            
            return globule_id
            
        except Exception as e:
            self.logger.error(f"Failed to cache input: {e}")
            raise
    
    async def _process_background(self, globule_id: str, text: str):
        """Process the globule in the background with embeddings and parsing."""
        try:
            self.logger.info(f"Starting background processing for {globule_id}")
            
            # Step 1: Parallel processing of embedding and parsing
            embedding_task = asyncio.create_task(self._embed_text(text))
            parsing_task = asyncio.create_task(self._parse_text(text))
            
            # Step 2: Gather results
            embedding, parsed_result = await asyncio.gather(
                embedding_task, 
                parsing_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(embedding, Exception):
                self.logger.error(f"Embedding failed: {embedding}")
                embedding = None
            
            if isinstance(parsed_result, Exception):
                self.logger.error(f"Parsing failed: {parsed_result}")
                parsed_result = None
            
            # Step 3: Extract entities and domain
            entities = []
            domain = "other"
            parsed_data = {}
            
            if parsed_result:
                entities = parsed_result.entities
                domain = detect_domain(parsed_result)
                parsed_data = parsed_result.model_dump()
            
            # Step 4: Update the globule with processed data
            processed_globule = Globule(
                id=globule_id,
                content=text,
                created_at=datetime.now(),
                embedding=embedding,
                parsed_data=parsed_data,
                entities=entities,
                domain=domain,
                metadata={"processing_version": "1.0"}
            )
            
            await self.storage.store_globule(processed_globule)
            self.logger.info(f"Completed background processing for {globule_id}")
            
        except Exception as e:
            self.logger.error(f"Background processing failed for {globule_id}: {e}")
    
    async def _embed_text(self, text: str):
        """Generate embedding for text."""
        if not self.embedder:
            raise RuntimeError("Embedder not initialized")
        
        return await self.embedder.embed_text(text)
    
    async def _parse_text(self, text: str):
        """Parse text with LLM."""
        if not self.parser:
            raise RuntimeError("Parser not initialized")
        
        return await self.parser.parse_text(text)
    
    async def get_globule(self, globule_id: str) -> Optional[Globule]:
        """Retrieve a globule by ID."""
        return await self.storage.retrieve_by_id(globule_id)
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self.embedder, 'close'):
            await self.embedder.close()
        if hasattr(self.parser, 'close'):
            await self.parser.close()
        self.logger.info("ThoughtProcessor closed")


class ProcessingManager:
    """Manager for the processing pipeline with singleton pattern."""
    
    _instance: Optional[ThoughtProcessor] = None
    _config: Optional[Config] = None
    
    @classmethod
    async def get_processor(cls, config: Optional[Config] = None) -> ThoughtProcessor:
        """Get or create the processor instance."""
        if cls._instance is None:
            if config is None:
                from .config import load_config
                config = load_config()
            
            cls._config = config
            cls._instance = ThoughtProcessor(config)
            await cls._instance.initialize()
        
        return cls._instance
    
    @classmethod
    async def close(cls):
        """Close the processor instance."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None
            cls._config = None


from opentelemetry import trace

tracer = trace.get_tracer(__name__)

# Convenience functions for the CLI
@tracer.start_as_current_span("process_thought")
async def process_thought(text: str, config: Optional[Config] = None) -> str:
    """Process a thought and return the globule ID."""
    processor = await ProcessingManager.get_processor(config)
    return await processor.process_input(text)


async def get_thought(globule_id: str, config: Optional[Config] = None) -> Optional[Globule]:
    """Retrieve a thought by ID."""
    processor = await ProcessingManager.get_processor(config)
    return await processor.get_globule(globule_id)


async def cleanup():
    """Clean up resources."""
    await ProcessingManager.close()


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('globule.log'),
        logging.StreamHandler()
    ]
)
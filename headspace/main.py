"""
Headspace System - Main Application
Cosmic document visualization system entry point
"""

import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import services
from headspace.services.database import DatabaseManager
from headspace.services.document_processor import DocumentProcessor

# Import API components
from headspace.api.middleware import setup_middleware
from headspace.api import routes

from headspace.api.enrichment_events import enrichment_event_bus
# Import supporting modules
from config_manager import ConfigManager
from model_monitor import ModelMonitor, ModelType, ModelStatus
from embeddings_engine import EmbeddingEngine
from tag_engine import TagEngine
from llm_chunker import LLMChunker

# Configuration
DATABASE_PATH = "headspace.db"
DOCUMENTS_FOLDER = "documents"
STATIC_FOLDER = "static"

# Ensure folders exist
Path(DOCUMENTS_FOLDER).mkdir(exist_ok=True)
Path(STATIC_FOLDER).mkdir(exist_ok=True)

# Initialize model monitor for comprehensive tracking
monitor = ModelMonitor(log_level="DEBUG")


def initialize_services():
    """Initialize all required services with proper error handling"""
    print("\n" + "=" * 80)
    print("🔧 INITIALIZING HEADSPACE SYSTEM SERVICES")
    print("=" * 80 + "\n")

    # Initialize configuration
    config_manager = ConfigManager()
    config_manager.print_status()

    # Run comprehensive service checks before initialization
    print("\n" + "=" * 80)
    print("🔍 PERFORMING PRE-INITIALIZATION SERVICE CHECKS")
    print("=" * 80)
    service_check_results = monitor.check_all_services({
        'gemini_api_key': config_manager.config.get('api_keys', {}).get('gemini'),
        'ollama_url': os.environ.get('OLLAMA_URL', 'http://localhost:11434')
    })

    # Initialize database
    print("\n📊 Initializing Database...")
    db = DatabaseManager(DATABASE_PATH)
    print("✅ Database initialized")

    # Initialize embedding engine with status monitoring
    print("\n🧠 Initializing Embedding Engine...")
    try:
        embedder = EmbeddingEngine(config_manager)
        embedder_config = config_manager.get_embedding_config()
        monitor.register_model(
            model_id="embedder-main",
            model_type=ModelType.EMBEDDING,
            provider=embedder_config.provider.value,
            model_name=embedder_config.name
        )

        # Update status based on service check
        provider_key = embedder_config.provider.value.replace('-', '_')
        if provider_key in service_check_results:
            status = service_check_results[provider_key].get('status', ModelStatus.UNKNOWN)
            # Convert status to string for serialization
            details = service_check_results[provider_key].copy()
            if 'status' in details and isinstance(details['status'], ModelStatus):
                details['status'] = details['status'].value
            monitor.update_model_status("embedder-main", status, details)

        print(f"✅ Embedding Engine initialized: {embedder_config.provider.value} ({embedder_config.name})")
    except Exception as e:
        print(f"❌ Failed to initialize Embedding Engine: {e}")
        monitor.update_model_status("embedder-main", ModelStatus.FAILED, {"error": str(e)})
        embedder = None

    # Initialize tag engine
    print("\n🏷️ Initializing Tag Engine...")
    try:
        tag_engine = TagEngine()
        monitor.register_model(
            model_id="tagger-main",
            model_type=ModelType.TAGGER,
            provider="ollama",
            model_name="llama2"
        )

        # Check if Ollama is available for tagging
        if 'ollama' in service_check_results:
            status = service_check_results['ollama'].get('status', ModelStatus.UNKNOWN)
            # Convert status to string for serialization
            details = service_check_results['ollama'].copy()
            if 'status' in details and isinstance(details['status'], ModelStatus):
                details['status'] = details['status'].value
            monitor.update_model_status("tagger-main", status, details)

        print("✅ Tag Engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Tag Engine: {e}")
        tag_engine = None

    # Initialize LLM chunker with status monitoring
    print("\n📝 Initializing LLM Chunker...")
    try:
        llm_chunker = LLMChunker(config_manager)
        chunker_config = config_manager.get_llm_config()
        monitor.register_model(
            model_id="chunker-main",
            model_type=ModelType.CHUNKER,
            provider=chunker_config.provider.value,
            model_name=chunker_config.name
        )

        # Update status based on service check
        provider_key = chunker_config.provider.value.replace('-', '_')
        if provider_key in service_check_results:
            status = service_check_results[provider_key].get('status', ModelStatus.UNKNOWN)
            # Convert status to string for serialization
            details = service_check_results[provider_key].copy()
            if 'status' in details and isinstance(details['status'], ModelStatus):
                details['status'] = details['status'].value
            monitor.update_model_status("chunker-main", status, details)
        else:
            # Check if it's using fallback method
            if chunker_config.provider.value == "fallback":
                monitor.update_model_status("chunker-main", ModelStatus.HEALTHY,
                                          {"info": "Using fallback chunking method"})

        print(f"✅ LLM Chunker initialized: {chunker_config.provider.value} ({chunker_config.name})")
    except Exception as e:
        print(f"❌ Failed to initialize LLM Chunker: {e}")
        monitor.update_model_status("chunker-main", ModelStatus.FAILED, {"error": str(e)})
        llm_chunker = None

    # Initialize document processor
    print("\n📄 Initializing Document Processor...")
    processor = DocumentProcessor(db, embedder, tag_engine, llm_chunker, config_manager, monitor)
    print("✅ Document Processor initialized")

    # Print initialization summary
    print("\n" + "=" * 80)
    print("📋 INITIALIZATION SUMMARY")
    print("=" * 80)
    monitor.print_status_dashboard()

    return db, config_manager, embedder, tag_engine, llm_chunker, processor


def load_all_documents(db, processor):
    """Load ALL documents from documents folder into the system"""
    docs_dir = Path(DOCUMENTS_FOLDER)

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True)
        print(f"📁 Created documents folder: {docs_dir}")
        return

    # Get all text files
    file_patterns = ['*.txt', '*.md', '*.py', '*.js', '*.rs', '*.java', '*.cpp']
    all_files = []
    for pattern in file_patterns:
        all_files.extend(docs_dir.glob(pattern))

    if not all_files:
        print("📄 No documents found in documents folder")
        return

    print(f"\n📚 Found {len(all_files)} document(s) in {DOCUMENTS_FOLDER}/")
    print("=" * 60)

    # Get existing documents to avoid reprocessing
    existing_docs = db.get_all_documents()
    existing_titles = {doc.title for doc in existing_docs}

    processed_count = 0
    skipped_count = 0

    for filepath in sorted(all_files):
        # Create title from filename
        title = filepath.stem.replace('_', ' ').replace('-', ' ').title()

        # Check if already processed
        if title in existing_titles:
            skipped_count += 1
            print(f"  ⏭️  Skipped (already loaded): {filepath.name}")
            continue

        try:
            # Read content
            content = filepath.read_text(encoding='utf-8', errors='ignore')

            # Determine document type
            if filepath.suffix in ['.py', '.js', '.rs', '.java', '.cpp', '.c', '.ts']:
                doc_type = "code"
            else:
                doc_type = "text"

            # Process with full pipeline (embeddings, tags, etc.)
            print(f"  🔄 Processing: {filepath.name} ({len(content)} chars)")
            doc_id = processor.process_document(title, content, doc_type)
            print(f"     ✅ Created: {doc_id}")

            processed_count += 1

        except Exception as e:
            print(f"     ❌ Error processing {filepath.name}: {e}")

    print("=" * 60)
    print(f"📊 Summary: {processed_count} processed, {skipped_count} skipped")
    print()


def create_app():
    """Create and configure the FastAPI application"""
    app = FastAPI(title="Headspace API", version="1.0.0")

    # Setup middleware
    setup_middleware(app)

    # Initialize services
    db, config_manager, embedder, tag_engine, llm_chunker, processor = initialize_services()

    # Store services in app state for access in routes
    app.state.db = db
    app.state.config_manager = config_manager
    app.state.embedder = embedder
    app.state.tag_engine = tag_engine
    app.state.llm_chunker = llm_chunker
    app.state.processor = processor
    app.state.monitor = monitor


    # Include API routes
    app.include_router(routes.router)

    # Mount static files - create proper instances with html=True for correct MIME types
    from pathlib import Path
    static_path = Path(STATIC_FOLDER).resolve()
    css_path = static_path / "css"
    js_path = static_path / "js"

    if css_path.exists():
        app.mount("/css", StaticFiles(directory=str(css_path), html=True), name="css")
    if js_path.exists():
        app.mount("/js", StaticFiles(directory=str(js_path), html=True), name="js")
    app.mount("/static", StaticFiles(directory=STATIC_FOLDER, html=True), name="static")

    # Load initial documents
    load_all_documents(db, processor)

    return app


if __name__ == "__main__":
    print("\n🌌 Headspace System Starting...")
    print("📂 Documents folder: ./documents/")
    print("🗄️ Database: headspace.db")
    print("🌐 Server: http://localhost:8000")
    print("\nAPI Endpoints:")
    print("  POST   /api/documents        - Create document")
    print("  GET    /api/documents        - List documents")
    print("  GET    /api/documents/:id    - Get document")
    print("  DELETE /api/documents/:id    - Delete document")
    print("  GET    /api/visualization    - Get viz data")
    print("  POST   /api/upload           - Upload file")
    print("\n🏥 Health Check Endpoints:")
    print("  GET    /api/health           - Basic health check")
    print("  GET    /api/health/models    - Model status overview")
    print("  GET    /api/health/detailed  - Comprehensive service check")

    # Create and run application
    app = create_app()

    print("🚀 Starting API server...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
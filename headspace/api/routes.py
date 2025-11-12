"""
API Routes for Headspace System
FastAPI endpoints for document management and visualization
"""

import requests
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, WebSocket, Depends, Request, BackgroundTasks, WebSocketDisconnect
from fastapi.responses import FileResponse

from headspace.models.api_models import (
    DocumentCreateRequest,
    DocumentResponse,
    VisualizationData,
    ChunkResponse,
    ChunkAttachmentRequest
)
from headspace.api.enrichment_events import enrichment_event_bus, EnrichmentEvent


router = APIRouter()


def get_db(request: Request):
    """Dependency to get database instance"""
    return request.app.state.db


def get_processor(request: Request):
    """Dependency to get processor instance"""
    return request.app.state.processor


def get_config_manager(request: Request):
    """Dependency to get config manager instance"""
    return request.app.state.config_manager


def get_monitor(request: Request):
    """Dependency to get monitor instance"""
    return request.app.state.monitor


@router.get("/api/storage/status")
async def storage_status():
    """Get storage mode status"""
    from headspace.services.storage_manager import StorageManager
    storage_manager = StorageManager()
    return {
        "current_mode": storage_manager.get_mode(),
        "cloud_available": storage_manager.can_use_cloud(),
        "local_available": True
    }


@router.get("/")
async def root():
    """Serve the main application"""
    return FileResponse("static/index.html")


@router.get("/headspace.html")
async def headspace():
    """Serve the headspace application"""
    return FileResponse("static/headspace.html")


@router.get("/api/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/api/health/models")
async def model_health(
    config_manager=Depends(get_config_manager),
    monitor=Depends(get_monitor)
):
    """Comprehensive model status endpoint"""
    try:
        # Get full status report from monitor
        status_report = monitor.get_full_status_report()

        # Add current service availability
        services = {
            "ollama": False,
            "sentence_transformers": False,
            "gemini": False
        }

        # Quick check for Ollama
        try:
            ollama_url = config_manager.config.get("llm", {}).get("providers", {}).get("ollama", {}).get("url",
                                                                                                        "http://localhost:11434")
            response = requests.get(f'{ollama_url}/api/tags', timeout=1)
            services["ollama"] = response.status_code == 200
        except:
            pass

        # Check if sentence-transformers is available
        try:
            import sentence_transformers
            services["sentence_transformers"] = True
        except:
            pass

        # Check Gemini API key
        gemini_key = config_manager.config.get('gemini', {}).get('api_key')
        services["gemini"] = bool(gemini_key and gemini_key not in ["YOUR_GEMINI_API_KEY", ""])

        return {
            "status": "operational",
            "services": services,
            "models": status_report["models"],
            "timestamp": status_report["timestamp"],
            "uptime_seconds": status_report["uptime_seconds"]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/api/health/detailed")
async def detailed_health(
    config_manager=Depends(get_config_manager),
    monitor=Depends(get_monitor)
):
    """Run comprehensive service checks and return detailed status"""
    try:
        # Run full service check
        results = monitor.check_all_services({
            'gemini_api_key': config_manager.config.get('gemini', {}).get('api_key')
        })

        # Convert ModelStatus enums to strings for JSON serialization
        for service, details in results.items():
            if 'status' in details and hasattr(details['status'], 'value'):
                details['status'] = details['status'].value

        return {
            "status": "check_complete",
            "services": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def enrich_document_background(processor, doc_id: str, content: str, doc_type: str):
    """Background task to enrich document chunks with embeddings and tags"""
    try:
        processor.monitor.logger.info(f"🔄 Starting enrichment for document {doc_id}")

        # Emit enrichment started event
        await enrichment_event_bus.emit(EnrichmentEvent(
            event_type="started",
            doc_id=doc_id,
            timestamp=datetime.now().isoformat()
        ))

        # Get existing chunks created by instant processor
        chunks = processor.db.get_chunks_by_document(doc_id)

        if not chunks:
            processor.monitor.logger.warning(f"No chunks found for document {doc_id}")
            return

        processor.monitor.logger.info(f"📊 Enriching {len(chunks)} chunks for document {doc_id}")

        # Extract chunk texts for batch processing
        chunk_texts = [chunk.content for chunk in chunks]

        # Generate embeddings for all chunks at once (more efficient)
        try:
            embeddings = processor.embedder.generate_embeddings(chunk_texts)
            positions_3d = processor._calculate_3d_positions(embeddings)

            processor.monitor.logger.info(f"✅ Generated {len(embeddings)} embeddings, updating chunks...")

            # Update each chunk with its embedding and calculated position
            enriched_count = 0
            for i, chunk in enumerate(chunks):
                # Generate tags (non-blocking, can fail silently)
                try:
                    tag_results = processor.tag_engine.generate_tags(chunk.content)
                    tags = tag_results.get('keywords', [])
                except Exception as tag_error:
                    processor.monitor.logger.debug(f"Tag generation failed for chunk {i}: {tag_error}")
                    tags = []

                # Generate color from embedding
                color = processor._get_chunk_color(embeddings[i].tolist())
                embedding_list = embeddings[i].tolist()

                # Update chunk with enriched data
                chunk.embedding = embedding_list
                chunk.position_3d = positions_3d[i].tolist()
                chunk.color = color
                chunk.metadata = {
                    **chunk.metadata,
                    "status": "enriched",
                    "tags": tags
                }
                processor.db.save_chunk(chunk)
                enriched_count += 1

                # Emit event for each enriched chunk (for real-time visualization)
                await enrichment_event_bus.emit(EnrichmentEvent(
                    event_type="chunk_enriched",
                    doc_id=doc_id,
                    chunk_id=chunk.id,
                    chunk_index=i,
                    embedding=embedding_list,
                    color=color,
                    position_3d=chunk.position_3d,
                    progress=int((enriched_count / len(chunks)) * 100),
                    total_chunks=len(chunks),
                    timestamp=datetime.now().isoformat()
                ))

                # Log progress for large documents
                if (i + 1) % 10 == 0:
                    processor.monitor.logger.debug(f"  Progress: {i + 1}/{len(chunks)} chunks enriched")

            processor.monitor.logger.info(f"✅ Enriched {enriched_count}/{len(chunks)} chunks for document {doc_id}")

        except Exception as e:
            processor.monitor.logger.error(f"❌ Enrichment failed for {doc_id}: {e}")
            import traceback
            processor.monitor.logger.error(traceback.format_exc())

            # Emit error event
            await enrichment_event_bus.emit(EnrichmentEvent(
                event_type="error",
                doc_id=doc_id,
                error=str(e),
                timestamp=datetime.now().isoformat()
            ))

            # Update document status to indicate failure
            doc = processor.db.get_document(doc_id)
            if doc:
                doc.metadata["status"] = "enrichment_failed"
                doc.metadata["error"] = str(e)
                processor.db.save_document(doc)
            return

        # Update document status
        doc = processor.db.get_document(doc_id)
        if doc:
            doc.metadata["status"] = "enriched"
            doc.metadata["enriched_at"] = datetime.now().isoformat()
            processor.db.save_document(doc)

            # Emit completion event
            await enrichment_event_bus.emit(EnrichmentEvent(
                event_type="completed",
                doc_id=doc_id,
                progress=100,
                total_chunks=len(chunks),
                timestamp=datetime.now().isoformat()
            ))

            processor.monitor.logger.info(f"✅ Document {doc_id} enrichment complete")

    except Exception as e:
        processor.monitor.logger.error(f"❌ Background enrichment failed for {doc_id}: {e}")
        import traceback
        processor.monitor.logger.error(traceback.format_exc())


@router.post("/api/documents")
async def create_document(
    doc: DocumentCreateRequest,
    processor=Depends(get_processor),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a new document with instant response and background enrichment"""
    try:
        # Estimate document size (rough chunk count)
        estimated_chunks = len([p for p in doc.content.split('\n\n') if p.strip()])
        
        # For small documents (< 10 chunks), enrich synchronously for instant visualization
        # For larger documents, use background enrichment
        if estimated_chunks <= 10 and len(doc.content) < 5000:
            processor.monitor.logger.info(f"📄 Small document detected ({estimated_chunks} chunks), enriching synchronously")
            # Use full processing pipeline for small docs
            doc_id = processor.process_document(doc.title, doc.content, doc.doc_type)
            return {
                "id": doc_id,
                "status": "enriched",
                "message": "Document created and enriched"
            }
        else:
            # Phase 1: Instant document creation (< 100ms)
            doc_id = processor.process_document_instant(doc.title, doc.content, doc.doc_type)

            # Phase 2: Queue background enrichment
            background_tasks.add_task(
                enrich_document_background,
                processor, doc_id, doc.content, doc.doc_type
            )

            return {
                "id": doc_id, 
                "status": "processing",
                "message": "Document created, enrichment in progress"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")


@router.get("/api/documents/{doc_id}/status")
async def get_document_status(doc_id: str, db=Depends(get_db)):
    """Get document enrichment status"""
    try:
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check chunk enrichment status
        chunks = db.get_chunks_by_document(doc_id)
        enriched_chunks = sum(1 for c in chunks if c.embedding and len(c.embedding) > 0)
        total_chunks = len(chunks)
        
        status = doc.metadata.get("status", "unknown")
        is_enriched = status == "enriched" and enriched_chunks == total_chunks
        
        return {
            "document_id": doc_id,
            "status": status,
            "is_enriched": is_enriched,
            "chunks": {
                "total": total_chunks,
                "enriched": enriched_chunks,
                "pending": total_chunks - enriched_chunks
            },
            "enriched_at": doc.metadata.get("enriched_at")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")


@router.get("/api/documents")
async def get_documents(db=Depends(get_db)):
    """Get all documents"""
    documents = db.get_all_documents()
    response = []
    for doc in documents:
        chunks = db.get_chunks_by_document(doc.id)
        response.append(DocumentResponse(
            id=doc.id,
            title=doc.title,
            doc_type=doc.doc_type,
            created_at=doc.created_at.isoformat() if isinstance(doc.created_at, datetime) else str(doc.created_at),
            chunk_count=len(chunks)
        ))
    return response


@router.get("/api/documents/{doc_id}")
async def get_document(doc_id: str, db=Depends(get_db)):
    """Get a specific document with chunks and validation"""
    try:
        # Validate document ID
        if not doc_id or not doc_id.strip():
            raise HTTPException(status_code=400, detail="Invalid document ID")

        # Get document
        doc = db.get_document(doc_id.strip())
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get chunks and connections
        chunks = db.get_chunks_by_document(doc_id.strip())
        chunk_ids = [c.id for c in chunks]
        connections = db.get_connections(chunk_ids)

        return {
            "document": doc.model_dump(),
            "chunks": [c.model_dump() for c in chunks],
            "connections": [c.model_dump() for c in connections]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


@router.get("/api/visualization")
async def get_visualization_data(
    db=Depends(get_db),
    processor=Depends(get_processor)
):
    """Get all data for 3D visualization"""
    documents = db.get_all_documents()
    all_chunks = []
    all_connections = []

    doc_responses = []
    for doc in documents:
        chunks = db.get_chunks_by_document(doc.id)
        chunk_ids = [c.id for c in chunks]
        connections = db.get_connections(chunk_ids)

        doc_responses.append(DocumentResponse(
            id=doc.id,
            title=doc.title,
            doc_type=doc.doc_type,
            created_at=doc.created_at.isoformat(),
            chunk_count=len(chunks)
        ))

        for chunk in chunks:
            all_chunks.append(ChunkResponse(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content[:200],  # Truncate for performance
                chunk_type=chunk.chunk_type,
                position_3d=chunk.position_3d,
                color=chunk.color,
                tags=chunk.tags,
                reasoning=chunk.reasoning,
                shape_3d=processor._get_shape_from_tags(chunk.tags),
                embedding=chunk.embedding,  # Include embedding for procedural geometry
                metadata=chunk.metadata
            ))

        all_connections.extend([c.model_dump() for c in connections])

    return VisualizationData(
        documents=doc_responses,
        chunks=all_chunks,
        connections=all_connections
    )


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, db=Depends(get_db)):
    """Delete a document with validation"""
    try:
        # Validate document ID
        if not doc_id or not doc_id.strip():
            raise HTTPException(status_code=400, detail="Invalid document ID")

        # Check if document exists
        try:
            doc = db.get_document(doc_id.strip())
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
        except Exception:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete the document
        db.delete_document(doc_id.strip())
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    processor=Depends(get_processor)
):
    """Upload a file for processing with security validation"""
    try:
        # Security validations
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check file size (10MB limit)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")

        # Check file extension
        allowed_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.log'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Decode content safely
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File contains invalid UTF-8 characters")

        # Determine file type based on extension
        doc_type = "text"
        if file.filename.endswith(('.py', '.js', '.rs', '.cpp', '.java')):
            doc_type = "code"

        doc_id = processor.process_document(
            title=file.filename,
            content=content_str,
            doc_type=doc_type
        )

        return {"id": doc_id, "message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chunks/{chunk_id}/attach")
async def attach_document_to_chunk(
    chunk_id: str,
    request: ChunkAttachmentRequest,
    db=Depends(get_db)
):
    """Attach a document to a chunk with validation"""
    try:
        # Validate chunk_id
        if not chunk_id or not chunk_id.strip():
            raise HTTPException(status_code=400, detail="Invalid chunk ID")

        # Verify chunk exists (basic validation)
        if len(chunk_id.strip()) < 5:  # Basic length check
            raise HTTPException(status_code=400, detail="Invalid chunk ID format")

        db.add_attachment(chunk_id.strip(), request.document_id)
        return {"message": "Document attached successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to attach document: {str(e)}")


@router.delete("/api/chunks/{chunk_id}/attach/{document_id}")
async def remove_attachment_from_chunk(
    chunk_id: str,
    document_id: str,
    db=Depends(get_db)
):
    """Remove a document attachment from a chunk"""
    try:
        db.remove_attachment(chunk_id, document_id)
        return {"message": "Attachment removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/chunks/{chunk_id}/attachments")
async def get_chunk_attachments(chunk_id: str, db=Depends(get_db)):
    """Get all documents attached to a chunk"""
    try:
        attachments = db.get_chunk_attachments(chunk_id)
        return [doc.model_dump() for doc in attachments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/enrichment/{doc_id}")
async def websocket_enrichment_stream(websocket: WebSocket, doc_id: str):
    """
    WebSocket endpoint for real-time enrichment streaming
    Sends embedding and shape data as chunks are enriched
    """
    await websocket.accept()
    queue = await enrichment_event_bus.subscribe(doc_id)

    try:
        while True:
            # Get event from queue
            event = await queue.get()
            # Send to client
            await websocket.send_json(event.to_json())
    except WebSocketDisconnect:
        await enrichment_event_bus.unsubscribe(doc_id, queue)
    except Exception as e:
        print(f"WebSocket error for {doc_id}: {e}")
        await enrichment_event_bus.unsubscribe(doc_id, queue)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates (legacy, kept for compatibility)"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo for now, can add real-time features
            await websocket.send_text(f"Echo: {data}")
    except:
        pass
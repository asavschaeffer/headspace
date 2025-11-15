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


async def enrich_document_background(processor, doc_id: str):
    """Background task to enrich document chunks with embeddings and tags"""
    try:
        processor.monitor.logger.info(f"🔄 Starting enrichment for document {doc_id}")
        existing_chunks = processor.db.get_chunks_by_document(doc_id)
        total_chunks = len(existing_chunks)

        await enrichment_event_bus.emit(EnrichmentEvent(
            event_type="started",
            doc_id=doc_id,
            total_chunks=total_chunks,
            progress=0,
            timestamp=datetime.now().isoformat()
        ))

        loop = asyncio.get_running_loop()

        def chunk_callback(chunk, index, total, stage):
            event_type = "chunk_enriched" if stage == "embedding" else "chunk_layout_updated"
            progress = int(((index + 1) / total) * 100) if stage == "embedding" and total > 0 else 100
            event = EnrichmentEvent(
                event_type=event_type,
                doc_id=doc_id,
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                embedding=chunk.embedding,
                color=chunk.color,
                position_3d=chunk.position_3d,
                umap_coordinates=chunk.umap_coordinates,
                cluster_id=chunk.cluster_id,
                cluster_confidence=chunk.cluster_confidence,
                cluster_label=chunk.cluster_label,
                nearest_chunk_ids=chunk.nearest_chunk_ids,
                progress=progress,
                total_chunks=total,
                timestamp=datetime.now().isoformat()
            )
            loop.create_task(enrichment_event_bus.emit(event))

        processor.enrich_document(doc_id, chunk_callback=chunk_callback)

        await enrichment_event_bus.emit(EnrichmentEvent(
            event_type="completed",
            doc_id=doc_id,
            progress=100,
            total_chunks=total_chunks,
            timestamp=datetime.now().isoformat()
        ))
        processor.monitor.logger.info(f"✅ Document {doc_id} enrichment complete")

    except Exception as e:
        processor.monitor.logger.error(f"❌ Background enrichment failed for {doc_id}: {e}")
        import traceback
        processor.monitor.logger.error(traceback.format_exc())
        await enrichment_event_bus.emit(EnrichmentEvent(
            event_type="error",
            doc_id=doc_id,
            error=str(e),
            timestamp=datetime.now().isoformat()
        ))
        doc = processor.db.get_document(doc_id)
        if doc:
            doc.metadata["status"] = "enrichment_failed"
            doc.metadata["error"] = str(e)
            processor.db.save_document(doc)


@router.post("/api/documents")
async def create_document(
    doc: DocumentCreateRequest,
    processor=Depends(get_processor),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create a new document and schedule enrichment in the background"""
    try:
        processor.monitor.logger.info(f"📄 Creating document: {doc.title}")
        processor.monitor.logger.debug(f"Content size: {len(doc.content)} chars, type: {doc.doc_type}")

        doc_id, placeholders = processor.create_document_placeholders(doc.title, doc.content, doc.doc_type)
        total_chunks = len(placeholders)
        processor.monitor.logger.info(f"🌱 Document {doc_id} queued for enrichment ({total_chunks} chunks)")

        background_tasks.add_task(enrich_document_background, processor, doc_id)

        return {
            "id": doc_id,
            "status": "processing",
            "chunks": total_chunks,
            "message": "Document accepted. Enrichment in progress."
        }
    except Exception as e:
        processor.monitor.logger.error(f"❌ Failed to create document: {e}", exc_info=True)
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
                metadata=chunk.metadata,
                cluster_id=chunk.cluster_id,
                cluster_confidence=chunk.cluster_confidence,
                cluster_label=chunk.cluster_label,
                umap_coordinates=chunk.umap_coordinates or chunk.position_3d,
                nearest_chunk_ids=chunk.nearest_chunk_ids
            ))

        all_connections.extend([c.model_dump() for c in connections])

    return VisualizationData(
        documents=doc_responses,
        chunks=all_chunks,
        connections=all_connections
    )


@router.get("/api/clusters")
async def get_clusters(db=Depends(get_db)):
    """Return aggregate cluster metadata for visualization legends."""
    try:
        cluster_records = db.get_cluster_metadata()
        response = []
        for record in cluster_records:
            cluster_id = record.get("cluster_id")
            response.append({
                "cluster_id": cluster_id,
                "label": record.get("label"),
                "size": record.get("chunk_count", record.get("size", 0)),
                "color": record.get("color") or ("#E8E8E8" if cluster_id is None else None),
                "updated_at": record.get("updated_at")
            })
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load cluster metadata: {exc}")


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
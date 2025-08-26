"""
Web Frontend for Globule using FastAPI.

This is a basic web interface that mirrors the TUI and CLI functionality
in a browser-accessible format. Currently a prototype/placeholder.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, Request, Form
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from globule.core.api import GlobuleAPI
from globule.core.layout_engine import LayoutEngine, CanvasModule
from globule.storage.sqlite_manager import SQLiteStorageManager

logger = logging.getLogger(__name__)


def create_app() -> "FastAPI":
    """
    Create and configure the FastAPI web application.
    
    Returns:
        FastAPI app instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn jinja2")
    
    app = FastAPI(
        title="Globule Web Interface",
        description="Web frontend for Globule thought management system",
        version="1.0.0"
    )
    
    # Initialize core components
    storage_manager = None
    api = None
    
    @app.on_event("startup")
    async def startup_event():
        nonlocal storage_manager, api
        try:
            # Initialize storage (using default path)
            storage_manager = SQLiteStorageManager()
            api = GlobuleAPI(storage_manager)
            logger.info("Web app initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web app: {e}")
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Main page with search and draft functionality."""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Globule Web Interface</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .header {
                    background: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                
                /* Layout Engine Grid System */
                .canvas-grid {
                    display: grid;
                    grid-template: 1fr 2fr 1fr / 1fr 2fr 1fr;
                    gap: 20px;
                    min-height: 600px;
                }
                
                /* Position classes matching layout engine */
                .position-top-left { grid-area: 1 / 1; }
                .position-top-center { grid-area: 1 / 2; }
                .position-top-right { grid-area: 1 / 3; }
                .position-center-left { grid-area: 2 / 1; }
                .position-center { grid-area: 2 / 2; }
                .position-center-right { grid-area: 2 / 3; }
                .position-bottom-left { grid-area: 3 / 1; }
                .position-bottom-center { grid-area: 3 / 2; }
                .position-bottom-right { grid-area: 3 / 3; }
                .position-full-width { grid-area: 2 / 1 / 2 / 4; }
                
                /* Schema-specific styling */
                .canvas-module {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid #3498db;
                }
                
                .canvas-module.valet {
                    border-left-color: #27ae60;
                    background: #f8fff8;
                }
                
                .canvas-module.academic {
                    border-left-color: #3498db;
                    background: #f8fbff;
                }
                
                .canvas-module.technical {
                    border-left-color: #f39c12;
                    background: #fffbf0;
                }
                
                /* Size classes */
                .size-small { max-height: 200px; overflow-y: auto; }
                .size-medium { max-height: 400px; overflow-y: auto; }
                .size-large { max-height: 600px; overflow-y: auto; }
                .size-auto { height: auto; }
                .size-full { height: 100%; }
                
                /* Control panels */
                .control-panel {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .search-box, .draft-box {
                    margin-bottom: 15px;
                }
                input, textarea, button {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 14px;
                    margin-bottom: 10px;
                }
                button {
                    background: #3498db;
                    color: white;
                    border: none;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                button:hover {
                    background: #2980b9;
                }
                .results {
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 4px;
                    min-height: 200px;
                    max-height: 400px;
                    overflow-y: auto;
                    font-family: monospace;
                    white-space: pre-wrap;
                }
                .status {
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
                .success { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
                .info { background: #d1ecf1; color: #0c5460; }
                
                .module-header {
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 10px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 5px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌐 Globule Web Interface</h1>
                <p>Layout-aware canvas with schema-based positioning</p>
            </div>
            
            <div class="canvas-grid" id="canvasGrid">
                <!-- Control panels positioned using layout engine -->
                <div class="control-panel position-top-left">
                    <h3>🔍 Search</h3>
                    <div class="search-box">
                        <input type="text" id="searchQuery" placeholder="Search globules..." />
                        <button onclick="performSearch()">Search</button>
                    </div>
                    <div id="searchResults" class="results size-small">Search results...</div>
                </div>
                
                <div class="control-panel position-top-right">
                    <h3>📝 Draft</h3>
                    <div class="draft-box">
                        <textarea id="draftContent" rows="3" placeholder="Add content..."></textarea>
                        <button onclick="addToDraft()">Add to Draft</button>
                        <button onclick="getDraftStats()">Stats</button>
                    </div>
                    <div id="draftResults" class="results size-small">Draft operations...</div>
                </div>
                
                <!-- Main canvas area -->
                <div class="control-panel position-center" id="mainCanvas">
                    <h3>📋 Canvas</h3>
                    <p>Schema-positioned modules will appear throughout the grid based on their configuration.</p>
                    <button onclick="clearCanvas()">Clear Canvas</button>
                    <button onclick="showLayoutInfo()">Show Layout Info</button>
                    <button onclick="saveCurrentSkeleton()">Save as Template</button>
                    <button onclick="loadSkeletonDialog()">Load Template</button>
                </div>
                
                <!-- Dynamic module areas - will be populated by layout engine -->
                <div class="position-top-center" id="topCenter"></div>
                <div class="position-center-left" id="centerLeft"></div>
                <div class="position-center-right" id="centerRight"></div>
                <div class="position-bottom-left" id="bottomLeft"></div>
                <div class="position-bottom-center" id="bottomCenter"></div>
                <div class="position-bottom-right" id="bottomRight"></div>
            </div>
            
            <div class="control-panel" style="margin-top: 20px;">
                <h3>ℹ️ Layout Engine Features</h3>
                <p><strong>Schema-Based Positioning:</strong> Modules are positioned according to their schema's canvas_config</p>
                <p><strong>Valet modules:</strong> Positioned at top-left (green styling)</p>
                <p><strong>Academic modules:</strong> Blue styling with flexible positioning</p>
                <p><strong>Technical modules:</strong> Orange styling for technical content</p>
                <p><strong>Consistent with TUI:</strong> Same layout rules as terminal interface</p>
            </div>
            
            <script>
                async function performSearch() {
                    const query = document.getElementById('searchQuery').value;
                    const resultsDiv = document.getElementById('searchResults');
                    
                    if (!query.trim()) {
                        resultsDiv.innerHTML = '<div class="error">Please enter a search query</div>';
                        return;
                    }
                    
                    resultsDiv.innerHTML = '<div class="info">Searching...</div>';
                    
                    try {
                        const response = await fetch('/api/search_positioned', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: query })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultsDiv.innerHTML = '<div class="success">Search completed - results positioned on canvas</div>';
                            
                            // Create positioned module on canvas
                            if (result.layout_info) {
                                createPositionedModule(result.layout_info, result.data, query);
                            }
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Search failed: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                function createPositionedModule(layoutInfo, content, query) {
                    const position = layoutInfo.position || 'center-left';
                    const schemaName = layoutInfo.schema_name || 'default';
                    const size = layoutInfo.size || 'medium';
                    
                    // Find the target container
                    const targetId = position.replace('-', '').replace('left', 'Left').replace('right', 'Right').replace('center', 'Center').replace('top', 'top').replace('bottom', 'bottom');
                    let targetContainer = document.getElementById(targetId) || document.getElementById('centerLeft');
                    
                    // Create module element
                    const moduleElement = document.createElement('div');
                    moduleElement.className = `canvas-module ${schemaName} size-${size}`;
                    moduleElement.innerHTML = `
                        <div class="module-header">${query} (${schemaName})</div>
                        <div class="module-content">
                            <pre>${content}</pre>
                        </div>
                        <button onclick="this.parentElement.remove()" style="position: absolute; top: 5px; right: 5px; background: #e74c3c; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer;">×</button>
                    `;
                    moduleElement.style.position = 'relative';
                    
                    // Add to target container
                    targetContainer.appendChild(moduleElement);
                }
                
                async function addToDraft() {
                    const content = document.getElementById('draftContent').value;
                    const resultsDiv = document.getElementById('draftResults');
                    
                    if (!content.trim()) {
                        resultsDiv.innerHTML = '<div class="error">Please enter content to add</div>';
                        return;
                    }
                    
                    resultsDiv.innerHTML = '<div class="info">Adding to draft...</div>';
                    
                    try {
                        const response = await fetch('/api/draft/add', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ content: content })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultsDiv.innerHTML = `<div class="success">Added to draft successfully</div>`;
                            document.getElementById('draftContent').value = '';
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Failed to add: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                async function getDraftStats() {
                    const resultsDiv = document.getElementById('draftResults');
                    resultsDiv.innerHTML = '<div class="info">Getting draft stats...</div>';
                    
                    try {
                        const response = await fetch('/api/draft/stats');
                        const result = await response.json();
                        
                        if (result.success) {
                            const stats = result.data;
                            resultsDiv.innerHTML = `
                                <div class="success">Draft Statistics</div>
                                <pre>Path: ${stats.path}
Size: ${stats.size_bytes} bytes
Lines: ${stats.line_count}
Words: ${stats.word_count}
Modified: ${stats.last_modified}</pre>`;
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Failed to get stats: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                async function exportDraft() {
                    const resultsDiv = document.getElementById('draftResults');
                    resultsDiv.innerHTML = '<div class="info">Exporting draft...</div>';
                    
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    const filename = `draft_export_${timestamp}.md`;
                    
                    try {
                        const response = await fetch('/api/draft/export', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ output_path: filename })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            resultsDiv.innerHTML = `<div class="success">Exported to: ${result.data.export_path}</div>`;
                        } else {
                            resultsDiv.innerHTML = `<div class="error">Export failed: ${result.message}</div>`;
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                    }
                }
                
                function clearCanvas() {
                    const positions = ['topCenter', 'centerLeft', 'centerRight', 'bottomLeft', 'bottomCenter', 'bottomRight'];
                    positions.forEach(pos => {
                        const container = document.getElementById(pos);
                        if (container) container.innerHTML = '';
                    });
                    document.getElementById('mainCanvas').querySelector('p').textContent = 'Canvas cleared. Schema-positioned modules will appear here.';
                }
                
                function showLayoutInfo() {
                    const infoText = `Layout Engine Information:
- Grid: 3x3 with dynamic positioning  
- Schemas: valet (top-left), academic (flexible), technical (flexible)
- Positions: top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right
- Sizes: small, medium, large, auto, full
- Styling: Schema-specific colors and borders`;
                    alert(infoText);
                }
                
                async function saveCurrentSkeleton() {
                    const name = prompt('Enter template name:');
                    if (!name) return;
                    
                    const description = prompt('Enter template description:') || 'User-created template';
                    const tags = prompt('Enter tags (comma-separated):') || '';
                    
                    try {
                        const response = await fetch('/api/skeleton/save', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ 
                                name: name,
                                description: description,
                                tags: tags.split(',').map(t => t.trim()).filter(t => t)
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            alert(`Template '${name}' saved successfully!`);
                        } else {
                            alert(`Failed to save template: ${result.message}`);
                        }
                    } catch (error) {
                        alert(`Error saving template: ${error.message}`);
                    }
                }
                
                async function loadSkeletonDialog() {
                    try {
                        // Get list of available skeletons
                        const response = await fetch('/api/skeleton/list');
                        const result = await response.json();
                        
                        if (!result.success || !result.data.length) {
                            alert('No templates available');
                            return;
                        }
                        
                        // Create selection dialog
                        let options = 'Available Templates:\\n\\n';
                        result.data.forEach((skeleton, index) => {
                            options += `${index + 1}. ${skeleton.name}\\n   ${skeleton.description}\\n   Modules: ${skeleton.module_count}, Usage: ${skeleton.usage_count}\\n\\n`;
                        });
                        
                        const selection = prompt(options + '\\nEnter template number to load:');
                        if (!selection) return;
                        
                        const index = parseInt(selection) - 1;
                        if (index >= 0 && index < result.data.length) {
                            const skeleton = result.data[index];
                            await loadSkeleton(skeleton.name);
                        } else {
                            alert('Invalid selection');
                        }
                        
                    } catch (error) {
                        alert(`Error loading templates: ${error.message}`);
                    }
                }
                
                async function loadSkeleton(skeletonName) {
                    try {
                        const response = await fetch('/api/skeleton/apply', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ 
                                skeleton_name: skeletonName,
                                query_data: {
                                    query: 'Template Applied',
                                    content: 'Template content will appear here',
                                    timestamp: new Date().toISOString()
                                }
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            // Clear existing canvas
                            clearCanvas();
                            
                            // Apply skeleton modules to canvas
                            if (result.modules && result.modules.length > 0) {
                                result.modules.forEach(module => {
                                    createPositionedModule({
                                        position: module.position,
                                        schema_name: module.schema_name,
                                        size: module.size
                                    }, module.content, module.title);
                                });
                            }
                            
                            alert(`Template '${skeletonName}' applied successfully!`);
                        } else {
                            alert(`Failed to apply template: ${result.message}`);
                        }
                        
                    } catch (error) {
                        alert(`Error applying template: ${error.message}`);
                    }
                }
                
                // Allow Enter key in search box
                document.getElementById('searchQuery').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        performSearch();
                    }
                });
            </script>
        </body>
        </html>
        """
        return html_content
    
    @app.post("/api/search")
    async def api_search(request: Request):
        """API endpoint for search functionality."""
        try:
            data = await request.json()
            query = data.get('query', '')
            
            if not query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.search(query)
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Search failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.post("/api/search_positioned")
    async def api_search_positioned(request: Request):
        """API endpoint for search with layout engine positioning."""
        try:
            data = await request.json()
            query = data.get('query', '')
            
            if not query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            # Perform regular search
            result = await api.search(query)
            
            if not result['success']:
                return JSONResponse(result)
            
            # Determine schema from query content
            schema_name = 'default'
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ['valet', 'parking', 'car', 'vehicle']):
                schema_name = 'valet'
            elif any(keyword in query_lower for keyword in ['academic', 'research', 'paper', 'study']):
                schema_name = 'academic' 
            elif any(keyword in query_lower for keyword in ['technical', 'code', 'programming', 'software']):
                schema_name = 'technical'
            
            # Get layout configuration from schema
            layout_engine = LayoutEngine()
            layout_config = layout_engine.get_layout_config(schema_name)
            
            # Prepare layout information for frontend
            layout_info = {
                'schema_name': schema_name,
                'position': 'center-left',  # default
                'size': 'medium',  # default
                'css_classes': []
            }
            
            if layout_config:
                layout_info.update({
                    'position': layout_config.position.value,
                    'size': layout_config.size.value,
                    'css_classes': layout_config.css_classes
                })
            
            # Add layout information to result
            result['layout_info'] = layout_info
            
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Positioned search API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Positioned search failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.post("/api/draft/add")
    async def api_draft_add(request: Request):
        """API endpoint for adding content to draft."""
        try:
            data = await request.json()
            content = data.get('content', '')
            
            if not content:
                raise HTTPException(status_code=400, detail="Content is required")
                
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.add_to_draft(content)
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Draft add API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Add to draft failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.get("/api/draft/stats")
    async def api_draft_stats():
        """API endpoint for getting draft statistics."""
        try:
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.get_draft_stats()
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Draft stats API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Get draft stats failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.post("/api/draft/export")
    async def api_draft_export(request: Request):
        """API endpoint for exporting draft."""
        try:
            data = await request.json()
            output_path = data.get('output_path', f'draft_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
            
            if api is None:
                raise HTTPException(status_code=500, detail="API not initialized")
                
            result = await api.export_draft(output_path)
            return JSONResponse(result)
            
        except Exception as e:
            logger.error(f"Draft export API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Export draft failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.post("/api/skeleton/save")
    async def api_skeleton_save(request: Request):
        """API endpoint for saving current canvas as skeleton."""
        try:
            data = await request.json()
            name = data.get('name', '')
            description = data.get('description', '')
            tags = data.get('tags', [])
            
            if not name:
                raise HTTPException(status_code=400, detail="Skeleton name is required")
            
            # For web interface, we'll create a simple placeholder skeleton
            # In a real implementation, we'd track the current canvas state
            layout_engine = LayoutEngine()
            
            # Create a basic skeleton with sample data
            # This would be replaced with actual canvas modules in production
            sample_modules = [
                layout_engine._create_sample_module("web_sample", "center", "default", "medium")
            ]
            
            success = layout_engine.save_canvas_as_skeleton(
                modules=sample_modules,
                name=name,
                description=description,
                author="web_user",
                tags=tags
            )
            
            if success:
                return JSONResponse({
                    "success": True,
                    "message": f"Skeleton '{name}' saved successfully",
                    "data": {"name": name}
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": "Failed to save skeleton",
                    "data": None
                })
                
        except Exception as e:
            logger.error(f"Save skeleton API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Save skeleton failed: {str(e)}",
                "data": None
            }, status_code=500)
    
    @app.get("/api/skeleton/list")
    async def api_skeleton_list():
        """API endpoint for listing available skeletons."""
        try:
            layout_engine = LayoutEngine()
            skeletons = layout_engine.list_skeletons()
            
            skeleton_data = []
            for skeleton in skeletons:
                skeleton_data.append({
                    "name": skeleton.name,
                    "description": skeleton.description,
                    "author": skeleton.author,
                    "created_at": skeleton.created_at.isoformat(),
                    "tags": skeleton.tags,
                    "module_count": len(skeleton.module_placeholders),
                    "usage_count": skeleton.usage_count,
                    "schemas": skeleton.primary_schemas
                })
            
            return JSONResponse({
                "success": True,
                "message": "Skeletons retrieved successfully",
                "data": skeleton_data
            })
            
        except Exception as e:
            logger.error(f"List skeletons API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"List skeletons failed: {str(e)}",
                "data": []
            }, status_code=500)
    
    @app.post("/api/skeleton/apply")
    async def api_skeleton_apply(request: Request):
        """API endpoint for applying skeleton to canvas."""
        try:
            data = await request.json()
            skeleton_name = data.get('skeleton_name', '')
            query_data = data.get('query_data', {})
            
            if not skeleton_name:
                raise HTTPException(status_code=400, detail="Skeleton name is required")
            
            layout_engine = LayoutEngine()
            
            # Apply skeleton to get module configurations
            modules = layout_engine.apply_skeleton_to_canvas(skeleton_name, query_data)
            
            if modules:
                # Convert modules to JSON-serializable format
                module_data = []
                for module in modules:
                    module_data.append({
                        "id": module.id,
                        "title": module.name,
                        "content": module.content,
                        "position": module.layout.position.value,
                        "schema_name": module.layout.schema_name,
                        "size": module.layout.size.value,
                        "layout_type": module.layout.layout_type.value
                    })
                
                return JSONResponse({
                    "success": True,
                    "message": f"Skeleton '{skeleton_name}' applied successfully",
                    "modules": module_data
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": f"Failed to apply skeleton '{skeleton_name}'",
                    "modules": []
                })
                
        except Exception as e:
            logger.error(f"Apply skeleton API error: {e}")
            return JSONResponse({
                "success": False,
                "message": f"Apply skeleton failed: {str(e)}",
                "modules": []
            }, status_code=500)
    
    return app


# For development/testing
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
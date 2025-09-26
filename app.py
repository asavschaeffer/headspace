#!/usr/bin/env python3
"""
Production WSGI application for Brain server
Compatible with Gunicorn for Render deployment
"""

import os
import json
import urllib.parse
from pathlib import Path
import mimetypes
import base64
from wsgiref.simple_server import make_server
from brain_core import Brain

# Global brain instance
brain = Brain()

def application(environ, start_response):
    """WSGI application entry point"""

    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')

    # Enable CORS
    headers = [
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
        ('Access-Control-Allow-Headers', 'Content-Type')
    ]

    # Handle OPTIONS for CORS preflight
    if method == 'OPTIONS':
        start_response('200 OK', headers)
        return [b'']

    # Route handling
    try:
        if method == 'GET':
            if path == '/' or path == '/index.html':
                return serve_interface(environ, start_response, headers)
            elif path == '/api/graph':
                return serve_graph_data(environ, start_response, headers)
            elif path == '/api/search':
                return serve_search(environ, start_response, headers)
            elif path.startswith('/api/file/'):
                return serve_file(environ, start_response, headers, path[10:])
            elif path == '/api/recent':
                return serve_recent(environ, start_response, headers)
            else:
                start_response('404 Not Found', headers)
                return [b'Not Found']

        elif method == 'POST':
            if path == '/api/save':
                return handle_save(environ, start_response, headers)
            else:
                start_response('404 Not Found', headers)
                return [b'Not Found']
        else:
            start_response('405 Method Not Allowed', headers)
            return [b'Method Not Allowed']

    except Exception as e:
        start_response('500 Internal Server Error', headers)
        return [json.dumps({'error': str(e)}).encode()]

def serve_interface(environ, start_response, headers):
    """Serve the main HTML interface"""
    html_path = Path(__file__).parent / 'brain_interface.html'

    if html_path.exists():
        with open(html_path, 'rb') as f:
            content = f.read()
    else:
        content = get_inline_html().encode()

    headers.append(('Content-Type', 'text/html'))
    headers.append(('Content-Length', str(len(content))))
    start_response('200 OK', headers)
    return [content]

def serve_graph_data(environ, start_response, headers):
    """Serve graph data as JSON"""
    graph_data = brain.get_graph_data()
    response = json.dumps(graph_data).encode()

    headers.append(('Content-Type', 'application/json'))
    start_response('200 OK', headers)
    return [response]

def serve_search(environ, start_response, headers):
    """Handle search requests"""
    query_string = environ.get('QUERY_STRING', '')
    query_params = urllib.parse.parse_qs(query_string)

    q = query_params.get('q', [''])[0]
    search_type = query_params.get('type', ['all'])[0]

    results = []

    if search_type in ['all', 'filename']:
        for file in brain.search_filename(q):
            results.append({
                'type': 'filename',
                'path': file.name,
                'match': file.name,
                'modified': file.stat().st_mtime
            })

    if search_type in ['all', 'content']:
        for file, snippet in brain.search_content(q):
            results.append({
                'type': 'content',
                'path': file.name,
                'match': snippet,
                'modified': file.stat().st_mtime
            })

    response = json.dumps({'query': q, 'results': results}).encode()

    headers.append(('Content-Type', 'application/json'))
    start_response('200 OK', headers)
    return [response]

def serve_file(environ, start_response, headers, filename):
    """Serve specific file content"""
    filepath = brain.base / filename

    if not filepath.exists() or not filepath.is_file():
        start_response('404 Not Found', headers)
        return [b'File not found']

    # Limit file size for web viewing
    if filepath.stat().st_size > 1024 * 1024:  # 1MB limit
        content = filepath.read_text(errors='ignore')[:1024*1024]
        content += "\n\n[File truncated for web viewing]"
    else:
        content = filepath.read_text(errors='ignore')

    response = json.dumps({
        'filename': filename,
        'content': content,
        'size': filepath.stat().st_size,
        'modified': filepath.stat().st_mtime
    }).encode()

    headers.append(('Content-Type', 'application/json'))
    start_response('200 OK', headers)
    return [response]

def serve_recent(environ, start_response, headers):
    """Get recently modified files"""
    recent = brain.search_by_time(hours_ago=24)

    results = []
    for file in recent[:50]:  # Limit to 50 most recent
        results.append({
            'path': file.name,
            'modified': file.stat().st_mtime,
            'size': file.stat().st_size
        })

    response = json.dumps({'files': results}).encode()

    headers.append(('Content-Type', 'application/json'))
    start_response('200 OK', headers)
    return [response]

def handle_save(environ, start_response, headers):
    """Handle saving new content"""
    try:
        content_length = int(environ.get('CONTENT_LENGTH', 0))
        post_data = environ['wsgi.input'].read(content_length)

        data = json.loads(post_data.decode())
        content = data.get('content', '')
        filename = data.get('filename', None)

        # Check if it's binary data (base64 encoded)
        if data.get('binary'):
            binary_data = base64.b64decode(data.get('data', ''))
            filepath = brain.save_binary(binary_data, filename or 'upload.bin')
            was_appended = False
        else:
            filepath, was_appended = brain.save_text(content, filename)

        response = json.dumps({
            'success': True,
            'filepath': filepath.name,
            'appended': was_appended,
            'message': f"{'Appended to' if was_appended else 'Saved as'}: {filepath.name}"
        }).encode()

        headers.append(('Content-Type', 'application/json'))
        start_response('200 OK', headers)
        return [response]

    except Exception as e:
        response = json.dumps({
            'success': False,
            'error': str(e)
        }).encode()

        headers.append(('Content-Type', 'application/json'))
        start_response('500 Internal Server Error', headers)
        return [response]

def get_inline_html():
    """Return minimal inline HTML for initial setup"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain - Living Memory Space</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background: #000;
            color: #0f0;
            font-family: 'Monaco', 'Courier New', monospace;
            overflow: hidden;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        #galaxy {
            flex: 1;
            position: relative;
            background: radial-gradient(circle at center, #001 0%, #000 100%);
        }

        canvas {
            width: 100%;
            height: 100%;
            display: block;
        }

        #input-container {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 600px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #0f0;
            border-radius: 4px;
            padding: 10px;
        }

        #brain-input {
            width: 100%;
            background: transparent;
            border: none;
            color: #0f0;
            font-family: inherit;
            font-size: 16px;
            outline: none;
            resize: vertical;
            min-height: 30px;
        }

        #status {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 12px;
            color: #0f0;
            opacity: 0.7;
        }

        .pulse {
            animation: pulse 0.5s ease-out;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
            100% { opacity: 1; transform: scale(1); }
        }
    </style>
</head>
<body>
    <div id="galaxy">
        <canvas id="canvas"></canvas>
        <div id="status">Initializing brain...</div>
        <div id="input-container">
            <textarea id="brain-input"
                      placeholder="Type anything... press Ctrl+Enter to save, type '/' to search"
                      rows="1"></textarea>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const input = document.getElementById('brain-input');
        const status = document.getElementById('status');

        // Auto-resize canvas
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Simple star field for now
        function drawStarField() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw some stars
            for (let i = 0; i < 100; i++) {
                const x = Math.random() * canvas.width;
                const y = Math.random() * canvas.height;
                const size = Math.random() * 2;
                const brightness = Math.random();

                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 255, 0, ${brightness})`;
                ctx.fill();
            }

            status.textContent = 'Brain ready. Start typing...';
        }

        drawStarField();

        // Handle input
        input.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                const content = input.value.trim();
                if (!content) return;

                try {
                    const response = await fetch('/api/save', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ content })
                    });

                    const result = await response.json();
                    if (result.success) {
                        status.textContent = result.message;
                        input.value = '';
                        input.classList.add('pulse');
                        setTimeout(() => input.classList.remove('pulse'), 500);
                    }
                } catch (error) {
                    status.textContent = 'Error: ' + error.message;
                }
            }
        });

        // Auto-resize textarea
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = input.scrollHeight + 'px';
        });
    </script>
</body>
</html>"""

# For local testing
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    with make_server('', port, application) as httpd:
        print(f'Serving on port {port}...')
        httpd.serve_forever()
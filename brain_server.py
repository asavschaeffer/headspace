#!/usr/bin/env python3
"""
Brain Server - Minimal web interface for the brain filesystem
Serves HTML interface and provides API endpoints
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from pathlib import Path
import mimetypes
import io
import base64
from brain_core import Brain

# Global brain instance
brain = Brain()

class BrainHandler(BaseHTTPRequestHandler):
    """HTTP request handler for brain operations"""

    def send_cors_headers(self):
        """Enable CORS for local development"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)

        # Route to appropriate handler
        if path == '/' or path == '/index.html':
            self.serve_interface()
        elif path == '/api/graph':
            self.serve_graph_data()
        elif path == '/api/search':
            self.serve_search(query)
        elif path.startswith('/api/file/'):
            self.serve_file(path[10:])  # Remove /api/file/ prefix
        elif path == '/api/recent':
            self.serve_recent()
        else:
            self.send_error(404)

    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/save':
            self.handle_save()
        else:
            self.send_error(404)

    def serve_interface(self):
        """Serve the main HTML interface"""
        html_path = Path(__file__).parent / 'brain_interface.html'

        if not html_path.exists():
            # Serve inline HTML if file doesn't exist yet
            html_content = self.get_inline_html()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(html_content.encode())
        else:
            # Serve from file
            with open(html_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(content)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content)

    def serve_graph_data(self):
        """Serve graph data as JSON"""
        graph_data = brain.get_graph_data()
        response = json.dumps(graph_data)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(response.encode())

    def serve_search(self, query_params):
        """Handle search requests"""
        q = query_params.get('q', [''])[0]
        search_type = query_params.get('type', ['all'])[0]

        results = []

        if search_type in ['all', 'filename']:
            # Search filenames
            for file in brain.search_filename(q):
                results.append({
                    'type': 'filename',
                    'path': file.name,
                    'match': file.name,
                    'modified': file.stat().st_mtime
                })

        if search_type in ['all', 'content']:
            # Search content
            for file, snippet in brain.search_content(q):
                results.append({
                    'type': 'content',
                    'path': file.name,
                    'match': snippet,
                    'modified': file.stat().st_mtime
                })

        response = json.dumps({'query': q, 'results': results})

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(response.encode())

    def serve_file(self, filename):
        """Serve specific file content"""
        filepath = brain.base / filename

        if not filepath.exists() or not filepath.is_file():
            self.send_error(404)
            return

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
        })

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(response.encode())

    def serve_recent(self):
        """Get recently modified files"""
        recent = brain.search_by_time(hours_ago=24)

        results = []
        for file in recent[:50]:  # Limit to 50 most recent
            results.append({
                'path': file.name,
                'modified': file.stat().st_mtime,
                'size': file.stat().st_size
            })

        response = json.dumps({'files': results})

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(response.encode())

    def handle_save(self):
        """Handle saving new content"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
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
            })

            self.send_response(200)
        except Exception as e:
            response = json.dumps({
                'success': False,
                'error': str(e)
            })
            self.send_response(500)

        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(response.encode())

    def get_inline_html(self):
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
        // Will be replaced with full implementation
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

def run_server(port=8888, host='localhost'):
    """Start the brain server"""
    server = HTTPServer((host, port), BrainHandler)
    print(f"""
╔══════════════════════════════════════════╗
║        BRAIN SERVER STARTED              ║
║                                          ║
║  Navigate to: http://{host}:{port:<8}║
║                                          ║
║  Shortcuts:                              ║
║  - Ctrl+Enter: Save thought              ║
║  - /search: Search mode                  ║
║  - Drag files: Import                    ║
║                                          ║
║  Brain location: {str(brain.base):<24}║
╚══════════════════════════════════════════╝
    """)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nBrain server stopped.")
        server.server_close()

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    port = 8888
    host = 'localhost'

    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    if len(sys.argv) > 2:
        host = sys.argv[2]

    run_server(port, host)
#!/usr/bin/env python3
"""
Brain Core - The living filesystem memory system
Handles save, collision-as-continuation, and metadata tracking
"""

import os
import re
import hashlib
import json
from pathlib import Path
from datetime import datetime
import subprocess
import mimetypes
from typing import Optional, Dict, List, Tuple

class Brain:
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the brain filesystem"""
        self.base = base_path or Path.home() / "brain"
        self.base.mkdir(exist_ok=True)
        self.metadata_dir = self.base / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # Track connections between files
        self.connections_file = self.metadata_dir / "connections.json"
        self.load_connections()

        # Track last accessed for connection building
        self.last_accessed = None

    def load_connections(self):
        """Load the connection graph from disk"""
        if self.connections_file.exists():
            with open(self.connections_file) as f:
                self.connections = json.load(f)
        else:
            self.connections = {}

    def save_connections(self):
        """Persist connection graph"""
        with open(self.connections_file, 'w') as f:
            json.dump(self.connections, f, indent=2)

    def generate_filename(self, content: str, extension: str = "txt") -> str:
        """
        Generate smart filename from content
        - First meaningful sentence (up to 50 chars)
        - Remove special chars, replace spaces with dashes
        - Detect special patterns (errors, people, code)
        """
        # Clean the content
        content = content.strip()
        if not content:
            return f"empty-{datetime.now():%Y%m%d-%H%M%S}.{extension}"

        # Try to extract semantic meaning
        filename = None

        # Pattern 1: Error or bug report
        if any(word in content.lower() for word in ['error', 'bug', 'crash', 'exception']):
            # Extract error type if possible
            error_match = re.search(r'(\w+Error|\w+Exception|bug|crash)', content, re.I)
            if error_match:
                filename = f"error-{error_match.group(1).lower()}"

        # Pattern 2: Meeting or person reference
        elif any(word in content.lower() for word in ['meeting', 'call', 'discussion']):
            # Extract person names (capitalized words)
            names = re.findall(r'\b[A-Z][a-z]+\b', content)[:2]  # Max 2 names
            if names:
                filename = f"meeting-{'-'.join(n.lower() for n in names)}"

        # Pattern 3: Code or function definition
        elif 'def ' in content or 'function ' in content or 'class ' in content:
            # Extract function/class name
            code_match = re.search(r'(def|function|class)\s+(\w+)', content)
            if code_match:
                filename = f"code-{code_match.group(2).lower()}"

        # Pattern 4: Question or help request
        elif content.startswith(('how', 'what', 'why', 'when', 'where', 'can')):
            filename = "question-" + content[:30]

        # Default: First sentence or line
        if not filename:
            # Get first sentence or line
            first_part = content.split('.')[0] if '.' in content else content.split('\n')[0]
            filename = first_part[:50]

        # Clean filename - remove special chars, spaces to dashes
        filename = re.sub(r'[^\w\s-]', '', filename.lower())
        filename = re.sub(r'[-\s]+', '-', filename).strip('-')

        # Ensure we have something
        if not filename:
            filename = f"thought-{datetime.now():%Y%m%d-%H%M%S}"

        return f"{filename}.{extension}"

    def calculate_content_hash(self, content: str) -> str:
        """Generate hash of content for duplicate detection"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def save_text(self, content: str, filename: Optional[str] = None) -> Tuple[Path, bool]:
        """
        Save text content to brain
        Returns (filepath, was_appended)

        Collision handling: Same filename = continuation of thought
        Appends with timestamp separator
        """
        if not filename:
            filename = self.generate_filename(content)

        filepath = self.base / filename
        was_appended = False

        # Check if file exists (collision)
        if filepath.exists():
            # This is a continuation of the same thought thread
            with open(filepath, 'a') as f:
                separator = f"\n\n{'='*50}\n[Continued {datetime.now():%Y-%m-%d %H:%M:%S}]\n{'='*50}\n\n"
                f.write(separator)
                f.write(content)
            was_appended = True
        else:
            # New file
            filepath.write_text(content)

        # Track metadata
        self.add_metadata(filepath, {
            'created': datetime.now().isoformat() if not was_appended else None,
            'appended': datetime.now().isoformat() if was_appended else None,
            'hash': self.calculate_content_hash(content),
            'size': len(content),
            'type': 'text'
        })

        # Build connections
        if self.last_accessed:
            self.add_connection(self.last_accessed, filepath, strength=0.5, type='temporal')
        self.last_accessed = filepath

        return filepath, was_appended

    def save_binary(self, data: bytes, filename: str, process: bool = True) -> Path:
        """
        Save binary data (images, audio, etc)
        Optionally process to extract searchable text
        """
        filepath = self.base / filename
        filepath.write_bytes(data)

        mime_type = mimetypes.guess_type(filename)[0]

        if process and mime_type:
            # Generate searchable text based on type
            text_content = None

            if mime_type.startswith('image/'):
                text_content = self.process_image(filepath)
            elif mime_type.startswith('audio/'):
                text_content = self.process_audio(filepath)
            elif filename.endswith('.pdf'):
                text_content = self.process_pdf(filepath)

            if text_content:
                # Save searchable text version
                text_filepath = filepath.with_suffix(filepath.suffix + '.txt')
                text_filepath.write_text(f"[Extracted from {filename}]\n\n{text_content}")

                # Track connection between original and text
                self.add_connection(filepath, text_filepath, strength=1.0, type='extraction')

        return filepath

    def process_image(self, filepath: Path) -> Optional[str]:
        """Extract text from image using OCR"""
        try:
            # Try tesseract if available
            result = subprocess.run(
                ['tesseract', str(filepath), '-'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: just note it's an image
        return f"[Image file: {filepath.name}]"

    def process_audio(self, filepath: Path) -> Optional[str]:
        """Transcribe audio using whisper or similar"""
        try:
            # Try whisper.cpp if available
            result = subprocess.run(
                ['whisper', str(filepath), '--text'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return f"[Audio file: {filepath.name}]"

    def process_pdf(self, filepath: Path) -> Optional[str]:
        """Extract text from PDF"""
        try:
            result = subprocess.run(
                ['pdftotext', str(filepath), '-'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return f"[PDF file: {filepath.name}]"

    def add_metadata(self, filepath: Path, metadata: Dict):
        """Store metadata for a file"""
        meta_file = self.metadata_dir / f"{filepath.name}.json"

        existing = {}
        if meta_file.exists():
            with open(meta_file) as f:
                existing = json.load(f)

        # Merge with existing metadata
        existing.update({k: v for k, v in metadata.items() if v is not None})

        with open(meta_file, 'w') as f:
            json.dump(existing, f, indent=2)

    def add_connection(self, file1: Path, file2: Path, strength: float = 1.0, type: str = 'reference'):
        """Track connection between two files"""
        key1 = str(file1.relative_to(self.base))
        key2 = str(file2.relative_to(self.base))

        if key1 not in self.connections:
            self.connections[key1] = {}

        if key2 not in self.connections[key1]:
            self.connections[key1][key2] = {
                'strength': 0,
                'types': []
            }

        # Update connection strength (max 1.0)
        self.connections[key1][key2]['strength'] = min(
            1.0,
            self.connections[key1][key2]['strength'] + strength * 0.5
        )

        # Track connection type
        if type not in self.connections[key1][key2]['types']:
            self.connections[key1][key2]['types'].append(type)

        self.save_connections()

    def search_filename(self, pattern: str) -> List[Path]:
        """Fast filename search using glob patterns"""
        results = []
        pattern_lower = pattern.lower()

        for file in self.base.glob("*"):
            if file.is_file() and not file.name.startswith('.'):
                if pattern_lower in file.name.lower():
                    results.append(file)

        # Sort by modification time (most recent first)
        results.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return results[:20]  # Limit results

    def search_content(self, query: str) -> List[Tuple[Path, str]]:
        """
        Search file contents using ripgrep if available, else grep
        Returns list of (filepath, matching_line)
        """
        results = []

        try:
            # Try ripgrep first (much faster)
            result = subprocess.run(
                ['rg', '--max-count', '3', '--no-heading', query, str(self.base)],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        filepath = Path(parts[0])
                        match_text = parts[1] if len(parts) > 1 else ""
                        results.append((filepath, match_text.strip()))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback to grep
            try:
                result = subprocess.run(
                    ['grep', '-r', '-l', query, str(self.base)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    for filename in result.stdout.strip().split('\n'):
                        if filename:
                            filepath = Path(filename)
                            # Get a snippet of the match
                            content = filepath.read_text()
                            idx = content.lower().find(query.lower())
                            if idx >= 0:
                                snippet = content[max(0, idx-30):idx+len(query)+30]
                                results.append((filepath, f"...{snippet}..."))
            except:
                pass

        return results[:20]  # Limit results

    def search_by_time(self, hours_ago: int = 24) -> List[Path]:
        """Find files modified within the last N hours"""
        import time
        cutoff = time.time() - (hours_ago * 3600)

        results = []
        for file in self.base.glob("*"):
            if file.is_file() and not file.name.startswith('.'):
                if file.stat().st_mtime > cutoff:
                    results.append(file)

        results.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return results

    def get_graph_data(self) -> Dict:
        """
        Generate graph data for visualization
        Returns nodes and edges in a format ready for D3.js/Canvas
        """
        nodes = []
        edges = []

        # Collect all files as nodes
        for file in self.base.glob("*"):
            if file.is_file() and not file.name.startswith('.'):
                stat = file.stat()

                # Load metadata if exists
                meta_file = self.metadata_dir / f"{file.name}.json"
                metadata = {}
                if meta_file.exists():
                    with open(meta_file) as f:
                        metadata = json.load(f)

                nodes.append({
                    'id': file.name,
                    'path': str(file.relative_to(self.base)),
                    'size': stat.st_size,
                    'created': stat.st_ctime,
                    'modified': stat.st_mtime,
                    'accessed': stat.st_atime,
                    'type': metadata.get('type', 'unknown'),
                    'preview': file.read_text()[:100] if file.suffix == '.txt' else '',
                    'mass': max(1, stat.st_size / 1024),  # KB as mass
                    'age': (datetime.now().timestamp() - stat.st_ctime) / 86400,  # Days old
                    'heat': 0  # Will be calculated based on recent access
                })

        # Add edges from connections
        for source, targets in self.connections.items():
            for target, conn_data in targets.items():
                edges.append({
                    'source': Path(source).name,
                    'target': Path(target).name,
                    'strength': conn_data['strength'],
                    'types': conn_data['types']
                })

        return {
            'nodes': nodes,
            'edges': edges,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Quick test
    brain = Brain()

    # Test saving
    filepath, was_appended = brain.save_text("This is a test thought about vector databases")
    print(f"Saved to: {filepath} (appended: {was_appended})")

    # Test collision handling
    filepath2, was_appended2 = brain.save_text("More thoughts about vector databases", "test-thought-about-vector-databases.txt")
    print(f"Saved to: {filepath2} (appended: {was_appended2})")

    # Test search
    results = brain.search_content("vector")
    print(f"Search results: {results}")

    # Get graph data
    graph = brain.get_graph_data()
    print(f"Graph has {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
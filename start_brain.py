#!/usr/bin/env python3
"""
Brain System Launcher
Quick start script for the living memory galaxy
"""

import sys
import os
import webbrowser
import time
from pathlib import Path

def main():
    # Check Python version
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher required")
        sys.exit(1)

    # Import and start server
    try:
        from brain_server import run_server
    except ImportError:
        print("Error: brain_server.py not found in current directory")
        sys.exit(1)

    # Default settings
    port = 8888
    host = 'localhost'

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("""
Brain - Living Memory Galaxy

Usage:
    python start_brain.py [port] [host]

Examples:
    python start_brain.py              # Start on localhost:8888
    python start_brain.py 3000         # Start on localhost:3000
    python start_brain.py 8888 0.0.0.0 # Allow network access

Your brain directory will be created at: ~/brain/
            """)
            sys.exit(0)
        else:
            port = int(sys.argv[1])

    if len(sys.argv) > 2:
        host = sys.argv[2]

    # Open browser after slight delay
    def open_browser():
        time.sleep(1)
        url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        print(f"\nOpening browser to {url}")
        webbrowser.open(url)

    # Start browser opener in background
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start server
    try:
        run_server(port, host)
    except KeyboardInterrupt:
        print("\n\nBrain server stopped.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
Seed Data for Headspace Demo
Populates the cosmos with demo planets/chunks for new users
"""

from datetime import datetime
from data_models import Document, Chunk
import numpy as np


DEMO_CONTENT = {
    "home_link": {
        "title": "🏠 Return Home",
        "content": "Click this planet to return to the main index page.",
        "doc_type": "link",
        "metadata": {
            "link_url": "/index.html",
            "is_external_link": True,
            "chunk_type": "portal"
        },
        "position": [0, 0, 0],  # Center position
        "color": "#ff6b6b"
    },
    "about_project": {
        "title": "About Headspace",
        "content": """Headspace is a cosmic knowledge visualization system that transforms your thoughts, documents, and ideas into a navigable 3D universe.

Each document becomes a constellation of connected chunks, floating in semantic space. Similar ideas cluster together, creating a beautiful map of your knowledge.

Features:
• Document management with AI-powered chunking
• 3D visualization using Three.js
• Semantic connections between related ideas
• Embedding-based positioning
• Tag-based organization

Built with FastAPI, Python, and modern web technologies.""",
        "doc_type": "text",
        "metadata": {
            "is_demo": True
        },
        "position": [30, 20, 10],
        "color": "#4ecdc4"
    },
    "poetry_1": {
        "title": "Cosmic Thoughts",
        "content": """In the vast expanse of digital space,
Where thoughts converge and ideas race,
Each chunk a planet, bright and true,
Connected by the threads we drew.

The cosmos watches, silent, vast,
As knowledge forms that will forever last.
A universe of meaning, growing slow,
Where every thought can find its glow.""",
        "doc_type": "text",
        "metadata": {
            "is_demo": True,
            "genre": "poetry"
        },
        "position": [-25, 15, -20],
        "color": "#95e1d3"
    },
    "poetry_2": {
        "title": "The Language of Stars",
        "content": """Words become planets,
Sentences form constellations,
Paragraphs create galaxies,
And documents light up the universe.

We are architects of meaning,
Building bridges between ideas,
Creating paths through the void,
Where knowledge waits to be discovered.""",
        "doc_type": "text",
        "metadata": {
            "is_demo": True,
            "genre": "poetry"
        },
        "position": [20, -30, 15],
        "color": "#f38181"
    },
    "welcome": {
        "title": "Welcome to Your Headspace",
        "content": """This is your personal knowledge cosmos. 

You can:
• Add your own documents and thoughts
• Watch them transform into a 3D visualization
• Explore connections between ideas
• Build your own universe of knowledge

Start by clicking the "+ Add Document" button in the sidebar, or explore the demo planets floating around you.

Everything you create here can be stored locally on your computer, or synced to the cloud to share with others.""",
        "doc_type": "text",
        "metadata": {
            "is_demo": True,
            "welcome_message": True
        },
        "position": [-15, 25, -10],
        "color": "#a8e6cf"
    }
}


def create_seed_documents(processor, db):
    """Create demo documents with chunks"""
    print("\n🌱 Creating seed data for demo...")
    
    # Check if demo data already exists
    existing_docs = db.get_all_documents()
    demo_titles = {doc.title for doc in existing_docs if doc.metadata.get("is_demo", False)}
    
    created_count = 0
    
    for doc_key, doc_data in DEMO_CONTENT.items():
        if doc_data["title"] in demo_titles:
            print(f"  ⏭️  Skipping {doc_data['title']} (already exists)")
            continue
            
        try:
            # Create document
            doc_id = processor.process_document_instant(
                title=doc_data["title"],
                content=doc_data["content"],
                doc_type=doc_data["doc_type"]
            )
            
            # Get the document and update metadata
            doc = db.get_document(doc_id)
            if doc:
                doc.metadata.update(doc_data.get("metadata", {}))
                doc.metadata["is_demo"] = True
                db.save_document(doc)
            
            # Update chunks with specific positions and colors
            chunks = db.get_chunks_by_document(doc_id)
            if chunks:
                # Use the first chunk and customize it
                chunk = chunks[0]
                
                # Set custom position if specified
                if "position" in doc_data:
                    chunk.position_3d = doc_data["position"]
                
                # Set custom color
                if "color" in doc_data:
                    chunk.color = doc_data["color"]
                
                # Add link metadata if present
                if "metadata" in doc_data and "link_url" in doc_data["metadata"]:
                    chunk.metadata["link_url"] = doc_data["metadata"]["link_url"]
                    chunk.metadata["is_external_link"] = doc_data["metadata"].get("is_external_link", False)
                
                # Update chunk metadata
                chunk.metadata.update(doc_data.get("metadata", {}))
                chunk.metadata["is_demo"] = True
                
                db.save_chunk(chunk)
                
                # For multi-chunk documents, adjust other chunks relative to first
                for i, other_chunk in enumerate(chunks[1:], 1):
                    # Position other chunks nearby
                    offset = np.array(doc_data.get("position", [0, 0, 0])) + np.random.randn(3) * 5
                    other_chunk.position_3d = offset.tolist()
                    other_chunk.color = doc_data.get("color", "#748ffc")
                    other_chunk.metadata["is_demo"] = True
                    db.save_chunk(other_chunk)
            
            print(f"  ✅ Created: {doc_data['title']}")
            created_count += 1
            
        except Exception as e:
            print(f"  ❌ Error creating {doc_data['title']}: {e}")
    
    print(f"\n📊 Seed data complete: {created_count} documents created")
    return created_count


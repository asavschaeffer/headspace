"""
Seed Headspace with example documents on first load
Creates Home Planet, Guide, and Creative Piece
"""

import hashlib
from datetime import datetime
from data_models import Document

# Read README for guide
def get_readme_content():
    """Get README.md content for the Headspace Guide"""
    try:
        with open('README.md', 'r') as f:
            return f.read()
    except:
        return """# Headspace Guide

Transform your documents into navigable 3D constellations where chunks are stars and knowledge becomes a cosmic memory palace.

## How to Use

1. **Click a Planet** - View its full content in document view
2. **Add Your Own** - Click the + button to create a new entry
3. **Watch it Form** - See embeddings materialize chunk by chunk
4. **Float to Place** - Your planet animates to its semantic position
5. **Explore** - Return to cosmos and discover connections

## Philosophy

Headspace is built on the idea that human memory works spatially. By converting your thoughts into a 3D semantic space, you create a "memory palace" - a navigable knowledge landscape where related ideas naturally cluster together.

Each document is a planet. Each chunk is a star system. The cosmos is your mind."""


def get_creative_piece():
    """A poetic entry to seed the system"""
    return """# A Thought on Silence

There is a kind of silence that speaks louder than words—the silence before understanding, when the mind is still preparing itself to receive a truth it did not know it was seeking.

In this silence, nothing moves. The stars hold their breath. The cosmos pauses, waiting.

And in that pause, we find ourselves. Not lost, but finally found. Standing in the center of our own constellation, where every thought we've ever had orbits in slow, eternal grace around the gravity of our being.

This is what Headspace seeks to capture: the moment when thought becomes space, when ideas become navigable, when your mind becomes a cosmos you can walk through and wonder at.

---

*Signed by the void that lives in all of us.*

Created in quiet contemplation, November 2024."""


SEED_DOCUMENTS = [
    {
        "title": "🏠 Home - Return to Index",
        "content": """Welcome to Headspace.

You are standing at the entrance to your cosmic mind.

The white foggy planet before you is your home. Click it anytime to return to the main website.

Around you float other thoughts—documents and diary entries, each one a unique world shaped by its own meaning and purpose.

Explore them. Add your own. Watch as your thoughts materialize before your eyes, chunk by chunk, embedding by embedding, until they find their place in the cosmos of your consciousness.

This is Headspace. Your mind as a universe to explore.""",
        "doc_type": "text",
        "metadata": {
            "is_home": True,
            "status": "seed",
            "signature": "The System"
        }
    },
    {
        "title": "📖 Headspace Guide",
        "content": get_readme_content(),
        "doc_type": "text",
        "metadata": {
            "is_guide": True,
            "status": "seed",
            "signature": "Documentation"
        }
    },
    {
        "title": "✨ A Thought on Silence",
        "content": get_creative_piece(),
        "doc_type": "text",
        "metadata": {
            "is_poetry": True,
            "status": "seed",
            "signature": "The Void"
        }
    }
]


def seed_headspace(db, processor, monitor):
    """
    Seed the database with example documents

    Args:
        db: Database manager instance
        processor: Document processor instance
        monitor: Model monitor instance
    """
    try:
        monitor.logger.info("🌱 Seeding Headspace with example documents...")

        # Check if already seeded
        existing_docs = db.get_all_documents()
        if existing_docs and len(existing_docs) > 0:
            monitor.logger.info("✅ Database already seeded, skipping...")
            return

        # Create seed documents
        for seed_doc in SEED_DOCUMENTS:
            try:
                # Generate document ID
                doc_id = hashlib.md5(
                    f"{seed_doc['title']}{seed_doc['content'][:100]}".encode()
                ).hexdigest()[:12]

                # Mark as seed
                metadata = seed_doc.get("metadata", {})
                metadata["is_seed"] = True

                # Create document
                doc = Document(
                    id=doc_id,
                    title=seed_doc["title"],
                    content=seed_doc["content"],
                    doc_type=seed_doc["doc_type"],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    metadata=metadata
                )

                monitor.logger.info(f"  Processing seed document: {seed_doc['title']}")

                # Process document (generate embeddings, chunks, etc.)
                if hasattr(processor, 'process_document_with_monitor'):
                    processor.process_document_with_monitor(
                        seed_doc["title"],
                        seed_doc["content"],
                        seed_doc["doc_type"],
                        monitor=monitor
                    )
                else:
                    # Fallback to regular processing
                    processor.process_document(
                        seed_doc["title"],
                        seed_doc["content"],
                        seed_doc["doc_type"]
                    )

                monitor.logger.info(f"  ✅ Seeded: {seed_doc['title']}")

            except Exception as e:
                monitor.logger.error(f"  ❌ Failed to seed {seed_doc['title']}: {e}")
                continue

        monitor.logger.info("🌱 Headspace seeding complete!")

    except Exception as e:
        monitor.logger.error(f"Seeding failed: {e}")
        import traceback
        monitor.logger.error(traceback.format_exc())

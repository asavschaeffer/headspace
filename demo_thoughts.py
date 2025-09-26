#!/usr/bin/env python3
"""
Demo script to populate the brain with sample thoughts
Shows collision handling and different content types
"""

from brain_core import Brain
import time
from datetime import datetime

def main():
    brain = Brain()
    print("Populating brain with demo thoughts...\n")

    # Sample thoughts to demonstrate features
    thoughts = [
        # Meeting notes (will create meeting-sarah cluster)
        ("Meeting with Sarah about the new vector database implementation. "
         "We need to consider performance implications and memory usage."),

        # Bug report (will be categorized as error)
        ("BUG: Auth middleware throwing 500 error when token expires. "
         "Stack trace shows null pointer exception in validateToken function."),

        # Code snippet (will be categorized as code)
        ("def calculate_similarity(vec1, vec2):\n"
         "    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))"),

        # Question (will be categorized as question)
        ("How do I implement lazy loading in React components? "
         "Need to optimize bundle size for the dashboard."),

        # Follow-up to meeting (collision - will append)
        ("Follow-up thoughts on vector database discussion with Sarah. "
         "Maybe we should prototype with FAISS first before committing."),

        # Another bug (will cluster with other bugs)
        ("ERROR: Memory leak detected in vector index builder. "
         "RSS memory grows unbounded when processing large datasets."),

        # Random thought
        ("The orbital mechanics visualization reminds me of Carl Sagan's Cosmos. "
         "Each thought is a star in the personal universe of memory."),

        # Technical note
        ("Performance benchmark: Current implementation processes "
         "1000 embeddings in 230ms on M1 MacBook Pro."),

        # Todo item
        ("TODO: Refactor the similarity calculation to use batch processing. "
         "Should improve throughput by 10x according to benchmarks."),

        # Personal reflection
        ("This breathing galaxy interface creates a meditative space for thought. "
         "The slow drift of ideas mirrors how memory actually works.")
    ]

    # Save each thought with slight delay to show temporal connections
    for i, thought in enumerate(thoughts):
        print(f"[{i+1}/{len(thoughts)}] Saving thought...")

        # Special case: make one deliberate collision
        if "Follow-up thoughts on vector" in thought:
            # This will append to the existing Sarah meeting file
            filepath, appended = brain.save_text(thought, "meeting-sarah-about-the-new-vector-database.txt")
            print(f"  → {'Appended to' if appended else 'Created'}: {filepath.name}")
        else:
            filepath, appended = brain.save_text(thought)
            print(f"  → {'Appended to' if appended else 'Created'}: {filepath.name}")

        # Small delay to create temporal connections
        time.sleep(0.5)

    print("\n" + "="*50)
    print("Demo thoughts saved!")
    print("="*50)

    # Show some statistics
    graph = brain.get_graph_data()
    print(f"\nBrain statistics:")
    print(f"  Total thoughts: {len(graph['nodes'])}")
    print(f"  Connections: {len(graph['edges'])}")

    # Demonstrate search
    print("\nSearch examples:")

    print("  Searching for 'vector':")
    results = brain.search_content("vector")
    for file, snippet in results[:3]:
        print(f"    - {file.name}: {snippet[:50]}...")

    print("\n  Searching for 'error':")
    results = brain.search_content("error")
    for file, snippet in results[:3]:
        print(f"    - {file.name}: {snippet[:50]}...")

    print("\n  Files from last hour:")
    recent = brain.search_by_time(1)
    for file in recent[:5]:
        print(f"    - {file.name}")

    print(f"\nBrain location: {brain.base}")
    print("Start the web interface with: python start_brain.py")

if __name__ == "__main__":
    main()
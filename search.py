import argparse
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

INDEX_FILE = "search_index.pkl"

def search(query: str, limit: int = 5):
    if not os.path.exists(INDEX_FILE):
        print(f"Index file '{INDEX_FILE}' not found. Please run indexer.py first.")
        return

    try:
        data = joblib.load(INDEX_FILE)
        vectorizer = data['vectorizer']
        tfidf_matrix = data['tfidf_matrix']
        paths = data['paths']
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # Transform query
    try:
        query_vec = vectorizer.transform([query])
    except Exception as e:
        print(f"Error processing query: {e}")
        return

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top N results
    related_docs_indices = cosine_similarities.argsort()[:-limit-1:-1]

    print(f"\nSearch results for: '{query}'\n")
    found = False
    for i in related_docs_indices:
        score = cosine_similarities[i]
        if score > 0:
            print(f"[{score:.4f}] {paths[i]}")
            found = True
    
    if not found:
        print("No relevant files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search indexed files using cached TF-IDF model.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    search(args.query, args.limit)

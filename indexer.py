import os
import argparse
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from llm_client import get_llm_client
from database import Database

# Load environment variables
load_dotenv()

IGNORED_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.gemini'}
IGNORED_EXTENSIONS = {'.pyc', '.db', '.sqlite', '.png', '.jpg', '.jpeg', '.gif', '.webp'}

def should_ignore(path: Path) -> bool:
    if path.name.startswith('.'):
        return True
    if path.name in IGNORED_DIRS:
        return True
    if path.suffix in IGNORED_EXTENSIONS:
        return True
    return False

def describe_file(client, path: Path) -> str:
    try:
        txt = path.read_text(errors="ignore")[:8000]
        if not txt.strip():
            return "Empty file."
        
        prompt = f"Summarize this file in 2 sentences and tag it with topics and type:\n\n{txt[:2000]}"
        return client.ask(prompt)
    except Exception as e:
        return f"Error processing file: {e}"

def index_directory(directory: str):
    root_path = Path(directory).resolve()
    if not root_path.exists():
        print(f"Directory not found: {root_path}")
        return

    print(f"Indexing directory: {root_path}")
    
    db = Database()
    client = get_llm_client()
    
    count = 0
    for f in root_path.rglob("*"):
        # Check if any parent part is ignored
        if any(part in IGNORED_DIRS or part.startswith('.') for part in f.relative_to(root_path).parts):
            continue
            
        if f.is_file() and not should_ignore(f) and f.stat().st_size < 5e6:
            print(f"Processing: {f.relative_to(root_path)}")
            summary = describe_file(client, f)
            
            # Basic parsing of the LLM response could go here to extract structured data
            # For now, we just store the raw summary text
            
            db.upsert_file(str(f), summary)
            count += 1

    print(f"Finished indexing. Processed {count} files.")
    build_search_index()

def build_search_index():
    print("Building search index...")
    db = Database()
    # We need to access the raw connection or add a method to get all files
    # For simplicity, let's just use the raw connection here since we are in the same package
    try:
        with db._get_conn() as conn:
            cursor = conn.execute("SELECT path, summary, type, topics FROM files")
            rows = cursor.fetchall()
    except Exception as e:
        print(f"Error reading from DB: {e}")
        return

    if not rows:
        print("No files to index.")
        return

    corpus = []
    paths = []
    for row in rows:
        path, summary, type_, topics = row
        text_content = f"{path} {summary or ''} {type_ or ''} {topics or ''}"
        corpus.append(text_content)
        paths.append(path)

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # Save both the vectorizer and the matrix, and the paths
        joblib.dump({
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'paths': paths
        }, 'search_index.pkl')
        print("Search index saved to search_index.pkl")
    except ValueError:
        print("Could not build search index (corpus too small).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-OS File Indexer")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to index (default: current directory)")
    args = parser.parse_args()
    
    index_directory(args.directory)

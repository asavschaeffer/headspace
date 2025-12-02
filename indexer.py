import os
import argparse
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from llm_client import get_llm_client
from database import Database
from core.analyzers import BaseFileAnalyzer, CodeFileAnalyzer

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


def index_directory(directory: str, use_llm: bool = True):
    root_path = Path(directory).resolve()
    if not root_path.exists():
        print(f"Directory not found: {root_path}")
        return

    print(f"Indexing directory: {root_path}")
    
    db = Database()
    client = get_llm_client() if use_llm else None
    
    count = 0
    for f in root_path.rglob("*"):
        # Check if any parent part is ignored
        if any(part in IGNORED_DIRS or part.startswith('.') for part in f.relative_to(root_path).parts):
            continue
            
        if f.is_file() and not should_ignore(f) and f.stat().st_size < 5e6:
            print(f"Processing: {f.relative_to(root_path)}")
            
            # Choose analyzer
            if f.suffix == '.py':
                analyzer = CodeFileAnalyzer(client)
            else:
                analyzer = BaseFileAnalyzer(client)
                
            metadata = analyzer.analyze(f)
            
            # Generate summary if not present (BaseFileAnalyzer doesn't do it yet, so we keep the old logic for now or move it)
            # The plan said "Extend FileAnalyzer with code-specific extractors".
            # Let's keep the existing summary logic for now but use the metadata.
            
            summary = metadata.get('summary', '')
            
            # Store in DB
            # We need to pass the extra metadata
            extra_metadata = {k: v for k, v in metadata.items() if k not in ['path', 'summary', 'type', 'topics', 'action']}
            
            # Determine type/topics from metadata or LLM
            # For now, we trust the analyzer for type if it set it
            file_type = metadata.get('type', '')
            
            # Extract topics from summary if possible
            topics = ""
            if "Topics:" in summary:
                try:
                    topics = summary.split("Topics:")[1].split(".")[0].strip()
                except:
                    pass
            elif "topics:" in summary.lower():
                 try:
                    topics = summary.lower().split("topics:")[1].split(".")[0].strip()
                 except:
                    pass

            db.upsert_file(str(f), summary, type_=file_type, topics=topics, extra_metadata=extra_metadata)
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

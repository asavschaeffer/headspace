import joblib
import os
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from core.reasoning import ReasoningStrategy, Decision

class SemanticClusteringReasoner(ReasoningStrategy):
    """
    Groups files based on semantic similarity using TF-IDF.
    """
    
    def __init__(self, index_path: str = "search_index.pkl"):
        self.index_path = index_path
        self.vectorizer = None
        self.tfidf_matrix = None
        self.paths = []
        self._load_index()
        
    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                data = joblib.load(self.index_path)
                self.vectorizer = data['vectorizer']
                self.tfidf_matrix = data['tfidf_matrix']
                self.paths = data['paths']
            except Exception as e:
                print(f"Error loading search index: {e}")
                
    def reason(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        if self.tfidf_matrix is None:
            return []
            
        decisions = []
        
        # Calculate cosine similarity matrix
        # Note: This computes similarity for ALL files in the index, 
        # but we should filter for the ones in file_metadata if possible.
        # For simplicity, we'll iterate over the index paths and check if they are in our scope.
        
        # Map paths to indices
        path_to_idx = {p: i for i, p in enumerate(self.paths)}
        
        # Threshold for grouping
        SIMILARITY_THRESHOLD = 0.7
        
        processed_pairs = set()
        
        for meta in file_metadata:
            path = meta['path']
            if path not in path_to_idx:
                continue
                
            idx = path_to_idx[path]
            
            # Get similarities for this file
            # This is efficient enough for now
            sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
            
            # Find similar files
            related_indices = sim_scores.argsort()[::-1] # Descending
            
            for other_idx in related_indices:
                if other_idx == idx:
                    continue
                    
                score = sim_scores[other_idx]
                
                if score < SIMILARITY_THRESHOLD:
                    break # Sorted, so we can stop
                    
                other_path = self.paths[other_idx]
                
                # Avoid duplicate pairs (A-B and B-A)
                pair = tuple(sorted([path, other_path]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                # Check if they are in different directories
                if Path(path).parent != Path(other_path).parent:
                    decisions.append(Decision(
                        action="group_files",
                        target_path=path,
                        destination_path=str(Path(other_path).parent), # Propose moving to the other's dir
                        reasoning=f"File is semantically similar ({score:.2f}) to {Path(other_path).name}.",
                        confidence=score,
                        metadata={"similarity": score, "related_file": other_path}
                    ))
                    
        return decisions

class DuplicateDetector(ReasoningStrategy):
    """
    Detects duplicate files based on content hash or high similarity.
    """
    
    def reason(self, file_metadata: List[Dict[str, Any]]) -> List[Decision]:
        decisions = []
        hashes = {}
        
        for meta in file_metadata:
            path = Path(meta['path'])
            if not path.exists() or not path.is_file():
                continue
                
            # Calculate hash
            try:
                # Use chunked reading for large files
                hasher = hashlib.md5()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()
                
                if file_hash in hashes:
                    original_path = hashes[file_hash]
                    decisions.append(Decision(
                        action="delete", # or deduplicate
                        target_path=str(path),
                        reasoning=f"Exact duplicate of {Path(original_path).name}.",
                        confidence=1.0,
                        metadata={"original": original_path}
                    ))
                else:
                    hashes[file_hash] = str(path)
                    
            except Exception:
                continue
        
        # TODO: Implement fuzzy matching using difflib or embeddings
        # for near-duplicate detection.
                
        return decisions

"""
Deterministic embedding generator for test data.

Creates reproducible embeddings based on content hashes that maintain
semantic relationships for proper integration testing.
"""

import hashlib
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


class DeterministicEmbeddingGenerator:
    """Generate deterministic embeddings that preserve semantic relationships."""
    
    def __init__(self, dimension: int = 1024, seed: int = 42):
        self.dimension = dimension
        self.seed = seed
        self.base_rng = np.random.RandomState(seed)
        
        # Create domain-specific base vectors for semantic clustering
        self.domain_bases = self._create_domain_bases()
        
    def _create_domain_bases(self) -> Dict[str, np.ndarray]:
        """Create base vectors for each domain to ensure semantic clustering."""
        domains = [
            "fitness", "software", "wellness", "learning", "finance", 
            "productivity", "creativity", "science"
        ]
        
        bases = {}
        for i, domain in enumerate(domains):
            # Create deterministic base vector for domain
            domain_seed = self.seed + hash(domain) % 10000
            domain_rng = np.random.RandomState(domain_seed)
            base_vector = domain_rng.randn(self.dimension).astype(np.float32)
            
            # Normalize to unit vector
            norm = np.linalg.norm(base_vector)
            if norm > 0:
                base_vector = base_vector / norm
                
            bases[domain] = base_vector
            
        return bases
        
    def generate_embedding(self, text_hash: str, domain: str, 
                          similarity_group: str = None, concept_keywords: List[str] = None) -> np.ndarray:
        """
        Generate a deterministic embedding based on text hash and domain.
        
        Args:
            text_hash: Hash string representing the content
            domain: Domain for semantic clustering
            similarity_group: Optional group for creating similar embeddings
            concept_keywords: Keywords that define the concept for cross-domain similarity
            
        Returns:
            Normalized embedding vector
        """
        # Create seed from hash
        hash_seed = int(hashlib.sha256(text_hash.encode()).hexdigest()[:8], 16) % (2**31)
        embedding_rng = np.random.RandomState(hash_seed)
        
        # Start with domain base vector
        if domain in self.domain_bases:
            base_vector = self.domain_bases[domain].copy()
        else:
            # Fallback to generic base
            base_vector = self.base_rng.randn(self.dimension).astype(np.float32)
            base_vector = base_vector / np.linalg.norm(base_vector)
        
        # Add concept-based components for cross-domain similarity
        if concept_keywords:
            concept_component = np.zeros(self.dimension, dtype=np.float32)
            for keyword in concept_keywords:
                keyword_seed = self.seed + hash(keyword) % 10000  # Use consistent seed
                keyword_rng = np.random.RandomState(keyword_seed)
                keyword_vector = keyword_rng.randn(self.dimension).astype(np.float32)
                keyword_vector = keyword_vector / np.linalg.norm(keyword_vector)
                concept_component += keyword_vector
            
            if np.linalg.norm(concept_component) > 0:
                concept_component = concept_component / np.linalg.norm(concept_component)
                # Increase concept influence for better semantic similarity
                base_vector = 0.5 * base_vector + 0.5 * concept_component
        
        # Add controlled noise based on content
        noise_scale = 0.2  # Reduced noise for better semantic clustering
        if similarity_group:
            # Reduce noise for similar concepts
            noise_scale = 0.1
            
        noise = embedding_rng.normal(0, noise_scale, self.dimension).astype(np.float32)
        embedding = base_vector + noise
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
        
    def create_similar_embeddings(self, base_hash: str, domain: str, 
                                count: int = 2) -> List[np.ndarray]:
        """Create multiple similar embeddings for testing similarity search."""
        embeddings = []
        
        for i in range(count):
            similar_hash = f"{base_hash}_similar_{i}"
            embedding = self.generate_embedding(similar_hash, domain, "similar_group")
            embeddings.append(embedding)
            
        return embeddings


def load_test_embeddings() -> Dict:
    """Load the test embeddings configuration."""
    test_data_path = Path(__file__).parent.parent / "data" / "test_embeddings.json"
    
    with open(test_data_path, 'r') as f:
        return json.load(f)


def generate_all_test_embeddings() -> Dict[str, np.ndarray]:
    """Generate all deterministic test embeddings with semantic relationships."""
    config = load_test_embeddings()
    generator = DeterministicEmbeddingGenerator()
    
    # Create embeddings with explicit similarity relationships
    embeddings = {}
    
    # Generate base embeddings for each domain
    for globule in config["test_globules"]:
        embedding = generator.generate_embedding(
            globule["embedding_hash"],
            globule["parsed_data"]["domain"]
        )
        embeddings[globule["id"]] = embedding
    
    # Now apply similarity relationships manually for testing purposes
    similarity_pairs = config["semantic_relationships"]["similar_pairs"]
    
    for pair in similarity_pairs:
        id1, id2, target_sim, reason = pair
        if id1 in embeddings and id2 in embeddings:
            emb1 = embeddings[id1]
            emb2 = embeddings[id2]
            
            # Create a shared component to increase similarity
            shared_component_seed = hash(f"{id1}_{id2}") % 10000
            shared_rng = np.random.RandomState(shared_component_seed)
            shared_vector = shared_rng.randn(generator.dimension).astype(np.float32)
            shared_vector = shared_vector / np.linalg.norm(shared_vector)
            
            # Mix in shared component to achieve target similarity
            mix_strength = 0.6  # Adjust to get desired similarity level
            
            new_emb1 = (1 - mix_strength) * emb1 + mix_strength * shared_vector
            new_emb2 = (1 - mix_strength) * emb2 + mix_strength * shared_vector
            
            # Normalize
            new_emb1 = new_emb1 / np.linalg.norm(new_emb1)
            new_emb2 = new_emb2 / np.linalg.norm(new_emb2)
            
            embeddings[id1] = new_emb1
            embeddings[id2] = new_emb2
        
    return embeddings


def generate_performance_embeddings(count: int = 100000) -> List[Dict]:
    """Generate large dataset for performance testing with proper semantic clustering."""
    generator = DeterministicEmbeddingGenerator()
    
    domains = ["fitness", "software", "wellness", "learning", "finance", 
               "productivity", "creativity", "science"]
    categories = ["concept", "technique", "practice", "idea", "note", "reflection"]
    
    dataset = []
    
    for i in range(count):
        domain = domains[i % len(domains)]
        category = categories[i % len(categories)]
        
        # Create deterministic hash for this entry
        content_hash = f"{domain}_{category}_{i:06d}_performance_test"
        
        # Generate embedding
        embedding = generator.generate_embedding(content_hash, domain)
        
        globule_data = {
            "id": f"perf_{i:06d}",
            "text": f"Performance test entry {i} in {domain} domain discussing {category} with substantial content to make search meaningful and realistic for testing purposes. This entry includes detailed information about {domain} concepts and {category} applications.",
            "embedding": embedding.tolist(),
            "embedding_confidence": 0.7 + (i % 30) * 0.01,  # 0.7 to 0.99
            "parsed_data": {
                "title": f"Performance Test Entry {i}",
                "domain": domain,
                "category": category,
                "keywords": [domain, category, "performance", "test", f"entry_{i}"],
                "metadata": {"parser_type": "test_parser", "confidence_score": 0.75 + (i % 25) * 0.01}
            },
            "parsing_confidence": 0.75 + (i % 25) * 0.01,
            "file_decision": {
                "semantic_path": f"{domain}/{category}",
                "filename": f"perf-test-{i:06d}.md",
                "confidence": 0.7 + (i % 30) * 0.01
            },
            "days_ago": i % 365
        }
        
        dataset.append(globule_data)
        
    return dataset


if __name__ == "__main__":
    # Test the embedding generation
    embeddings = generate_all_test_embeddings()
    print(f"Generated {len(embeddings)} test embeddings")
    
    # Test similarity relationships
    config = load_test_embeddings()
    for pair in config["semantic_relationships"]["similar_pairs"]:
        id1, id2, expected_sim, reason = pair
        if id1 in embeddings and id2 in embeddings:
            # Calculate cosine similarity
            emb1 = embeddings[id1]
            emb2 = embeddings[id2]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"{id1} <-> {id2}: {similarity:.3f} (expected ~{expected_sim}) - {reason}")
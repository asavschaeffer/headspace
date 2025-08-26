"""
Semantic Clustering Engine for Phase 2 Intelligence.

This module implements intelligent clustering of thoughts and ideas using
semantic similarity and machine learning techniques. It automatically
discovers themes, groups related content, and provides meaningful cluster
labels based on content analysis.

Features:
- K-means clustering on semantic embeddings
- Automatic optimal cluster number detection  
- Intelligent cluster naming using content analysis
- Temporal clustering (recent vs historical)
- Dynamic re-clustering as content grows
- Cross-domain theme detection

Author: Globule Team
Version: 2.0.0
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter
import json


from globule.core.models import ProcessedGlobule
from globule.storage.sqlite_manager import SQLiteStorageManager


@dataclass
class SemanticCluster:
    """
    Represents a semantic cluster of related thoughts.
    
    Contains the cluster metadata, representative content,
    and intelligence about the common themes.
    """
    id: str
    label: str
    description: str
    size: int
    centroid: np.ndarray  # NOTE: For HDBSCAN, this is the geometric center of the cluster members, not a defining parameter of the algorithm.
    member_ids: List[str]
    keywords: List[str]
    domains: List[str]
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    representative_samples: List[str] = field(default_factory=list)
    theme_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "size": self.size,
            "member_ids": self.member_ids,
            "keywords": self.keywords,
            "domains": self.domains,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "representative_samples": self.representative_samples,
            "theme_analysis": self.theme_analysis
        }


@dataclass
class ClusteringAnalysis:
    """
    Complete clustering analysis results.
    
    Contains all discovered clusters plus metadata about
    the clustering process and quality metrics.
    """
    clusters: List[SemanticCluster]
    total_globules: int
    clustering_method: str
    dbcv_score: float
    processing_time_ms: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    cross_cluster_relationships: Dict[str, List[str]] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "total_globules": self.total_globules,
            "clustering_method": self.clustering_method,
            "dbcv_score": self.dbcv_score,
            "processing_time_ms": self.processing_time_ms,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "cross_cluster_relationships": self.cross_cluster_relationships,
            "temporal_patterns": self.temporal_patterns,
            "quality_metrics": self.quality_metrics
        }


class SemanticClusteringEngine:
    """
    Intelligent semantic clustering engine for Phase 2.
    
    Discovers themes and groups related thoughts using advanced machine learning
    techniques combined with content analysis and domain knowledge.
    """

    def __init__(self, storage_manager: SQLiteStorageManager):
        """Initialize the clustering engine."""
        self.storage = storage_manager
        self.logger = logging.getLogger(__name__)
        
        # Clustering parameters
        self.min_cluster_size = 2
        self.max_clusters = 20
        self.min_similarity_threshold = 0.3
        self.temporal_window_days = 30

    async def analyze_semantic_clusters(
        self, 
        min_globules: int = 5,
        force_recalculation: bool = False
    ) -> ClusteringAnalysis:
        """
        Perform comprehensive semantic clustering analysis.
        
        Args:
            min_globules: Minimum number of globules required for clustering
            force_recalculation: Force recalculation even if recent results exist
            
        Returns:
            ClusteringAnalysis with discovered clusters and metadata
        """
        start_time = datetime.now()
        
        try:
            # Get all globules with embeddings
            globules = await self._get_clusterable_globules()
            
            if len(globules) < min_globules:
                self.logger.warning(f"Insufficient globules for clustering: {len(globules)} < {min_globules}")
                return self._create_empty_analysis(len(globules))
            
            self.logger.info(f"Starting semantic clustering analysis on {len(globules)} globules")
            
            # Extract embeddings and prepare data
            embeddings_matrix, globule_map = self._prepare_clustering_data(globules)
            
            # Perform HDBSCAN density-based clustering
            cluster_labels = self._perform_clustering(embeddings_matrix)
            
            # Calculate DBCV score for quality assessment
            dbcv_score = self._calculate_dbcv_score(embeddings_matrix, cluster_labels)
            
            # Create semantic clusters with intelligent labeling
            semantic_clusters = await self._create_semantic_clusters(
                globules, cluster_labels, embeddings_matrix, globule_map
            )
            
            # Analyze cross-cluster relationships
            cross_relationships = self._analyze_cross_cluster_relationships(semantic_clusters)
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(globules, cluster_labels)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                semantic_clusters, embeddings_matrix, cluster_labels
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            analysis = ClusteringAnalysis(
                clusters=semantic_clusters,
                total_globules=len(globules),
                clustering_method="hdbscan_density_based",
                dbcv_score=dbcv_score,
                processing_time_ms=processing_time,
                cross_cluster_relationships=cross_relationships,
                temporal_patterns=temporal_patterns,
                quality_metrics=quality_metrics
            )
            
            self.logger.info(f"Clustering analysis completed: {len(semantic_clusters)} clusters, DBCV={dbcv_score:.3f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {e}")
            raise

    

    def _prepare_clustering_data(
        self, 
        globules: List[ProcessedGlobule]
    ) -> Tuple[np.ndarray, Dict[int, ProcessedGlobule]]:
        """Prepare embeddings matrix and globule mapping for clustering."""
        embeddings = []
        globule_map = {}
        
        for i, globule in enumerate(globules):
            if globule.embedding is not None:
                # Normalize embeddings for consistent clustering
                normalized_embedding = globule.embedding / np.linalg.norm(globule.embedding)
                embeddings.append(normalized_embedding)
                globule_map[i] = globule
        
        embeddings_matrix = np.vstack(embeddings)
        
        # Optional: Apply dimensionality reduction for very high-dimensional spaces
        # For now, we'll work directly with the embeddings
        
        return embeddings_matrix, globule_map

    def _perform_clustering(self, embeddings_matrix: np.ndarray) -> np.ndarray:
        """Perform HDBSCAN density-based clustering."""
        try:
            import hdbscan
            
            # Initialize HDBSCAN with appropriate parameters
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,  # Minimum points to form a cluster
                metric='cosine',     # Use cosine distance for semantic similarity
                cluster_selection_epsilon=0.0,  # Use default cluster selection
                min_samples=None     # Use min_cluster_size as default
            )
            
            # Perform clustering
            cluster_labels = clusterer.fit_predict(embeddings_matrix)
            
            self.logger.info(f"HDBSCAN clustering: found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters, "
                           f"{np.sum(cluster_labels == -1)} noise points")
            
            return cluster_labels
            
        except ImportError:
            self.logger.error("hdbscan library not available. Install with: pip install hdbscan")
            raise Exception("HDBSCAN clustering requires hdbscan library")
        except Exception as e:
            self.logger.error(f"HDBSCAN clustering failed: {e}")
            raise
    
    def _calculate_dbcv_score(self, embeddings_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate Density-Based Clustering Validation (DBCV) score."""
        try:
            import hdbscan
            
            # Calculate DBCV score using hdbscan's built-in validation
            dbcv_score = hdbscan.validity.validity_index(embeddings_matrix, cluster_labels, metric='cosine')
            
            return dbcv_score if dbcv_score is not None else 0.0
            
        except ImportError:
            self.logger.warning("DBCV score requires hdbscan library")
            return 0.0
        except Exception as e:
            self.logger.warning(f"DBCV calculation failed: {e}")
            return 0.0

    async def _create_semantic_clusters(
        self,
        globules: List[ProcessedGlobule],
        cluster_labels: np.ndarray,
        embeddings_matrix: np.ndarray,
        globule_map: Dict[int, ProcessedGlobule]
    ) -> List[SemanticCluster]:
        """Create SemanticCluster objects with intelligent labeling."""
        
        clusters = []
        unique_labels = set(cluster_labels)
        noise_globules = []
        
        for cluster_id in unique_labels:
            # Handle noise points separately
            if cluster_id == -1:
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                noise_globules = [globule_map[i] for i in cluster_indices]
                self.logger.info(f"Found {len(noise_globules)} noise points (outliers)")
                continue
            # Get globules in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_globules = [globule_map[i] for i in cluster_indices]
            
            if len(cluster_globules) < self.min_cluster_size:
                continue  # Skip very small clusters
            
            # Calculate cluster centroid
            cluster_embeddings = embeddings_matrix[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find the most representative globule (closest to centroid)
            representative_globule = self._find_representative_globule(cluster_globules, cluster_embeddings, centroid)
            
            # Use the representative globule for cluster metadata
            label = self._create_cluster_label_from_globule(representative_globule)
            description = f"Theme represented by: {representative_globule.text[:100]}{'...' if len(representative_globule.text) > 100 else ''}"
            keywords = self._extract_cluster_keywords(cluster_globules)
            domains = self._analyze_cluster_domains(cluster_globules)
            confidence = self._calculate_cluster_confidence(cluster_globules, cluster_embeddings)
            representative_samples = [representative_globule.text] + [g.text for g in cluster_globules[:2] if g != representative_globule]
            theme_analysis = self._analyze_cluster_themes(cluster_globules)
            
            cluster = SemanticCluster(
                id=f"cluster_{cluster_id}",
                label=label,
                description=description,
                size=len(cluster_globules),
                centroid=centroid,
                member_ids=[g.id for g in cluster_globules],
                keywords=keywords,
                domains=domains,
                confidence_score=confidence,
                representative_samples=representative_samples,
                theme_analysis=theme_analysis
            )
            
            clusters.append(cluster)
        
        # Handle noise points by creating an "Uncategorized" cluster if any exist
        if noise_globules:
            noise_cluster = SemanticCluster(
                id="noise_cluster",
                label="Uncategorized",
                description=f"Outliers and noise points that don't fit into any theme ({len(noise_globules)} items)",
                size=len(noise_globules),
                centroid=np.zeros(embeddings_matrix.shape[1]) if len(embeddings_matrix) > 0 else np.array([]),
                member_ids=[g.id for g in noise_globules],
                keywords=["outlier", "uncategorized"],
                domains=["mixed"],
                confidence_score=0.1,  # Low confidence for noise
                representative_samples=[g.text[:100] + "..." if len(g.text) > 100 else g.text for g in noise_globules[:3]],
                theme_analysis={"temporal": {}, "content": {"avg_length": np.mean([len(g.text) for g in noise_globules])}}
            )
            clusters.append(noise_cluster)
        
        # Sort clusters by size (largest first) 
        clusters.sort(key=lambda c: c.size, reverse=True)
        
        return clusters

    def _find_representative_globule(
        self, 
        cluster_globules: List[ProcessedGlobule], 
        cluster_embeddings: np.ndarray, 
        centroid: np.ndarray
    ) -> ProcessedGlobule:
        """
        Find the globule whose embedding is closest to the cluster centroid.
        
        Uses cosine distance (not Euclidean) because in high-dimensional embedding
        spaces, cosine similarity is a better measure of semantic closeness.
        
        This gives us a real, user-written note as the cluster representative
        instead of an abstract mathematical centroid.
        """
        min_distance = float('inf')
        representative_globule = cluster_globules[0]  # fallback
        
        for i, globule in enumerate(cluster_globules):
            # Calculate cosine distance to centroid (1 - cosine similarity)
            embedding = cluster_embeddings[i]
            
            # Normalize vectors for cosine similarity calculation
            norm_embedding = embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
            norm_centroid = centroid / np.linalg.norm(centroid) if np.linalg.norm(centroid) > 0 else centroid
            
            # Cosine distance = 1 - cosine similarity
            cosine_similarity = np.dot(norm_embedding, norm_centroid)
            cosine_distance = 1 - cosine_similarity
            
            if cosine_distance < min_distance:
                min_distance = cosine_distance
                representative_globule = globule
        
        return representative_globule
    
    def _create_cluster_label_from_globule(self, globule: ProcessedGlobule) -> str:
        """
        Create a meaningful cluster label from the representative globule.
        
        Uses the globule's parsed title if available, otherwise creates one
        from the first part of the text.
        """
        # Try to use parsed title first
        if globule.parsed_data and 'title' in globule.parsed_data:
            title = globule.parsed_data['title']
            if len(title.strip()) > 0:
                return title.strip()
        
        # Fallback: create title from text
        text = globule.text.strip()
        if len(text) == 0:
            return "Empty Theme"
        
        # Use first sentence or first 50 characters, whichever is shorter
        sentences = text.split('.')
        first_sentence = sentences[0].strip()
        
        if len(first_sentence) <= 50:
            return first_sentence
        else:
            # Truncate at word boundary
            words = first_sentence.split()
            truncated = []
            char_count = 0
            
            for word in words:
                if char_count + len(word) + 1 > 47:  # Leave room for "..."
                    break
                truncated.append(word)
                char_count += len(word) + 1
            
            return ' '.join(truncated) + "..."

    async def _generate_cluster_label(self, globules: List[ProcessedGlobule]) -> Tuple[str, str]:
        """Generate intelligent label and description for a cluster."""
        
        # Analyze common themes in the cluster
        all_keywords = []
        all_domains = []
        all_categories = []
        
        for globule in globules:
            if globule.parsed_data:
                all_keywords.extend(globule.parsed_data.get('keywords', []))
                all_domains.append(globule.parsed_data.get('domain', 'other'))
                all_categories.append(globule.parsed_data.get('category', 'note'))
        
        # Find most common keywords, domains, categories
        top_keywords = [word for word, count in Counter(all_keywords).most_common(3)]
        top_domain = Counter(all_domains).most_common(1)[0][0] if all_domains else 'mixed'
        top_category = Counter(all_categories).most_common(1)[0][0] if all_categories else 'thoughts'
        
        # Generate label based on content analysis
        if top_keywords:
            # Use top keywords to create meaningful label
            if len(top_keywords) >= 2:
                label = f"{top_keywords[0].title()} & {top_keywords[1].title()}"
            else:
                label = f"{top_keywords[0].title()} {top_category.title()}"
        else:
            # Fallback to domain/category based label
            label = f"{top_domain.title()} {top_category.title()}"
        
        # Generate description
        size = len(globules)
        description = f"A cluster of {size} {top_category}s primarily focused on {top_domain} themes"
        
        if top_keywords:
            description += f", with key concepts: {', '.join(top_keywords[:3])}"
        
        return label, description

    def _extract_cluster_keywords(self, globules: List[ProcessedGlobule]) -> List[str]:
        """Extract the most representative keywords for a cluster."""
        all_keywords = []
        
        for globule in globules:
            if globule.parsed_data and 'keywords' in globule.parsed_data:
                all_keywords.extend(globule.parsed_data['keywords'])
        
        # Return top 5 most common keywords
        keyword_counts = Counter(all_keywords)
        return [word for word, count in keyword_counts.most_common(5)]

    def _analyze_cluster_domains(self, globules: List[ProcessedGlobule]) -> List[str]:
        """Analyze the domains represented in a cluster."""
        domains = []
        
        for globule in globules:
            if globule.parsed_data and 'domain' in globule.parsed_data:
                domains.append(globule.parsed_data['domain'])
        
        # Return unique domains, sorted by frequency
        domain_counts = Counter(domains)
        return [domain for domain, count in domain_counts.most_common()]

    def _calculate_cluster_confidence(
        self, 
        globules: List[ProcessedGlobule], 
        embeddings: np.ndarray
    ) -> float:
        """Calculate confidence score for cluster quality."""
        
        # Factor 1: Embedding coherence (how similar are the embeddings?)
        if len(embeddings) > 1:
            centroid = np.mean(embeddings, axis=0)
            similarities = [np.dot(emb, centroid) for emb in embeddings]
            embedding_coherence = np.mean(similarities)
        else:
            embedding_coherence = 1.0
        
        # Factor 2: Content quality (parsing confidence)
        parsing_confidences = [g.parsing_confidence for g in globules if g.parsing_confidence > 0]
        content_quality = np.mean(parsing_confidences) if parsing_confidences else 0.5
        
        # Factor 3: Size factor (larger clusters are generally more reliable)
        size_factor = min(1.0, len(globules) / 10)  # Plateau at 10 members
        
        # Combine factors
        confidence = (embedding_coherence * 0.5 + content_quality * 0.3 + size_factor * 0.2)
        
        return min(1.0, max(0.0, confidence))

    def _select_representative_samples(self, globules: List[ProcessedGlobule]) -> List[str]:
        """Select representative text samples from the cluster."""
        
        # Sort by parsing confidence and select top examples
        sorted_globules = sorted(globules, key=lambda g: g.parsing_confidence, reverse=True)
        
        # Take up to 3 representative samples
        samples = []
        for globule in sorted_globules[:3]:
            # Truncate long texts for readability
            sample = globule.text[:100] + "..." if len(globule.text) > 100 else globule.text
            samples.append(sample)
        
        return samples

    def _analyze_cluster_themes(self, globules: List[ProcessedGlobule]) -> Dict[str, Any]:
        """Perform deeper thematic analysis of the cluster."""
        
        # Analyze temporal patterns
        creation_dates = [g.created_at for g in globules if g.created_at]
        
        temporal_analysis = {}
        if creation_dates:
            temporal_analysis = {
                "earliest": min(creation_dates).isoformat(),
                "latest": max(creation_dates).isoformat(),
                "span_days": (max(creation_dates) - min(creation_dates)).days,
                "recent_activity": sum(1 for d in creation_dates if d > datetime.now() - timedelta(days=7))
            }
        
        # Analyze content characteristics
        content_analysis = {
            "avg_length": np.mean([len(g.text) for g in globules]),
            "total_words": sum(len(g.text.split()) for g in globules),
            "sentiment_distribution": self._analyze_sentiment_distribution(globules),
            "category_distribution": self._analyze_category_distribution(globules)
        }
        
        return {
            "temporal": temporal_analysis,
            "content": content_analysis,
            "cluster_density": self._calculate_cluster_density(globules),
            "cross_domain_score": self._calculate_cross_domain_score(globules)
        }

    def _analyze_sentiment_distribution(self, globules: List[ProcessedGlobule]) -> Dict[str, int]:
        """Analyze sentiment distribution in the cluster."""
        sentiments = []
        
        for globule in globules:
            if globule.parsed_data and 'metadata' in globule.parsed_data:
                sentiment = globule.parsed_data['metadata'].get('sentiment', 'neutral')
                sentiments.append(sentiment)
        
        return dict(Counter(sentiments))

    def _analyze_category_distribution(self, globules: List[ProcessedGlobule]) -> Dict[str, int]:
        """Analyze category distribution in the cluster."""
        categories = []
        
        for globule in globules:
            if globule.parsed_data:
                category = globule.parsed_data.get('category', 'note')
                categories.append(category)
        
        return dict(Counter(categories))

    def _calculate_cluster_density(self, globules: List[ProcessedGlobule]) -> float:
        """Calculate how tightly clustered the content is."""
        # For now, return a placeholder based on size
        # In future versions, this could use embedding distances
        size = len(globules)
        return min(1.0, size / 20)  # Density increases with size, plateaus at 20

    def _calculate_cross_domain_score(self, globules: List[ProcessedGlobule]) -> float:
        """Calculate how much this cluster spans multiple domains."""
        domains = set()
        
        for globule in globules:
            if globule.parsed_data:
                domain = globule.parsed_data.get('domain', 'other')
                domains.add(domain)
        
        # Higher score = more cross-domain (potentially more interesting)
        return len(domains) / max(1, len(set(['creative', 'technical', 'personal', 'academic', 'business', 'philosophy'])))

    def _analyze_cross_cluster_relationships(self, clusters: List[SemanticCluster]) -> Dict[str, List[str]]:
        """Analyze relationships between different clusters."""
        relationships = {}
        
        for i, cluster_a in enumerate(clusters):
            related_clusters = []
            
            for j, cluster_b in enumerate(clusters):
                if i != j:
                    # Calculate centroid similarity
                    similarity = np.dot(cluster_a.centroid, cluster_b.centroid)
                    
                    # Also check keyword overlap
                    keyword_overlap = len(set(cluster_a.keywords) & set(cluster_b.keywords))
                    
                    # Check domain overlap
                    domain_overlap = len(set(cluster_a.domains) & set(cluster_b.domains))
                    
                    # Combined relationship score
                    relationship_score = similarity * 0.6 + (keyword_overlap / 5) * 0.3 + (domain_overlap / 3) * 0.1
                    
                    if relationship_score > 0.3:  # Threshold for "related"
                        related_clusters.append(cluster_b.id)
            
            relationships[cluster_a.id] = related_clusters
        
        return relationships

    def _analyze_temporal_patterns(
        self, 
        globules: List[ProcessedGlobule], 
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in the clustering."""
        
        # Group by time periods
        now = datetime.now()
        recent_window = now - timedelta(days=7)
        medium_window = now - timedelta(days=30)
        
        temporal_stats = {
            "recent_clusters": set(),
            "active_clusters": set(),
            "historical_clusters": set(),
            "temporal_distribution": {}
        }
        
        for i, globule in enumerate(globules):
            if globule.created_at:
                cluster_id = f"cluster_{cluster_labels[i]}"
                
                if globule.created_at > recent_window:
                    temporal_stats["recent_clusters"].add(cluster_id)
                elif globule.created_at > medium_window:
                    temporal_stats["active_clusters"].add(cluster_id)
                else:
                    temporal_stats["historical_clusters"].add(cluster_id)
        
        # Convert sets to lists for JSON serialization
        temporal_stats["recent_clusters"] = list(temporal_stats["recent_clusters"])
        temporal_stats["active_clusters"] = list(temporal_stats["active_clusters"])
        temporal_stats["historical_clusters"] = list(temporal_stats["historical_clusters"])
        
        return temporal_stats

    def _calculate_quality_metrics(
        self,
        clusters: List[SemanticCluster],
        embeddings_matrix: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various quality metrics for the clustering."""
        
        metrics = {}
        
        # Basic metrics
        metrics["num_clusters"] = len(clusters)
        metrics["avg_cluster_size"] = np.mean([c.size for c in clusters]) if clusters else 0
        metrics["largest_cluster_size"] = max([c.size for c in clusters]) if clusters else 0
        metrics["smallest_cluster_size"] = min([c.size for c in clusters]) if clusters else 0
        
        # Confidence metrics
        metrics["avg_cluster_confidence"] = np.mean([c.confidence_score for c in clusters]) if clusters else 0
        metrics["high_confidence_clusters"] = sum(1 for c in clusters if c.confidence_score > 0.7)
        
        # Domain diversity
        all_domains = set()
        for cluster in clusters:
            all_domains.update(cluster.domains)
        metrics["domain_diversity"] = len(all_domains)
        
        # Cross-domain clusters (potentially interesting insights)
        metrics["cross_domain_clusters"] = sum(1 for c in clusters if len(c.domains) > 1)
        
        return metrics

    

    def _create_empty_analysis(self, total_globules: int) -> ClusteringAnalysis:
        """Create empty analysis when clustering is not possible."""
        return ClusteringAnalysis(
            clusters=[],
            total_globules=total_globules,
            clustering_method="insufficient_data",
            dbcv_score=0.0,
            processing_time_ms=0.0,
            cross_cluster_relationships={},
            temporal_patterns={},
            quality_metrics={"reason": "insufficient_globules_for_clustering"}
        )

    async def get_cluster_by_id(self, cluster_id: str) -> Optional[SemanticCluster]:
        """Retrieve a specific cluster by ID (would need caching/storage)."""
        # For now, this would need to re-run clustering
        # In production, we'd cache clustering results
        analysis = await self.analyze_semantic_clusters()
        
        for cluster in analysis.clusters:
            if cluster.id == cluster_id:
                return cluster
        
        return None

    async def find_globule_cluster(self, globule_id: str) -> Optional[SemanticCluster]:
        """Find which cluster a specific globule belongs to."""
        analysis = await self.analyze_semantic_clusters()
        
        for cluster in analysis.clusters:
            if globule_id in cluster.member_ids:
                return cluster
        
        return None
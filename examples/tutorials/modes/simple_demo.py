"""
Simple Demo Glass Engine Mode - Minimal Implementation

This is a simplified version of the demo mode that focuses on core functionality
without complex formatting to avoid syntax issues.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..glass_engine_core import AbstractGlassEngine, GlassEngineMode
from globule.core.models import EnrichedInput


class SimpleDemoGlassEngine(AbstractGlassEngine):
    """
    Simplified Demo Glass Engine implementation for technical showcases.
    
    This class provides a basic demonstration of Globule's capabilities
    with minimal formatting complexity.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the Simple Demo Glass Engine."""
        super().__init__(console)
        self.demo_scenarios = [
            {
                "category": "Creative Writing",
                "input": "The concept of 'progressive overload' in fitness could apply to creative stamina. Just as muscles grow stronger when gradually challenged, perhaps our creative capacity expands when we consistently push slightly beyond our comfort zone. Today I will on the ground, cross my legs, straighten my back, and think about absolutely nothing. This will last for approximately 1.2 seconds. Maybe tomorrow it will be 1.23 seconds.",
                "context": "Cross-domain thinking and metaphorical reasoning - showcasing creative domain detection"
            },
            {
                "category": "Technical Insight", 
                "input": "Instead of preventing all edge cases, design systems that gracefully degrade. When the unexpected happens, the system should fail in a predictable, controlled manner rather than catastrophically.",
                "context": "Systems thinking and resilience engineering - showcasing technical domain detection"
            },
            {
                "category": "Question Analysis",
                "input": "How can we measure the effectiveness of knowledge management systems in creative workflows?",
                "context": "Interrogative content analysis - showcasing question categorization"
            },
            {
                "category": "Personal Reflection",
                "input": "I feel like I'm constantly switching between different tools for note-taking, and it's becoming overwhelming. Need a unified system that actually works for my scattered thinking.",
                "context": "Personal domain classification and sentiment analysis"
            },
            {
                "category": "Fast Success Test",
                "input": "Quick test for LLM parsing speed.",
                "context": "Lightweight scenario designed to succeed with fast models - demonstrates full LLM integration when available"
            }
        ]
        
    def get_mode(self) -> GlassEngineMode:
        """Return the Demo Glass Engine mode."""
        return GlassEngineMode.DEMO
    
    async def execute_tutorial_flow(self) -> None:
        """Execute the simplified demo tutorial flow."""
        self.logger.info("Starting simplified demo showcase")
        
        # Phase 1: Welcome
        self.console.print(Panel.fit(
            "[bold blue]Globule Demo: Professional System Showcase[/bold blue]\n\n"
            "[dim]This demo showcases both SUCCESS and FAILURE paths:\n"
            "- LLM parsing (when Ollama + fast model available)\n"
            "- Intelligent fallback parsing (when offline)\n"
            "- Granular performance metrics for diagnosis[/dim]",
            title="Glass Engine Demo Mode"
        ))
        
        # Phase 2: Configuration
        await self._show_configuration()
        
        # Phase 3: Process demo scenarios
        await self._process_demo_scenarios()
        
        # Phase 4: Vector Search Demonstration
        await self._demonstrate_vector_search()
        
        # Phase 5: Clustering Demonstration
        await self._demonstrate_clustering()
        
        # Phase 6: Results
        self.console.print(Panel.fit(
            "[bold green]Demo Complete![/bold green]",
            title="Success"
        ))
        
        self.logger.info("Simplified demo showcase completed")
    
    async def _show_configuration(self) -> None:
        """Show basic system configuration."""
        self.console.print("\n[bold]System Configuration:[/bold]")
        
        config_table = Table(title="Current Settings")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        storage_dir = self.config.get_storage_dir()
        config_table.add_row("Storage Directory", str(storage_dir))
        config_table.add_row("Ollama URL", self.config.ollama_base_url)
        config_table.add_row("Embedding Model", self.config.default_embedding_model)
        config_table.add_row("Parsing Model", self.config.default_parsing_model)
        
        self.console.print(config_table)
        
        # Glass Engine educational note
        self.console.print("\n[dim]TIP: Glass Engine Tip: For fast SUCCESS demos, try:")
        self.console.print("[dim]   ollama pull tinyllama    # 650MB, very fast")
        self.console.print("[dim]   Then set default_parsing_model: 'tinyllama' in config.yaml[/dim]")
    
    async def _process_demo_scenarios(self) -> None:
        """Process the demo scenarios."""
        self.console.print("\n[bold]Processing Demo Scenarios:[/bold]")
        
        for i, scenario in enumerate(self.demo_scenarios, 1):
            self.console.print(f"\nScenario {i}: {scenario['category']}")
            self.console.print(f"Input: {scenario['input']}")
            
            try:
                # Create enriched input
                enriched_input = self.create_test_input(
                    scenario["input"], 
                    f"demo_scenario_{i}"
                )
                
                # Process with detailed timing breakdown
                self.console.print(f"[yellow]PROCESSING[/yellow] Starting globule processing...")
                
                total_start = datetime.now()
                result = await self.orchestrator.process_globule(enriched_input)
                processing_time = (datetime.now() - total_start).total_seconds() * 1000
                
                # Extract detailed timing from orchestration
                timing_data = result.processing_time_ms
                
                # Store the result with timing
                storage_start = datetime.now()
                globule_id = await self.storage.store_globule(result)
                storage_time = (datetime.now() - storage_start).total_seconds() * 1000
                
                # Show granular performance breakdown
                self.console.print(f"[green]SUCCESS[/green] Total processing: {processing_time:.1f}ms")
                
                # Detailed timing breakdown (your suggestion!)
                embed_time = timing_data.get('embedding_ms', 0)
                parse_time = timing_data.get('parsing_ms', 0)
                orchestration_time = timing_data.get('orchestration_ms', 0)
                
                self.console.print(f"[cyan]METRICS[/cyan] Embedding time: {embed_time:.1f}ms")
                self.console.print(f"[cyan]METRICS[/cyan] Parsing time: {parse_time:.1f}ms")
                self.console.print(f"[cyan]METRICS[/cyan] Storage time: {storage_time:.1f}ms")
                self.console.print(f"[cyan]METRICS[/cyan] Orchestration overhead: {orchestration_time:.1f}ms")
                
                self.console.print(f"[green]SUCCESS[/green] Stored as globule: {str(globule_id)[:8]}...")
                
                if result.embedding is not None:
                    self.console.print(f"[green]SUCCESS[/green] Generated {len(result.embedding)}-dimensional embedding")
                
                # Show intelligent parsing results (Phase 2 feature)
                if result.parsed_data:
                    parsed = result.parsed_data
                    self.console.print(f"[cyan]INTELLIGENCE[/cyan] Title: '{parsed.get('title', 'N/A')}'")
                    self.console.print(f"[cyan]INTELLIGENCE[/cyan] Domain: {parsed.get('domain', 'N/A')} | Category: {parsed.get('category', 'N/A')}")
                    
                    # Show keywords if available
                    keywords = parsed.get('keywords', [])
                    if keywords:
                        self.console.print(f"[cyan]INTELLIGENCE[/cyan] Keywords: {', '.join(keywords[:3])}")
                    
                    # Show metadata confidence and type
                    metadata = parsed.get('metadata', {})
                    if metadata:
                        parser_type = metadata.get('parser_type', 'unknown')
                        confidence = metadata.get('confidence_score', 0)
                        self.console.print(f"[cyan]INTELLIGENCE[/cyan] Parser: {parser_type} (confidence: {confidence:.2f})")
                        
                        if 'sentiment' in metadata:
                            self.console.print(f"[cyan]INTELLIGENCE[/cyan] Sentiment: {metadata['sentiment']}")
                
                # Record test result
                self.metrics.test_results.append({
                    "test": f"demo_scenario_{i}",
                    "input": scenario["input"],
                    "success": True,
                    "processing_time_ms": processing_time,
                    "globule_id": str(globule_id)
                })
                
            except Exception as e:
                self.console.print(f"[red]ERROR[/red]: {e}")
                self.metrics.add_error(e, f"demo_scenario_{i}")
                self.metrics.test_results.append({
                    "test": f"demo_scenario_{i}",
                    "input": scenario["input"],
                    "success": False,
                    "error": str(e)
                })

    async def _demonstrate_vector_search(self) -> None:
        """
        Phase 2: Demonstrate intelligent vector search capabilities.
        
        Shows semantic similarity search in action using the globules
        we just created in the demo scenarios.
        """
        self.console.print("\n[bold]Phase 2: Semantic Vector Search Demo[/bold]")
        self.console.print("[dim]Demonstrating intelligent discovery of related thoughts...[/dim]")
        
        try:
            # Use a search query that should find semantic relationships
            search_query = "creative development and growth"
            
            self.console.print(f"\n🔍 Search Query: '{search_query}'")
            
            # Generate embedding for search query
            self.console.print("[yellow]PROCESSING[/yellow] Generating search embedding...")
            search_embedding = await self.embedding_provider.embed(search_query)
            
            # Perform vector search
            self.console.print("[yellow]PROCESSING[/yellow] Searching semantic database...")
            search_results = await self.storage.search_by_embedding(
                search_embedding, 
                limit=3, 
                similarity_threshold=0.3
            )
            
            if search_results:
                self.console.print(f"[green]SUCCESS[/green] Found {len(search_results)} semantically related thoughts:")
                
                for i, (globule, similarity) in enumerate(search_results, 1):
                    similarity_pct = similarity * 100
                    similarity_bar = "█" * max(1, int(similarity * 15))
                    
                    self.console.print(f"\n{i}. [{similarity_pct:.1f}% {similarity_bar}]")
                    
                    # Show content preview
                    preview = globule.text[:80] + "..." if len(globule.text) > 80 else globule.text
                    self.console.print(f"   {preview}")
                    
                    # Show intelligence metadata
                    if globule.parsed_data:
                        domain = globule.parsed_data.get('domain', 'unknown')
                        category = globule.parsed_data.get('category', 'unknown')
                        self.console.print(f"   [cyan]INTELLIGENCE[/cyan] {domain}/{category}")
                        
                        keywords = globule.parsed_data.get('keywords', [])
                        if keywords:
                            self.console.print(f"   [cyan]KEYWORDS[/cyan] {', '.join(keywords[:3])}")
                
                self.console.print(f"\n[green]SUCCESS[/green] Vector search completed using {len(search_embedding)}-dimensional semantic space")
                
                # Record search success
                self.metrics.test_results.append({
                    "test": "vector_search_demo",
                    "query": search_query,
                    "results_found": len(search_results),
                    "success": True,
                    "avg_similarity": sum(score for _, score in search_results) / len(search_results)
                })
                
            else:
                self.console.print("[yellow]INFO[/yellow] No semantically similar content found (threshold too high)")
                self.console.print("[dim]This is normal with limited demo data - try 'globule search' with more content![/dim]")
                
                self.metrics.test_results.append({
                    "test": "vector_search_demo", 
                    "query": search_query,
                    "results_found": 0,
                    "success": True,
                    "note": "No results above similarity threshold - expected with limited data"
                })
                
        except Exception as e:
            self.console.print(f"[red]ERROR[/red] Vector search failed: {e}")
            self.metrics.add_error(e, "vector_search_demo")
            self.metrics.test_results.append({
                "test": "vector_search_demo",
                "success": False,
                "error": str(e)
            })
    
    async def _demonstrate_clustering(self) -> None:
        """Phase 2: Demonstrate semantic clustering."""
        self.console.print("\n[bold]Phase 2: Semantic Clustering Demo[/bold]")
        self.console.print("[dim]Demonstrating automatic theme detection...[/dim]")
        
        try:
            from globule.clustering.semantic_clustering import SemanticClusteringEngine
            clustering_engine = SemanticClusteringEngine(self.storage)
            
            self.console.print("\n[yellow]PROCESSING[/yellow] Analyzing semantic clusters...")
            analysis = await clustering_engine.analyze_semantic_clusters(min_globules=3)
            
            if analysis.clusters:
                self.console.print(f"[green]SUCCESS[/green] Discovered {len(analysis.clusters)} semantic clusters:")
                
                cluster_table = Table(title="Semantic Clusters")
                cluster_table.add_column("Cluster Label", style="cyan")
                cluster_table.add_column("Size", style="green")
                cluster_table.add_column("Keywords", style="yellow")
                cluster_table.add_column("Confidence", style="dim")
                
                for cluster in analysis.clusters:
                    cluster_table.add_row(
                        cluster.label,
                        str(cluster.size),
                        ', '.join(cluster.keywords[:3]),
                        f"{cluster.confidence_score:.2f}"
                    )
                
                self.console.print(cluster_table)
                self.console.print(f"Overall Quality (Silhouette Score): {analysis.silhouette_score:.3f}")

                self.metrics.test_results.append({
                    "test": "clustering_demo",
                    "clusters_found": len(analysis.clusters),
                    "success": True,
                    "silhouette_score": analysis.silhouette_score
                })
            else:
                self.console.print("[yellow]INFO[/yellow] Not enough related content to form distinct clusters.")
                self.metrics.test_results.append({
                    "test": "clustering_demo",
                    "clusters_found": 0,
                    "success": True,
                    "note": "Not enough data to form clusters."
                })

        except Exception as e:
            self.console.print(f"[red]ERROR[/red] Clustering failed: {e}")
            self.metrics.add_error(e, "clustering_demo")
            self.metrics.test_results.append({
                "test": "clustering_demo",
                "success": False,
                "error": str(e)
            })
    
    def present_results(self) -> None:
        """Present demo results in a simple format."""
        self.console.print("\n" + "=" * 60)
        self.console.print(Panel.fit(
            "[bold blue]Demo Results Summary[/bold blue]",
            title="Glass Engine Results"
        ))
        
        # Test results
        results_table = Table(title="Test Results")
        results_table.add_column("Test", style="cyan")
        results_table.add_column("Status", style="green") 
        results_table.add_column("Time (ms)", style="yellow")
        
        for result in self.metrics.test_results:
            status = "PASS" if result.get("success", False) else "FAIL"
            time_ms = result.get("processing_time_ms", 0)
            results_table.add_row(
                result.get("test", "unknown"),
                status,
                f"{time_ms:.1f}" if time_ms else "N/A"
            )
        
        self.console.print(results_table)
        
        # Summary
        success_count = sum(1 for r in self.metrics.test_results if r.get("success", False))
        total_count = len(self.metrics.test_results)
        
        self.console.print(f"\nOverall Success Rate: {success_count}/{total_count}")
        self.console.print(f"Total Duration: {self.metrics.total_duration_ms:.1f}ms")
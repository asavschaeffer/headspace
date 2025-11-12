"""
Enrichment Event System for Real-Time Streaming
Manages WebSocket connections for live embedding calculation feedback
"""

from typing import Dict, List, Set, Callable
from dataclasses import dataclass, asdict
import asyncio
import json
from datetime import datetime


@dataclass
class EnrichmentEvent:
    """Event emitted during chunk enrichment"""
    event_type: str  # "started", "chunk_enriched", "completed", "error"
    doc_id: str
    chunk_id: str = ""
    chunk_index: int = -1
    embedding: List[float] = None
    color: str = ""
    position_3d: List[float] = None
    progress: int = 0  # 0-100
    total_chunks: int = 0
    error: str = ""
    timestamp: str = ""

    def to_json(self):
        """Convert to JSON-serializable dict"""
        data = asdict(self)
        if self.timestamp == "":
            data["timestamp"] = datetime.now().isoformat()
        return data


class EnrichmentEventBus:
    """Manages enrichment event subscriptions and broadcasting"""

    def __init__(self):
        # doc_id -> Set[queue]
        self.subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self.active_enrichments: Dict[str, dict] = {}

    async def subscribe(self, doc_id: str) -> asyncio.Queue:
        """Subscribe to enrichment events for a document"""
        queue = asyncio.Queue()
        if doc_id not in self.subscribers:
            self.subscribers[doc_id] = set()
        self.subscribers[doc_id].add(queue)
        return queue

    async def unsubscribe(self, doc_id: str, queue: asyncio.Queue):
        """Unsubscribe from enrichment events"""
        if doc_id in self.subscribers:
            self.subscribers[doc_id].discard(queue)
            if not self.subscribers[doc_id]:
                del self.subscribers[doc_id]

    async def emit(self, event: EnrichmentEvent):
        """Broadcast an enrichment event to all subscribers"""
        if event.doc_id not in self.subscribers:
            return

        # Put event in all subscriber queues (non-blocking)
        for queue in list(self.subscribers[event.doc_id]):  # Copy list to avoid modification during iteration
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Queue is full, skip this subscriber (they're too slow)
                pass

    def get_active_enrichments(self) -> Dict[str, dict]:
        """Get status of all active enrichments"""
        return self.active_enrichments.copy()

    def set_enrichment_progress(self, doc_id: str, progress: int, total: int):
        """Update enrichment progress"""
        if doc_id not in self.active_enrichments:
            self.active_enrichments[doc_id] = {}
        self.active_enrichments[doc_id]["progress"] = progress
        self.active_enrichments[doc_id]["total"] = total

    def clear_enrichment(self, doc_id: str):
        """Clear enrichment progress when complete"""
        if doc_id in self.active_enrichments:
            del self.active_enrichments[doc_id]


# Global event bus instance
enrichment_event_bus = EnrichmentEventBus()

// Global State Management

class AppState {
    constructor() {
        this.currentDocument = null;
        this.documents = [];
        this.chunks = [];
        this.connections = [];
        this.currentView = 'document';
        this.parentChunkId = null;
        this.parentChunkInfo = null;
        this.selectedChunk = null;
        this.hoveredChunk = null;
    }

    setCurrentDocument(doc) {
        this.currentDocument = doc;
    }

    setDocuments(docs) {
        this.documents = docs;
    }

    setChunks(chunks) {
        this.chunks = chunks;
    }

    setConnections(connections) {
        this.connections = connections;
    }

    setCurrentView(view) {
        this.currentView = view;
    }

    setParentChunk(chunkId, chunkInfo) {
        this.parentChunkId = chunkId;
        this.parentChunkInfo = chunkInfo;
    }

    clearParentChunk() {
        this.parentChunkId = null;
        this.parentChunkInfo = null;
    }

    setSelectedChunk(chunk) {
        this.selectedChunk = chunk;
    }

    setHoveredChunk(chunk) {
        this.hoveredChunk = chunk;
    }
}

// Export singleton instance
export const state = new AppState();

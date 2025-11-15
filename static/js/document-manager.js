// Document Management - Fetches data for the cosmos view

import { state } from './state.js';
import { fetchDocuments } from './api.js';
import { updateStatus } from './utils.js';
import { updateCosmosData } from './cosmos-renderer.js';

export async function loadDocuments() {
    updateStatus('Loading documents...');
    try {
        const docs = await fetchDocuments();
        // In our new simplified model, documents are the planets.
        // We will treat them like the "chunks" of the old system for rendering.
        state.setChunks(docs); 
        
        updateCosmosData();
        updateStatus('Cosmos populated');

    } catch (error) {
        console.error('Failed to load documents:', error);
        updateStatus('Failed to connect to data source');
    }
}

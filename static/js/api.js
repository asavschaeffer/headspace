// API Layer - Centralized fetch wrapper

import { API_BASE } from './config.js';

export async function fetchDocuments() {
    const response = await fetch(`${API_BASE}/documents`);
    return await response.json();
}

export async function fetchDocument(docId) {
    const response = await fetch(`${API_BASE}/documents/${docId}`);
    return await response.json();
}

export async function createDocument(title, content, docType) {
    const response = await fetch(`${API_BASE}/documents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, content, doc_type: docType })
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

export async function deleteDocument(docId) {
    const response = await fetch(`${API_BASE}/documents/${docId}`, {
        method: 'DELETE'
    });
    return await response.json();
}

export async function attachDocumentToChunk(chunkId, documentId) {
    const response = await fetch(`${API_BASE}/chunks/${chunkId}/attach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: documentId })
    });

    if (response.ok) {
        return await response.json();
    } else {
        throw new Error('Failed to attach document');
    }
}

export async function fetchChunkAttachments(chunkId) {
    const response = await fetch(`${API_BASE}/chunks/${chunkId}/attachments`);
    return await response.json();
}

export async function removeChunkAttachment(chunkId, documentId) {
    const response = await fetch(`${API_BASE}/chunks/${chunkId}/attach/${documentId}`, {
        method: 'DELETE'
    });
    return await response.json();
}

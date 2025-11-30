// Document Management - CRUD operations and rendering

import { state } from './state.js';
import { fetchDocuments, fetchDocument } from './api.js';
import { updateStatus } from './utils.js';
import { renderDocument } from './chunk-renderer.js';
import { updateCosmosData } from './cosmos-renderer.js';

export async function loadDocuments() {
    try {
        const docs = await fetchDocuments();
        state.setDocuments(docs);
        renderDocumentList();

        if (docs.length > 0 && !state.currentDocument) {
            await loadDocument(docs[0].id);
        }
    } catch (error) {
        console.error('Failed to load documents:', error);
        updateStatus('Failed to connect to server');
    }
}

export async function loadDocument(docId) {
    updateStatus('Loading document...');

    try {
        const data = await fetchDocument(docId);

        state.setCurrentDocument(data.document);
        state.setChunks(data.chunks);
        state.setConnections(data.connections);

        // Update UI
        document.querySelectorAll('.document-item').forEach(item => {
            item.classList.toggle('active', item.dataset.docId === docId);
        });

        renderDocument();
        updateCosmosData();
        updateStatus('Document loaded');
    } catch (error) {
        console.error('Failed to load document:', error);
        updateStatus('Failed to load document');
    }
}

export function renderDocumentList() {
    const container = document.getElementById('document-list');
    container.innerHTML = '';

    state.documents.forEach(doc => {
        const item = document.createElement('div');
        item.className = 'document-item';
        item.dataset.docId = doc.id;
        item.onclick = () => loadDocument(doc.id);

        item.innerHTML = `
            <div class="document-title">${doc.title}</div>
            <div class="document-meta">${doc.chunk_count} chunks • ${doc.doc_type}</div>
        `;

        container.appendChild(item);
    });
}

export function updateBreadcrumb() {
    const breadcrumbEl = document.getElementById('breadcrumb');
    const cosmosBreadcrumbEl = document.getElementById('cosmos-breadcrumb');

    if (!state.currentDocument) {
        breadcrumbEl.style.display = 'none';
        cosmosBreadcrumbEl.style.display = 'none';
        return;
    }

    const breadcrumbHTML = `
        <span class="breadcrumb-item" onclick="window.appFunctions.loadDocuments(); window.appFunctions.switchView('document')">All Documents</span>
        <span class="breadcrumb-separator">›</span>
        <span class="breadcrumb-current">${state.currentDocument.title}</span>
    `;

    breadcrumbEl.style.display = 'block';
    breadcrumbEl.innerHTML = breadcrumbHTML;

    cosmosBreadcrumbEl.style.display = 'block';
    cosmosBreadcrumbEl.innerHTML = breadcrumbHTML;
}

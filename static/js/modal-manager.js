// Modal Management - Add Document Modal

import { state } from './state.js';
import { createDocument, attachDocumentToChunk } from './api.js';
import { loadDocuments, loadDocument } from './document-manager.js';
import { updateStatus } from './utils.js';

export function showAddModal(chunkId = null) {
    document.getElementById('add-modal').classList.add('active');

    // Clear previous values
    document.getElementById('doc-title-input').value = '';
    document.getElementById('doc-content-input').value = '';
    document.getElementById('doc-signature-input').value = '';
    document.getElementById('file-upload-input').value = '';
    document.getElementById('doc-type-select').value = 'text';

    // Set parent chunk if provided
    if (chunkId) {
        const chunk = state.chunks.find(c => c.id === chunkId);
        if (chunk) {
            state.setParentChunk(chunkId, chunk);
            document.getElementById('parent-node-container').style.display = 'block';
            document.getElementById('parent-node-display').textContent =
                `${chunk.chunk_type} • "${chunk.content.substring(0, 50)}${chunk.content.length > 50 ? '...' : ''}"`;
        }
    } else {
        clearParentNode();
    }
}

export function hideAddModal() {
    document.getElementById('add-modal').classList.remove('active');
    clearParentNode();
    document.getElementById('doc-title-input').value = '';
    document.getElementById('doc-content-input').value = '';
    document.getElementById('doc-signature-input').value = '';
    document.getElementById('file-upload-input').value = '';
}

export function clearParentNode() {
    state.clearParentChunk();
    document.getElementById('parent-node-container').style.display = 'none';
    document.getElementById('parent-node-display').textContent = 'None';
}

export async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        const content = await file.text();

        const titleInput = document.getElementById('doc-title-input');
        const contentInput = document.getElementById('doc-content-input');
        const typeSelect = document.getElementById('doc-type-select');

        if (!titleInput.value) {
            titleInput.value = file.name.replace(/\.[^/.]+$/, "");
        }

        contentInput.value = content;

        if (file.name.match(/\.(py|js|rs|java|cpp|c|ts)$/)) {
            typeSelect.value = 'code';
        } else {
            typeSelect.value = 'text';
        }
    } catch (error) {
        console.error('Error reading file:', error);
        alert('Failed to read file: ' + error.message);
    }
}

export async function addDocument() {
    const titleInput = document.getElementById('doc-title-input');
    const contentInput = document.getElementById('doc-content-input');
    const signatureInput = document.getElementById('doc-signature-input');
    const docTypeSelect = document.getElementById('doc-type-select');

    let title = titleInput.value.trim();
    const content = contentInput.value.trim();
    const signature = signatureInput ? signatureInput.value.trim() : '';
    const docType = docTypeSelect ? docTypeSelect.value : 'text';

    if (!content) {
        alert('Please enter some content');
        return;
    }

    // Set title to untitled if not provided
    if (!title) {
        title = 'Untitled Thought';
    }

    // Append signature to content if provided
    let finalContent = content;
    if (signature) {
        finalContent = `${content}\n\n— ${signature}`;
    }

    updateStatus('Processing thought...');

    try {
        const result = await createDocument(title, finalContent, docType);

        // If there's a parent chunk, attach the new document to it
        if (state.parentChunkId) {
            updateStatus('Attaching to parent chunk...');
            await attachDocumentToChunk(state.parentChunkId, result.id);
        }

        hideAddModal();
        updateStatus('Reloading documents...');
        await loadDocuments();

        await loadDocument(result.id);

        updateStatus(state.parentChunkId ? 'Document added and attached!' : 'Document added successfully!');
    } catch (error) {
        console.error('Failed to add document:', error);
        alert('Failed to add document: ' + error.message);
        updateStatus('Failed to add document');
    }
}

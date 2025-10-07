// Chunk Rendering and Interactions

import { state } from './state.js';
import { updateBreadcrumb } from './document-manager.js';
import { hexToRgb, showChunkTooltip, hideChunkTooltip } from './utils.js';
import { focusChunkInCosmos } from './cosmos-renderer.js';
import { showAddModal } from './modal-manager.js';
import { fetchChunkAttachments } from './api.js';
import { loadDocument } from './document-manager.js';

export function renderDocument() {
    if (!state.currentDocument) return;

    updateBreadcrumb();

    // Update header
    document.getElementById('doc-title').textContent = state.currentDocument.title;
    document.getElementById('doc-meta').textContent =
        `${state.chunks.length} chunks • ${state.currentDocument.doc_type} • ${new Date(state.currentDocument.created_at).toLocaleDateString()}`;

    // Render chunks
    const container = document.getElementById('chunks-container');
    container.innerHTML = '';

    state.chunks.forEach((chunk, index) => {
        const chunkEl = document.createElement('div');
        chunkEl.className = `chunk ${chunk.chunk_type}`;
        chunkEl.dataset.chunkId = chunk.id;
        chunkEl.style.borderLeftColor = chunk.color;

        // Apply subtle background based on embedding color
        const rgb = hexToRgb(chunk.color);
        if (rgb) {
            chunkEl.style.background = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.03)`;
        }

        // Attachment badge
        const attachmentCount = chunk.attachments ? chunk.attachments.length : 0;
        const attachmentBadge = document.createElement('div');
        attachmentBadge.className = attachmentCount > 0 ? 'attachment-badge has-attachments' : 'attachment-badge';
        attachmentBadge.textContent = attachmentCount > 0 ? attachmentCount : '+';
        attachmentBadge.onclick = (e) => {
            e.stopPropagation();
            toggleAttachments(chunk.id);
        };
        chunkEl.appendChild(attachmentBadge);

        // Chunk ID (shows on hover)
        const chunkId = document.createElement('div');
        chunkId.className = 'chunk-id';
        chunkId.textContent = chunk.id;
        chunkEl.appendChild(chunkId);

        const content = document.createElement('div');
        content.className = 'chunk-content';
        content.textContent = chunk.content;
        chunkEl.appendChild(content);

        // Attachments panel
        const attachmentsPanel = document.createElement('div');
        attachmentsPanel.className = 'attachments-panel';
        attachmentsPanel.id = `attachments-${chunk.id}`;
        attachmentsPanel.innerHTML = `
            <div class="attachments-title">Attached Documents</div>
            <div class="attachments-list" id="attachments-list-${chunk.id}">
                ${attachmentCount > 0 ? '' : '<div style="color: #666; font-size: 12px;">No attachments yet</div>'}
            </div>
            <button class="attach-new-btn" data-chunk-id="${chunk.id}">+ Attach Document</button>
        `;
        chunkEl.appendChild(attachmentsPanel);

        const badge = document.createElement('div');
        badge.className = 'chunk-badge';
        badge.innerHTML = `
            <button class="cosmos-btn" data-chunk-id="${chunk.id}" title="View in cosmos">
                🪐
            </button>
        `;
        chunkEl.appendChild(badge);

        // Hover tooltip
        chunkEl.onmouseenter = (e) => showChunkTooltip(e, chunk);
        chunkEl.onmouseleave = () => hideChunkTooltip();

        container.appendChild(chunkEl);
    });

    // Add event listeners using delegation
    setupChunkEventListeners();
}

function setupChunkEventListeners() {
    const container = document.getElementById('chunks-container');

    // Chunk click - focus in cosmos
    container.addEventListener('click', (e) => {
        const chunkEl = e.target.closest('.chunk');
        if (chunkEl && !e.target.closest('.cosmos-btn') && !e.target.closest('.attachment-badge') && !e.target.closest('.attach-new-btn')) {
            const chunkId = chunkEl.dataset.chunkId;
            focusChunkInCosmos(chunkId);
        }
    });

    // Cosmos button click
    container.addEventListener('click', (e) => {
        const cosmosBtn = e.target.closest('.cosmos-btn');
        if (cosmosBtn) {
            e.stopPropagation();
            const chunkId = cosmosBtn.dataset.chunkId;
            focusChunkInCosmos(chunkId);
        }
    });

    // Attach button click
    container.addEventListener('click', (e) => {
        const attachBtn = e.target.closest('.attach-new-btn');
        if (attachBtn) {
            e.stopPropagation();
            const chunkId = attachBtn.dataset.chunkId;
            showAddModal(chunkId);
        }
    });
}

export async function toggleAttachments(chunkId) {
    const panel = document.getElementById(`attachments-${chunkId}`);
    if (!panel) return;

    const isVisible = panel.classList.contains('visible');

    // Close all other panels
    document.querySelectorAll('.attachments-panel').forEach(p => {
        p.classList.remove('visible');
    });

    if (!isVisible) {
        panel.classList.add('visible');
        await loadAttachments(chunkId);
    }
}

async function loadAttachments(chunkId) {
    try {
        const attachments = await fetchChunkAttachments(chunkId);

        const listEl = document.getElementById(`attachments-list-${chunkId}`);
        if (!listEl) return;

        if (attachments.length === 0) {
            listEl.innerHTML = '<div style="color: #666; font-size: 12px;">No attachments yet</div>';
        } else {
            listEl.innerHTML = attachments.map(doc => `
                <div class="attachment-item" data-doc-id="${doc.id}">
                    <div class="attachment-title">${doc.title}</div>
                    <div class="attachment-preview">${doc.content.substring(0, 100)}...</div>
                </div>
            `).join('');

            // Add click handlers to attachment items
            listEl.querySelectorAll('.attachment-item').forEach(item => {
                item.onclick = () => loadDocument(item.dataset.docId);
            });
        }
    } catch (error) {
        console.error('Failed to load attachments:', error);
    }
}

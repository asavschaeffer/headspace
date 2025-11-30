// Main Application Entry Point

import { loadDocuments } from './document-manager.js';
import { initCosmos, switchView } from './cosmos-renderer.js';
import { showAddModal, hideAddModal, clearParentNode, handleFileUpload, addDocument } from './modal-manager.js';

// Initialize app
async function init() {
    console.log('🏁 App initializing...');
    try {
        await loadDocuments();
        console.log('📚 Documents loaded');

        initCosmos();
        console.log('✨ Initialization complete');
    } catch (error) {
        console.error('❌ Initialization error:', error);
    }
}

// Global functions that need to be accessible from HTML
// (until we fully migrate to event delegation)
window.appFunctions = {
    loadDocuments,
    switchView,
    showAddModal,
    hideAddModal,
    clearParentNode,
    handleFileUpload,
    addDocument
};

// Start app when DOM is ready
window.addEventListener('DOMContentLoaded', init);

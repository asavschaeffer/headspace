// Main Application Entry Point

import { loadDocuments } from './document-manager.js';
import { initCosmos } from './cosmos-renderer.js';

// Initialize app
async function init() {
    console.log('🏁 App initializing...');
    try {
        initCosmos();
        console.log('✨ Cosmos initialized');
        
        await loadDocuments();
        console.log('📚 Documents loaded and rendered');

    } catch (error) {
        console.error('❌ Initialization error:', error);
    }
}

// Start app when DOM is ready
window.addEventListener('DOMContentLoaded', init);

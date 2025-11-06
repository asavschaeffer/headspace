// Storage Mode Manager
// Handles switching between local and cloud storage

let currentStorageMode = localStorage.getItem('storageMode') || 'local';

export function initStorageSelector() {
    const localBtn = document.getElementById('storage-local');
    const cloudBtn = document.getElementById('storage-cloud');
    const statusEl = document.getElementById('storage-status');
    
    if (!localBtn || !cloudBtn || !statusEl) return;
    
    // Set initial state
    updateStorageUI();
    
    // Add click handlers
    localBtn.addEventListener('click', () => {
        setStorageMode('local');
    });
    
    cloudBtn.addEventListener('click', () => {
        // Check if cloud is available
        checkCloudAvailability().then(available => {
            if (available) {
                setStorageMode('cloud');
            } else {
                alert('Cloud storage not configured. Please set SUPABASE_URL and SUPABASE_KEY environment variables.');
            }
        });
    });
}

function setStorageMode(mode) {
    currentStorageMode = mode;
    localStorage.setItem('storageMode', mode);
    updateStorageUI();
    
    // Reload documents with new storage mode
    if (window.appFunctions && window.appFunctions.loadDocuments) {
        window.appFunctions.loadDocuments();
    }
}

function updateStorageUI() {
    const localBtn = document.getElementById('storage-local');
    const cloudBtn = document.getElementById('storage-cloud');
    const statusEl = document.getElementById('storage-status');
    
    if (!localBtn || !cloudBtn || !statusEl) return;
    
    // Update button states
    localBtn.classList.toggle('active', currentStorageMode === 'local');
    cloudBtn.classList.toggle('active', currentStorageMode === 'cloud');
    
    // Update status text
    if (currentStorageMode === 'local') {
        statusEl.textContent = 'Using local storage';
    } else {
        statusEl.textContent = 'Using cloud storage';
    }
}

async function checkCloudAvailability() {
    try {
        const response = await fetch('/api/storage/status');
        const data = await response.json();
        return data.cloud_available || false;
    } catch (error) {
        console.error('Error checking cloud availability:', error);
        return false;
    }
}

export function getStorageMode() {
    return currentStorageMode;
}

export function isCloudMode() {
    return currentStorageMode === 'cloud';
}

export function isLocalMode() {
    return currentStorageMode === 'local';
}


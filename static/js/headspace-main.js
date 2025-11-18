import { state } from './state.js';
import {
    initCosmos,
    updateCosmosData,
    getChunkMeshes,
    addCustomObject,
    switchView,
    focusChunkInCosmos
} from './cosmos-renderer.js';
import { initializeSearch } from './search.js';
const statusEl = () => document.getElementById('status-text');
const modalEl = () => document.getElementById('thought-modal');
const feedbackEl = () => document.getElementById('thought-feedback');
const submitBtn = () => document.getElementById('submit-thought');

function setStatus(message) {
    const el = statusEl();
    if (el) {
        el.textContent = message;
    }
}

async function fetchVisualization() {
    const response = await fetch('/api/visualization');
    if (!response.ok) {
        throw new Error(`Visualization request failed: ${response.status}`);
    }
    return response.json();
}

async function refreshCosmos(focusDocId = null) {
    const data = await fetchVisualization();
    state.setDocuments(data.documents || []);
    state.setChunks(data.chunks || []);
    state.setConnections(data.connections || []);
    updateCosmosData();

    if (focusDocId) {
        // Give the renderer a frame to create meshes before animating
        setTimeout(() => animateNewDocument(focusDocId), 120);
    }
}

function attachHomePlanet() {
    if (!window.HomePlanetGenerator) return;

    const generator = new window.HomePlanetGenerator();
    const homePlanet = generator.generateHomePlanet();
    homePlanet.position.set(0, -6, -42);

    addCustomObject(homePlanet);
}

function openModal() {
    const modal = modalEl();
    if (!modal) return;
    modal.classList.add('active');
}

function closeModal(resetForm = false) {
    const modal = modalEl();
    if (!modal) return;
    modal.classList.remove('active');

    if (resetForm) {
        const form = document.getElementById('thought-form');
        if (form) {
            form.reset();
        }
        showFeedback('');
    }
}

function showFeedback(message, isError = true) {
    const el = feedbackEl();
    if (!el) return;
    el.textContent = message;
    el.style.color = isError ? 'rgba(255, 160, 160, 0.85)' : 'rgba(144, 220, 255, 0.85)';
}

function animateNewDocument(docId) {
    const chunks = state.chunks.filter(chunk => chunk.document_id === docId);
    if (!chunks.length) return;

    const meshMap = getChunkMeshes();
    const origin = new THREE.Vector3(0, 0, 0);

    chunks.forEach((chunk, index) => {
        const mesh = meshMap.get(chunk.id);
        if (!mesh) {
            return;
        }

        const targetPosition = mesh.position.clone();
        mesh.position.copy(origin);
        mesh.scale.setScalar(0.1);
        if (mesh.material) {
            mesh.material.transparent = true;
            mesh.material.opacity = 0.2;
            mesh.material.needsUpdate = true;
        }

        const startDelay = index * 180;
        const duration = 1600;
        const startTime = performance.now() + startDelay;

        function step(currentTime) {
            const elapsed = currentTime - startTime;
            if (elapsed < 0) {
                requestAnimationFrame(step);
                return;
            }

            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);

            mesh.position.lerpVectors(origin, targetPosition, eased);
            mesh.scale.setScalar(0.1 + 0.9 * eased);
            if (mesh.material) {
                mesh.material.opacity = 0.2 + 0.78 * eased;
            }

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                mesh.position.copy(targetPosition);
                mesh.scale.setScalar(1);
                if (mesh.material) {
                    mesh.material.opacity = 1;
                    mesh.material.transparent = false;
                    mesh.material.needsUpdate = true;
                }
            }
        }

        requestAnimationFrame(step);

        if (window.ShapeMorphingAnimator && chunk.embedding && chunk.embedding.length) {
            const morphAnimator = new window.ShapeMorphingAnimator(mesh, chunk.embedding, 1800);
            setTimeout(() => morphAnimator.start(), startDelay);
        }
    });

    focusChunkInCosmos(chunks[0].id);
}

async function handleThoughtSubmit(event) {
    event.preventDefault();

    const titleInput = document.getElementById('thought-title');
    const contentInput = document.getElementById('thought-content');
    const signatureInput = document.getElementById('thought-signature');
    const submit = submitBtn();

    if (!contentInput || !submit) return;

    const rawContent = contentInput.value.trim();
    if (!rawContent) {
        showFeedback('Your thought needs a little substance. ✨');
        return;
    }

    const title = (titleInput?.value || '').trim() || 'Untitled Thought';
    const signature = (signatureInput?.value || '').trim();

    let content = rawContent;
    if (signature) {
        content = `${rawContent}\n\n— ${signature}`;
    }

    submit.disabled = true;
    showFeedback('');
    setStatus('Encoding your thought…');

    try {
        const result = await createDocumentViaApi(title, content);
        const docId = result.id;
        const status = result.status || 'enriched';

        setStatus(status === 'processing' ? 'Awaiting live enrichment…' : 'Sculpting its planetary shell…');
        await refreshCosmos(docId);

        const chunkMeshes = getChunkMeshes();
        if (status === 'processing' && window.startEnrichmentStreaming) {
            window.startEnrichmentStreaming(docId, chunkMeshes);
        } else {
            setStatus('Thought anchored in orbit ✨');
        }

        closeModal(true);
    } catch (error) {
        console.error('Failed to create document', error);
        showFeedback('Launch failed. Please try again.');
        setStatus('Launch failed');
    } finally {
        submit.disabled = false;
    }
}

function registerFileUpload() {
    const uploadBtn = document.getElementById('upload-file-btn');
    const fileInput = document.getElementById('file-input');
    const contentTextarea = document.getElementById('thought-content');
    const titleInput = document.getElementById('thought-title');
    const wrapper = document.querySelector('.thought-input-wrapper');

    if (!uploadBtn || !fileInput || !contentTextarea) return;

    // Upload button click
    uploadBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileUpload(e.target.files[0], contentTextarea, titleInput);
        fileInput.value = ''; // Reset file input
    });

    // Drag and drop
    contentTextarea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        wrapper.classList.add('drag-over');
    });

    contentTextarea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        wrapper.classList.remove('drag-over');
    });

    contentTextarea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        wrapper.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0], contentTextarea, titleInput);
        }
    });
}

function handleFileUpload(file, textarea, titleInput) {
    if (!file) return;

    // Check if file is text-based
    const validTypes = ['text/plain', 'text/markdown', 'application/octet-stream'];
    const validExtensions = ['.txt', '.md', '.markdown'];
    const fileName = file.name.toLowerCase();

    const isValid = validTypes.includes(file.type) ||
                   validExtensions.some(ext => fileName.endsWith(ext));

    if (!isValid && file.type !== '') {
        showFeedback('Please upload a text file (.txt, .md, .markdown)', true);
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        const content = event.target.result;
        textarea.value = content;

        // Auto-populate title from filename if title is empty
        if (titleInput && !titleInput.value.trim()) {
            const titleFromFile = file.name.replace(/\.[^/.]+$/, ''); // Remove extension
            titleInput.value = titleFromFile;
        }

        showFeedback('File loaded successfully', false);
    };

    reader.onerror = () => {
        showFeedback('Failed to read file', true);
    };

    reader.readAsText(file);
}

function registerUI() {
    const openBtn = document.getElementById('open-thought');
    const cancelBtn = document.getElementById('cancel-thought');
    const closeBtn = document.getElementById('close-thought');
    const form = document.getElementById('thought-form');

    if (openBtn) {
        openBtn.addEventListener('click', () => {
            openModal();
            showFeedback('');
        });
    }

    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => closeModal(true));
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', () => closeModal(false));
    }

    if (form) {
        form.addEventListener('submit', handleThoughtSubmit);
    }

    registerFileUpload();

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closeModal(false);
        }
    });
}

async function initialize() {
    try {
        setStatus('Summoning cosmos…');
        state.setCurrentView('cosmos');

        await initCosmos();
        switchView('cosmos');
        attachHomePlanet();

        await refreshCosmos();
        registerUI();
        initializeSearch();
        setStatus('Ready to capture a new constellation');
    } catch (error) {
        console.error('Failed to initialize headspace', error);
        setStatus('Initialization failed');
    }
}

document.addEventListener('DOMContentLoaded', initialize);

async function createDocumentViaApi(title, content, docType = 'text') {
    const response = await fetch('/api/documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            title,
            content,
            doc_type: docType
        })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Document creation failed: ${response.status} ${errorText}`);
    }

    return response.json();
}

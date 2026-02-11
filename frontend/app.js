// ===== Headspace Frontend =====

const API = '';
const CLUSTER_COLORS = [
    '#6366f1', '#f472b6', '#34d399', '#fbbf24',
    '#60a5fa', '#fb923c', '#a78bfa', '#38bdf8',
];
const NOISE_COLOR = '#4b5568';

// ---------- State ----------
let documents = [];
let clusters = [];
let selectedDoc = null;
let hoveredPoint = null;
let animFrame = null;

// ---------- DOM ----------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const statusBadge = $('#statusBadge');
const statusText = statusBadge.querySelector('.status-text');
const docCount = $('#docCount');
const fileTree = $('#fileTree');
const fileCount = $('#fileCount');
const clusterCanvas = $('#clusterCanvas');
const vizLegend = $('#vizLegend');
const vizTooltip = $('#vizTooltip');
const searchInput = $('#searchInput');
const searchResults = $('#searchResults');
const detailContent = $('#detailContent');
const ingestModal = $('#ingestModal');
const ingestPath = $('#ingestPath');

// ---------- Init ----------
async function init() {
    setupEventListeners();
    await refreshData();
    startPolling();
    drawClusters();
}

function setupEventListeners() {
    // Ingest modal
    $('#ingestBtn').addEventListener('click', () => {
        ingestModal.classList.add('visible');
        ingestPath.focus();
    });
    $('#closeModal').addEventListener('click', () => ingestModal.classList.remove('visible'));
    $('#cancelIngest').addEventListener('click', () => ingestModal.classList.remove('visible'));
    $('#confirmIngest').addEventListener('click', startIngest);
    ingestPath.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') startIngest();
    });

    // Close modal on overlay click
    ingestModal.addEventListener('click', (e) => {
        if (e.target === ingestModal) ingestModal.classList.remove('visible');
    });

    // Tabs
    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.tab').forEach(t => t.classList.remove('active'));
            $$('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            $(`#${tab.dataset.tab}Tab`).classList.add('active');
            if (tab.dataset.tab === 'viz') drawClusters();
        });
    });

    // Search
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runSearch();
    });

    // Canvas interactions
    clusterCanvas.addEventListener('mousemove', onCanvasMouseMove);
    clusterCanvas.addEventListener('click', onCanvasClick);
    clusterCanvas.addEventListener('mouseleave', () => {
        hoveredPoint = null;
        vizTooltip.classList.remove('visible');
    });

    // Close detail
    $('#closeDetail').addEventListener('click', () => {
        selectedDoc = null;
        detailContent.innerHTML = '<div class="empty-state"><p>Select a file to view details</p></div>';
    });

    // Resize
    window.addEventListener('resize', drawClusters);
}

// ---------- API Calls ----------
async function fetchJSON(url) {
    const res = await fetch(API + url);
    return res.json();
}

async function refreshData() {
    try {
        const status = await fetchJSON('/api/status');
        updateStatus(status);

        if (status.document_count > 0) {
            documents = await fetchJSON('/api/documents');
            clusters = await fetchJSON('/api/clusters');
            renderFileTree();
            drawClusters();
        }
    } catch (e) {
        console.error('Failed to refresh:', e);
    }
}

function updateStatus(status) {
    docCount.textContent = `${status.document_count} document${status.document_count !== 1 ? 's' : ''}`;
    fileCount.textContent = status.document_count;

    if (status.is_ingesting) {
        statusBadge.classList.add('ingesting');
        statusText.textContent = 'Ingesting...';
    } else {
        statusBadge.classList.remove('ingesting');
        statusText.textContent = status.document_count > 0 ? 'Ready' : 'No data';
    }
}

async function startIngest() {
    const path = ingestPath.value.trim();
    if (!path) return;

    ingestModal.classList.remove('visible');

    try {
        const res = await fetch(API + '/api/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });

        if (!res.ok) {
            const text = await res.text();
            alert(`Ingestion failed: ${text}`);
            return;
        }

        statusBadge.classList.add('ingesting');
        statusText.textContent = 'Ingesting...';
    } catch (e) {
        alert(`Error: ${e.message}`);
    }
}

function startPolling() {
    setInterval(async () => {
        try {
            const status = await fetchJSON('/api/status');
            const wasIngesting = statusBadge.classList.contains('ingesting');
            updateStatus(status);

            // Refresh data when ingestion finishes
            if (wasIngesting && !status.is_ingesting) {
                await refreshData();
            }
        } catch (e) {
            // Silent fail on poll
        }
    }, 2000);
}

// ---------- File Tree ----------
function renderFileTree() {
    if (documents.length === 0) {
        fileTree.innerHTML = `<div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48" opacity="0.3">
                <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
            </svg>
            <p>No files ingested yet</p>
            <p class="hint">Click <strong>Ingest</strong> to scan a directory</p>
        </div>`;
        return;
    }

    // Build tree structure
    const tree = {};
    documents.forEach(doc => {
        const parts = doc.rel_path.split('/');
        let node = tree;
        for (let i = 0; i < parts.length - 1; i++) {
            if (!node[parts[i]]) node[parts[i]] = {};
            node = node[parts[i]];
        }
        node[parts[parts.length - 1]] = doc;
    });

    fileTree.innerHTML = renderTreeNode(tree, '');
}

function renderTreeNode(node, prefix) {
    let html = '';
    const entries = Object.entries(node).sort(([a, va], [b, vb]) => {
        const aIsDir = typeof va === 'object' && !va.id;
        const bIsDir = typeof vb === 'object' && !vb.id;
        if (aIsDir !== bIsDir) return aIsDir ? -1 : 1;
        return a.localeCompare(b);
    });

    for (const [name, value] of entries) {
        if (value && value.id) {
            // File
            html += `<div class="tree-file" data-id="${value.id}" onclick="selectDocument('${value.id}')">
                <svg class="tree-file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                </svg>
                <span class="tree-file-name">${name}</span>
                ${value.extension ? `<span class="tree-ext">.${value.extension}</span>` : ''}
            </div>`;
        } else {
            // Directory
            html += `<div class="tree-dir" onclick="toggleDir(this, event)">
                <svg class="tree-dir-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
                </svg>
                <span class="tree-dir-name">${name}</span>
                <div class="tree-children">
                    ${renderTreeNode(value, prefix + name + '/')}
                </div>
            </div>`;
        }
    }

    return html;
}

window.toggleDir = function (el, event) {
    if (event.target.closest('.tree-file')) return;
    el.classList.toggle('collapsed');
};

window.selectDocument = async function (id) {
    // Highlight in tree
    $$('.tree-file').forEach(f => f.classList.remove('active'));
    const el = $(`.tree-file[data-id="${id}"]`);
    if (el) el.classList.add('active');

    try {
        const doc = await fetchJSON(`/api/document/${id}`);
        selectedDoc = doc;
        renderDetail(doc);
    } catch (e) {
        console.error('Failed to load document:', e);
    }
};

// ---------- Document Detail ----------
function renderDetail(doc) {
    const clusterColor = doc.cluster_id >= 0
        ? CLUSTER_COLORS[doc.cluster_id % CLUSTER_COLORS.length]
        : NOISE_COLOR;

    detailContent.innerHTML = `
        <div class="detail-meta">
            <div class="detail-title">${escapeHtml(doc.name)}</div>
            <div class="detail-path">${escapeHtml(doc.rel_path)}</div>
            <div class="detail-tags">
                ${doc.extension ? `<span class="detail-tag">.${doc.extension}</span>` : ''}
                <span class="detail-tag">${formatBytes(doc.content_length)}</span>
                <span class="detail-tag" style="border-color: ${clusterColor}; color: ${clusterColor}">
                    ${doc.cluster_id >= 0 ? `Cluster ${doc.cluster_id}` : 'Unclustered'}
                </span>
            </div>
        </div>
        <div class="detail-content">
            <pre>${escapeHtml(doc.content_preview)}</pre>
        </div>
    `;
}

// ---------- Cluster Visualization ----------
function drawClusters() {
    const canvas = clusterCanvas;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const padding = 50;

    // Clear
    ctx.fillStyle = '#0a0e17';
    ctx.fillRect(0, 0, w, h);

    if (clusters.length === 0) {
        ctx.fillStyle = '#4b5568';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data to visualize', w / 2, h / 2 - 10);
        ctx.font = '12px Inter, sans-serif';
        ctx.fillStyle = '#374151';
        ctx.fillText('Ingest a directory to see document clusters', w / 2, h / 2 + 14);
        return;
    }

    // Draw grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
        const x = padding + (w - 2 * padding) * (i / 10);
        const y = padding + (h - 2 * padding) * (i / 10);
        ctx.beginPath(); ctx.moveTo(x, padding); ctx.lineTo(x, h - padding); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(padding, y); ctx.lineTo(w - padding, y); ctx.stroke();
    }

    // Draw points
    const uniqueClusters = new Set();
    for (const point of clusters) {
        const px = padding + point.x * (w - 2 * padding);
        const py = padding + point.y * (h - 2 * padding);
        const color = point.cluster_id >= 0
            ? CLUSTER_COLORS[point.cluster_id % CLUSTER_COLORS.length]
            : NOISE_COLOR;

        uniqueClusters.add(point.cluster_id);

        const isHovered = hoveredPoint && hoveredPoint.id === point.id;
        const radius = isHovered ? 7 : 4;

        // Glow
        if (isHovered) {
            ctx.beginPath();
            ctx.arc(px, py, 16, 0, Math.PI * 2);
            ctx.fillStyle = color + '30';
            ctx.fill();
        }

        // Point
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Store screen coords for hit testing
        point._px = px;
        point._py = py;
    }

    // Render legend
    const legendItems = [...uniqueClusters].sort((a, b) => a - b);
    vizLegend.innerHTML = legendItems.map(id => {
        const color = id >= 0 ? CLUSTER_COLORS[id % CLUSTER_COLORS.length] : NOISE_COLOR;
        const label = id >= 0 ? `Cluster ${id}` : 'Noise';
        const count = clusters.filter(c => c.cluster_id === id).length;
        return `<span class="legend-item">
            <span class="legend-dot" style="background:${color}"></span>
            ${label} (${count})
        </span>`;
    }).join('');
}

function onCanvasMouseMove(e) {
    const rect = clusterCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let closest = null;
    let minDist = 20; // Hit radius in px

    for (const point of clusters) {
        if (point._px === undefined) continue;
        const dist = Math.sqrt((point._px - mx) ** 2 + (point._py - my) ** 2);
        if (dist < minDist) {
            minDist = dist;
            closest = point;
        }
    }

    hoveredPoint = closest;

    if (closest) {
        vizTooltip.innerHTML = `
            <strong>${escapeHtml(closest.name)}</strong>
            <div class="tooltip-path">${escapeHtml(closest.rel_path)}</div>
        `;
        vizTooltip.style.left = (e.clientX - clusterCanvas.parentElement.getBoundingClientRect().left + 12) + 'px';
        vizTooltip.style.top = (e.clientY - clusterCanvas.parentElement.getBoundingClientRect().top - 10) + 'px';
        vizTooltip.classList.add('visible');
        clusterCanvas.style.cursor = 'pointer';
    } else {
        vizTooltip.classList.remove('visible');
        clusterCanvas.style.cursor = 'crosshair';
    }

    drawClusters();
}

function onCanvasClick() {
    if (hoveredPoint) {
        selectDocument(hoveredPoint.id);
    }
}

// ---------- Search ----------
async function runSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    searchResults.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Searching...</p></div>';

    try {
        const results = await fetchJSON(`/api/search?q=${encodeURIComponent(query)}&limit=20`);

        if (results.length === 0) {
            searchResults.innerHTML = '<div class="empty-state"><p>No results found</p></div>';
            return;
        }

        searchResults.innerHTML = results.map(r => `
            <div class="search-result" onclick="selectDocument('${r.id}')">
                <div class="result-header">
                    <span class="result-name">${escapeHtml(r.name)}</span>
                    <span class="result-score">${(r.score * 100).toFixed(1)}%</span>
                </div>
                <div class="result-path">${escapeHtml(r.rel_path)}</div>
                <div class="result-preview">${escapeHtml(r.content_preview)}</div>
            </div>
        `).join('');
    } catch (e) {
        searchResults.innerHTML = `<div class="empty-state"><p>Search failed: ${e.message}</p></div>`;
    }
}

// ---------- Helpers ----------
function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ---------- Boot ----------
init();

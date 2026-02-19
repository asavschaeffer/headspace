const API = "";
const CLUSTER_COLORS = [
    "#6366f1", "#f472b6", "#34d399", "#fbbf24",
    "#60a5fa", "#fb923c", "#a78bfa", "#38bdf8",
];
const NOISE_COLOR = "#4b5568";

let documents = [];
let clusters = [];
let reviewQueue = [];
let selectedDoc = null;
let hoveredPoint = null;
let activeTab = "viz";
let progressSource = null;
let lastEnrichLine = null;
let reviewCurrentId = null;
let approvedThisSession = 0;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const statusBadge = $("#statusBadge");
const statusText = statusBadge.querySelector(".status-text");
const docCount = $("#docCount");
const fileTree = $("#fileTree");
const fileCount = $("#fileCount");
const clusterCanvas = $("#clusterCanvas");
const vizLegend = $("#vizLegend");
const vizTooltip = $("#vizTooltip");
const searchInput = $("#searchInput");
const searchResults = $("#searchResults");
const detailContent = $("#detailContent");
const ingestModal = $("#ingestModal");
const ingestPath = $("#ingestPath");
const reviewBadge = $("#reviewBadge");
const reviewGroups = $("#reviewGroups");
const reviewNotice = $("#reviewNotice");
const finishReviewMeta = $("#finishReviewMeta");
const finishReviewBtn = $("#finishReviewBtn");
const progressLog = $("#progressLog");
const progressLogBody = $("#progressLogBody");

async function init() {
    setupEventListeners();
    await refreshData();
    startPolling();
    drawClusters();
}

function setupEventListeners() {
    $("#ingestBtn").addEventListener("click", () => {
        ingestModal.classList.add("visible");
        ingestPath.focus();
    });
    $("#closeModal").addEventListener("click", () => ingestModal.classList.remove("visible"));
    $("#cancelIngest").addEventListener("click", () => ingestModal.classList.remove("visible"));
    $("#confirmIngest").addEventListener("click", startIngest);
    ingestPath.addEventListener("keydown", (e) => {
        if (e.key === "Enter") startIngest();
    });
    ingestModal.addEventListener("click", (e) => {
        if (e.target === ingestModal) ingestModal.classList.remove("visible");
    });

    $$(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            const tabName = tab.dataset.tab;
            activeTab = tabName;
            $$(".tab").forEach((t) => t.classList.remove("active"));
            $$(".tab-content").forEach((c) => c.classList.remove("active"));
            tab.classList.add("active");
            $(`#${tabName}Tab`).classList.add("active");
            if (tabName === "viz") drawClusters();
            if (tabName === "review") focusCurrentReviewCard();
        });
    });

    searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") runSearch();
    });

    clusterCanvas.addEventListener("mousemove", onCanvasMouseMove);
    clusterCanvas.addEventListener("click", onCanvasClick);
    clusterCanvas.addEventListener("mouseleave", () => {
        hoveredPoint = null;
        vizTooltip.classList.remove("visible");
    });

    $("#closeDetail").addEventListener("click", () => {
        selectedDoc = null;
        detailContent.innerHTML = '<div class="empty-state"><p>Select a file to view details</p></div>';
    });

    finishReviewBtn.addEventListener("click", finishReview);

    window.addEventListener("resize", drawClusters);

    document.addEventListener("keydown", onReviewKeydown);
}

function onReviewKeydown(e) {
    if (activeTab !== "review") return;
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;

    const key = e.key.toLowerCase();
    if (key === "a" || e.key === "ArrowRight") {
        approveCurrentCard();
    } else if (key === "r" || e.key === "Delete") {
        rejectCurrentCard();
    } else if (e.key === " " || e.key === "ArrowDown") {
        e.preventDefault();
        nextCard();
    }
}

async function fetchJSON(url, options = undefined) {
    const res = await fetch(API + url, options);
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed (${res.status})`);
    }
    const text = await res.text();
    if (!text) return null;
    return JSON.parse(text);
}

async function refreshData() {
    try {
        const status = await fetchJSON("/api/status");
        updateStatus(status);
        documents = await fetchJSON("/api/documents");
        clusters = await fetchJSON("/api/clusters");
        await loadReviewQueue();
        renderFileTree();
        drawClusters();
    } catch (e) {
        console.error("Failed to refresh:", e);
    }
}

function updateStatus(status) {
    docCount.textContent = `${status.approved_count} approved / ${status.pending_review_count} pending`;
    fileCount.textContent = documents.length;

    if (status.is_ingesting) {
        statusBadge.classList.add("ingesting");
        statusText.textContent = "Ingesting...";
    } else {
        statusBadge.classList.remove("ingesting");
        statusText.textContent = status.approved_count > 0 ? "Ready" : "No approved data";
    }

    const statsEl = $("#ingestStats");
    if (status.last_ingest && !status.is_ingesting) {
        const s = status.last_ingest;
        const parts = [];
        if (s.new_files > 0) parts.push(`${s.new_files} new`);
        if (s.changed_files > 0) parts.push(`${s.changed_files} changed`);
        if (s.deleted_files > 0) parts.push(`${s.deleted_files} deleted`);
        if (s.unchanged_files > 0) parts.push(`${s.unchanged_files} unchanged`);
        if (parts.length > 0) {
            statsEl.textContent = parts.join(" | ");
            statsEl.classList.add("visible");
        } else {
            statsEl.classList.remove("visible");
        }
    } else {
        statsEl.classList.remove("visible");
    }
}

async function startIngest() {
    const path = ingestPath.value.trim();
    if (!path) return;

    ingestModal.classList.remove("visible");

    try {
        await fetchJSON("/api/ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
        });
        statusBadge.classList.add("ingesting");
        statusText.textContent = "Ingesting...";
        startProgressStream();
    } catch (e) {
        alert(`Ingestion failed: ${e.message}`);
    }
}

function startProgressStream() {
    if (progressSource) {
        progressSource.close();
        progressSource = null;
    }
    lastEnrichLine = null;
    progressLogBody.innerHTML = "";
    showProgressLog();

    progressSource = new EventSource("/api/ingest/stream");
    progressSource.onmessage = async (e) => {
        const event = JSON.parse(e.data);
        appendProgressLine(event);
        if (event.phase === "complete" || event.phase === "error") {
            progressSource.close();
            progressSource = null;
            setTimeout(() => hideProgressLog(), 1200);
            await refreshData();
        }
    };
    progressSource.onerror = () => {
        if (progressSource) {
            progressSource.close();
            progressSource = null;
        }
    };
}

function showProgressLog() {
    progressLog.classList.add("visible");
}

function hideProgressLog() {
    progressLog.classList.remove("visible");
}

function appendProgressLine(event) {
    const lineText = formatProgressLine(event);
    if (!lineText) return;

    if (event.phase === "enrich") {
        if (!lastEnrichLine) {
            lastEnrichLine = document.createElement("div");
            lastEnrichLine.className = "progress-line progress-enrich";
            progressLogBody.appendChild(lastEnrichLine);
        }
        lastEnrichLine.textContent = lineText;
    } else {
        lastEnrichLine = null;
        const line = document.createElement("div");
        line.className = `progress-line progress-${event.phase}`;
        line.textContent = lineText;
        progressLogBody.appendChild(line);
    }
    progressLogBody.scrollTop = progressLogBody.scrollHeight;
}

function formatProgressLine(event) {
    if (event.phase === "discover") return `OK ${event.message}`;
    if (event.phase === "diff") return `NEXT ${event.message}`;
    if (event.phase === "enrich") return `RUN ${event.message}`;
    if (event.phase === "complete") return `OK ${event.message}`;
    if (event.phase === "error") return `ERR ${event.message}`;
    return event.message || "";
}

function startPolling() {
    setInterval(async () => {
        try {
            const status = await fetchJSON("/api/status");
            const wasIngesting = statusBadge.classList.contains("ingesting");
            updateStatus(status);
            if (wasIngesting && !status.is_ingesting) {
                await refreshData();
            }
        } catch (_e) {
            // ignore poll errors
        }
    }, 2000);
}

async function loadReviewQueue() {
    reviewQueue = await fetchJSON("/api/review/queue");
    updateReviewBadge();
    renderReviewQueue();
}

function updateReviewBadge() {
    reviewBadge.textContent = reviewQueue.length > 0 ? String(reviewQueue.length) : "";
}

function renderReviewQueue() {
    if (!reviewQueue.length) {
        reviewNotice.innerHTML = "";
        reviewGroups.innerHTML = `<div class="empty-state"><p>No files pending review</p></div>`;
        reviewCurrentId = null;
        updateFinishReviewBar();
        return;
    }

    const hashGroups = groupBy(reviewQueue, (item) => item.content_hash);
    const duplicateSets = Object.values(hashGroups).filter((items) => items.length > 1);
    if (duplicateSets.length) {
        const dupes = duplicateSets.reduce((n, items) => n + items.length, 0);
        reviewNotice.innerHTML = `
            <div class="duplicate-notice">
                <span>${dupes} pending files have exact duplicates.</span>
                <button class="btn btn-ghost" id="rejectDupesBtn">Reject duplicates, keep newest</button>
            </div>
        `;
        $("#rejectDupesBtn").addEventListener("click", rejectDuplicateSets);
    } else {
        reviewNotice.innerHTML = "";
    }

    const dirGroups = groupBy(reviewQueue, (item) => item.dir_path || "(root)");
    const sortedDirs = Object.keys(dirGroups).sort((a, b) => a.localeCompare(b));

    const html = sortedDirs.map((dir) => {
        const items = dirGroups[dir];
        const ids = items.map((item) => item.id);
        const cards = items.map(renderReviewCard).join("");
        return `
            <section class="review-group" data-dir="${escapeHtml(dir)}">
                <header class="review-group-header">
                    <div>
                        <div class="review-dir">${escapeHtml(dir)}</div>
                        <div class="review-dir-count">${items.length} pending</div>
                    </div>
                    <div class="review-group-actions">
                        <button class="btn btn-ghost review-bulk-approve" data-ids="${ids.join(",")}">Approve all</button>
                        <button class="btn btn-ghost review-bulk-reject" data-ids="${ids.join(",")}">Reject all</button>
                    </div>
                </header>
                <div class="review-card-list">${cards}</div>
            </section>
        `;
    }).join("");

    reviewGroups.innerHTML = html;

    $$(".review-approve-btn").forEach((btn) => {
        btn.addEventListener("click", () => approveReviewItem(btn.dataset.id));
    });
    $$(".review-reject-btn").forEach((btn) => {
        btn.addEventListener("click", () => rejectReviewItem(btn.dataset.id));
    });
    $$(".review-bulk-approve").forEach((btn) => {
        btn.addEventListener("click", () => bulkReviewAction(splitIds(btn.dataset.ids), "approve"));
    });
    $$(".review-bulk-reject").forEach((btn) => {
        btn.addEventListener("click", () => bulkReviewAction(splitIds(btn.dataset.ids), "reject"));
    });
    $$(".review-card").forEach((card) => {
        card.addEventListener("click", () => {
            reviewCurrentId = card.dataset.id;
            highlightCurrentReviewCard();
        });
    });

    const ids = reviewQueue.map((item) => item.id);
    if (!reviewCurrentId || !ids.includes(reviewCurrentId)) {
        reviewCurrentId = ids[0];
    }
    highlightCurrentReviewCard();
    updateFinishReviewBar();
}

function renderReviewCard(item) {
    const extTag = item.extension ? `.${item.extension}` : "(none)";
    const topicTags = (item.topics || []).slice(0, 4).map((topic) => `<span class="review-topic">${escapeHtml(topic)}</span>`).join("");
    const statusClass = statusClassName(item.status);
    const dupe = item.dupe_count > 0 ? `<span class="review-dupe">${item.dupe_count + 1} copies</span>` : "";
    return `
        <article class="review-card" data-id="${item.id}" tabindex="0">
            <div class="review-card-head">
                <div class="review-name">${escapeHtml(item.name)}</div>
                <div class="review-meta">
                    <span class="review-ext">${escapeHtml(extTag)}</span>
                    <span class="review-size">${formatBytes(item.content_length)}</span>
                    ${dupe}
                </div>
            </div>
            <div class="review-status-row">
                <span class="status-pill ${statusClass}">${escapeHtml(item.status || "Unknown")}</span>
            </div>
            <p class="review-summary">${escapeHtml(item.summary || "No summary available.")}</p>
            <div class="review-topics">${topicTags}</div>
            <div class="review-actions">
                <button class="btn btn-ghost review-reject-btn" data-id="${item.id}">Reject (R)</button>
                <button class="btn btn-primary review-approve-btn" data-id="${item.id}">Approve (A)</button>
            </div>
        </article>
    `;
}

function statusClassName(status) {
    const normalized = String(status || "").toLowerCase();
    if (normalized === "draft") return "status-draft";
    if (normalized === "active") return "status-active";
    if (normalized === "reference") return "status-reference";
    return "status-rot";
}

function splitIds(raw) {
    if (!raw) return [];
    return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

async function approveCurrentCard() {
    if (!reviewCurrentId) return;
    await approveReviewItem(reviewCurrentId);
}

async function rejectCurrentCard() {
    if (!reviewCurrentId) return;
    await rejectReviewItem(reviewCurrentId);
}

function nextCard() {
    if (!reviewQueue.length || !reviewCurrentId) return;
    const ids = reviewQueue.map((item) => item.id);
    const current = ids.indexOf(reviewCurrentId);
    if (current < 0) {
        reviewCurrentId = ids[0];
    } else {
        reviewCurrentId = ids[(current + 1) % ids.length];
    }
    highlightCurrentReviewCard();
    focusCurrentReviewCard();
}

function highlightCurrentReviewCard() {
    $$(".review-card").forEach((card) => {
        card.classList.toggle("current", card.dataset.id === reviewCurrentId);
    });
}

function focusCurrentReviewCard() {
    const current = $(`.review-card[data-id="${reviewCurrentId}"]`);
    if (current) current.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function removeReviewQueueItem(id) {
    reviewQueue = reviewQueue.filter((item) => item.id !== id);
    const ids = reviewQueue.map((item) => item.id);
    if (!ids.includes(reviewCurrentId)) {
        reviewCurrentId = ids.length ? ids[0] : null;
    }
    updateReviewBadge();
    renderReviewQueue();
}

async function approveReviewItem(id) {
    const card = $(`.review-card[data-id="${id}"]`);
    if (card) card.classList.add("removing");
    try {
        await fetchJSON(`/api/review/${id}/approve`, { method: "POST" });
        approvedThisSession += 1;
        removeReviewQueueItem(id);
        await refreshApprovedViews();
    } catch (e) {
        console.error(e);
        await loadReviewQueue();
    }
}

async function rejectReviewItem(id) {
    const card = $(`.review-card[data-id="${id}"]`);
    if (card) card.classList.add("removing");
    try {
        await fetchJSON(`/api/review/${id}/reject`, { method: "POST" });
        removeReviewQueueItem(id);
    } catch (e) {
        console.error(e);
        await loadReviewQueue();
    }
}

async function bulkReviewAction(ids, action) {
    if (!ids.length) return;
    try {
        const result = await fetchJSON("/api/review/bulk", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids, action }),
        });
        if (action === "approve") {
            approvedThisSession += Number(result.approved || 0);
            await refreshApprovedViews();
        }
        await loadReviewQueue();
    } catch (e) {
        alert(`Bulk ${action} failed: ${e.message}`);
    }
}

async function rejectDuplicateSets() {
    const hashGroups = groupBy(reviewQueue, (item) => item.content_hash);
    const rejectIds = [];
    for (const items of Object.values(hashGroups)) {
        if (items.length <= 1) continue;
        const sorted = [...items].sort((a, b) => Number(b.modified_at || 0) - Number(a.modified_at || 0));
        for (let i = 1; i < sorted.length; i += 1) {
            rejectIds.push(sorted[i].id);
        }
    }
    if (rejectIds.length) {
        await bulkReviewAction(rejectIds, "reject");
    }
}

function updateFinishReviewBar() {
    finishReviewMeta.textContent = `${approvedThisSession} approved this session`;
    finishReviewBtn.disabled = approvedThisSession === 0;
}

async function finishReview() {
    if (approvedThisSession === 0) return;
    try {
        const result = await fetchJSON("/api/review/finish", { method: "POST" });
        approvedThisSession = 0;
        updateFinishReviewBar();
        await refreshApprovedViews();
        alert(`Clustering complete: ${result.cluster_count} clusters across ${result.approved_documents} approved documents.`);
    } catch (e) {
        alert(`Finish review failed: ${e.message}`);
    }
}

async function refreshApprovedViews() {
    try {
        documents = await fetchJSON("/api/documents");
        clusters = await fetchJSON("/api/clusters");
        renderFileTree();
        drawClusters();
    } catch (e) {
        console.error("Failed to refresh approved views:", e);
    }
}

function renderFileTree() {
    if (documents.length === 0) {
        fileTree.innerHTML = `<div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48" opacity="0.3">
                <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
            </svg>
            <p>No approved files yet</p>
            <p class="hint">Approve files from the Review tab</p>
        </div>`;
        return;
    }

    const tree = {};
    documents.forEach((doc) => {
        const parts = doc.rel_path.split("/");
        let node = tree;
        for (let i = 0; i < parts.length - 1; i += 1) {
            if (!node[parts[i]]) node[parts[i]] = {};
            node = node[parts[i]];
        }
        node[parts[parts.length - 1]] = doc;
    });

    fileTree.innerHTML = renderTreeNode(tree);
    fileCount.textContent = documents.length;
}

function renderTreeNode(node) {
    let html = "";
    const entries = Object.entries(node).sort(([a, va], [b, vb]) => {
        const aIsDir = typeof va === "object" && !va.id;
        const bIsDir = typeof vb === "object" && !vb.id;
        if (aIsDir !== bIsDir) return aIsDir ? -1 : 1;
        return a.localeCompare(b);
    });

    for (const [name, value] of entries) {
        if (value && value.id) {
            html += `<div class="tree-file" data-id="${value.id}" onclick="selectDocument('${value.id}')">
                <svg class="tree-file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                </svg>
                <span class="tree-file-name">${escapeHtml(name)}</span>
                ${value.extension ? `<span class="tree-ext">.${escapeHtml(value.extension)}</span>` : ""}
            </div>`;
        } else {
            html += `<div class="tree-dir" onclick="toggleDir(this, event)">
                <svg class="tree-dir-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
                </svg>
                <span class="tree-dir-name">${escapeHtml(name)}</span>
                <div class="tree-children">${renderTreeNode(value)}</div>
            </div>`;
        }
    }
    return html;
}

window.toggleDir = function toggleDir(el, event) {
    if (event.target.closest(".tree-file")) return;
    el.classList.toggle("collapsed");
};

window.selectDocument = async function selectDocument(id) {
    $$(".tree-file").forEach((f) => f.classList.remove("active"));
    const el = $(`.tree-file[data-id="${id}"]`);
    if (el) el.classList.add("active");
    try {
        const doc = await fetchJSON(`/api/document/${id}`);
        selectedDoc = doc;
        renderDetail(doc);
    } catch (e) {
        console.error("Failed to load document:", e);
    }
};

function renderDetail(doc) {
    const clusterColor = doc.cluster_id >= 0
        ? CLUSTER_COLORS[doc.cluster_id % CLUSTER_COLORS.length]
        : NOISE_COLOR;

    detailContent.innerHTML = `
        <div class="detail-meta">
            <div class="detail-title">${escapeHtml(doc.name)}</div>
            <div class="detail-path">${escapeHtml(doc.rel_path)}</div>
            <div class="detail-tags">
                ${doc.extension ? `<span class="detail-tag">.${escapeHtml(doc.extension)}</span>` : ""}
                <span class="detail-tag">${formatBytes(doc.content_length)}</span>
                <span class="detail-tag">${escapeHtml(doc.review_status || "pending_review")}</span>
                <span class="detail-tag" style="border-color: ${clusterColor}; color: ${clusterColor}">
                    ${doc.cluster_id >= 0 ? `Cluster ${doc.cluster_id}` : "Unclustered"}
                </span>
            </div>
        </div>
        <div class="detail-content">
            <pre>${escapeHtml(doc.content_preview)}</pre>
        </div>
    `;
}

function drawClusters() {
    const canvas = clusterCanvas;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const padding = 50;

    ctx.fillStyle = "#0a0e17";
    ctx.fillRect(0, 0, w, h);

    if (!clusters.length) {
        ctx.fillStyle = "#4b5568";
        ctx.font = "14px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("No approved data to visualize", w / 2, h / 2 - 10);
        ctx.font = "12px Inter, sans-serif";
        ctx.fillStyle = "#374151";
        ctx.fillText("Approve files and click Finish Review", w / 2, h / 2 + 14);
        return;
    }

    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i += 1) {
        const x = padding + ((w - 2 * padding) * (i / 10));
        const y = padding + ((h - 2 * padding) * (i / 10));
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, h - padding);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(w - padding, y);
        ctx.stroke();
    }

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

        if (isHovered) {
            ctx.beginPath();
            ctx.arc(px, py, 16, 0, Math.PI * 2);
            ctx.fillStyle = `${color}30`;
            ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        point._px = px;
        point._py = py;
    }

    const legendItems = [...uniqueClusters].sort((a, b) => a - b);
    vizLegend.innerHTML = legendItems.map((id) => {
        const color = id >= 0 ? CLUSTER_COLORS[id % CLUSTER_COLORS.length] : NOISE_COLOR;
        const label = id >= 0 ? `Cluster ${id}` : "Noise";
        const count = clusters.filter((c) => c.cluster_id === id).length;
        return `<span class="legend-item">
            <span class="legend-dot" style="background:${color}"></span>
            ${label} (${count})
        </span>`;
    }).join("");
}

function onCanvasMouseMove(e) {
    const rect = clusterCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let closest = null;
    let minDist = 20;

    for (const point of clusters) {
        if (point._px === undefined) continue;
        const dist = Math.hypot(point._px - mx, point._py - my);
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
        vizTooltip.style.left = `${e.clientX - clusterCanvas.parentElement.getBoundingClientRect().left + 12}px`;
        vizTooltip.style.top = `${e.clientY - clusterCanvas.parentElement.getBoundingClientRect().top - 10}px`;
        vizTooltip.classList.add("visible");
        clusterCanvas.style.cursor = "pointer";
    } else {
        vizTooltip.classList.remove("visible");
        clusterCanvas.style.cursor = "crosshair";
    }

    drawClusters();
}

function onCanvasClick() {
    if (hoveredPoint) {
        selectDocument(hoveredPoint.id);
    }
}

async function runSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    searchResults.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Searching...</p></div>';
    try {
        const results = await fetchJSON(`/api/search?q=${encodeURIComponent(query)}&limit=20`);
        if (!results.length) {
            searchResults.innerHTML = '<div class="empty-state"><p>No results found</p></div>';
            return;
        }
        searchResults.innerHTML = results.map((r) => `
            <div class="search-result" onclick="selectDocument('${r.id}')">
                <div class="result-header">
                    <span class="result-name">${escapeHtml(r.name)}</span>
                    <span class="result-score">${(r.score * 100).toFixed(1)}%</span>
                </div>
                <div class="result-path">${escapeHtml(r.rel_path)}</div>
                <div class="result-preview">${escapeHtml(r.content_preview)}</div>
            </div>
        `).join("");
    } catch (e) {
        searchResults.innerHTML = `<div class="empty-state"><p>Search failed: ${escapeHtml(e.message)}</p></div>`;
    }
}

function groupBy(items, getKey) {
    const out = {};
    items.forEach((item) => {
        const key = getKey(item);
        if (!out[key]) out[key] = [];
        out[key].push(item);
    });
    return out;
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
}

init();

(() => {
    const H = window.Headspace;
    const {
        dom,
        state,
        $,
        fetchJSON,
        esc,
        escAttr,
        norm,
        seeded,
        parentDir,
        shortPath,
        statusClass,
        sleep,
        showError,
        showWarningBanner,
    } = H;

    let pollErrorCount = 0;
    const POLL_ERROR_THRESHOLD = 3;

    async function init() {
        setupEvents();
        H.canvas.bindEvents();
        H.canvas.resize();
        state.handlers.onNodeActivate = selectNode;

        await Promise.all([refreshCanvas(true), refreshStatus(), refreshOverseen()]);
        renderTerritory();
        updateAmbientStatus();

        H.canvas.startLoop();
        startStatusPolling();
    }

    function setupEvents() {
        document.addEventListener("keydown", onKey);

        $("#ingestBtn").addEventListener("click", () => {
            dom.ingestModal.classList.add("visible");
            dom.ingestPath.focus();
        });
        $("#closeModal").addEventListener("click", () => dom.ingestModal.classList.remove("visible"));
        $("#cancelIngest").addEventListener("click", () => dom.ingestModal.classList.remove("visible"));
        $("#confirmIngest").addEventListener("click", startIngest);
        dom.ingestPath.addEventListener("keydown", (event) => {
            if (event.key === "Enter") startIngest();
        });
        dom.ingestModal.addEventListener("click", (event) => {
            if (event.target === dom.ingestModal) dom.ingestModal.classList.remove("visible");
        });

        $("#cardClose").addEventListener("click", closeCard);
        dom.cardNote.addEventListener("input", () => {
            dom.enrichBtn.hidden = dom.cardNote.value.trim().length === 0;
        });
        dom.enrichBtn.addEventListener("click", enrichSelected);

        $("#lassoApprove").addEventListener("click", () => bulkLasso("approve"));
        $("#lassoReject").addEventListener("click", () => bulkLasso("reject"));
        $("#lassoClear").addEventListener("click", H.canvas.clearLasso);

        dom.ambientStatus.addEventListener("click", onAmbientClick);
        dom.finalizeBtn.addEventListener("click", finishReview);
        dom.searchInput.addEventListener("input", () => {
            renderSearch();
        });
        dom.searchInput.addEventListener("focus", renderFilterChips);
    }

    async function refreshCanvas(fit = false) {
        const fresh = await fetchJSON("/api/canvas");
        const prevById = new Map(state.nodes.map((node) => [node.id, node]));
        state.nodes = fresh.map((node) => ({
            ...node,
            topics: state.topicOverrides.get(node.id) || node.topics || [],
            _born: prevById.get(node.id)?._born || Date.now(),
            _seed: prevById.get(node.id)?._seed || seeded(node.id),
        }));

        if ((fit || !state.fitDone) && state.nodes.length) {
            H.canvas.fitView();
            state.fitDone = true;
        }

        if (state.selectedId) {
            const selected = state.nodes.find((node) => node.id === state.selectedId);
            if (selected) renderCard(selected);
            else closeCard();
        }
    }

    async function refreshStatus() {
        state.lastStatus = await fetchJSON("/api/status");
        const pending = state.nodes.filter((node) => node.review_status === "pending_review").length;
        const approved = state.nodes.filter((node) => node.review_status === "approved").length;
        dom.finalizeBtn.hidden = !(pending === 0 && approved > 0);

        if (state.lastStatus.is_ingesting && !state.isIngesting) startIngestPoll();
        if (!state.lastStatus.is_ingesting && state.isIngesting) stopIngestPoll();

        if (!state.lastStatus.has_embeddings) {
            showWarningBanner("Semantic search is unavailable \u2014 set EMBEDDING_API_KEY (or NVIDIA_API_KEY) to enable.");
        } else {
            showWarningBanner(null);
        }
    }

    async function refreshOverseen() {
        try {
            state.overseenDirs = await fetchJSON("/api/overseen");
        } catch (error) {
            console.warn("Failed to load overseen directories:", error);
            state.overseenDirs = [];
        }
    }

    async function bulkLasso(action) {
        if (!state.lassoSelected.size) return;
        try {
            await fetchJSON("/api/review/bulk", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ids: [...state.lassoSelected], action }),
            });
            H.canvas.clearLasso();
            await Promise.all([refreshCanvas(), refreshStatus()]);
            renderTerritory();
            updateAmbientStatus();
        } catch (error) {
            showError(`Bulk action failed: ${error.message}`);
        }
    }

    function selectNode(id, center = true, open = true) {
        const node = state.nodes.find((item) => item.id === id);
        if (!node) return;
        state.selectedId = id;
        if (center) H.canvas.centerNode(node);
        if (open) openCard(node);
    }

    function openCard(node) {
        state.selectedId = node.id;
        dom.cardPanel.hidden = false;
        dom.cardPanel.classList.remove("card-placing");
        dom.cardPanel.classList.add("card-entering");
        setTimeout(() => dom.cardPanel.classList.remove("card-entering"), 260);
        renderCard(node);
    }

    function closeCard() {
        state.selectedId = null;
        dom.cardPanel.hidden = true;
        dom.cardPanel.classList.remove("card-entering", "card-placing");
    }

    function renderCard(node) {
        dom.cardContentZone.innerHTML = `
            <div class="card-file">${esc(node.name || "(untitled)")}</div>
            <div class="card-path">${esc(node.abs_path || node.rel_path || "")}</div>
            <div class="card-preview">${esc(node.content_preview || "")}</div>
        `;

        const status = String(node.status || "").toLowerCase();
        const topics = (node.topics || [])
            .map((topic) => `<span class="topic-chip">${esc(topic)}<button type="button" data-topic="${encodeURIComponent(topic)}">x</button></span>`)
            .join("") || '<span class="topic-chip">no topics</span>';

        const warnings = node.enrichment_warnings || [];
        const warningHtml = warnings.length
            ? `<div class="warning-chip-row">${warnings.map((w) => `<span class="warning-chip">${esc(w)}</span>`).join("")}</div>`
            : "";

        dom.cardAiZone.innerHTML = `
            <p class="card-summary">${esc(node.summary || "No summary yet.")}</p>
            <div class="card-status-row">
                <span class="status-pill ${statusClass(status)}">${esc(node.status || "Unknown")}</span>
                ${topics}
            </div>
            ${warningHtml}
        `;

        dom.cardAiZone.querySelectorAll(".topic-chip button").forEach((button) => {
            button.addEventListener("click", () => {
                const topic = decodeURIComponent(button.dataset.topic || "");
                const filtered = (node.topics || []).filter((value) => value !== topic);
                state.topicOverrides.set(node.id, filtered);
                const live = state.nodes.find((item) => item.id === node.id);
                if (live) {
                    live.topics = filtered;
                    renderCard(live);
                }
            });
        });

        dom.cardNote.value = node.user_note || "";
        dom.enrichBtn.hidden = dom.cardNote.value.trim().length === 0;
        dom.enrichBtn.disabled = false;
        dom.enrichBtn.textContent = "Enrich with context (rerun)";

        if (node.review_status === "pending_review") {
            dom.cardActions.innerHTML = `
                <button class="danger" id="discardBtn">Discard</button>
                <button class="primary" id="placeBtn">Place -></button>
            `;
            $("#discardBtn").addEventListener("click", () => rejectNode(node.id));
            $("#placeBtn").addEventListener("click", () => placeNode(node.id));
        } else {
            dom.cardActions.innerHTML = `
                <button class="danger" id="removeBtn">Remove</button>
                <button id="closeCardBtn">Close</button>
            `;
            $("#removeBtn").addEventListener("click", () => rejectNode(node.id));
            $("#closeCardBtn").addEventListener("click", closeCard);
        }
    }

    async function enrichSelected() {
        if (!state.selectedId) return;
        try {
            dom.enrichBtn.disabled = true;
            dom.enrichBtn.textContent = "Enriching...";
            await fetchJSON(`/api/document/${state.selectedId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ note: dom.cardNote.value }),
            });
            await fetchJSON(`/api/review/${state.selectedId}/enrich`, { method: "POST" });
            await Promise.all([refreshCanvas(), refreshStatus()]);
            renderTerritory();
            updateAmbientStatus();
            const updated = state.nodes.find((node) => node.id === state.selectedId);
            if (updated) openCard(updated);
        } catch (error) {
            showError(`Enrichment failed: ${error.message}`);
        } finally {
            dom.enrichBtn.disabled = false;
            dom.enrichBtn.textContent = "Enrich with context (rerun)";
        }
    }

    async function placeNode(id) {
        try {
            dom.cardPanel.classList.add("card-placing");
            await fetchJSON(`/api/review/${id}/approve`, { method: "POST" });
            await sleep(350);
            closeCard();
            await Promise.all([refreshCanvas(), refreshStatus()]);
            renderTerritory();
            updateAmbientStatus();
        } catch (error) {
            dom.cardPanel.classList.remove("card-placing");
            showError(`Approve failed: ${error.message}`);
        }
    }

    async function rejectNode(id) {
        try {
            await fetchJSON(`/api/review/${id}/reject`, { method: "POST" });
            closeCard();
            await Promise.all([refreshCanvas(), refreshStatus()]);
            renderTerritory();
            updateAmbientStatus();
        } catch (error) {
            showError(`Reject failed: ${error.message}`);
        }
    }

    function renderTerritory() {
        if (!state.nodes.length) {
            dom.territoryTree.innerHTML = '<div class="territory-empty">No territory yet. Click + to begin.</div>';
            return;
        }

        const byDir = new Map();
        for (const node of state.nodes) {
            const dir = parentDir(node.abs_path || node.rel_path || "");
            if (!byDir.has(dir)) byDir.set(dir, { pending: 0, approved: 0 });
            const entry = byDir.get(dir);
            if (node.review_status === "pending_review") entry.pending += 1;
            if (node.review_status === "approved") entry.approved += 1;
        }

        const root = String(state.lastStatus?.root_path || "");
        const overseen = new Set((state.overseenDirs || []).map((entry) => norm(entry.path)));

        dom.territoryTree.innerHTML = [...byDir.entries()]
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([dir, info]) => {
                const normalized = norm(dir);
                const active = state.directorySpotlight && norm(state.directorySpotlight) === normalized ? "active" : "";
                const ingesting = state.isIngesting && root && normalized.startsWith(norm(root)) ? "ingesting" : "";
                const over = overseen.has(normalized) ? '<span class="territory-badge overseen">&#x25C8; overseen</span>' : "";
                return `
                    <div class="territory-dir ${active} ${ingesting}" data-dir="${escAttr(dir)}">
                        <div class="territory-path">${esc(shortPath(dir))}</div>
                        <div class="territory-meta">
                            <span class="territory-badge pending">pending ${info.pending}</span>
                            <span class="territory-badge approved">approved ${info.approved}</span>
                            ${over}
                        </div>
                    </div>
                `;
            })
            .join("");

        dom.territoryTree.querySelectorAll(".territory-dir").forEach((element) => {
            element.addEventListener("click", () => {
                const dir = element.dataset.dir || "";
                state.directorySpotlight =
                    (state.directorySpotlight && norm(state.directorySpotlight) === norm(dir)) ? "" : dir;
                renderTerritory();
            });
        });
    }

    function updateAmbientStatus() {
        const pending = state.nodes.filter((node) => node.review_status === "pending_review").length;
        const approved = state.nodes.filter((node) => node.review_status === "approved").length;

        dom.ambientStatus.classList.remove("hidden", "clickable", "ingesting");
        dom.ambientStatus.dataset.action = "";

        if (state.isIngesting || state.lastStatus?.is_ingesting) {
            dom.ambientStatus.classList.add("ingesting");
            dom.ambientStatus.innerHTML = '<span class="dot"></span>discovering your space...';
            return;
        }

        if (!state.nodes.length) {
            dom.ambientStatus.innerHTML = "Click + to begin";
            dom.ambientStatus.style.opacity = "0.55";
            return;
        }

        dom.ambientStatus.style.opacity = "1";
        if (pending > 0) {
            dom.ambientStatus.classList.add("clickable");
            dom.ambientStatus.dataset.action = "pending";
            dom.ambientStatus.textContent = `${pending} files waiting for you`;
            return;
        }
        if (approved > 0) {
            dom.ambientStatus.classList.add("clickable");
            dom.ambientStatus.dataset.action = "finalize";
            dom.ambientStatus.textContent = "Finalize space";
            return;
        }
        dom.ambientStatus.classList.add("hidden");
    }

    function onAmbientClick() {
        const action = dom.ambientStatus.dataset.action;
        if (action === "pending") {
            const next = state.nodes.find((node) => node.review_status === "pending_review");
            if (next) selectNode(next.id, true, true);
        } else if (action === "finalize") {
            finishReview();
        }
    }

    async function finishReview() {
        try {
            await fetchJSON("/api/review/finish", { method: "POST" });
            await Promise.all([refreshCanvas(), refreshStatus()]);
            renderTerritory();
            updateAmbientStatus();
        } catch (error) {
            showError(`Finalize failed: ${error.message}`);
        }
    }

    function onKey(event) {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
            event.preventDefault();
            if (dom.searchOverlay.hidden) {
                dom.searchOverlay.hidden = false;
                dom.searchInput.value = "";
                state.activeFilters = { extension: null, status: null };
                renderFilterChips();
                renderSearch();
                dom.searchInput.focus();
            } else {
                dom.searchOverlay.hidden = true;
                state.searchMatches = null;
                state.activeFilters = { extension: null, status: null };
            }
            return;
        }

        if (event.key === "Escape") {
            if (!dom.searchOverlay.hidden) {
                dom.searchOverlay.hidden = true;
                state.searchMatches = null;
                state.activeFilters = { extension: null, status: null };
                return;
            }
            if (!dom.cardPanel.hidden) {
                closeCard();
                return;
            }
            if (state.lassoSelected.size) H.canvas.clearLasso();
        }
    }

    function renderFilterChips() {
        if (!dom.searchFilters) return;

        // Collect unique extensions from loaded nodes (top 8 most common)
        const extCounts = new Map();
        for (const node of state.nodes) {
            const ext = (node.extension || "").toLowerCase();
            if (ext) extCounts.set(ext, (extCounts.get(ext) || 0) + 1);
        }
        const topExts = [...extCounts.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8)
            .map(([ext]) => ext);

        const statusOptions = [
            { label: "pending", value: "pending_review" },
            { label: "approved", value: "approved" },
        ];

        function chip(label, type, value) {
            const active = state.activeFilters[type] === value ? "active" : "";
            return `<button class="filter-chip ${active}" data-type="${escAttr(type)}" data-value="${escAttr(value)}">${esc(label)}</button>`;
        }

        const extChips = topExts.map((ext) => chip(`.${ext}`, "extension", ext)).join("");
        const statusChips = statusOptions.map(({ label, value }) => chip(label, "status", value)).join("");

        dom.searchFilters.innerHTML = `
            <div class="filter-chip-group">${statusChips}${extChips}</div>
        `;

        dom.searchFilters.querySelectorAll(".filter-chip").forEach((btn) => {
            btn.addEventListener("click", () => {
                const type = btn.dataset.type;
                const value = btn.dataset.value;
                // Toggle: clicking active chip clears it
                state.activeFilters[type] = state.activeFilters[type] === value ? null : value;
                renderFilterChips();
                renderSearch();
            });
        });
    }

    function renderSearch() {
        const query = dom.searchInput.value.trim().toLowerCase();
        const hasFilters = state.activeFilters.extension || state.activeFilters.status;

        if (!query && !hasFilters) {
            state.searchMatches = null;
            dom.searchResults.innerHTML =
                '<div class="search-item">Type to filter nodes by name, path, summary, topic, or note.</div>';
            return;
        }

        const filtered = state.nodes.filter((node) => {
            if (state.activeFilters.extension &&
                (node.extension || "").toLowerCase() !== state.activeFilters.extension) return false;
            if (state.activeFilters.status &&
                node.review_status !== state.activeFilters.status) return false;
            return true;
        });

        const ranked = filtered
            .map((node) => {
                if (!query) return { node, index: 0 };
                const haystack = [
                    node.name,
                    node.rel_path,
                    node.abs_path,
                    node.summary,
                    (node.topics || []).join(" "),
                    node.user_note,
                ].join(" ").toLowerCase();
                const index = haystack.indexOf(query);
                return index < 0 ? null : { node, index };
            })
            .filter(Boolean)
            .sort((a, b) => a.index - b.index)
            .slice(0, 40);

        state.searchMatches = ranked.length ? new Set(ranked.map((item) => item.node.id)) : new Set();
        if (!ranked.length) {
            dom.searchResults.innerHTML = '<div class="search-item">No matches.</div>';
            return;
        }

        dom.searchResults.innerHTML = ranked.map(({ node }) => `
            <div class="search-item" data-id="${node.id}">
                <div class="search-name">${esc(node.name)}</div>
                <div class="search-path">${esc(node.rel_path || node.abs_path || "")}</div>
            </div>
        `).join("");

        dom.searchResults.querySelectorAll(".search-item[data-id]").forEach((element) => {
            element.addEventListener("click", () => {
                selectNode(element.dataset.id, true, true);
                dom.searchOverlay.hidden = true;
                state.searchMatches = null;
            });
        });
    }

    async function startIngest() {
        const path = dom.ingestPath.value.trim();
        if (!path) return;
        try {
            await fetchJSON("/api/ingest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path }),
            });
            dom.ingestModal.classList.remove("visible");
            startIngestPoll();
        } catch (error) {
            showError(`Ingestion failed: ${error.message}`);
        }
    }

    function startIngestPoll() {
        if (state.ingestPollTimer) return;
        state.isIngesting = true;
        updateAmbientStatus();
        state.ingestPollTimer = setInterval(async () => {
            try {
                const [fresh, status] = await Promise.all([
                    fetchJSON("/api/canvas"),
                    fetchJSON("/api/status"),
                ]);

                const prevById = new Map(state.nodes.map((node) => [node.id, node]));
                state.nodes = fresh.map((node) => ({
                    ...node,
                    _born: prevById.get(node.id)?._born || Date.now(),
                    _seed: prevById.get(node.id)?._seed || seeded(node.id),
                }));

                state.lastStatus = status;
                state.isIngesting = Boolean(status.is_ingesting);
                renderTerritory();
                updateAmbientStatus();
                if (!status.is_ingesting) stopIngestPoll();
                pollErrorCount = 0;
            } catch (_error) {
                pollErrorCount++;
                if (pollErrorCount >= POLL_ERROR_THRESHOLD) {
                    showWarningBanner("Connection lost \u2014 retrying...");
                }
            }
        }, 3000);
    }

    function stopIngestPoll() {
        if (state.ingestPollTimer) clearInterval(state.ingestPollTimer);
        state.ingestPollTimer = null;
        state.isIngesting = false;
        updateAmbientStatus();
    }

    function startStatusPolling() {
        if (state.statusPollTimer) clearInterval(state.statusPollTimer);
        state.statusPollTimer = setInterval(async () => {
            if (state.isIngesting) return;
            try {
                await Promise.all([refreshCanvas(), refreshStatus(), refreshOverseen()]);
                renderTerritory();
                updateAmbientStatus();
                pollErrorCount = 0;
            } catch (_error) {
                pollErrorCount++;
                if (pollErrorCount >= POLL_ERROR_THRESHOLD) {
                    showWarningBanner("Connection lost \u2014 retrying...");
                }
            }
        }, 5000);
    }

    init();
})();

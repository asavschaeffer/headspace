(() => {
    const H = window.Headspace;
    const {
        CANVAS_W,
        CANVAS_H,
        MIN_SCALE,
        MAX_SCALE,
        clamp,
        inDir,
        inPoly,
        truncate,
        state,
        dom,
    } = H;

    const ctx = dom.canvas.getContext("2d");

    function world(node) {
        return { x: (node.x || 0) * CANVAS_W, y: (node.y || 0) * CANVAS_H };
    }

    function worldToScreen(wx, wy) {
        return { x: wx * state.scale + state.panX, y: wy * state.scale + state.panY };
    }

    function resize() {
        const rect = dom.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        dom.canvas.width = Math.max(1, Math.round(rect.width * dpr));
        dom.canvas.height = Math.max(1, Math.round(rect.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (!state.fitDone && state.nodes.length) fitView();
    }

    function fitView() {
        const rect = dom.canvas.getBoundingClientRect();
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        for (const node of state.nodes) {
            const point = world(node);
            minX = Math.min(minX, point.x);
            maxX = Math.max(maxX, point.x);
            minY = Math.min(minY, point.y);
            maxY = Math.max(maxY, point.y);
        }

        const width = Math.max(120, maxX - minX);
        const height = Math.max(120, maxY - minY);
        state.scale = clamp(Math.min((rect.width * 0.72) / width, (rect.height * 0.72) / height), MIN_SCALE, MAX_SCALE);
        state.panX = rect.width / 2 - (minX + width / 2) * state.scale;
        state.panY = rect.height / 2 - (minY + height / 2) * state.scale;
    }

    function centerNode(node) {
        const rect = dom.canvas.getBoundingClientRect();
        const point = world(node);
        state.panX = rect.width / 2 - point.x * state.scale;
        state.panY = rect.height / 2 - point.y * state.scale;
    }

    function startLoop() {
        requestAnimationFrame(loop);
    }

    function loop(ts) {
        state.animTime = ts;
        draw();
        requestAnimationFrame(loop);
    }

    function draw() {
        const rect = dom.canvas.getBoundingClientRect();
        ctx.clearRect(0, 0, rect.width, rect.height);
        drawGrid(rect.width, rect.height);
        state.nodeBounds.clear();

        for (const node of state.nodes) {
            const screen = worldToScreen(world(node).x, world(node).y);
            const nodeW = Math.max(22, (node.review_status === "pending_review" ? 122 : 136) * state.scale);
            const nodeH = Math.max(16, 38 * state.scale);
            const x = screen.x - nodeW / 2;
            const y = screen.y - nodeH / 2;

            if (x > rect.width + 160 || y > rect.height + 80 || x + nodeW < -160 || y + nodeH < -80) {
                continue;
            }

            let alpha = 1;
            if (state.directorySpotlight) alpha *= inDir(node, state.directorySpotlight) ? 1 : 0.16;
            if (state.searchMatches) alpha *= state.searchMatches.has(node.id) ? 1 : 0.1;
            alpha *= Math.min(1, (Date.now() - (node._born || Date.now())) / 600);

            drawNode(node, screen.x, screen.y, nodeW, nodeH, alpha);
            state.nodeBounds.set(node.id, { x, y, w: nodeW, h: nodeH });
        }

        if (state.isLassoing && state.lassoPath.length > 1) drawLasso();
    }

    function drawGrid(width, height) {
        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,.03)";
        const step = 80;
        for (let x = (state.panX % step); x < width; x += step) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = (state.panY % step); y < height; y += step) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        ctx.restore();
    }

    function drawNode(node, cx, cy, width, height, alpha) {
        const isPending = node.review_status === "pending_review";
        const status = String(node.status || "").toLowerCase();
        const pulse = 0.55 + 0.25 * Math.sin(state.animTime / 800 + node._seed);

        ctx.save();
        if (isPending) {
            ctx.globalAlpha = alpha * pulse;
            ctx.fillStyle = "rgba(120,180,255,.12)";
            ctx.strokeStyle = "#7eb8ff";
            ctx.lineWidth = Math.max(1, state.scale * 0.7);
            ctx.setLineDash([Math.max(2, state.scale * 2.4), Math.max(2, state.scale * 2.4)]);
        } else {
            ctx.globalAlpha = alpha;
            ctx.fillStyle = status === "rot" ? "#bcb7ab" : "#f2ead8";
            ctx.strokeStyle = "rgba(255,255,255,.2)";
            ctx.lineWidth = Math.max(1, state.scale * 0.45);
            ctx.shadowBlur = Math.max(6, state.scale * 8);
            ctx.shadowColor = "rgba(0,0,0,.45)";
        }

        roundRect(cx - width / 2, cy - height / 2, width, height, Math.max(4, state.scale * 4));
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = isPending ? "#a8c8ff" : (status === "rot" ? "#554f47" : "#1a1612");
        const fontSize = Math.max(9, Math.min(16, state.scale * 11));
        ctx.font = `${fontSize}px 'Space Grotesk', sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const chars = Math.max(8, Math.floor((width - 16) / Math.max(6, fontSize * 0.54)));
        ctx.fillText(truncate(node.name, chars), cx, cy + 1);

        if (node.id === state.hoveredId || node.id === state.selectedId) {
            ctx.globalAlpha = alpha * 0.75;
            ctx.strokeStyle = "#6366f1";
            ctx.lineWidth = Math.max(2, state.scale * 1.8);
            roundRect(cx - width / 2 - 2, cy - height / 2 - 2, width + 4, height + 4, Math.max(6, state.scale * 5));
            ctx.stroke();
        }

        if (state.lassoSelected.has(node.id)) {
            ctx.globalAlpha = alpha * 0.85;
            ctx.strokeStyle = "#f472b6";
            ctx.lineWidth = Math.max(2, state.scale * 1.8);
            roundRect(cx - width / 2 - 3, cy - height / 2 - 3, width + 6, height + 6, Math.max(7, state.scale * 5.4));
            ctx.stroke();
        }
        ctx.restore();
    }

    function drawLasso() {
        ctx.save();
        ctx.strokeStyle = "rgba(244,114,182,.85)";
        ctx.fillStyle = "rgba(244,114,182,.12)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(state.lassoPath[0][0], state.lassoPath[0][1]);
        for (let i = 1; i < state.lassoPath.length; i += 1) {
            ctx.lineTo(state.lassoPath[i][0], state.lassoPath[i][1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }

    function roundRect(x, y, width, height, radius) {
        const r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + width, y, x + width, y + height, r);
        ctx.arcTo(x + width, y + height, x, y + height, r);
        ctx.arcTo(x, y + height, x, y, r);
        ctx.arcTo(x, y, x + width, y, r);
        ctx.closePath();
    }

    function bindEvents() {
        window.addEventListener("resize", resize);
        dom.canvas.addEventListener("wheel", onWheel, { passive: false });
        dom.canvas.addEventListener("mousedown", onDown);
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        dom.canvas.addEventListener("mouseleave", () => {
            if (!state.isPanning && !state.isLassoing) state.hoveredId = null;
        });
    }

    function onWheel(event) {
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.1 : 0.9;
        const target = clamp(state.scale * factor, MIN_SCALE, MAX_SCALE);
        const wx = (event.offsetX - state.panX) / state.scale;
        const wy = (event.offsetY - state.panY) / state.scale;
        state.scale = target;
        state.panX = event.offsetX - wx * state.scale;
        state.panY = event.offsetY - wy * state.scale;
    }

    function onDown(event) {
        state.pointerDown = {
            x: event.offsetX,
            y: event.offsetY,
            moved: false,
            id: hitNode(event.offsetX, event.offsetY),
        };
        if (event.shiftKey) {
            state.isLassoing = true;
            state.lassoPath = [[event.offsetX, event.offsetY]];
            dom.canvas.style.cursor = "crosshair";
            return;
        }
        state.isPanning = true;
        state.panStart = { x: event.offsetX - state.panX, y: event.offsetY - state.panY };
        dom.canvas.style.cursor = "grabbing";
    }

    function onMove(event) {
        if (state.pointerDown && !state.pointerDown.moved) {
            const delta = Math.hypot(state.pointerDown.x - event.offsetX, state.pointerDown.y - event.offsetY);
            if (delta > 3) state.pointerDown.moved = true;
        }

        if (state.isPanning && state.panStart) {
            state.panX = event.offsetX - state.panStart.x;
            state.panY = event.offsetY - state.panStart.y;
            return;
        }

        if (state.isLassoing) {
            const last = state.lassoPath[state.lassoPath.length - 1];
            if (Math.hypot(last[0] - event.offsetX, last[1] - event.offsetY) > 2) {
                state.lassoPath.push([event.offsetX, event.offsetY]);
            }
            return;
        }

        state.hoveredId = hitNode(event.offsetX, event.offsetY);
        dom.canvas.style.cursor = state.hoveredId ? "pointer" : "crosshair";
    }

    function onUp(event) {
        if (state.isLassoing) {
            state.isLassoing = false;
            finalizeLasso();
        }
        if (state.isPanning) {
            state.isPanning = false;
            state.panStart = null;
            dom.canvas.style.cursor = state.hoveredId ? "pointer" : "crosshair";
        }

        if (state.pointerDown && !state.pointerDown.moved && !event.shiftKey && state.pointerDown.id) {
            state.handlers.onNodeActivate?.(state.pointerDown.id, false, true);
        }
        state.pointerDown = null;
    }

    function hitNode(x, y) {
        let found = null;
        for (const node of state.nodes) {
            const bounds = state.nodeBounds.get(node.id);
            if (!bounds) continue;
            if (x >= bounds.x && x <= bounds.x + bounds.w && y >= bounds.y && y <= bounds.y + bounds.h) {
                found = node.id;
            }
        }
        return found;
    }

    function finalizeLasso() {
        if (state.lassoPath.length < 3) {
            // Shift+click (no drag): toggle the node under cursor
            const downInfo = state.pointerDown;
            if (downInfo && !downInfo.moved && downInfo.id) {
                if (state.lassoSelected.has(downInfo.id)) {
                    state.lassoSelected.delete(downInfo.id);
                } else {
                    state.lassoSelected.add(downInfo.id);
                }
            }
            updateLassoBar();
            return;
        }
        // Shift+drag (polygon): replace selection with enclosed nodes
        state.lassoSelected.clear();
        for (const node of state.nodes) {
            const point = worldToScreen(world(node).x, world(node).y);
            if (inPoly(point.x, point.y, state.lassoPath)) state.lassoSelected.add(node.id);
        }
        updateLassoBar();
    }

    function updateLassoBar() {
        const hasSelection = state.lassoSelected.size > 0;
        dom.lassoExpand.hidden = !hasSelection;
        if (hasSelection) {
            dom.lassoToggle.classList.add("active");
            // Close search expand if open — only one tool at a time
            dom.searchExpand.hidden = true;
            dom.searchDropdown.hidden = true;
            dom.searchToggle.classList.remove("active");
        } else {
            dom.lassoToggle.classList.remove("active");
        }
        dom.lassoCount.textContent = `${state.lassoSelected.size} selected`;
    }

    function clearLasso() {
        state.lassoSelected.clear();
        state.lassoPath = [];
        dom.lassoExpand.hidden = true;
        dom.lassoToggle.classList.remove("active");
        dom.lassoCount.textContent = "0 selected";
    }

    H.canvas = {
        bindEvents,
        resize,
        fitView,
        startLoop,
        centerNode,
        world,
        worldToScreen,
        updateLassoBar,
        clearLasso,
    };
})();

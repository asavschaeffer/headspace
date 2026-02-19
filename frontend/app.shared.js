(() => {
    const API = "";
    const CANVAS_W = 1000;
    const CANVAS_H = 1000;
    const MIN_SCALE = 0.2;
    const MAX_SCALE = 8;

    const $ = (selector) => document.querySelector(selector);

    const dom = {
        canvas: $("#spaceCanvas"),
        ambientStatus: $("#ambientStatus"),
        territoryTree: $("#territoryTree"),
        finalizeBtn: $("#finalizeBtn"),
        cardPanel: $("#cardPanel"),
        cardContentZone: $("#cardContentZone"),
        cardAiZone: $("#cardAiZone"),
        cardNote: $("#cardNote"),
        cardActions: $("#cardActions"),
        enrichBtn: $("#enrichBtn"),
        lassoBar: $("#lassoBar"),
        lassoCount: $("#lassoCount"),
        searchOverlay: $("#searchOverlay"),
        searchInput: $("#searchInput"),
        searchResults: $("#searchResults"),
        ingestModal: $("#ingestModal"),
        ingestPath: $("#ingestPath"),
    };

    const state = {
        nodes: [],
        selectedId: null,
        hoveredId: null,
        lassoSelected: new Set(),
        lassoPath: [],
        searchMatches: null,
        directorySpotlight: "",
        overseenDirs: [],
        topicOverrides: new Map(),
        nodeBounds: new Map(),
        lastStatus: null,
        panX: 0,
        panY: 0,
        scale: 1,
        isPanning: false,
        isLassoing: false,
        panStart: null,
        pointerDown: null,
        isIngesting: false,
        ingestPollTimer: null,
        statusPollTimer: null,
        animTime: 0,
        fitDone: false,
        handlers: {
            onNodeActivate: null,
        },
    };

    async function fetchJSON(url, options) {
        const response = await fetch(API + url, options);
        const text = await response.text();
        if (!response.ok) throw new Error(text || `Request failed (${response.status})`);
        try { return text ? JSON.parse(text) : null; }
        catch { throw new Error(`Invalid JSON from ${url}`); }
    }

    function inDir(node, dir) {
        return norm(node.abs_path || "").startsWith(norm(dir));
    }

    function parentDir(path) {
        if (!path) return "(root)";
        const trimmed = path.replace(/[/\\]+$/, "");
        const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
        return idx < 0 ? "(root)" : (trimmed.slice(0, idx) || "(root)");
    }

    function shortPath(path) {
        if (!path || path === "(root)") return path;
        if (path.length <= 38) return path;
        const parts = path.split(/[/\\]/).filter(Boolean);
        if (parts.length < 3) return `...${path.slice(-35)}`;
        return `.../${parts.slice(-3).join("/")}`;
    }

    function seeded(value) {
        let hash = 0;
        for (let i = 0; i < value.length; i += 1) {
            hash = ((hash << 5) - hash) + value.charCodeAt(i);
            hash |= 0;
        }
        return Math.abs(hash % 1000) / 100;
    }

    function inPoly(x, y, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0];
            const yi = polygon[i][1];
            const xj = polygon[j][0];
            const yj = polygon[j][1];
            const intersects = ((yi > y) !== (yj > y))
                && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9) + xi);
            if (intersects) inside = !inside;
        }
        return inside;
    }

    function statusClass(status) {
        if (status === "draft") return "draft";
        if (status === "active") return "active";
        if (status === "reference") return "reference";
        return "rot";
    }

    function truncate(value, maxChars) {
        if (!value) return "";
        return value.length <= maxChars ? value : `${value.slice(0, Math.max(1, maxChars - 1))}...`;
    }

    function esc(value) {
        if (value === null || value === undefined) return "";
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function escAttr(value) {
        return esc(value).replace(/`/g, "&#96;");
    }

    function norm(value) {
        return String(value || "").replace(/\\/g, "/").toLowerCase();
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function showError(msg) {
        const container = document.getElementById("toastContainer");
        if (!container) return;
        const toast = document.createElement("div");
        toast.className = "toast";
        toast.textContent = msg;
        const close = document.createElement("button");
        close.className = "toast-close";
        close.textContent = "\u00d7";
        close.addEventListener("click", () => toast.remove());
        toast.appendChild(close);
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 8000);
    }

    function showWarningBanner(msg) {
        const banner = document.getElementById("warningBanner");
        if (!banner) return;
        if (msg) {
            banner.textContent = msg;
            banner.hidden = false;
        } else {
            banner.hidden = true;
        }
    }

    window.Headspace = {
        API,
        CANVAS_W,
        CANVAS_H,
        MIN_SCALE,
        MAX_SCALE,
        $,
        dom,
        state,
        fetchJSON,
        inDir,
        parentDir,
        shortPath,
        seeded,
        inPoly,
        statusClass,
        truncate,
        esc,
        escAttr,
        norm,
        clamp,
        sleep,
        showError,
        showWarningBanner,
    };
})();

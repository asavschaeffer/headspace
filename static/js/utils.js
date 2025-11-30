// Utility Functions

export function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

export function updateStatus(message) {
    const statusEl = document.getElementById('status-text');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

export function showChunkTooltip(event, chunk) {
    const tooltip = document.getElementById('chunk-tooltip');
    tooltip.innerHTML = `
        <div>Type: ${chunk.chunk_type}</div>
        <div>Index: ${chunk.chunk_index}</div>
        <div>ID: ${chunk.id}</div>
    `;
    tooltip.style.left = event.clientX + 10 + 'px';
    tooltip.style.top = event.clientY + 10 + 'px';
    tooltip.classList.add('visible');
}

export function hideChunkTooltip() {
    document.getElementById('chunk-tooltip').classList.remove('visible');
}

export function createElement(tag, className, innerHTML = '') {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (innerHTML) el.innerHTML = innerHTML;
    return el;
}

export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

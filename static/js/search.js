/**
 * Search functionality for Headspace
 * Handles hybrid search (vector + keyword) and displays results
 */

import { state } from './state.js';

let selectedSearchResult = null;

export function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const closeSearchBtn = document.getElementById('close-search-results');

    // Search on button click
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
    }

    // Search on Enter key
    if (searchInput) {
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }

    // Close search results
    if (closeSearchBtn) {
        closeSearchBtn.addEventListener('click', closeSearchResults);
    }
}

async function performSearch() {
    const searchInput = document.getElementById('search-input');
    const query = searchInput.value.trim();

    if (!query) {
        showSearchMessage('Enter a search query');
        return;
    }

    try {
        // Show loading state
        const resultsList = document.getElementById('search-results-list');
        resultsList.innerHTML = '<div class="search-results__empty">Searching…</div>';
        openSearchResults();

        // Fetch search results
        const response = await fetch(`/api/search?query=${encodeURIComponent(query)}&top_k=15`);

        if (!response.ok) {
            throw new Error(`Search failed: ${response.status}`);
        }

        const data = await response.json();
        displaySearchResults(data.results, query);
    } catch (error) {
        console.error('Search error:', error);
        showSearchMessage('Search failed. Please try again.');
    }
}

function displaySearchResults(results, query) {
    const resultsList = document.getElementById('search-results-list');

    if (!results || results.length === 0) {
        resultsList.innerHTML = '<div class="search-results__empty">No results found for "' + escapeHtml(query) + '"</div>';
        return;
    }

    resultsList.innerHTML = '';

    results.forEach((result) => {
        const resultItem = createSearchResultItem(result);
        resultsList.appendChild(resultItem);
    });
}

function createSearchResultItem(result) {
    const item = document.createElement('div');
    item.className = 'search-result-item';

    // Format score as percentage
    const scorePercent = Math.round(result.score * 100);

    item.innerHTML = `
        <div class="search-result-item__title">${escapeHtml(result.chunk_type)} #${result.chunk_index + 1}</div>
        <div class="search-result-item__content">${escapeHtml(result.content)}</div>
        <div class="search-result-item__score">Relevance: ${scorePercent}%</div>
    `;

    // Click to select chunk
    item.addEventListener('click', () => {
        selectSearchResult(result);
    });

    return item;
}

function selectSearchResult(result) {
    selectedSearchResult = result;

    // Find and focus the chunk in the cosmos view
    if (window.focusChunkInCosmos) {
        try {
            // We need to trigger the selection through the cosmos view
            // Dispatch a custom event that cosmos-renderer can handle
            const event = new CustomEvent('searchResultSelected', {
                detail: { chunkId: result.id, chunk: result }
            });
            document.getElementById('cosmos-canvas').dispatchEvent(event);
        } catch (error) {
            console.error('Failed to focus chunk:', error);
        }
    }

    // Highlight the selected item
    document.querySelectorAll('.search-result-item').forEach((item) => {
        item.style.background = 'rgba(255, 255, 255, 0.05)';
    });

    event.currentTarget.style.background = 'rgba(111, 139, 255, 0.2)';
    event.currentTarget.style.borderColor = 'rgba(111, 139, 255, 0.5)';
}

function openSearchResults() {
    const searchResultsPanel = document.getElementById('search-results');
    if (searchResultsPanel) {
        searchResultsPanel.classList.add('visible');
    }
}

function closeSearchResults() {
    const searchResultsPanel = document.getElementById('search-results');
    if (searchResultsPanel) {
        searchResultsPanel.classList.remove('visible');
    }
}

function showSearchMessage(message) {
    const resultsList = document.getElementById('search-results-list');
    if (resultsList) {
        resultsList.innerHTML = `<div class="search-results__empty">${escapeHtml(message)}</div>`;
    }
    openSearchResults();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

export function handleSearchResultSelection(chunkId) {
    // This is called from cosmos-renderer when a search result is clicked
    // Focus the chunk in the cosmos view
    if (window.focusChunkInCosmos) {
        window.focusChunkInCosmos(chunkId);
    }
}

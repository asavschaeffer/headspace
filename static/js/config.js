// Configuration and Constants

export const API_BASE = 'http://localhost:8000/api';

export const COLORS = {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#16a085',
    error: '#e74c3c'
};

export const CHUNK_TYPES = {
    heading_1: { color: '#e74c3c', label: 'Heading 1' },
    heading_2: { color: '#f39c12', label: 'Heading 2' },
    heading_3: { color: '#f1c40f', label: 'Heading 3' },
    paragraph: { color: '#3498db', label: 'Paragraph' },
    list: { color: '#16a085', label: 'List' },
    code: { color: '#667eea', label: 'Code' },
    function: { color: '#9b59b6', label: 'Function' }
};

export const COSMOS_SETTINGS = {
    gravity: false,
    animationSpeed: 1.0,
    showConnections: true,
    showNebulae: true,
    maxConnections: 500,
    gravityThrottle: 3,
    maxGravityRange: 100
};

/**
 * SIMPLIFIED Cosmos Renderer - Basic Three.js 3D Visualization
 *
 * Features:
 * - Simple sphere meshes for each chunk
 * - Position from embedding PCA (3D coordinates)
 * - Color from chunk metadata
 * - Basic orbit camera controls
 * - Click detection for chunk selection
 * - Comprehensive logging for debugging
 */

import { state } from './state.js';
import { COSMOS_SETTINGS } from './config.js';

// Three.js scene variables
let scene, camera, renderer, controls;
let chunkMeshes = new Map();
let selectedMesh = null;

// Logging helper
function logViz(level, msg) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] [VIZ] [${level}] ${msg}`);
}

/**
 * Initialize the 3D cosmos visualization
 */
export async function initCosmos() {
    logViz('INFO', 'Initializing cosmos visualization...');

    try {
        const container = document.getElementById('cosmos-container');
        if (!container) {
            logViz('ERROR', 'No cosmos-container element found');
            return;
        }

        // Create scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x001a33); // Dark blue
        scene.fog = new THREE.Fog(0x001a33, 500, 2000);
        logViz('INFO', 'Scene created');

        // Create camera
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 10000);
        camera.position.set(0, 0, 100);
        logViz('INFO', `Camera created: ${width}x${height}`);

        // Create renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        logViz('INFO', 'Renderer created');

        // Add simple orbit controls
        if (window.OrbitControls) {
            controls = new window.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = false;
            logViz('INFO', 'OrbitControls enabled');
        } else {
            logViz('WARN', 'OrbitControls not available');
        }

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0xffffff, 0.8);
        pointLight.position.set(100, 100, 100);
        scene.add(pointLight);
        logViz('INFO', 'Lighting added');

        // Add starfield background
        addStarfield();

        // Handle window resize
        window.addEventListener('resize', onWindowResize);

        // Handle mouse clicks for selection
        document.addEventListener('click', onCanvasClick);

        logViz('INFO', '✅ Cosmos initialized successfully');

        // Start animation loop
        animate();

    } catch (error) {
        logViz('ERROR', `Failed to initialize cosmos: ${error.message}`);
        throw error;
    }
}

/**
 * Add a starfield background
 */
function addStarfield() {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];

    for (let i = 0; i < 1000; i++) {
        vertices.push(
            (Math.random() - 0.5) * 2000,
            (Math.random() - 0.5) * 2000,
            (Math.random() - 0.5) * 2000
        );
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
    const material = new THREE.PointsMaterial({ color: 0xffffff, size: 2, sizeAttenuation: true });
    const stars = new THREE.Points(geometry, material);
    scene.add(stars);
    logViz('DEBUG', 'Starfield added');
}

/**
 * Add a chunk to the visualization
 * @param {Object} chunk - Chunk object with id, content, embedding, position_3d, color
 */
export function addChunk(chunk) {
    if (!chunk.id) {
        logViz('WARN', 'Chunk missing ID');
        return;
    }

    try {
        // Create sphere geometry (simple, not cached)
        const geometry = new THREE.SphereGeometry(5, 32, 32);

        // Parse color
        let color = 0x748ffc; // Default purple
        if (chunk.color && chunk.color.startsWith('#')) {
            color = parseInt(chunk.color.slice(1), 16);
        }

        const material = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.2,
            roughness: 0.7,
            metalness: 0.2
        });

        const mesh = new THREE.Mesh(geometry, material);

        // Set position from 3D coordinates
        if (chunk.position_3d && Array.isArray(chunk.position_3d) && chunk.position_3d.length === 3) {
            mesh.position.set(chunk.position_3d[0], chunk.position_3d[1], chunk.position_3d[2]);
        } else {
            logViz('WARN', `Chunk ${chunk.id} has invalid position_3d: ${JSON.stringify(chunk.position_3d)}`);
            // Place at origin
            mesh.position.set(0, 0, 0);
        }

        // Store metadata on mesh
        mesh.userData = {
            chunkId: chunk.id,
            content: chunk.content || '',
            embedding: chunk.embedding || [],
            tags: chunk.tags || []
        };

        // Add to scene and map
        scene.add(mesh);
        chunkMeshes.set(chunk.id, mesh);

        logViz('DEBUG', `Added chunk ${chunk.id} at (${mesh.position.x.toFixed(1)}, ${mesh.position.y.toFixed(1)}, ${mesh.position.z.toFixed(1)})`);

    } catch (error) {
        logViz('ERROR', `Failed to add chunk ${chunk.id}: ${error.message}`);
    }
}

/**
 * Update a chunk's visualization
 */
export function updateChunk(chunk) {
    const mesh = chunkMeshes.get(chunk.id);
    if (!mesh) {
        logViz('WARN', `Chunk ${chunk.id} not found, adding new`);
        addChunk(chunk);
        return;
    }

    // Update position if provided
    if (chunk.position_3d && Array.isArray(chunk.position_3d)) {
        mesh.position.set(chunk.position_3d[0], chunk.position_3d[1], chunk.position_3d[2]);
    }

    // Update color if provided
    if (chunk.color && chunk.color.startsWith('#')) {
        const color = parseInt(chunk.color.slice(1), 16);
        mesh.material.color.setHex(color);
        mesh.material.emissive.setHex(color);
    }

    // Update metadata
    mesh.userData.content = chunk.content || mesh.userData.content;
    mesh.userData.embedding = chunk.embedding || mesh.userData.embedding;
    mesh.userData.tags = chunk.tags || mesh.userData.tags;

    logViz('DEBUG', `Updated chunk ${chunk.id}`);
}

/**
 * Load and render all chunks for a document
 */
export async function loadDocument(documentId) {
    logViz('INFO', `Loading document ${documentId}...`);

    try {
        const response = await fetch(`/api/documents/${documentId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        logViz('INFO', `Got document: ${data.title}, ${data.chunks.length} chunks`);

        // Clear existing chunks
        chunkMeshes.forEach(mesh => {
            scene.remove(mesh);
            mesh.geometry.dispose();
            mesh.material.dispose();
        });
        chunkMeshes.clear();

        // Add all chunks
        let successCount = 0;
        for (const chunk of data.chunks) {
            try {
                addChunk(chunk);
                successCount++;
            } catch (error) {
                logViz('WARN', `Failed to add chunk: ${error.message}`);
            }
        }

        logViz('INFO', `✅ Loaded ${successCount}/${data.chunks.length} chunks`);

        // Fit camera to scene
        fitCameraToScene();

    } catch (error) {
        logViz('ERROR', `Failed to load document: ${error.message}`);
        throw error;
    }
}

/**
 * Fit camera to view all objects
 */
function fitCameraToScene() {
    if (chunkMeshes.size === 0) {
        logViz('WARN', 'No chunks to fit camera to');
        return;
    }

    const box = new THREE.Box3();
    chunkMeshes.forEach(mesh => {
        box.expandByObject(mesh);
    });

    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5; // Add some padding

    const center = box.getCenter(new THREE.Vector3());
    camera.position.copy(center);
    camera.position.z += cameraZ;

    if (controls) {
        controls.target.copy(center);
        controls.update();
    }

    logViz('DEBUG', `Camera fitted to scene (${chunkMeshes.size} chunks)`);
}

/**
 * Handle canvas click for chunk selection
 */
function onCanvasClick(event) {
    if (!renderer || !camera) return;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    // Calculate mouse position in normalized device coordinates
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Update raycaster
    raycaster.setFromCamera(mouse, camera);

    // Get intersected objects
    const meshArray = Array.from(chunkMeshes.values());
    const intersects = raycaster.intersectObjects(meshArray);

    if (intersects.length > 0) {
        const mesh = intersects[0].object;
        selectChunk(mesh);
    } else {
        deselectChunk();
    }
}

/**
 * Select a chunk and highlight it
 */
function selectChunk(mesh) {
    // Deselect previous
    if (selectedMesh) {
        selectedMesh.material.emissiveIntensity = 0.2;
    }

    // Select new
    selectedMesh = mesh;
    mesh.material.emissiveIntensity = 0.8;

    const chunkId = mesh.userData.chunkId;
    const content = mesh.userData.content;

    logViz('INFO', `Selected chunk: ${chunkId}`);

    // Update UI with chunk info
    updateChunkInfo(chunkId, content);
}

/**
 * Deselect current chunk
 */
function deselectChunk() {
    if (selectedMesh) {
        selectedMesh.material.emissiveIntensity = 0.2;
        selectedMesh = null;
    }
    clearChunkInfo();
}

/**
 * Update UI with chunk information
 */
function updateChunkInfo(chunkId, content) {
    const infoEl = document.getElementById('chunk-info');
    if (infoEl) {
        infoEl.innerHTML = `
            <div class="chunk-info-card">
                <h3>Chunk: ${chunkId}</h3>
                <p>${content.substring(0, 200)}${content.length > 200 ? '...' : ''}</p>
            </div>
        `;
        infoEl.style.display = 'block';
    }
}

/**
 * Clear chunk info from UI
 */
function clearChunkInfo() {
    const infoEl = document.getElementById('chunk-info');
    if (infoEl) {
        infoEl.innerHTML = '';
        infoEl.style.display = 'none';
    }
}

/**
 * Handle window resize
 */
function onWindowResize() {
    if (!renderer || !camera) return;

    const container = document.getElementById('cosmos-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);

    logViz('DEBUG', `Window resized: ${width}x${height}`);
}

/**
 * Animation loop
 */
function animate() {
    requestAnimationFrame(animate);

    if (controls) {
        controls.update();
    }

    // Subtle rotation of all chunks
    chunkMeshes.forEach(mesh => {
        mesh.rotation.x += 0.0002;
        mesh.rotation.y += 0.0003;
    });

    renderer.render(scene, camera);
}

/**
 * Get chunk meshes map for external use
 */
export function getChunkMeshes() {
    return chunkMeshes;
}

/**
 * Dispose of resources
 */
export function disposeCosmos() {
    logViz('INFO', 'Disposing cosmos...');

    if (renderer) {
        renderer.dispose();
    }

    chunkMeshes.forEach(mesh => {
        mesh.geometry.dispose();
        mesh.material.dispose();
    });
    chunkMeshes.clear();

    logViz('INFO', '✅ Cosmos disposed');
}

// Cosmos Renderer - Shared Three.js scene management

import { state } from './state.js';

let scene;
let camera;
let renderer;
let controls;
let animationId = null;
let raycaster;
let mouse;

const chunkMeshes = new Map();
const customObjects = [];
let frameCount = 0;
const LOD_UPDATE_INTERVAL = 30;
let geometryGeneratorInstance = null;

function normalizeShapeSignature(rawSignature) {
    if (!rawSignature) return null;
    if (typeof rawSignature === 'object') {
        return rawSignature;
    }
    if (typeof rawSignature === 'string') {
        try {
            return JSON.parse(rawSignature);
        } catch (error) {
            console.warn('Failed to parse shape signature string', error);
            return null;
        }
    }
    return null;
}

function getGeometryGenerator() {
    if (!geometryGeneratorInstance && typeof window !== 'undefined' && window.ProceduralGeometryGenerator) {
        geometryGeneratorInstance = new window.ProceduralGeometryGenerator();
    }
    return geometryGeneratorInstance;
}

function getStatusElement() {
    return document.getElementById('status-text');
}

function setStatus(message) {
    const el = getStatusElement();
    if (el) {
        el.textContent = message;
    }
}

export function getChunkMeshes() {
    return chunkMeshes;
}

export function addCustomObject(object3D) {
    if (!scene || !object3D) return;
    if (!customObjects.includes(object3D)) {
        customObjects.push(object3D);
    }
    scene.add(object3D);
}

export async function initCosmos() {
    const container = document.getElementById('cosmos-view');
    if (!container) {
        console.warn('Cosmos view container not found');
        return;
    }

    setStatus('Initializing cosmos…');

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000010, 0.00035);

    // Camera
    camera = new THREE.PerspectiveCamera(72, container.clientWidth / container.clientHeight, 0.1, 10000);
    camera.position.set(0, 18, 70);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('cosmos-canvas'),
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance'
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    if ('outputColorSpace' in renderer) {
        renderer.outputColorSpace = THREE.SRGBColorSpace;
    } else if ('outputEncoding' in renderer) {
        renderer.outputEncoding = THREE.sRGBEncoding;
    }
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 8;
    controls.maxDistance = 320;
    controls.maxPolarAngle = Math.PI;

    // Lighting
    const ambient = new THREE.AmbientLight(0x667eea, 0.45);
    scene.add(ambient);

    const keyLight = new THREE.PointLight(0xffffff, 1.2, 400);
    keyLight.position.set(60, 120, 80);
    scene.add(keyLight);

    const fillLight = new THREE.PointLight(0x82a0ff, 0.6, 300);
    fillLight.position.set(-90, -40, -30);
    scene.add(fillLight);

    const rimLight = new THREE.PointLight(0xff8a65, 0.5, 260);
    rimLight.position.set(0, 80, -120);
    scene.add(rimLight);

    // Raycaster for interaction
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 3;
    mouse = new THREE.Vector2();

    // Starfield background
    createStarfield();

    // Event listeners
    window.addEventListener('resize', onWindowResize);
    const canvas = document.getElementById('cosmos-canvas');
    canvas.addEventListener('mousemove', onCosmosMouseMove);
    canvas.addEventListener('click', onCosmosClick);

    // Begin animation loop
    state.setCurrentView('cosmos');
    animateCosmos();
    setStatus('Cosmos ready');
}

function createStarfield() {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];

    for (let i = 0; i < 15000; i++) {
        vertices.push(
            (Math.random() - 0.5) * 2500,
            (Math.random() - 0.5) * 2500,
            (Math.random() - 0.5) * 2500
        );

        const color = new THREE.Color();
        color.setHSL(0.55 + Math.random() * 0.1, 0.35, 0.6 + Math.random() * 0.2);
        colors.push(color.r, color.g, color.b);
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 1.2,
        vertexColors: true,
        transparent: true,
        opacity: 0.8
    });

    const stars = new THREE.Points(geometry, material);
    stars.name = 'starfield';
    scene.add(stars);
}

function materialSupportsEmissive(material) {
    return material && ('emissive' in material) && ('emissiveIntensity' in material);
}

function createChunkMaterial(chunk) {
    const signature = normalizeShapeSignature(chunk?.shape_3d);
    const baseColor = new THREE.Color();
    const fallbackColor = (chunk?.color && typeof chunk.color === 'string' && chunk.color.trim())
        ? chunk.color
        : '#748ffc';
    baseColor.setStyle(fallbackColor);
    const emissive = baseColor.clone().multiplyScalar(0.25);

    let metalness = 0.2;
    let roughness = 0.65;
    if (signature?.texture === 'crystalline') {
        metalness = 0.35;
        roughness = 0.4;
        emissive.multiplyScalar(1.15);
    } else if (signature?.texture === 'nebula') {
        metalness = 0.1;
        roughness = 0.85;
    }

    const material = new THREE.MeshStandardMaterial({
        color: baseColor,
        vertexColors: false,
        emissive,
        emissiveIntensity: 0.55,
        metalness,
        roughness,
        transparent: true,
        opacity: 0.97
    });
    material.toneMapped = true;
    return material;
}

function createPlaceholderGeometry(chunk) {
    const signature = normalizeShapeSignature(chunk?.shape_3d) || {};
    const detail = signature.detail ?? 16;
    const scale = signature.scale ?? 3.4;
    const geometry = new THREE.SphereGeometry(1, detail, detail);
    geometry.scale(scale, scale, scale);
    return geometry;
}

function createGeometryForChunk(chunk) {
    const generator = getGeometryGenerator();
    const signature = normalizeShapeSignature(chunk?.shape_3d);
    if (generator && signature && typeof generator.generatePlanetaryGeometryFromSignature === 'function') {
        try {
            return generator.generatePlanetaryGeometryFromSignature(signature);
        } catch (error) {
            console.warn('Failed to generate geometry from signature', error);
        }
    }
    return createPlaceholderGeometry(chunk);
}

export function updateCosmosData() {
    if (!scene) {
        console.warn('Cosmos scene not initialized yet');
        return;
    }

    // Remove existing meshes
    chunkMeshes.forEach(mesh => {
        scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
    });
    chunkMeshes.clear();

    const chunks = state.chunks || [];
    chunks.forEach((chunk) => {
        const geometry = createGeometryForChunk(chunk);
        const material = createChunkMaterial(chunk);
        const mesh = new THREE.Mesh(geometry, material);

        const targetPosition = Array.isArray(chunk.position_3d) && chunk.position_3d.length === 3
            ? new THREE.Vector3(chunk.position_3d[0], chunk.position_3d[1], chunk.position_3d[2])
            : new THREE.Vector3(
                (Math.random() - 0.5) * 180,
                (Math.random() - 0.5) * 120,
                (Math.random() - 0.5) * 180
            );

        mesh.position.copy(targetPosition);
        mesh.userData = {
            chunk,
            chunkId: chunk.id,
            documentId: chunk.document_id,
            clickHandler: chunk.metadata?.link_url ? () => handleChunkLink(chunk) : null,
            shapeSignature: chunk.shape_3d
        };

        scene.add(mesh);
        chunkMeshes.set(chunk.id || chunk.chunk_id, mesh);
    });

    setStatus(`Cosmos populated · ${chunkMeshes.size} nodes`);
}

function handleChunkLink(chunk) {
    const { metadata } = chunk;
    if (!metadata || !metadata.link_url) return;

    if (metadata.is_external_link) {
        window.open(metadata.link_url, '_blank');
    } else {
        window.location.href = metadata.link_url;
    }
}

export function switchView(view) {
    state.setCurrentView(view);

    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.remove('active');
        if ((view === 'document' && btn.textContent.includes('📄')) ||
            (view === 'cosmos' && btn.textContent.includes('🌌'))) {
            btn.classList.add('active');
        }
    });

    const documentViewEl = document.getElementById('document-view');
    if (documentViewEl) {
        documentViewEl.classList.toggle('active', view === 'document');
    }

    const cosmosViewEl = document.getElementById('cosmos-view');
    if (cosmosViewEl) {
        cosmosViewEl.classList.toggle('active', view === 'cosmos');
    }

    if (view === 'cosmos' && !animationId) {
        animateCosmos();
    }
}

export function focusChunkInCosmos(chunkId) {
    const mesh = chunkMeshes.get(chunkId);
    if (!mesh) return;

    switchView('cosmos');

    const startPos = camera.position.clone();
    const startTarget = controls.target.clone();
    const endTarget = mesh.position.clone();
    const endPos = mesh.position.clone().add(new THREE.Vector3(0, 12, 26));
    let progress = 0;

    function animateCamera() {
        progress += 0.03;
        const eased = 1 - Math.pow(1 - Math.min(progress, 1), 3);
        camera.position.lerpVectors(startPos, endPos, eased);
        controls.target.lerpVectors(startTarget, endTarget, eased);
        controls.update();
        if (progress < 1) {
            requestAnimationFrame(animateCamera);
        }
    }

    animateCamera();
}

function animateCosmos(time = 0) {
    animationId = requestAnimationFrame(animateCosmos);

    frameCount++;
    if (frameCount % LOD_UPDATE_INTERVAL === 0) {
        controls.update();
    }

    chunkMeshes.forEach((mesh, index) => {
        mesh.rotation.y += 0.0008;
        mesh.rotation.x += 0.0004;
        if (materialSupportsEmissive(mesh.material)) {
            const pulse = Math.sin(time * 0.0012 + index * 0.15) * 0.1 + 0.45;
            mesh.material.emissiveIntensity = pulse;
        }
    });

    customObjects.forEach((object) => {
        if (object?.userData && typeof object.userData.animate === 'function') {
            object.userData.animate(time * 0.001);
        }
    });

    const starfield = scene.getObjectByName('starfield');
    if (starfield) {
        starfield.rotation.y += 0.00003;
    }

    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    if (!camera || !renderer) return;
    const container = document.getElementById('cosmos-view');
    if (!container) return;

    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

let hoveredMesh = null;
let selectedMesh = null;

function onCosmosMouseMove(event) {
    if (!scene || !camera) return;

    const canvas = document.getElementById('cosmos-canvas');
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersectables = Array.from(chunkMeshes.values()).concat(customObjects);
    const intersects = raycaster.intersectObjects(intersectables);

    if (hoveredMesh && hoveredMesh !== selectedMesh) {
        hoveredMesh.scale.setScalar(1.0);
        if (materialSupportsEmissive(hoveredMesh.material)) {
            hoveredMesh.material.emissiveIntensity = 0.45;
        }
        hoveredMesh = null;
    }

    if (intersects.length > 0) {
        const mesh = intersects[0].object;
        if (chunkMeshes.has(mesh.userData?.chunkId)) {
            if (mesh !== selectedMesh) {
                mesh.scale.setScalar(1.4);
                if (materialSupportsEmissive(mesh.material)) {
                    mesh.material.emissiveIntensity = 1.2;
                }
            }
            hoveredMesh = mesh;
            showCosmosInfo(mesh.userData.chunk || mesh.userData);
            canvas.style.cursor = 'pointer';
            return;
        } else if (mesh.userData?.clickHandler) {
            canvas.style.cursor = 'pointer';
            return;
        }
    }

    hideCosmosInfo();
    canvas.style.cursor = 'default';
}

function onCosmosClick(event) {
    if (!scene || !camera) return;

    const canvas = document.getElementById('cosmos-canvas');
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersectables = Array.from(chunkMeshes.values()).concat(customObjects);
    const intersects = raycaster.intersectObjects(intersectables);

    if (selectedMesh) {
        selectedMesh.scale.setScalar(1.0);
        if (materialSupportsEmissive(selectedMesh.material)) {
            selectedMesh.material.emissiveIntensity = 0.45;
        }
        selectedMesh = null;
    }

    if (intersects.length > 0) {
        const mesh = intersects[0].object;
        if (mesh.userData?.clickHandler) {
            mesh.userData.clickHandler();
            return;
        }

        if (chunkMeshes.has(mesh.userData?.chunkId)) {
            selectedMesh = mesh;
            mesh.scale.setScalar(1.7);
            if (materialSupportsEmissive(mesh.material)) {
                mesh.material.emissiveIntensity = 1.5;
            }
            showCosmosInfo(mesh.userData.chunk || mesh.userData);
            focusChunkInCosmos(mesh.userData.chunkId || mesh.userData.chunk?.id);
        }
    } else {
        hideCosmosInfo();
    }
}

function showCosmosInfo(chunk) {
    const panel = document.getElementById('cosmos-info');
    if (!panel || !chunk) return;

    const chunkId = chunk.id || chunk.chunk_id || 'unknown';
    document.getElementById('cosmos-info-id').textContent = chunkId;

    const metaText = [];
    if (chunk.chunk_type) metaText.push(chunk.chunk_type);
    if (typeof chunk.chunk_index === 'number') metaText.push(`Index ${chunk.chunk_index}`);
    if (chunk.cluster_label) metaText.push(chunk.cluster_label);
    document.getElementById('cosmos-info-meta').textContent = metaText.join(' • ') || 'Semantic node';

    const preview = chunk.content || chunk.metadata?.description || 'No description available.';
    document.getElementById('cosmos-info-text').textContent = preview.slice(0, 240) + (preview.length > 240 ? '…' : '');

    panel.classList.add('visible');
}

function hideCosmosInfo() {
    const panel = document.getElementById('cosmos-info');
    if (panel) {
        panel.classList.remove('visible');
    }
}
// Cosmos Renderer - Shared Three.js scene management

import { state } from './state.js';

if (typeof THREE !== 'undefined' && THREE.ColorManagement) {
    THREE.ColorManagement.enabled = true;
}

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

const tmpAxis = new THREE.Vector3();

function randomUnitVector() {
    tmpAxis.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5);
    if (tmpAxis.lengthSq() === 0) {
        tmpAxis.set(1, 0, 0);
    }
    return tmpAxis.normalize().clone();
}

function applyArrivalPulse(mesh) {
    const start = performance.now();
    const duration = 900;
    const initialScale = mesh.scale.x;

    const animate = () => {
        const elapsed = performance.now() - start;
        const t = Math.min(1, elapsed / duration);
        const damped = 1 - Math.pow(1 - t, 4);
        const pulse = 1 + 0.04 * Math.sin(t * Math.PI * 3) * (1 - t);
        mesh.scale.setScalar(initialScale * pulse);
        if (mesh.userData?.fillMesh) {
            mesh.userData.fillMesh.scale.setScalar(initialScale * pulse);
        }
        if (t < 1) {
            requestAnimationFrame(animate);
        } else {
            mesh.scale.setScalar(initialScale);
            if (mesh.userData?.fillMesh) {
                mesh.userData.fillMesh.scale.setScalar(initialScale);
            }
        }
    };
    animate();
}

function getDebugConfig() {
    return {
        lightHelpers: false,
        normalHelpers: false,
        forceDoubleSide: false
    };
}

function getHexStringSafe(value) {
    if (value === undefined || value === null) {
        return 'null';
    }

    if (typeof value === 'string') {
        return value.startsWith('#') ? value.slice(1) : value;
    }

    if (typeof value === 'number') {
        return value.toString(16);
    }

    if (typeof value === 'object') {
        if (typeof value.getHexString === 'function') {
            return value.getHexString();
        }
        if (typeof value.getHex === 'function') {
            const hex = value.getHex();
            if (typeof hex === 'number') {
                return hex.toString(16);
            }
            return String(hex);
        }
    }

    return String(value);
}

function resolvePositionOverlap(target, usedPositions, options = {}) {
    const {
        minDistance = 6,
        maxAttempts = 18,
        jitterRadius = 12
    } = options;

    if (!target) {
        return new THREE.Vector3();
    }

    const candidate = target.clone();
    let attempts = 0;

    const isTooClose = (vec) => usedPositions.some(pos => pos.distanceTo(vec) < minDistance);

    if (!isTooClose(candidate)) {
        return candidate;
    }

    while (attempts < maxAttempts) {
        const radius = jitterRadius * (0.35 + (attempts / maxAttempts));
        candidate.set(
            target.x + (Math.random() - 0.5) * radius,
            target.y + (Math.random() - 0.5) * radius,
            target.z + (Math.random() - 0.5) * radius
        );

        if (!isTooClose(candidate)) {
            return candidate;
        }

        attempts += 1;
    }

    // As a fallback, push along Z-axis slightly
    return target.clone().add(new THREE.Vector3(0, 0, minDistance * 0.75));
}

function analyzeGeometryNormals(geometry, label = '') {
    if (!geometry || !geometry.attributes) {
        console.warn(`[GEOMETRY] ${label} missing geometry attributes`);
        return {
            hasIssues: true,
            reason: 'missing-attributes'
        };
    }

    const normalsAttr = geometry.attributes.normal;
    if (!normalsAttr) {
        console.warn(`[GEOMETRY] ${label} missing normal attribute`);
        return {
            hasIssues: true,
            reason: 'missing-normal-attribute'
        };
    }

    const array = normalsAttr.array;
    if (!array || array.length === 0) {
        console.warn(`[GEOMETRY] ${label} normal attribute empty`);
        return {
            hasIssues: true,
            reason: 'empty-normal-array'
        };
    }

    let zeroCount = 0;
    let nanCount = 0;
    let minLength = Infinity;
    let maxLength = 0;
    let totalLength = 0;
    const vector = new THREE.Vector3();

    for (let i = 0; i < array.length; i += 3) {
        vector.set(array[i], array[i + 1], array[i + 2]);
        const length = vector.length();

        if (!Number.isFinite(length)) {
            nanCount += 1;
            continue;
        }

        if (length < 1e-6) {
            zeroCount += 1;
        }

        minLength = Math.min(minLength, length);
        maxLength = Math.max(maxLength, length);
        totalLength += length;
    }

    const sampleCount = array.length / 3;
    const avgLength = totalLength / sampleCount;
    const zeroRatio = sampleCount > 0 ? zeroCount / sampleCount : 0;

    console.log(`[GEOMETRY] Normals for ${label}: min=${minLength.toFixed(3)}, max=${maxLength.toFixed(3)}, avg=${avgLength.toFixed(3)}, zero=${zeroCount} (${(zeroRatio * 100).toFixed(2)}%), nan=${nanCount}, vertices=${sampleCount}`);

    const MIN_LENGTH_THRESHOLD = 0.001;
    const hasIssues =
        nanCount > 0 ||
        zeroRatio > 0.02 ||
        (minLength < MIN_LENGTH_THRESHOLD && zeroRatio > 0.005);
    if (hasIssues) {
        console.warn(`[GEOMETRY] Suspicious normals detected for ${label}. Consider recomputing or checking deformation pipeline.`);
    }

    return {
        hasIssues,
        zeroCount,
        zeroRatio,
        nanCount,
        minLength,
        maxLength,
        avgLength,
        sampleCount
    };
}

function repairGeometryNormals(geometry, label = '') {
    if (!geometry || typeof geometry.computeVertexNormals !== 'function') {
        return;
    }

    geometry.deleteAttribute('normal');
    geometry.computeVertexNormals();
    if (typeof geometry.normalizeNormals === 'function') {
        geometry.normalizeNormals();
    }
    if (geometry.attributes?.normal) {
        geometry.attributes.normal.needsUpdate = true;
    }
    console.log(`[GEOMETRY] Recomputed normals for ${label}`);
}

function getMaterialOverride() {
    if (typeof window === 'undefined') {
        return null;
    }

    try {
        const params = new URLSearchParams(window.location.search);
        const overrideParam = params.get('cosmosMaterial');

        const debugMaterial =
            window.__COSMOS_DEBUG__?.material ||
            (window.__COSMOS_DEBUG__?.forcePhongMaterial ? 'phong' : null) ||
            window.__COSMOS_DEBUG__?.defaultMaterial;

        const fallback = window.__COSMOS_CONFIG__?.defaultMaterial || null;

        const selected = (overrideParam || debugMaterial || fallback || '').toLowerCase();

        if (selected === 'standard' || selected === 'meshstandard') {
            return null;
        }

        if (selected === 'basic' || selected === 'phong' || selected === 'lambert' ||
            selected === 'meshbasic' || selected === 'meshphong' || selected === 'meshlambert') {
            return selected.startsWith('mesh') ? selected.replace('mesh', '') : selected;
        }

        return null;
    } catch (error) {
        console.warn('[MATERIAL] Unable to read material override preference:', error);
        return null;
    }
}

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
    const debug = getDebugConfig();

    const globalLights = [];
    object3D.traverse((child) => {
        if (child?.userData?.isCosmosGlobalLight && child.isLight) {
            child.distance = 0;
            child.decay = 1.4;
            child.intensity = 260;
            child.color = new THREE.Color(0xffb36d);
        }
    });

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
    scene.background = new THREE.Color(0x00040e);
    scene.fog = new THREE.Fog(0x00040e, 600, 2400);

    // Camera
    camera = new THREE.PerspectiveCamera(72, container.clientWidth / container.clientHeight, 0.1, 10000);
    camera.position.set(0, 2, -65);  // Positioned behind the home planet at (0, -6, -42)

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('cosmos-canvas'),
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance'
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    if ('physicallyCorrectLights' in renderer) {
        renderer.physicallyCorrectLights = true;
    }
    if ('outputColorSpace' in renderer) {
        renderer.outputColorSpace = THREE.SRGBColorSpace;
    } else if ('outputEncoding' in renderer) {
        renderer.outputEncoding = THREE.sRGBEncoding;
    }
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.85;
    const debug = getDebugConfig();
    if (debug.logRenderer) {
        console.log('[RENDERER] Config:', {
            physicallyCorrectLights: renderer.physicallyCorrectLights,
            toneMapping: renderer.toneMapping,
            toneMappingExposure: renderer.toneMappingExposure,
            outputColorSpace: renderer.outputColorSpace ?? renderer.outputEncoding,
            colorManagement: THREE.ColorManagement?.enabled
        });
    }

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -6, -42);  // Point at home planet initially
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 8;
    controls.maxDistance = 320;
    controls.maxPolarAngle = Math.PI;

    // Lighting
    const sunLight = new THREE.PointLight(0xffffff, 420, 0, 1.8);
    sunLight.position.set(0, 0, 0);
    sunLight.castShadow = false;
    sunLight.name = 'cosmos-sun-light';
    scene.add(sunLight);
    console.log('[LIGHT] Sun light:', {
        intensity: sunLight.intensity,
        decay: sunLight.decay,
        distance: sunLight.distance,
        position: sunLight.position.toArray()
    });
    if (debug.lightHelpers) {
        const helperColor = 0xffaa33;
        const sunHelper = new THREE.PointLightHelper(sunLight, 40, helperColor);
        sunHelper.userData.isCosmosLightHelper = true;
        scene.add(sunHelper);
    }

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
    canvas.addEventListener('touchstart', onCosmosTouchStart, false);

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
    const material = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        wireframe: true,
        toneMapped: false,
        transparent: true,
        opacity: 0.65
    });
    material.name = 'ChunkWireframe';
    return material;
}

function createFillMaterial(chunk) {
    let colorHex = 0xffffff;
    if (chunk.color && typeof chunk.color === 'string' && chunk.color.startsWith('#')) {
        colorHex = chunk.color.trim();
    }

    const color = new THREE.Color(colorHex);
    if (typeof color.convertSRGBToLinear === 'function') {
        color.convertSRGBToLinear();
    }

    const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.75,
        metalness: 0.05,
        side: THREE.DoubleSide,
        flatShading: false
    });
    material.toneMapped = true;
    material.name = 'ChunkFillMaterial';
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

    console.log(`[GEOMETRY] Generator available: ${!!generator}, signature valid: ${!!signature}`);
    if (signature) {
        console.log(`[GEOMETRY] Signature type=${signature.type}, texture=${signature.texture}, detail=${signature.detail}`);
    }

    if (generator && signature && typeof generator.generatePlanetaryGeometryFromSignature === 'function') {
        try {
            console.log(`[GEOMETRY] Attempting to generate procedural geometry...`);
            const geom = generator.generatePlanetaryGeometryFromSignature(signature);
            console.log(`[GEOMETRY] SUCCESS: Generated geometry with ${geom.attributes.position.count} vertices`);
            const normalStats = analyzeGeometryNormals(geom, `chunk ${chunk?.id ?? 'unknown'}`);
            if (normalStats?.hasIssues) {
                repairGeometryNormals(geom, `chunk ${chunk?.id ?? 'unknown'}`);
                analyzeGeometryNormals(geom, `chunk ${chunk?.id ?? 'unknown'} (post-repair)`);
            }
            return geom;
        } catch (error) {
            console.warn(`[GEOMETRY] ERROR generating geometry:`, error.message);
        }
    } else {
        console.log(`[GEOMETRY] Falling back to placeholder - generator=${!!generator}, hasMethod=${generator && typeof generator.generatePlanetaryGeometryFromSignature === 'function'}`);
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
        if (mesh.userData?.fillMesh) {
            mesh.userData.fillMesh.geometry.dispose();
            mesh.userData.fillMesh.material.dispose();
        }
        mesh.geometry.dispose();
        mesh.material.dispose();
    });
    chunkMeshes.clear();

    const chunks = state.chunks || [];
    console.log(`[COSMOS] Starting mesh creation with ${chunks.length} chunks`);

    const usedPositions = [];

    chunks.forEach((chunk, idx) => {
        console.log(`[COSMOS] Chunk ${idx} raw data:`, {
            id: chunk.id,
            color: chunk.color,
            shape_3d: chunk.shape_3d,
            position_3d: chunk.position_3d
        });

        const geometry = createGeometryForChunk(chunk);
        const material = createChunkMaterial(chunk);
        const mesh = new THREE.Mesh(geometry, material);
        const baseFinalGeometry = geometry.clone();
        const fillMaterial = createFillMaterial(chunk);
        const fillMesh = new THREE.Mesh(baseFinalGeometry.clone(), fillMaterial);
        fillMesh.castShadow = false;
        fillMesh.receiveShadow = true;
        fillMesh.name = 'chunk-fill';
        mesh.add(fillMesh);

        const rotationAxis = randomUnitVector();
        const rotationSpeed = 0.0002 + Math.random() * 0.0009;
        const baseScale = 0.9 + Math.random() * 0.25;
        mesh.scale.setScalar(baseScale);
        fillMesh.scale.setScalar(baseScale);

        console.log(`[COSMOS] Chunk ${idx} (id=${chunk.id}): material=${material.type}, color=${getHexStringSafe(material.color)}`);
        console.log(`[COSMOS]   Geometry vertices: ${geometry.attributes?.position?.count || 'N/A'}`);

        const targetPosition = Array.isArray(chunk.position_3d) && chunk.position_3d.length === 3
            ? new THREE.Vector3(chunk.position_3d[0], chunk.position_3d[1], chunk.position_3d[2])
            : new THREE.Vector3(
                (Math.random() - 0.5) * 180,
                (Math.random() - 0.5) * 120,
                (Math.random() - 0.5) * 180
            );

        const resolvedPosition = resolvePositionOverlap(
            targetPosition,
            usedPositions,
            {
                minDistance: 8,
                maxAttempts: 20,
                jitterRadius: 18
            }
        );
        if (!resolvedPosition.equals(targetPosition)) {
            console.log(`[COSMOS] Applied jitter to chunk ${chunk.id}: original=${targetPosition.toArray().map(v => v.toFixed(2))} resolved=${resolvedPosition.toArray().map(v => v.toFixed(2))}`);
        }

        mesh.position.copy(resolvedPosition);
        mesh.userData = {
            chunk,
            chunkId: chunk.id,
            documentId: chunk.document_id,
            clickHandler: chunk.metadata?.link_url ? () => handleChunkLink(chunk) : null,
            shapeSignature: chunk.shape_3d,
            originalPosition: targetPosition.clone(),
            resolvedPosition: resolvedPosition.clone(),
            baseFinalGeometry,
            fillMesh,
            rotationAxis,
            rotationSpeed,
            baseScale
        };

        usedPositions.push(resolvedPosition.clone());

        applyArrivalPulse(mesh);

        console.log(`[COSMOS]   Position diagnostics: position=(${mesh.position.x.toFixed(3)}, ${mesh.position.y.toFixed(3)}, ${mesh.position.z.toFixed(3)}), dist=${mesh.position.length().toFixed(3)}`);

        scene.add(mesh);
        chunkMeshes.set(chunk.id || chunk.chunk_id, mesh);
        console.log(`[COSMOS] Added mesh to scene: uuid=${mesh.uuid}, visible=${mesh.visible}, material.color=${getHexStringSafe(mesh.material.color)}`);

        setTimeout(() => {
            if (chunkMeshes.has(chunk.id || chunk.chunk_id)) {
                const m = chunkMeshes.get(chunk.id || chunk.chunk_id);
                const inScene = scene.children.includes(m);
                console.log(`[COSMOS] After add to scene (10ms later): mesh.material.color=${getHexStringSafe(m.material.color)}, in scene=${inScene}, mesh.visible=${m.visible}`);

                if (renderer) {
                    renderer.render(scene, camera);
                    console.log(`[COSMOS] Rendered scene - should see the mesh now`);
                }
            }
        }, 10);
    });

    console.log(`[COSMOS] Scene has ${chunkMeshes.size} meshes. Camera pos: ${camera.position.x.toFixed(1)}, ${camera.position.y.toFixed(1)}, ${camera.position.z.toFixed(1)}`);
    console.log(`[COSMOS] Lights in scene: ${scene.children.filter(c => c.isLight).length}`);
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

    // Find all chunks in the same document (the "group")
    const documentId = mesh.userData.documentId;
    const groupMeshes = Array.from(chunkMeshes.values()).filter(
        m => m.userData.documentId === documentId
    );

    // The target (center point) is the selected chunk
    const targetPos = mesh.position.clone();

    // Calculate centroid of OTHER chunks to determine viewing direction
    const otherChunks = groupMeshes.filter(m => m !== mesh);
    let otherCenter = targetPos.clone();
    if (otherChunks.length > 0) {
        otherCenter = new THREE.Vector3();
        otherChunks.forEach(m => {
            otherCenter.add(m.position);
        });
        otherCenter.divideScalar(otherChunks.length);
    }

    // Direction from selected chunk toward other chunks (in horizontal plane)
    const directionToOthers = otherCenter.clone().sub(targetPos);
    directionToOthers.y = 0; // Keep in horizontal plane only
    directionToOthers.normalize();

    // Position camera opposite to other chunks, slightly elevated
    const viewDistance = 25;
    const endPos = targetPos.clone()
        .sub(directionToOthers.multiplyScalar(viewDistance))
        .add(new THREE.Vector3(0, 8, 0)); // Slight elevation for better perspective

    const startPos = camera.position.clone();
    const startTarget = controls.target.clone();

    let progress = 0;
    const duration = 1.5; // Slower animation

    function animateCamera() {
        progress += 1 / 60 / duration;
        if (progress > 1) progress = 1;

        // Smoother easing function for less jarring motion
        const eased = progress < 0.5
            ? 2 * progress * progress
            : -1 + (4 - 2 * progress) * progress;

        // Interpolate position and target together for unified motion
        camera.position.lerpVectors(startPos, endPos, eased);
        controls.target.lerpVectors(startTarget, targetPos, eased);

        // Force upright orientation
        camera.up.set(0, 1, 0);
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
    if (frameCount === 1) {
        console.log(`[RENDER] First frame: renderer.outputColorSpace=${renderer.outputColorSpace}, toneMapping=${renderer.toneMapping}, scene.background=${scene.background ? scene.background.getHexString() : 'null'}`);
        console.log(`[RENDER] Scene has ${chunkMeshes.size} meshes visible`);
        const lights = scene.children.filter(c => c.isLight);
        console.log(`[RENDER] Lights: ${lights.map(l => `${l.type}(intensity=${l.intensity}, color=${l.color.getHexString()})`).join(', ')}`);
        chunkMeshes.forEach((mesh, idx) => {
            if (idx < 3) {
        console.log(`[RENDER] Mesh ${idx}: pos=(${mesh.position.x.toFixed(1)},${mesh.position.y.toFixed(1)},${mesh.position.z.toFixed(1)}), color=${getHexStringSafe(mesh.material.color)}, emissive=${getHexStringSafe(mesh.material.emissive)}, emissiveIntensity=${mesh.material.emissiveIntensity}, metalness=${mesh.material.metalness}, roughness=${mesh.material.roughness}`);
            }
        });
    }

    if (frameCount % LOD_UPDATE_INTERVAL === 0) {
        controls.update();
    }

    chunkMeshes.forEach((mesh) => {
        const axis = mesh.userData?.rotationAxis;
        const speed = mesh.userData?.rotationSpeed ?? 0.0004;
        if (axis) {
            mesh.rotateOnAxis(axis, speed);
        } else {
            mesh.rotation.y += speed;
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
let selectedChunkId = null;

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
            // On hover: show title near mouse position
            showChunkTooltip(mesh.userData.chunk || mesh.userData, event);
            canvas.style.cursor = 'pointer';
            return;
        } else if (mesh.userData?.clickHandler) {
            // Show tooltip for clickable custom objects like home planet
            if (mesh.userData?.isHomePlanet) {
                showChunkTooltip({ title: 'Return to index' }, event);
            }
            canvas.style.cursor = 'pointer';
            return;
        }
    }

    hideChunkTooltip();
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
            // On click: hide tooltip and show full info panel
            hideChunkTooltip();
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
    selectedChunkId = chunkId;

    document.getElementById('cosmos-info-id').textContent = chunkId;

    const metaText = [];
    if (chunk.chunk_type) metaText.push(chunk.chunk_type);
    if (typeof chunk.chunk_index === 'number') metaText.push(`Chunk ${chunk.chunk_index + 1}`);
    if (chunk.cluster_label) metaText.push(chunk.cluster_label);
    document.getElementById('cosmos-info-meta').textContent = metaText.join(' • ') || 'Semantic node';

    const fullContent = chunk.content || chunk.metadata?.description || 'No description available.';
    document.getElementById('cosmos-info-text').textContent = fullContent;

    // Setup navigation buttons
    updateNavigationButtons(chunk);

    panel.classList.add('visible');
}

function updateNavigationButtons(currentChunk) {
    if (!currentChunk) return;

    const documentId = currentChunk.document_id;
    const currentIndex = currentChunk.chunk_index;

    // Get all chunks in the same document, sorted by index
    const documentChunks = state.chunks
        .filter(c => c.document_id === documentId)
        .sort((a, b) => (a.chunk_index ?? 0) - (b.chunk_index ?? 0));

    const currentPosition = documentChunks.findIndex(c => (c.id || c.chunk_id) === selectedChunkId);
    const hasPrevious = currentPosition > 0;
    const hasNext = currentPosition < documentChunks.length - 1;

    const navPanel = document.getElementById('cosmos-nav');
    const prevBtn = document.getElementById('cosmos-nav-prev');
    const nextBtn = document.getElementById('cosmos-nav-next');

    // Show nav panel if there are multiple chunks
    if (navPanel) {
        if (documentChunks.length > 1) {
            navPanel.classList.add('visible');
        } else {
            navPanel.classList.remove('visible');
        }
    }

    if (prevBtn) {
        prevBtn.disabled = !hasPrevious;
        prevBtn.onclick = () => {
            if (hasPrevious) {
                const prevChunk = documentChunks[currentPosition - 1];
                const prevMesh = chunkMeshes.get(prevChunk.id || prevChunk.chunk_id);
                if (prevMesh) {
                    selectAndFocusChunk(prevMesh, prevChunk);
                }
            }
        };
    }

    if (nextBtn) {
        nextBtn.disabled = !hasNext;
        nextBtn.onclick = () => {
            if (hasNext) {
                const nextChunk = documentChunks[currentPosition + 1];
                const nextMesh = chunkMeshes.get(nextChunk.id || nextChunk.chunk_id);
                if (nextMesh) {
                    selectAndFocusChunk(nextMesh, nextChunk);
                }
            }
        };
    }
}

function selectAndFocusChunk(mesh, chunk) {
    if (selectedMesh) {
        selectedMesh.scale.setScalar(1.0);
        if (materialSupportsEmissive(selectedMesh.material)) {
            selectedMesh.material.emissiveIntensity = 0.45;
        }
    }

    selectedMesh = mesh;
    mesh.scale.setScalar(1.7);
    if (materialSupportsEmissive(mesh.material)) {
        mesh.material.emissiveIntensity = 1.5;
    }

    hideChunkTooltip();
    showCosmosInfo(chunk);
    focusChunkInCosmos(chunk.id || chunk.chunk_id);
}

function hideCosmosInfo() {
    const panel = document.getElementById('cosmos-info');
    if (panel) {
        panel.classList.remove('visible');
    }
    const navPanel = document.getElementById('cosmos-nav');
    if (navPanel) {
        navPanel.classList.remove('visible');
    }
}

function showChunkTooltip(chunk, event) {
    const tooltip = document.getElementById('chunk-tooltip');
    if (!tooltip || !chunk) return;

    // Get chunk title
    const chunkTitle = chunk.title || chunk.content?.split('\n')[0]?.slice(0, 60) || 'Untitled chunk';

    tooltip.textContent = chunkTitle;

    // Position tooltip near mouse, with small offset to avoid cursor
    const offsetX = 12;
    const offsetY = 12;
    tooltip.style.left = (event.clientX + offsetX) + 'px';
    tooltip.style.top = (event.clientY + offsetY) + 'px';

    tooltip.classList.add('visible');
}

function hideChunkTooltip() {
    const tooltip = document.getElementById('chunk-tooltip');
    if (tooltip) {
        tooltip.classList.remove('visible');
    }
}

function onCosmosTouchStart(event) {
    // Handle touch interaction same as click
    if (!scene || !camera || !event.touches || event.touches.length === 0) return;

    const touch = event.touches[0];
    const canvas = document.getElementById('cosmos-canvas');
    const rect = canvas.getBoundingClientRect();

    // Convert touch coordinates to normalized device coordinates
    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;

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
            // On touch: hide tooltip and show full info panel
            hideChunkTooltip();
            showCosmosInfo(mesh.userData.chunk || mesh.userData);
            focusChunkInCosmos(mesh.userData.chunkId || mesh.userData.chunk?.id);
        }
    } else {
        hideCosmosInfo();
    }
}
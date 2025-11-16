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

function getDebugConfig() {
    const defaults = {
        lightHelpers: true,
        normalHelpers: true,
        logRenderer: true,
        forceDoubleSide: false
    };

    if (typeof window === 'undefined') {
        return defaults;
    }

    window.__COSMOS_DEBUG__ = window.__COSMOS_DEBUG__ || {};
    return Object.assign({}, defaults, window.__COSMOS_DEBUG__);
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

function createNormalsDebugHelper(mesh, length = 1.8) {
    const debug = getDebugConfig();
    if (!debug.normalHelpers || !mesh.geometry?.attributes?.position || !mesh.geometry?.attributes?.normal) {
        return;
    }

    const positions = mesh.geometry.attributes.position;
    const normals = mesh.geometry.attributes.normal;
    const linePositions = new Float32Array(normals.count * 2 * 3);

    const start = new THREE.Vector3();
    const normal = new THREE.Vector3();

    for (let i = 0; i < normals.count; i++) {
        start.set(
            positions.getX(i),
            positions.getY(i),
            positions.getZ(i)
        );
        normal.set(
            normals.getX(i),
            normals.getY(i),
            normals.getZ(i)
        ).normalize().multiplyScalar(length);

        linePositions[i * 6 + 0] = start.x;
        linePositions[i * 6 + 1] = start.y;
        linePositions[i * 6 + 2] = start.z;

        linePositions[i * 6 + 3] = start.x + normal.x;
        linePositions[i * 6 + 4] = start.y + normal.y;
        linePositions[i * 6 + 5] = start.z + normal.z;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));

    const material = new THREE.LineBasicMaterial({ color: 0xff33ff, toneMapped: false });
    const helper = new THREE.LineSegments(geometry, material);
    helper.userData.isCosmosNormalHelper = true;
    mesh.add(helper);
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
            globalLights.push(child);
        }
    });

    scene.add(object3D);

    globalLights.forEach((light) => {
        if (light.parent) {
            light.parent.remove(light);
        }
        light.position.set(0, 0, 0);
        light.distance = 0;
        light.decay = 2;
        light.intensity = 520;
        light.castShadow = false;
        if (!scene.children.includes(light)) {
            scene.add(light);
        }
        if (debug.lightHelpers) {
            const helper = new THREE.PointLightHelper(light, 40, 0xffaa33);
            helper.userData.isCosmosLightHelper = true;
            scene.add(helper);
        }
    });
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
    scene.background = new THREE.Color(0x001a33);
    scene.fog = new THREE.Fog(0x001a33, 500, 2000);

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
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 8;
    controls.maxDistance = 320;
    controls.maxPolarAngle = Math.PI;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.42);
    scene.add(ambientLight);

    const rimLight = new THREE.PointLight(0xffffff, 140, 0, 2);
    rimLight.position.set(120, 220, 260);
    scene.add(rimLight);

    const sunLight = new THREE.PointLight(0xffffff, 520, 0, 2);
    sunLight.position.set(0, 0, 0);
    sunLight.castShadow = false;
    scene.add(sunLight);
    console.log('[LIGHT] Ambient intensity:', ambientLight.intensity);
    console.log('[LIGHT] Rim light:', {
        intensity: rimLight.intensity,
        decay: rimLight.decay,
        distance: rimLight.distance,
        position: rimLight.position.toArray()
    });
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

        const rimHelper = new THREE.PointLightHelper(rimLight, 30, 0x33aaff);
        rimHelper.userData.isCosmosLightHelper = true;
        scene.add(rimHelper);
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
    let colorHex = 0x748ffc;
    if (chunk.color && typeof chunk.color === 'string' && chunk.color.startsWith('#')) {
        colorHex = chunk.color.trim();
    }

    const logColor = typeof colorHex === 'string' ? colorHex : `#${colorHex.toString(16)}`;
    const srgbColor = new THREE.Color(colorHex);
    const linearColor = srgbColor.clone();
    if (typeof linearColor.convertSRGBToLinear === 'function') {
        linearColor.convertSRGBToLinear();
    }
    const emissiveLinear = linearColor.clone().multiplyScalar(0.15);

    const override = getMaterialOverride();
    if (override === 'basic') {
        console.log(`[MATERIAL] Using MeshBasicMaterial override with color=${srgbColor.getHexString()}`);
        return new THREE.MeshBasicMaterial({
            color: srgbColor,
            transparent: false
        });
    }

    if (override === 'phong') {
        console.log(`[MATERIAL] Using MeshPhongMaterial override with color=${srgbColor.getHexString()}`);
        return new THREE.MeshPhongMaterial({
            color: srgbColor,
            emissive: linearColor.clone().multiplyScalar(0.18),
            emissiveIntensity: 1.0,
            shininess: 36,
            specular: new THREE.Color(0xffffff)
        });
    }

    if (override === 'lambert') {
        console.log(`[MATERIAL] Using MeshLambertMaterial override with color=${linearColor.getHexString()}`);
        return new THREE.MeshLambertMaterial({
            color: linearColor.clone(),
            emissive: linearColor.clone().multiplyScalar(0.18),
            emissiveIntensity: 1.0
        });
    }

    console.log(`[MATERIAL] Creating MeshStandardMaterial with color=${logColor}, colorLinear=${linearColor.getHexString()}`);
    const material = new THREE.MeshStandardMaterial({
        color: linearColor,
        emissive: emissiveLinear,
        emissiveIntensity: 1.0,
        roughness: 0.42,
        metalness: 0.12,
        envMapIntensity: 0.45
    });
    material.toneMapped = true;
    material.needsUpdate = true;
    console.log(`[MATERIAL] Created material type=${material.type}`);
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

        console.log(`[COSMOS] Chunk ${idx} (id=${chunk.id}): color=${chunk.color}`);
        console.log(`[COSMOS]   Geometry type: ${geometry.type}, vertices: ${geometry.attributes?.position?.count || 'N/A'}`);
        console.log(`[COSMOS]   Material type: ${material.type}`);
        console.log(`[COSMOS]   Material.color=${getHexStringSafe(material.color)}`);
        console.log(`[COSMOS]   Material.emissive=${getHexStringSafe(material.emissive)}`);
    const colorVector = material.color
        ? `(${material.color.r.toFixed(4)}, ${material.color.g.toFixed(4)}, ${material.color.b.toFixed(4)})`
        : 'null';
    const emissiveVector = material.emissive
        ? `(${material.emissive.r.toFixed(4)}, ${material.emissive.g.toFixed(4)}, ${material.emissive.b.toFixed(4)})`
        : 'null';
    console.log(`[COSMOS]   Material numeric values: color=${colorVector}, emissive=${emissiveVector}, emissiveIntensity=${material.emissiveIntensity?.toFixed?.(4) ?? material.emissiveIntensity}, roughness=${material.roughness ?? 'n/a'}, metalness=${material.metalness ?? 'n/a'}, opacity=${material.opacity}`);

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
            resolvedPosition: resolvedPosition.clone()
        };

        usedPositions.push(resolvedPosition.clone());

        const debug = getDebugConfig();
        if (debug.forceDoubleSide && mesh.material) {
            mesh.material.side = THREE.DoubleSide;
            mesh.material.needsUpdate = true;
        }

    console.log(`[COSMOS]   Position diagnostics: position=(${mesh.position.x.toFixed(3)}, ${mesh.position.y.toFixed(3)}, ${mesh.position.z.toFixed(3)}), dist=${mesh.position.length().toFixed(3)}`);

        scene.add(mesh);
        chunkMeshes.set(chunk.id || chunk.chunk_id, mesh);
        console.log(`[COSMOS] Added mesh to scene: uuid=${mesh.uuid}, visible=${mesh.visible}, material.color=${getHexStringSafe(mesh.material.color)}`);
        console.log('[COSMOS] Material diagnostics:', {
            type: mesh.material.type,
            color: getHexStringSafe(mesh.material.color),
            emissive: getHexStringSafe(mesh.material.emissive),
            opacity: mesh.material.opacity,
            transparent: mesh.material.transparent,
            depthWrite: mesh.material.depthWrite,
            depthTest: mesh.material.depthTest,
            side: mesh.material.side,
            colorWrite: mesh.material.colorWrite,
            needsUpdate: mesh.material.needsUpdate
        });

        createNormalsDebugHelper(mesh);

    if (debug.cloneBasicPreview && mesh.material) {
        const previewMaterial = new THREE.MeshBasicMaterial({
            color: mesh.material.color.clone ? mesh.material.color.clone() : new THREE.Color(0xffffff),
            wireframe: true,
            toneMapped: false
        });
        const preview = new THREE.Mesh(mesh.geometry.clone(), previewMaterial);
        preview.scale.multiplyScalar(1.01);
        preview.position.copy(mesh.position);
        preview.userData.isCosmosDebugPreview = true;
        scene.add(preview);
        console.log('[COSMOS] Added debug basic preview mesh');
    }

        // Log initial rendering state
        setTimeout(() => {
            if (chunkMeshes.has(chunk.id || chunk.chunk_id)) {
                const m = chunkMeshes.get(chunk.id || chunk.chunk_id);
                const inScene = scene.children.includes(m);
                console.log(`[COSMOS] After add to scene (10ms later): mesh.material.color=${getHexStringSafe(m.material.color)}, emissive=${getHexStringSafe(m.material.emissive)}, in scene=${inScene}, mesh.visible=${m.visible}`);

                // Try to render once to see what color we get
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
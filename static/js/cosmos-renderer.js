// Cosmos Renderer - Three.js 3D Visualization
// Note: This file uses THREE global from CDN script

import { state } from './state.js';
import { COSMOS_SETTINGS } from './config.js';
import { showAddModal } from './modal-manager.js';

// Three.js variables
let scene, camera, renderer, controls;
let chunkMeshes = new Map();
let connectionLines = [];
let animationId = null;
let gravityEnabled = COSMOS_SETTINGS.gravity;
let animationSpeed = COSMOS_SETTINGS.animationSpeed;
let nebulae = [];
let raycaster, mouse;

// Shaders
const starVertexShader = `
    attribute float size;
    attribute vec3 customColor;
    varying vec3 vColor;
    void main() {
        vColor = customColor;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const starFragmentShader = `
    varying vec3 vColor;
    void main() {
        vec2 cxy = 2.0 * gl_PointCoord - 1.0;
        float r = dot(cxy, cxy);
        if (r > 1.0) { discard; }
        float glow = pow(1.0 - r, 2.5);
        float core = pow(1.0 - r * 0.5, 4.0);
        float alpha = glow + core * 0.5;
        vec3 finalColor = vColor * (1.0 + core);
        gl_FragColor = vec4(finalColor, alpha);
    }
`;

const nebulaVertexShader = `
    attribute float alpha;
    attribute float size;
    uniform float time;
    varying float vAlpha;
    void main() {
        vAlpha = alpha;
        vec3 animatedPos = position;
        animatedPos.x += sin(time * 0.1 + position.y * 0.01) * 2.0;
        animatedPos.y += cos(time * 0.1 + position.x * 0.01) * 2.0;
        vec4 mvPosition = modelViewMatrix * vec4(animatedPos, 1.0);
        gl_PointSize = size * (200.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const nebulaFragmentShader = `
    uniform vec3 nebulaColor;
    varying float vAlpha;
    void main() {
        vec2 cxy = 2.0 * gl_PointCoord - 1.0;
        float r = dot(cxy, cxy);
        if (r > 1.0) { discard; }
        float softness = 1.0 - pow(r, 0.5);
        float alpha = softness * vAlpha * 0.3;
        vec3 color = nebulaColor * (0.8 + 0.4 * softness);
        gl_FragColor = vec4(color, alpha);
    }
`;

export function initCosmos() {
    const container = document.getElementById('cosmos-view');

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000010, 0.0003);

    // Camera
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 10000);
    camera.position.set(0, 50, 150);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('cosmos-canvas'),
        antialias: true,
        alpha: true,
        powerPreference: "high-performance"
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 10;
    controls.maxDistance = 500;
    controls.maxPolarAngle = Math.PI;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404060, 0.3);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(100, 100, 50);
    scene.add(directionalLight);

    const lightColors = [0x667eea, 0xff6b6b, 0x4ecdc4];
    lightColors.forEach((color, i) => {
        const light = new THREE.PointLight(color, 0.3, 200);
        light.position.set(
            Math.cos(i * Math.PI * 2 / 3) * 100,
            Math.sin(i * Math.PI * 2 / 3) * 50,
            50
        );
        scene.add(light);
    });

    const pointLight = new THREE.PointLight(0xffffff, 0.5);
    camera.add(pointLight);
    scene.add(camera);

    // Raycaster for interactions
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 3;
    mouse = new THREE.Vector2();

    // Starfield
    createStarfield();

    // Event listeners
    window.addEventListener('resize', onWindowResize);
    document.getElementById('show-connections').addEventListener('change', updateConnections);
    document.getElementById('show-nebulae').addEventListener('change', toggleNebulae);
    document.getElementById('enable-gravity').addEventListener('change', (e) => {
        gravityEnabled = e.target.checked;
    });
    document.getElementById('animation-speed').addEventListener('input', (e) => {
        animationSpeed = parseFloat(e.target.value);
    });

    // Cosmos interaction listeners
    const canvas = document.getElementById('cosmos-canvas');
    canvas.addEventListener('mousemove', onCosmosMouseMove);
    canvas.addEventListener('click', onCosmosClick);
}

function createStarfield() {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    const sizes = [];

    for (let i = 0; i < 15000; i++) {
        const cluster = Math.random() < 0.3;
        const spread = cluster ? 500 : 2000;

        vertices.push(
            (Math.random() - 0.5) * spread,
            (Math.random() - 0.5) * spread,
            (Math.random() - 0.5) * spread
        );

        const color = new THREE.Color();
        const hue = Math.random() * 0.15 + 0.55;
        const saturation = cluster ? 0.6 : 0.3;
        const lightness = Math.random() * 0.4 + 0.6;
        color.setHSL(hue, saturation, lightness);
        colors.push(color.r, color.g, color.b);

        sizes.push(Math.random() * 1.5 + 0.5);
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

    const material = new THREE.PointsMaterial({
        size: 0.8,
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true
    });

    const stars = new THREE.Points(geometry, material);
    stars.name = 'starfield';
    scene.add(stars);
}

export function updateCosmosData() {
    // Clear existing meshes
    chunkMeshes.forEach(mesh => scene.remove(mesh));
    chunkMeshes.clear();

    connectionLines.forEach(line => scene.remove(line));
    connectionLines = [];

    // Geometries
    const geometries = {
        sphere: new THREE.SphereGeometry(3, 32, 32),
        cube: new THREE.BoxGeometry(5, 5, 5),
        icosahedron: new THREE.IcosahedronGeometry(4, 0),
        torus: new THREE.TorusGeometry(3, 1, 16, 100)
    };

    // Create chunk meshes
    state.chunks.forEach((chunk, index) => {
        const geometry = geometries[chunk.shape_3d] || geometries.sphere;
        const color = new THREE.Color(chunk.color);

        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.95,
            shininess: 100,
            specular: new THREE.Color(0xffffff)
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...chunk.position_3d);
        mesh.userData = chunk;
        mesh.userData.velocity = new THREE.Vector3(0, 0, 0);

        // Glow layers
        const glowGeometry1 = new THREE.SphereGeometry(4.5, 16, 16);
        const glowMaterial1 = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.15,
            side: THREE.BackSide,
            blending: THREE.AdditiveBlending
        });
        const glow1 = new THREE.Mesh(glowGeometry1, glowMaterial1);
        mesh.add(glow1);

        const glowGeometry2 = new THREE.SphereGeometry(6, 16, 16);
        const glowMaterial2 = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.05,
            side: THREE.BackSide,
            blending: THREE.AdditiveBlending
        });
        const glow2 = new THREE.Mesh(glowGeometry2, glowMaterial2);
        mesh.add(glow2);

        scene.add(mesh);
        chunkMeshes.set(chunk.id, mesh);
    });

    createNebulae();
    updateConnections();
}

function createNebulae() {
    nebulae.forEach(n => scene.remove(n));
    nebulae = [];

    if (state.chunks.length < 3 || state.chunks.length > 100) return;

    const numClusters = Math.min(Math.max(Math.floor(state.chunks.length / 20), 2), 4);
    const clusterCenters = [];

    for (let i = 0; i < numClusters; i++) {
        const randomChunk = state.chunks[Math.floor(Math.random() * state.chunks.length)];
        clusterCenters.push({
            position: randomChunk.position_3d,
            color: randomChunk.color
        });
    }

    clusterCenters.forEach((cluster, i) => {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const alphas = [];
        const sizes = [];

        const particleCount = state.chunks.length > 50 ? 500 : 1000;
        const center = new THREE.Vector3(...cluster.position);
        const clusterRadius = 40;

        for (let j = 0; j < particleCount; j++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = Math.pow(Math.random(), 0.5) * clusterRadius;

            const x = center.x + r * Math.sin(phi) * Math.cos(theta);
            const y = center.y + r * Math.sin(phi) * Math.sin(theta);
            const z = center.z + r * Math.cos(phi);

            positions.push(x, y, z);

            const normalizedR = r / clusterRadius;
            const alpha = Math.pow(1.0 - normalizedR, 2);
            alphas.push(alpha);

            sizes.push(Math.random() * 3 + 1);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('alpha', new THREE.Float32BufferAttribute(alphas, 1));
        geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        const material = new THREE.ShaderMaterial({
            uniforms: {
                nebulaColor: { value: new THREE.Color(cluster.color) },
                time: { value: 0 }
            },
            vertexShader: nebulaVertexShader,
            fragmentShader: nebulaFragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        const nebula = new THREE.Points(geometry, material);
        nebula.userData = { clusterId: i };
        nebulae.push(nebula);
        scene.add(nebula);
    });
}

function updateConnections() {
    connectionLines.forEach(line => scene.remove(line));
    connectionLines = [];

    if (!document.getElementById('show-connections').checked) return;

    const MAX_CONNECTIONS = COSMOS_SETTINGS.maxConnections;
    const connectionsToRender = state.connections.length > MAX_CONNECTIONS
        ? state.connections.slice(0, MAX_CONNECTIONS)
        : state.connections;

    connectionsToRender.forEach(conn => {
        const fromMesh = chunkMeshes.get(conn.from_chunk_id);
        const toMesh = chunkMeshes.get(conn.to_chunk_id);

        if (fromMesh && toMesh) {
            const material = new THREE.LineBasicMaterial({
                color: conn.connection_type === 'semantic' ? 0xff40ff : 0x4080ff,
                transparent: true,
                opacity: conn.strength * 0.4,
                blending: THREE.AdditiveBlending,
                linewidth: 2
            });

            const points = [fromMesh.position, toMesh.position];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, material);

            line.userData = { from: fromMesh, to: toMesh };
            connectionLines.push(line);
            scene.add(line);
        }
    });
}

function toggleNebulae() {
    const visible = document.getElementById('show-nebulae').checked;
    nebulae.forEach(nebula => {
        nebula.visible = visible;
    });
}

export function focusChunkInCosmos(chunkId) {
    switchView('cosmos');

    setTimeout(() => {
        const mesh = chunkMeshes.get(chunkId);
        if (mesh) {
            animateCameraToTarget(mesh.position);
        }
    }, 100);
}

export function switchView(view) {
    state.setCurrentView(view);

    // Update buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.remove('active');
        if ((view === 'document' && btn.textContent.includes('📄')) ||
            (view === 'cosmos' && btn.textContent.includes('🌌'))) {
            btn.classList.add('active');
        }
    });

    // Update views
    document.getElementById('document-view').classList.toggle('active', view === 'document');
    document.getElementById('cosmos-view').classList.toggle('active', view === 'cosmos');

    if (view === 'cosmos') {
        const container = document.getElementById('cosmos-view');
        const width = container.clientWidth;
        const height = container.clientHeight;

        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);

        if (!animationId) {
            animateCosmos();
        }
    }
}

function animateCameraToTarget(targetPos) {
    const startPos = camera.position.clone();
    const endPos = targetPos.clone().add(new THREE.Vector3(0, 10, 30));
    const startTarget = controls.target.clone();

    let progress = 0;
    const animate = () => {
        progress += 0.02;
        if (progress <= 1) {
            const t = 1 - Math.pow(1 - progress, 3);
            camera.position.lerpVectors(startPos, endPos, t);
            controls.target.lerpVectors(startTarget, targetPos, t);
            controls.update();
            requestAnimationFrame(animate);
        }
    };
    animate();
}

let frameCount = 0;
function animateCosmos() {
    animationId = requestAnimationFrame(animateCosmos);

    if (state.currentView !== 'cosmos') return;

    const time = Date.now() * 0.001;

    // Rotate chunks
    chunkMeshes.forEach((mesh, i) => {
        mesh.rotation.y += 0.001 * animationSpeed;
        const pulse = Math.sin(time * 0.5 + i * 0.1) * 0.1 + 0.5;
        mesh.material.emissiveIntensity = pulse;
    });

    // Animate nebulae
    nebulae.forEach((nebula, i) => {
        nebula.rotation.y += 0.0001 * animationSpeed;
        nebula.rotation.z += 0.0002 * animationSpeed;
        if (nebula.material.uniforms && nebula.material.uniforms.time) {
            nebula.material.uniforms.time.value = time;
        }
    });

    // Rotate starfield
    const starfield = scene.getObjectByName('starfield');
    if (starfield) {
        starfield.rotation.y += 0.00005 * animationSpeed;
    }

    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('cosmos-view');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function onCosmosMouseMove(event) {
    if (state.currentView !== 'cosmos') return;

    const canvas = document.getElementById('cosmos-canvas');
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const meshArray = Array.from(chunkMeshes.values());
    const intersects = raycaster.intersectObjects(meshArray);

    // Reset previous hover
    if (state.hoveredChunk && state.hoveredChunk !== state.selectedChunk) {
        const mesh = state.hoveredChunk;
        mesh.scale.setScalar(1.0);
        mesh.material.emissiveIntensity = 0.5;
    }

    // Highlight new hover
    if (intersects.length > 0) {
        const mesh = intersects[0].object;
        const chunk = mesh.userData;

        if (mesh !== state.selectedChunk) {
            mesh.scale.setScalar(1.5);
            mesh.material.emissiveIntensity = 1.2;
        }

        showCosmosInfo(chunk);
        state.setHoveredChunk(mesh);
        canvas.style.cursor = 'pointer';
    } else {
        if (!state.selectedChunk) {
            hideCosmosInfo();
        }
        state.setHoveredChunk(null);
        canvas.style.cursor = 'default';
    }
}

function onCosmosClick(event) {
    if (state.currentView !== 'cosmos') return;

    const canvas = document.getElementById('cosmos-canvas');
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const meshArray = Array.from(chunkMeshes.values());
    const intersects = raycaster.intersectObjects(meshArray);

    // Reset previous selection
    if (state.selectedChunk) {
        state.selectedChunk.scale.setScalar(1.0);
        state.selectedChunk.material.emissiveIntensity = 0.5;
    }

    if (intersects.length > 0) {
        const mesh = intersects[0].object;
        const chunk = mesh.userData;

        state.setSelectedChunk(mesh);
        mesh.scale.setScalar(1.8);
        mesh.material.emissiveIntensity = 1.5;

        showCosmosInfo(chunk);
        animateCameraToTarget(mesh.position);
    } else {
        state.setSelectedChunk(null);
        hideCosmosInfo();
    }
}

function showCosmosInfo(chunk) {
    const panel = document.getElementById('cosmos-info');
    document.getElementById('cosmos-info-id').textContent = chunk.id;

    let tagsHTML = '';
    if (chunk.tags && chunk.tags.length > 0) {
        tagsHTML = `<strong>Tags:</strong> ${chunk.tags.join(', ')}<br>`;
    }

    let reasoningHTML = '';
    if (chunk.reasoning) {
        reasoningHTML = `<strong>Reasoning:</strong> ${chunk.reasoning}`;
    }

    document.getElementById('cosmos-info-meta').innerHTML =
        `Type: ${chunk.chunk_type} • Index: ${chunk.chunk_index}<br>${tagsHTML}${reasoningHTML}`;

    document.getElementById('cosmos-info-text').textContent =
        chunk.content.substring(0, 200) + (chunk.content.length > 200 ? '...' : '');

    // Update attach button
    const attachBtn = document.getElementById('cosmos-attach-btn');
    attachBtn.onclick = () => showAddModal(chunk.id);

    panel.classList.add('visible');
}

function hideCosmosInfo() {
    const panel = document.getElementById('cosmos-info');
    panel.classList.remove('visible');
}

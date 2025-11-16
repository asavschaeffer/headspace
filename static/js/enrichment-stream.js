/**
 * Real-time Enrichment Streaming
 * Listens to WebSocket events for live embedding calculation and shape materialization
 */

class EnrichmentStreamListener {
    constructor(docId, onChunkEnriched, onProgress, onComplete, onError) {
        this.docId = docId;
        this.onChunkEnriched = onChunkEnriched;
        this.onProgress = onProgress;
        this.onComplete = onComplete;
        this.onError = onError;
        this.websocket = null;
        this.isConnected = false;
        this.enrichmentStartTime = null;
    }

    /**
     * Connect to enrichment WebSocket stream
     */
    connect() {
        return new Promise((resolve, reject) => {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/enrichment/${this.docId}`;

                console.log(`🔌 Connecting to enrichment stream: ${wsUrl}`);
                this.websocket = new WebSocket(wsUrl);

                this.websocket.onopen = () => {
                    this.isConnected = true;
                    this.enrichmentStartTime = performance.now();
                    console.log('✅ Connected to enrichment stream');
                    resolve();
                };

                this.websocket.onmessage = (event) => {
                    try {
                        const eventData = JSON.parse(event.data);
                        this.handleEvent(eventData);
                    } catch (error) {
                        console.error('Failed to parse enrichment event:', error);
                    }
                };

                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnected = false;
                    if (this.onError) {
                        this.onError(error);
                    }
                    reject(error);
                };

                this.websocket.onclose = () => {
                    console.log('WebSocket closed');
                    this.isConnected = false;
                };
            } catch (error) {
                console.error('Failed to connect to enrichment stream:', error);
                reject(error);
            }
        });
    }

    /**
     * Handle incoming enrichment events
     */
    handleEvent(eventData) {
        const {
            event_type,
            doc_id,
            chunk_id,
            chunk_index,
            embedding,
            color,
            position_3d,
            umap_coordinates,
            progress,
            total_chunks,
            error,
            timestamp
        } = eventData;

        console.log(`📡 Event: ${event_type} (${progress}% complete)`);

        switch (event_type) {
            case 'started':
                console.log(`🚀 Enrichment started for document ${doc_id}`);
                if (this.onProgress) {
                    this.onProgress(0, 0);
                }
                break;

            case 'chunk_enriched':
            case 'chunk_layout_updated':
                console.log(`✨ Chunk ${chunk_index} enriched:`, { chunk_id, embedding_dims: embedding.length, color });
                if (this.onChunkEnriched) {
                    this.onChunkEnriched({
                        chunk_id,
                        chunk_index,
                        embedding,
                        color,
                        position_3d,
                        umap_coordinates,
                        timestamp,
                        stage: event_type
                    });
                }
                if (this.onProgress && event_type === 'chunk_enriched') {
                    this.onProgress(progress, total_chunks);
                }
                break;

            case 'completed':
                const elapsedTime = (performance.now() - this.enrichmentStartTime) / 1000;
                console.log(`🎉 Enrichment completed! (${elapsedTime.toFixed(1)}s)`);
                if (this.onComplete) {
                    this.onComplete({ total_chunks, elapsed_seconds: elapsedTime });
                }
                if (this.onProgress) {
                    this.onProgress(100, total_chunks);
                }
                break;

            case 'error':
                console.error(`❌ Enrichment error: ${error}`);
                if (this.onError) {
                    this.onError(error);
                }
                break;

            default:
                console.warn(`Unknown event type: ${event_type}`);
        }
    }

    /**
     * Disconnect from stream
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
    }
}

let geometryGeneratorInstance = null;

function materialSupportsEmissive(material) {
    return !!(
        material &&
        typeof material === 'object' &&
        'emissive' in material &&
        'emissiveIntensity' in material &&
        material.emissive !== undefined
    );
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

/**
 * Shape Morphing Animation
 * Animates the transformation of a chunk's geometry from placeholder sphere to final deformed shape
 */
class ShapeMorphingAnimator {
    constructor(chunkMesh, targetSignature, duration = 1500) {
        this.mesh = chunkMesh;
        this.targetSignature = targetSignature;
        this.duration = duration; // milliseconds
        this.startTime = null;
        this.originalGeometry = null;
        this.targetGeometry = null;
        this.isAnimating = false;
        this.geometryGenerator = new ProceduralGeometryGenerator();
    }

    /**
     * Start morphing animation
     * Returns promise that resolves when animation completes
     */
    async start() {
        return new Promise((resolve) => {
            if (!this.mesh) {
                resolve();
                return;
            }

            // Store original geometry
            this.originalGeometry = this.mesh.geometry.clone();

            // Generate target geometry from server signature
            if (this.targetSignature && typeof this.targetSignature === 'object' &&
                typeof this.geometryGenerator.generatePlanetaryGeometryFromSignature === 'function') {
                this.targetGeometry = this.geometryGenerator.generatePlanetaryGeometryFromSignature(this.targetSignature);
            } else if (Array.isArray(this.targetSignature) && this.targetSignature.length) {
                this.targetGeometry = this.geometryGenerator.generatePlanetaryGeometry(this.targetSignature);
            } else {
                this.targetGeometry = this.geometryGenerator.generatePlanetaryGeometry([]);
            }

            // Ensure both geometries have same vertex count
            if (this.originalGeometry.attributes.position.count !== this.targetGeometry.attributes.position.count) {
                console.warn('Geometry vertex count mismatch, using target geometry directly');
                if (this.mesh.geometry && typeof this.mesh.geometry.dispose === 'function') {
                    this.mesh.geometry.dispose();
                }
                this.mesh.geometry = this.targetGeometry;
                resolve();
                return;
            }

            this.isAnimating = true;
            this.startTime = performance.now();
            this.animationFrameId = null;

            const animate = (currentTime) => {
                const elapsed = currentTime - this.startTime;
                const progress = Math.min(elapsed / this.duration, 1);

                this.morphGeometry(progress);

                if (progress < 1) {
                    this.animationFrameId = requestAnimationFrame(animate);
                } else {
                    this.isAnimating = false;
                    if (this.mesh.geometry && typeof this.mesh.geometry.dispose === 'function') {
                        this.mesh.geometry.dispose();
                    }
                    this.mesh.geometry = this.targetGeometry;
                    if (this.originalGeometry && typeof this.originalGeometry.dispose === 'function') {
                        this.originalGeometry.dispose();
                        this.originalGeometry = null;
                    }
                    resolve();
                }
            };

            this.animationFrameId = requestAnimationFrame(animate);
        });
    }

    /**
     * Morph geometry between original and target
     */
    morphGeometry(progress) {
        if (!this.originalGeometry || !this.targetGeometry || !this.mesh) {
            return;
        }

        const originalPos = this.originalGeometry.attributes.position;
        const targetPos = this.targetGeometry.attributes.position;
        const currentPos = this.mesh.geometry.attributes.position;

        // Linear interpolation with easing
        const easeProgress = this.easeInOutCubic(progress);

        for (let i = 0; i < originalPos.count; i++) {
            const origX = originalPos.getX(i);
            const origY = originalPos.getY(i);
            const origZ = originalPos.getZ(i);

            const targetX = targetPos.getX(i);
            const targetY = targetPos.getY(i);
            const targetZ = targetPos.getZ(i);

            const morphX = origX + (targetX - origX) * easeProgress;
            const morphY = origY + (targetY - origY) * easeProgress;
            const morphZ = origZ + (targetZ - origZ) * easeProgress;

            currentPos.setXYZ(i, morphX, morphY, morphZ);
        }

        currentPos.needsUpdate = true;
        this.mesh.geometry.computeVertexNormals();
    }

    /**
     * Easing function: cubic ease-in-out
     */
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    /**
     * Stop animation
     */
    stop() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        this.isAnimating = false;
    }
}

/**
 * Integration with Cosmos Renderer
 * Call this to start streaming enrichment and update shapes in real-time
 */
function startEnrichmentStreaming(docId, chunkMeshMap) {
    console.log(`[ENRICHMENT] Starting enrichment streaming for doc=${docId}, meshes available=${chunkMeshMap.size}`);
    const enrichmentListener = new EnrichmentStreamListener(
        docId,
        // onChunkEnriched - Update shape with new embedding
        async (chunkData) => {
            console.log(`[ENRICHMENT] Event received: type=${chunkData.event_type}, chunk_id=${chunkData.chunk_id}`);

            const MAX_RETRIES = 25;
            const RETRY_DELAY_MS = 120;

            const applyUpdateToMesh = async (mesh, resolvedData) => {
                const { color, position_3d, umap_coordinates, event_type } = resolvedData;
                const shapeSignature = normalizeShapeSignature(resolvedData.shape_3d);

                console.log(`🎨 Handling ${event_type} for chunk ${resolvedData.chunk_id}`);

                const effectiveColor = (typeof color === 'string' && color.trim())
                    ? color
                    : mesh.userData?.chunk?.color;
                if (effectiveColor && mesh.material && mesh.material.color && typeof mesh.material.color.setStyle === 'function') {
                    console.log(`[ENRICHMENT] Setting color for chunk ${resolvedData.chunk_id}: material.type=${mesh.material.type}, effectiveColor=${effectiveColor}`);
                    mesh.material.color.setStyle(effectiveColor);
                    if (typeof mesh.material.color.convertSRGBToLinear === 'function' && mesh.material.type === 'MeshStandardMaterial') {
                        mesh.material.color.convertSRGBToLinear();
                    }
                    console.log(`[ENRICHMENT] After setStyle: material.color=${mesh.material.color.getHexString()}`);
                    if (materialSupportsEmissive(mesh.material) && typeof THREE !== 'undefined') {
                        console.log(`[ENRICHMENT] Material supports emissive, setting it...`);
                        const emissiveColor = new THREE.Color(effectiveColor);
                        if (typeof mesh.material.emissive.convertSRGBToLinear === 'function') {
                            emissiveColor.convertSRGBToLinear();
                        }
                        mesh.material.emissive.copy(emissiveColor.multiplyScalar(0.15));
                        mesh.material.emissiveIntensity = 1.0;
                        console.log(`[ENRICHMENT] Set emissive: ${mesh.material.emissive.getHexString()}`);
                    } else {
                        console.log(`[ENRICHMENT] Material does NOT support emissive (type=${mesh.material.type})`);
                    }
                    mesh.material.needsUpdate = true;
                } else {
                    console.log(`[ENRICHMENT] Could not set color - mesh.material=${mesh.material?.type}, has color=${!!mesh.material?.color}`);
                }

                if (position_3d && position_3d.length === 3) {
                    mesh.position.set(position_3d[0], position_3d[1], position_3d[2]);
                } else if (umap_coordinates && umap_coordinates.length === 3) {
                    mesh.position.set(umap_coordinates[0], umap_coordinates[1], umap_coordinates[2]);
                }

                mesh.userData = {
                    ...mesh.userData,
                    shapeSignature: shapeSignature || mesh.userData?.shapeSignature,
                };
                if (mesh.userData && mesh.userData.chunk) {
                    mesh.userData.chunk = {
                        ...mesh.userData.chunk,
                        color: effectiveColor || mesh.userData.chunk.color,
                        position_3d: position_3d || mesh.userData.chunk.position_3d,
                        umap_coordinates: umap_coordinates || mesh.userData.chunk.umap_coordinates,
                        shape_3d: shapeSignature || mesh.userData.chunk.shape_3d,
                    };
                }

                if (shapeSignature && event_type === 'chunk_enriched') {
                    const animator = new ShapeMorphingAnimator(mesh, shapeSignature);
                    await animator.start();
                    addShapeCompletionGlow(mesh);
                } else if (shapeSignature && event_type === 'chunk_layout_updated') {
                    const generator = getGeometryGenerator();
                    if (generator && typeof generator.generatePlanetaryGeometryFromSignature === 'function') {
                        try {
                            const newGeometry = generator.generatePlanetaryGeometryFromSignature(shapeSignature);
                            if (newGeometry) {
                                if (mesh.geometry && typeof mesh.geometry.dispose === 'function') {
                                    mesh.geometry.dispose();
                                }
                                mesh.geometry = newGeometry;
                            }
                        } catch (error) {
                            console.warn('Failed to update geometry during layout stage', error);
                        }
                    }
                }
            };

            const attemptUpdate = (resolvedData, attempt = 0) => {
                const mesh = chunkMeshMap.get(resolvedData.chunk_id);
                if (attempt === 0) {
                    console.log(`[ENRICHMENT] Looking for mesh chunk_id=${resolvedData.chunk_id}, found=${!!mesh}, map size=${chunkMeshMap.size}`);
                }
                if (!mesh) {
                    if (attempt < MAX_RETRIES) {
                        setTimeout(() => attemptUpdate(resolvedData, attempt + 1), RETRY_DELAY_MS);
                    } else {
                        console.warn(`[ENRICHMENT] Mesh not found for chunk ${resolvedData.chunk_id} after ${MAX_RETRIES} retries`);
                    }
                    return;
                }
                applyUpdateToMesh(mesh, resolvedData);
            };

            attemptUpdate(chunkData);
        },

        // onProgress - Update progress indicator
        (progress, totalChunks) => {
            const percent = Math.round(progress);
            const statusEl = document.getElementById('status-text');
            if (statusEl) {
                statusEl.textContent = `Enriching: ${percent}% (${progress}/${totalChunks} chunks)`;
            }
            console.log(`📊 Progress: ${percent}%`);
        },

        // onComplete - Enrichment finished
        (data) => {
            const { total_chunks, elapsed_seconds } = data;
            const statusEl = document.getElementById('status-text');
            if (statusEl) {
                statusEl.textContent = `✨ Enrichment complete! (${total_chunks} chunks in ${elapsed_seconds.toFixed(1)}s)`;
            }
            console.log(`✅ All ${total_chunks} shapes materialized!`);
        },

        // onError - Error occurred
        (error) => {
            const statusEl = document.getElementById('status-text');
            if (statusEl) {
                statusEl.textContent = `❌ Enrichment error: ${error}`;
            }
            console.error('Enrichment stream error:', error);
        }
    );

    // Connect and start listening
    enrichmentListener.connect().catch((error) => {
        console.error('Failed to connect to enrichment stream:', error);
    });

    return enrichmentListener;
}

/**
 * Add completion glow effect to a mesh
 */
function addShapeCompletionGlow(mesh) {
    console.log(`[GLOW] addShapeCompletionGlow called. materialSupportsEmissive=${materialSupportsEmissive(mesh.material)}, material.type=${mesh.material?.type}`);
    if (!materialSupportsEmissive(mesh.material) || typeof THREE === 'undefined') {
        console.log(`[GLOW] Returning early - material doesn't support emissive`);
        return;
    }

    // Temporarily brighten the material
    const originalEmissive = mesh.material.emissive.getHex ? mesh.material.emissive.getHex() : 0x000000;
    const originalIntensity = mesh.material.emissiveIntensity || 0;

    console.log(`[GLOW] Setting glow - emissive to white, intensity to 0.3`);
    mesh.material.emissive = new THREE.Color(0xffffff);
    mesh.material.emissiveIntensity = 0.3;

    // Fade back to normal over 500ms
    let glowStart = performance.now();
    const glowDuration = 500;

    const glowAnimation = (currentTime) => {
        const elapsed = currentTime - glowStart;
        const progress = Math.min(elapsed / glowDuration, 1);
        const easeProgress = 1 - progress; // Ease out

        if (mesh.material) {
            mesh.material.emissiveIntensity = originalIntensity + (0.3 - originalIntensity) * easeProgress;
        }

        if (progress < 1) {
            requestAnimationFrame(glowAnimation);
        }
    };

    requestAnimationFrame(glowAnimation);
}

// Export for use
if (typeof window !== 'undefined') {
    window.EnrichmentStreamListener = EnrichmentStreamListener;
    window.ShapeMorphingAnimator = ShapeMorphingAnimator;
    window.startEnrichmentStreaming = startEnrichmentStreaming;
}

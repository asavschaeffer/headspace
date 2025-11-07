// Geometry Worker - Off-thread geometry and texture generation
// Prevents main thread blocking during expensive procedural generation

// Load THREE.js first (required by geometry generators)
importScripts('https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js');

// Import the procedural geometry generator
// Note: These are loaded as inline scripts in the worker context
importScripts('/js/procedural-geometry.js');
importScripts('/js/texture-generator.js');

/**
 * Main worker message handler
 * Receives: {action, chunkId, embedding, detail, tags}
 * Returns: {success, chunkId, detail, data: {positions, normals, colors, indices}}
 */
self.onmessage = async (event) => {
    const message = event.data;

    if (!message || typeof message !== 'object') {
        console.error('Invalid message format');
        return;
    }

    const { action, chunkId, embedding, detail, tags } = message;

    try {
        if (action === 'generateGeometryTexture') {
            generateGeometryWithTexture(chunkId, embedding, detail, tags);
        } else if (action === 'ping') {
            // Simple health check
            self.postMessage({ success: true, action: 'pong' });
        } else {
            self.postMessage({
                success: false,
                chunkId,
                error: `Unknown action: ${action}`
            });
        }
    } catch (error) {
        self.postMessage({
            success: false,
            chunkId,
            error: error.message || 'Unknown error'
        });
    }
};

/**
 * Generate procedural geometry and apply texture coloring
 * @param {string} chunkId
 * @param {Array<number>} embedding
 * @param {number} detail
 * @param {Array<string>} tags
 */
function generateGeometryWithTexture(chunkId, embedding, detail, tags) {
    // Step 1: Generate base procedural geometry
    const generator = new ProceduralGeometryGenerator();

    // Adjust smoothing based on detail level
    const smoothing = Math.max(1, Math.round(detail / 8));
    generator.updateParameters({ detail, smoothing });

    // Generate planetary geometry from embedding
    const geometry = generator.generatePlanetaryGeometry(embedding);

    // Step 2: Generate texture (biome colors) based on geometry
    const textureGenerator = new TextureGenerator();
    const colors = textureGenerator.generateVertexColors(geometry, {
        embedding,
        tags
    });

    // Step 3: Extract geometry data for transfer
    try {
        const positions = geometry.attributes.position.array;
        const normals = geometry.attributes.normal.array;

        // Convert colors from Uint8Array to normalized 0-1 for colors array
        // (Three.js expects Float32Array for vertex colors)
        const colorFloat = new Float32Array(colors.length);
        for (let i = 0; i < colors.length; i++) {
            colorFloat[i] = colors[i] / 255;
        }

        // Prepare indices if available
        let indices = null;
        if (geometry.index && geometry.index.array) {
            indices = Array.from(geometry.index.array);
        }

        // Send back to main thread
        self.postMessage({
            success: true,
            chunkId,
            detail,
            data: {
                positions: Array.from(positions),
                normals: Array.from(normals),
                colors: Array.from(colorFloat),
                indices
            }
        });
    } catch (error) {
        self.postMessage({
            success: false,
            chunkId,
            error: `Failed to extract geometry: ${error.message}`
        });
    }
}

// Worker ready signal
self.postMessage({
    ready: true,
    message: 'Geometry worker initialized'
});

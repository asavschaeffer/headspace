// Texture Generator - Biome Mapping from Geometry
// Analyzes deformed geometry altitude and slope to generate terrain-like textures

class TextureGenerator {
    constructor() {
        this.biomeColors = {
            deepOcean: { r: 0, g: 26, b: 77 },      // #001a4d
            beaches: { r: 194, g: 167, b: 108 },    // #c2a76c
            grasslands: { r: 45, g: 80, b: 22 },    // #2d5016
            forests: { r: 26, g: 58, b: 10 },       // #1a3a0a
            plateaus: { r: 107, g: 107, b: 71 },    // #6b6b47
            mountains: { r: 42, g: 42, b: 26 },     // #2a2a1a
            snow: { r: 240, g: 248, b: 255 }        // #f0f8ff
        };
    }

    /**
     * Generate vertex colors based on biome mapping from geometry
     * @param {THREE.BufferGeometry} geometry - Deformed planetary geometry
     * @param {Object} options - {embedding, tags}
     * @returns {Uint8Array} RGBA color data for vertices
     */
    generateVertexColors(geometry, options = {}) {
        const { embedding = [], tags = [] } = options;

        // Calculate altitude and slope for each vertex
        const altitudes = this.calculateAltitudes(geometry);
        const slopes = this.calculateSlopes(geometry);

        // Get embedding-based color tint (subtle influence)
        const embeddingTint = this.getEmbeddingTint(embedding);

        // Map vertices to biome colors
        const positions = geometry.attributes.position;
        const colors = new Uint8Array(positions.count * 3);

        for (let i = 0; i < positions.count; i++) {
            const altitude = altitudes[i];
            const slope = slopes[i];

            // Get base biome color
            const biomeColor = this.getBiomeColor(altitude, slope);

            // Apply embedding tint for variety
            const finalColor = this.applyTint(biomeColor, embeddingTint);

            colors[i * 3] = finalColor.r;
            colors[i * 3 + 1] = finalColor.g;
            colors[i * 3 + 2] = finalColor.b;
        }

        return colors;
    }

    /**
     * Calculate altitude (elevation) for each vertex
     * Altitude = normalized distance from center
     */
    calculateAltitudes(geometry) {
        const positions = geometry.attributes.position;
        const altitudes = new Float32Array(positions.count);

        let maxAltitude = 0;

        // Calculate raw altitudes
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            const z = positions.getZ(i);
            const altitude = Math.sqrt(x * x + y * y + z * z);
            altitudes[i] = altitude;
            if (altitude > maxAltitude) {
                maxAltitude = altitude;
            }
        }

        // Normalize to 0-1
        if (maxAltitude > 0) {
            for (let i = 0; i < altitudes.length; i++) {
                altitudes[i] /= maxAltitude;
            }
        }

        return altitudes;
    }

    /**
     * Calculate slope (steepness) for each vertex
     * Slope = average elevation difference with neighbors
     */
    calculateSlopes(geometry) {
        const positions = geometry.attributes.position;
        const altitudes = this.calculateAltitudes(geometry);
        const slopes = new Float32Array(positions.count);

        // For each vertex, compare to nearby vertices in the attribute array
        for (let i = 0; i < positions.count; i++) {
            const altitude_i = altitudes[i];

            // Sample nearby vertices (simple approach)
            let slopeDiff = 0;
            let sampleCount = 0;

            // Look at vertices within a range
            const range = Math.min(10, Math.max(1, Math.floor(positions.count / 100)));
            for (let j = Math.max(0, i - range); j < Math.min(positions.count, i + range); j++) {
                if (j === i) continue;

                const altitude_j = altitudes[j];
                slopeDiff += Math.abs(altitude_i - altitude_j);
                sampleCount++;
            }

            slopes[i] = sampleCount > 0 ? slopeDiff / sampleCount : 0;
        }

        // Normalize slopes to 0-1
        let maxSlope = 0;
        for (let i = 0; i < slopes.length; i++) {
            if (slopes[i] > maxSlope) {
                maxSlope = slopes[i];
            }
        }

        if (maxSlope > 0) {
            for (let i = 0; i < slopes.length; i++) {
                slopes[i] /= maxSlope;
            }
        }

        return slopes;
    }

    /**
     * Map altitude and slope to biome color
     * Creates terrain-like color progression
     */
    getBiomeColor(altitude, slope) {
        // Elevation-based biome selection
        if (altitude < 0.3) {
            // Deep ocean
            return this.biomeColors.deepOcean;
        } else if (altitude < 0.5) {
            // Beaches and shallow water
            return this.biomeColors.beaches;
        } else if (altitude < 0.7) {
            // Grasslands vs forests based on slope
            if (slope < 0.4) {
                return this.biomeColors.grasslands;
            } else {
                return this.biomeColors.forests;
            }
        } else if (altitude < 0.9) {
            // Plateaus vs mountains based on slope
            if (slope < 0.5) {
                return this.biomeColors.plateaus;
            } else {
                return this.biomeColors.mountains;
            }
        } else {
            // Snow at highest elevations
            return this.biomeColors.snow;
        }
    }

    /**
     * Extract color tint from embedding vector
     * Uses first 3 dimensions as RGB influence
     */
    getEmbeddingTint(embedding) {
        if (!embedding || embedding.length < 3) {
            return { r: 1.0, g: 1.0, b: 1.0 };
        }

        // Find min/max of first 3 dimensions
        const slice = embedding.slice(0, 3);
        const min = Math.min(...slice);
        const max = Math.max(...slice);
        const range = max - min || 1;

        // Normalize to 0.7-1.3 range (subtle tint)
        return {
            r: 0.7 + 0.6 * (embedding[0] - min) / range,
            g: 0.7 + 0.6 * (embedding[1] - min) / range,
            b: 0.7 + 0.6 * (embedding[2] - min) / range
        };
    }

    /**
     * Apply tint to a color
     */
    applyTint(color, tint) {
        return {
            r: Math.min(255, Math.floor(color.r * tint.r)),
            g: Math.min(255, Math.floor(color.g * tint.g)),
            b: Math.min(255, Math.floor(color.b * tint.b))
        };
    }
}

// Export for use in other modules (only in main thread, not in Web Worker)
if (typeof window !== 'undefined') {
    window.TextureGenerator = TextureGenerator;
}

// Enable use in Web Worker context
if (typeof self !== 'undefined') {
    self.TextureGenerator = TextureGenerator;
}

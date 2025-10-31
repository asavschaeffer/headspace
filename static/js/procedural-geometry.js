// Procedural Geometry Generator from Embedding Vectors
// Transforms high-dimensional embedding vectors into unique 3D planetary shapes

class ProceduralGeometryGenerator {
    constructor() {
        this.sphereDetail = 32; // Base sphere subdivision
        this.deformationScale = 0.3; // How much vertices can move (0-1)
        this.smoothingFactor = 2; // Smoothing for organic shapes
    }

    /**
     * Creates a unique planetary geometry from an embedding vector
     * @param {Array} embedding - The embedding vector (e.g., 384 or 1536 dimensions)
     * @returns {THREE.BufferGeometry} The generated geometry
     */
    generatePlanetaryGeometry(embedding) {
        if (!embedding || embedding.length === 0) {
            console.warn('No embedding provided, using default sphere');
            return new THREE.SphereGeometry(1, this.sphereDetail, this.sphereDetail);
        }

        // Create base sphere
        const geometry = new THREE.SphereGeometry(1, this.sphereDetail, this.sphereDetail);
        const positions = geometry.attributes.position;
        const vertex = new THREE.Vector3();

        // Map embedding dimensions to deformation parameters
        const embeddingDimensions = embedding.length;
        const deformationMap = this.createDeformationMap(embedding);

        // Deform each vertex
        for (let i = 0; i < positions.count; i++) {
            vertex.fromBufferAttribute(positions, i);

            // Get spherical coordinates
            const spherical = new THREE.Spherical();
            spherical.setFromVector3(vertex);

            // Calculate deformation based on embedding
            const deformation = this.calculateVertexDeformation(
                spherical.theta,
                spherical.phi,
                deformationMap,
                embeddingDimensions
            );

            // Apply deformation
            spherical.radius *= (1 + deformation * this.deformationScale);
            vertex.setFromSpherical(spherical);

            // Update position
            positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
        }

        // Smooth the geometry for organic appearance
        this.smoothGeometry(geometry);

        // Recalculate normals for proper lighting
        geometry.computeVertexNormals();

        return geometry;
    }

    /**
     * Creates a deformation map from the embedding vector
     * Groups embedding dimensions into spherical harmonics-like patterns
     */
    createDeformationMap(embedding) {
        const map = [];
        const normalizedEmbedding = this.normalizeEmbedding(embedding);

        // Group dimensions into harmonic patterns
        const harmonicCount = Math.min(16, Math.floor(Math.sqrt(embedding.length)));
        const dimensionsPerHarmonic = Math.floor(embedding.length / harmonicCount);

        for (let h = 0; h < harmonicCount; h++) {
            const startIdx = h * dimensionsPerHarmonic;
            const endIdx = Math.min(startIdx + dimensionsPerHarmonic, embedding.length);

            // Average values for this harmonic
            let harmonicValue = 0;
            for (let i = startIdx; i < endIdx; i++) {
                harmonicValue += normalizedEmbedding[i];
            }
            harmonicValue /= (endIdx - startIdx);

            map.push({
                frequency: h + 1,
                amplitude: harmonicValue,
                phaseShift: (h * Math.PI) / harmonicCount
            });
        }

        return map;
    }

    /**
     * Calculates deformation for a specific vertex based on its spherical coordinates
     */
    calculateVertexDeformation(theta, phi, deformationMap, embeddingDimensions) {
        let deformation = 0;

        // Apply each harmonic pattern
        for (const harmonic of deformationMap) {
            // Create complex deformation patterns using spherical harmonics
            const pattern = Math.sin(harmonic.frequency * theta + harmonic.phaseShift) *
                           Math.cos(harmonic.frequency * phi);

            deformation += pattern * harmonic.amplitude;
        }

        // Add fine detail based on embedding dimensionality
        const detail = this.generateFineDetail(theta, phi, embeddingDimensions);
        deformation += detail * 0.1;

        return deformation;
    }

    /**
     * Generates fine surface detail based on embedding dimensionality
     */
    generateFineDetail(theta, phi, dimensions) {
        // Use dimension count to determine detail frequency
        const detailFrequency = Math.log2(dimensions + 1);

        // Create turbulence-like pattern
        const noise1 = Math.sin(detailFrequency * 4 * theta) * Math.cos(detailFrequency * 4 * phi);
        const noise2 = Math.sin(detailFrequency * 8 * theta + 1) * Math.cos(detailFrequency * 8 * phi + 1);
        const noise3 = Math.sin(detailFrequency * 16 * theta + 2) * Math.cos(detailFrequency * 16 * phi + 2);

        return (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2);
    }

    /**
     * Normalizes embedding vector to [-1, 1] range
     */
    normalizeEmbedding(embedding) {
        // Find min and max
        let min = Math.min(...embedding);
        let max = Math.max(...embedding);

        // Prevent division by zero
        if (max === min) {
            return embedding.map(() => 0);
        }

        // Normalize to [-1, 1]
        return embedding.map(val => {
            return 2 * ((val - min) / (max - min)) - 1;
        });
    }

    /**
     * Smooths geometry for more organic appearance
     */
    smoothGeometry(geometry) {
        const positions = geometry.attributes.position;
        const tempPositions = new Float32Array(positions.array);

        for (let iteration = 0; iteration < this.smoothingFactor; iteration++) {
            // For each vertex, average with neighbors
            for (let i = 0; i < positions.count; i++) {
                const neighbors = this.findNeighbors(i, positions.count, this.sphereDetail);
                let avgX = 0, avgY = 0, avgZ = 0;

                for (const n of neighbors) {
                    avgX += tempPositions[n * 3];
                    avgY += tempPositions[n * 3 + 1];
                    avgZ += tempPositions[n * 3 + 2];
                }

                avgX /= neighbors.length;
                avgY /= neighbors.length;
                avgZ /= neighbors.length;

                // Blend with original position
                const blend = 0.5;
                tempPositions[i * 3] = tempPositions[i * 3] * (1 - blend) + avgX * blend;
                tempPositions[i * 3 + 1] = tempPositions[i * 3 + 1] * (1 - blend) + avgY * blend;
                tempPositions[i * 3 + 2] = tempPositions[i * 3 + 2] * (1 - blend) + avgZ * blend;
            }

            // Copy back
            positions.array.set(tempPositions);
        }

        positions.needsUpdate = true;
    }

    /**
     * Finds neighboring vertices for smoothing
     */
    findNeighbors(index, vertexCount, detail) {
        const neighbors = [];
        const row = Math.floor(index / (detail + 1));
        const col = index % (detail + 1);

        // Simple grid-based neighbors
        const offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]];

        for (const [dr, dc] of offsets) {
            const newRow = row + dr;
            const newCol = col + dc;

            if (newRow >= 0 && newRow <= detail && newCol >= 0 && newCol <= detail) {
                const neighborIdx = newRow * (detail + 1) + newCol;
                if (neighborIdx < vertexCount && neighborIdx !== index) {
                    neighbors.push(neighborIdx);
                }
            }
        }

        return neighbors.length > 0 ? neighbors : [index];
    }

    /**
     * Updates geometry deformation parameters
     */
    updateParameters(params) {
        if (params.detail !== undefined) this.sphereDetail = params.detail;
        if (params.deformation !== undefined) this.deformationScale = params.deformation;
        if (params.smoothing !== undefined) this.smoothingFactor = params.smoothing;
    }
}

// Export for use in other modules
window.ProceduralGeometryGenerator = ProceduralGeometryGenerator;
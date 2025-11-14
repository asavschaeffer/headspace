/**
 * Positioning Utilities for Semantic Space
 * Functions to calculate positions for new chunks based on similarity to existing ones
 */

/**
 * Calculate cosine similarity between two embeddings
 */
function cosineSimilarity(vec1, vec2) {
    if (!vec1 || !vec2 || vec1.length === 0 || vec2.length === 0) {
        return 0;
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < Math.min(vec1.length, vec2.length); i++) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) {
        return 0;
    }

    return dotProduct / (norm1 * norm2);
}

/**
 * Find nearest neighbors to a given embedding
 */
function findNearestNeighbors(embedding, allChunks, k = 3) {
    const similarities = allChunks.map((chunk, index) => {
        const similarity = cosineSimilarity(embedding, chunk.embedding);
        return { chunk, index, similarity };
    });

    // Sort by similarity (descending)
    similarities.sort((a, b) => b.similarity - a.similarity);

    // Return top k
    return similarities.slice(0, k).filter(s => s.similarity > 0.1);
}

/**
 * Calculate position for a new chunk based on its nearest neighbors
 * Uses average position of 3 nearest neighbors
 */
function calculatePositionFromNeighbors(embedding, allChunks) {
    if (!allChunks || allChunks.length === 0) {
        // No neighbors, random position
        return [
            (Math.random() - 0.5) * 100,
            (Math.random() - 0.5) * 100,
            (Math.random() - 0.5) * 100
        ];
    }

    const neighbors = findNearestNeighbors(embedding, allChunks, 3);

    if (neighbors.length === 0) {
        // No similar neighbors, random position nearby
        const randomChunk = allChunks[Math.floor(Math.random() * allChunks.length)];
        const offset = Math.random() * 30;
        return [
            randomChunk.position_3d[0] + (Math.random() - 0.5) * offset,
            randomChunk.position_3d[1] + (Math.random() - 0.5) * offset,
            randomChunk.position_3d[2] + (Math.random() - 0.5) * offset
        ];
    }

    // Average position of nearest neighbors
    let x = 0, y = 0, z = 0;
    neighbors.forEach(n => {
        x += n.chunk.position_3d[0];
        y += n.chunk.position_3d[1];
        z += n.chunk.position_3d[2];
    });

    x /= neighbors.length;
    y /= neighbors.length;
    z /= neighbors.length;

    // Add small offset to avoid perfect overlap
    const offset = Math.random() * 20;
    x += (Math.random() - 0.5) * offset;
    y += (Math.random() - 0.5) * offset;
    z += (Math.random() - 0.5) * offset;

    return [x, y, z];
}

/**
 * Smoothly animate a mesh from starting position to target position
 */
function animatePositionTransition(mesh, startPos, targetPos, duration = 1500) {
    return new Promise((resolve) => {
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing: ease-out-cubic
            const easeProgress = 1 - Math.pow(1 - progress, 3);

            mesh.position.x = startPos[0] + (targetPos[0] - startPos[0]) * easeProgress;
            mesh.position.y = startPos[1] + (targetPos[1] - startPos[1]) * easeProgress;
            mesh.position.z = startPos[2] + (targetPos[2] - startPos[2]) * easeProgress;

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                mesh.position.set(...targetPos);
                resolve();
            }
        };

        requestAnimationFrame(animate);
    });
}

// Export for use
if (typeof window !== 'undefined') {
    window.cosineSimilarity = cosineSimilarity;
    window.findNearestNeighbors = findNearestNeighbors;
    window.calculatePositionFromNeighbors = calculatePositionFromNeighbors;
    window.animatePositionTransition = animatePositionTransition;
}

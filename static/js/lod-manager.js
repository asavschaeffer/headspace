// LOD Manager - Level of Detail system based on camera distance
// Determines appropriate geometry detail level for viewport optimization

class LODManager {
    constructor(camera = null) {
        this.camera = camera;

        // LOD distance thresholds and corresponding detail levels
        this.lodLevels = [
            { name: 'ultra', distance: Infinity, detail: 48 },  // Ultra high (far away, not shown)
            { name: 'high', distance: 50, detail: 32 },         // High detail (close)
            { name: 'medium', distance: 150, detail: 24 },      // Medium detail (medium distance)
            { name: 'low', distance: 300, detail: 16 },         // Low detail (far)
            { name: 'minimal', distance: Infinity, detail: 8 }  // Minimal (very far or off-screen)
        ];

        // Default detail for when camera isn't initialized
        this.defaultDetail = 24;
    }

    /**
     * Set or update the camera reference
     * @param {THREE.Camera} camera
     */
    setCamera(camera) {
        this.camera = camera;
    }

    /**
     * Get appropriate detail level for a given distance
     * @param {number} distance - Distance from camera to object in world units
     * @param {number} currentDetail - Current detail level (for hysteresis)
     * @returns {number} Detail level (subdivision count for sphere)
     */
    getDetailForDistance(distance, currentDetail = null) {
        // Handle missing distance
        if (typeof distance !== 'number' || distance < 0) {
            return this.defaultDetail;
        }

        // Find appropriate LOD level
        for (const lod of this.lodLevels) {
            if (distance < lod.distance) {
                return lod.detail;
            }
        }

        return this.lodLevels[this.lodLevels.length - 1].detail;
    }

    /**
     * Calculate detail level for a 3D position
     * @param {THREE.Vector3} position - World position of object
     * @param {number} currentDetail - Current detail level
     * @returns {number} Detail level
     */
    getDetailForPosition(position, currentDetail = null) {
        if (!this.camera) {
            return this.defaultDetail;
        }

        const distance = this.camera.position.distanceTo(position);
        return this.getDetailForDistance(distance, currentDetail);
    }

    /**
     * Get the LOD level name for a given detail value
     * @param {number} detail - Detail level
     * @returns {string} LOD level name
     */
    getLODNameForDetail(detail) {
        for (const lod of this.lodLevels) {
            if (lod.detail === detail) {
                return lod.name;
            }
        }
        return 'unknown';
    }

    /**
     * Get all LOD levels (for UI or debugging)
     * @returns {Array} Array of LOD level definitions
     */
    getAllLODLevels() {
        return this.lodLevels;
    }

    /**
     * Check if detail level change requires significant update
     * (Useful for hysteresis/avoiding unnecessary re-computation)
     * @param {number} currentDetail
     * @param {number} newDetail
     * @returns {boolean}
     */
    requiresUpdate(currentDetail, newDetail) {
        if (!currentDetail) {
            return true;
        }

        // Always update if detail changes
        return currentDetail !== newDetail;
    }

    /**
     * Adjust LOD parameters dynamically
     * Useful for performance tuning
     * @param {number} multiplier - 0.5-2.0 to scale all details
     */
    adjustQuality(multiplier = 1.0) {
        // Clamp multiplier
        multiplier = Math.max(0.5, Math.min(2.0, multiplier));

        for (const lod of this.lodLevels) {
            // Round to nearest even number (valid for sphere subdivision)
            lod.detail = Math.round((lod.detail * multiplier) / 2) * 2;
            lod.detail = Math.max(4, Math.min(64, lod.detail)); // Keep in reasonable range
        }
    }

    /**
     * Reset quality settings to defaults
     */
    resetQuality() {
        this.lodLevels = [
            { name: 'ultra', distance: Infinity, detail: 48 },
            { name: 'high', distance: 50, detail: 32 },
            { name: 'medium', distance: 150, detail: 24 },
            { name: 'low', distance: 300, detail: 16 },
            { name: 'minimal', distance: Infinity, detail: 8 }
        ];
    }

    /**
     * Get recommended detail based on performance constraints
     * @param {number} targetFPS - Target frame rate (default 60)
     * @param {number} meshCount - Estimated number of meshes to render
     * @returns {number} Recommended base detail level
     */
    getRecommendedDetail(targetFPS = 60, meshCount = 10) {
        // Very rough heuristic: more meshes = lower detail
        // This is a starting point and should be tuned based on actual performance
        const detailMultiplier = Math.max(0.5, 1.0 - (meshCount - 5) * 0.05);
        return Math.round(this.defaultDetail * detailMultiplier);
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.LODManager = LODManager;
}

// ES module export
export { LODManager };

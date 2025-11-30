// Geometry Cache - IndexedDB persistent storage for generated geometry
// Avoids re-computing expensive procedural geometry on revisits

class GeometryCache {
    constructor() {
        this.dbName = 'HeadspaceGeometryCache';
        this.storeName = 'geometries';
        this.db = null;
        this.initPromise = this.init();
    }

    /**
     * Initialize IndexedDB with version management
     * @returns {Promise<void>}
     */
    init() {
        return new Promise((resolve, reject) => {
            if (!window.indexedDB) {
                console.warn('IndexedDB not available');
                resolve();
                return;
            }

            const request = window.indexedDB.open(this.dbName, 1);

            request.onerror = () => {
                console.error('IndexedDB initialization failed:', request.error);
                reject(request.error);
            };

            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    const store = db.createObjectStore(this.storeName, { keyPath: 'key' });
                    store.createIndex('chunkId', 'chunkId', { unique: false });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                }
            };
        });
    }

    /**
     * Generate cache key from chunkId and detail level
     */
    generateKey(chunkId, detail) {
        return `${chunkId}_detail_${detail}`;
    }

    /**
     * Retrieve geometry from cache
     * @param {string} chunkId - Unique chunk identifier
     * @param {number} detail - LOD detail level
     * @returns {Promise<Object|null>} Cached geometry data or null if not found
     */
    get(chunkId, detail) {
        return this.initPromise.then(() => {
            return new Promise((resolve, reject) => {
                if (!this.db) {
                    resolve(null);
                    return;
                }

                try {
                    const transaction = this.db.transaction([this.storeName], 'readonly');
                    const store = transaction.objectStore(this.storeName);
                    const key = this.generateKey(chunkId, detail);
                    const request = store.get(key);

                    request.onsuccess = () => {
                        const result = request.result;
                        if (result) {
                            // Return just the geometry data, not the metadata
                            resolve({
                                positions: new Float32Array(result.positions),
                                normals: new Float32Array(result.normals),
                                colors: new Float32Array(result.colors),
                                indices: result.indices ? new Uint32Array(result.indices) : null
                            });
                        } else {
                            resolve(null);
                        }
                    };

                    request.onerror = () => reject(request.error);
                } catch (error) {
                    console.warn('Cache get failed:', error);
                    resolve(null);
                }
            });
        });
    }

    /**
     * Store geometry in cache
     * @param {string} chunkId - Unique chunk identifier
     * @param {number} detail - LOD detail level
     * @param {Object} geometryData - {positions, normals, colors, indices}
     * @returns {Promise<void>}
     */
    set(chunkId, detail, geometryData) {
        return this.initPromise.then(() => {
            return new Promise((resolve, reject) => {
                if (!this.db) {
                    resolve();
                    return;
                }

                try {
                    const transaction = this.db.transaction([this.storeName], 'readwrite');
                    const store = transaction.objectStore(this.storeName);
                    const key = this.generateKey(chunkId, detail);

                    const data = {
                        key,
                        chunkId,
                        detail,
                        timestamp: Date.now(),
                        positions: Array.from(geometryData.positions),
                        normals: Array.from(geometryData.normals),
                        colors: Array.from(geometryData.colors),
                        indices: geometryData.indices ? Array.from(geometryData.indices) : null
                    };

                    const request = store.put(data);

                    request.onsuccess = () => resolve();
                    request.onerror = () => reject(request.error);
                } catch (error) {
                    console.warn('Cache set failed:', error);
                    resolve(); // Don't fail the whole process
                }
            });
        });
    }

    /**
     * Clear all cached geometries for a chunk
     * @param {string} chunkId - Unique chunk identifier
     * @returns {Promise<void>}
     */
    clearChunk(chunkId) {
        return this.initPromise.then(() => {
            return new Promise((resolve, reject) => {
                if (!this.db) {
                    resolve();
                    return;
                }

                try {
                    const transaction = this.db.transaction([this.storeName], 'readwrite');
                    const store = transaction.objectStore(this.storeName);
                    const index = store.index('chunkId');
                    const request = index.openCursor(IDBKeyRange.only(chunkId));

                    request.onsuccess = (event) => {
                        const cursor = event.target.result;
                        if (cursor) {
                            cursor.delete();
                            cursor.continue();
                        } else {
                            resolve();
                        }
                    };

                    request.onerror = () => reject(request.error);
                } catch (error) {
                    console.warn('Cache clear failed:', error);
                    resolve();
                }
            });
        });
    }

    /**
     * Clear entire cache (use sparingly)
     * @returns {Promise<void>}
     */
    clearAll() {
        return this.initPromise.then(() => {
            return new Promise((resolve, reject) => {
                if (!this.db) {
                    resolve();
                    return;
                }

                try {
                    const transaction = this.db.transaction([this.storeName], 'readwrite');
                    const store = transaction.objectStore(this.storeName);
                    const request = store.clear();

                    request.onsuccess = () => resolve();
                    request.onerror = () => reject(request.error);
                } catch (error) {
                    console.warn('Cache clear all failed:', error);
                    resolve();
                }
            });
        });
    }

    /**
     * Get cache statistics
     * @returns {Promise<Object>} {totalEntries, approximateSizeKB}
     */
    async getStats() {
        await this.initPromise;

        return new Promise((resolve) => {
            if (!this.db) {
                resolve({ totalEntries: 0, approximateSizeKB: 0 });
                return;
            }

            try {
                const transaction = this.db.transaction([this.storeName], 'readonly');
                const store = transaction.objectStore(this.storeName);
                const request = store.count();

                request.onsuccess = () => {
                    // Rough estimate: ~1.5KB per geometry entry average
                    const estimatedSize = (request.result * 1.5);
                    resolve({
                        totalEntries: request.result,
                        approximateSizeKB: Math.round(estimatedSize)
                    });
                };

                request.onerror = () => {
                    resolve({ totalEntries: 0, approximateSizeKB: 0 });
                };
            } catch (error) {
                console.warn('Cache stats failed:', error);
                resolve({ totalEntries: 0, approximateSizeKB: 0 });
            }
        });
    }
}

// Export for use in other modules
window.GeometryCache = GeometryCache;

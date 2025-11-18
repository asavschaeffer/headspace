/**
 * Home Planet - Loads a 3D GLB model as the special entry point
 */

class HomePlanetGenerator {
    constructor() {
        this.modelPath = '/assets/home-planet.glb';
    }

    /**
     * Load the home planet from GLB file
     */
    async generateHomePlanet() {
        return new Promise((resolve, reject) => {
            const loader = new THREE.GLTFLoader();
            loader.load(
                this.modelPath,
                (gltf) => {
                    const model = gltf.scene;

                    // Scale the model 200x to make it visible
                    model.scale.set(200, 200, 200);

                    // Set up userData properties for interaction
                    model.userData.isHomePlanet = true;
                    model.userData.clickHandler = () => {
                        window.location.href = 'https://asaschaeffer.com/index.html';
                    };

                    // Optional: Set up animation if the model has animations
                    if (gltf.animations && gltf.animations.length > 0) {
                        model.userData.animations = gltf.animations;
                    }

                    // Add light at the peak of the mountain for atmospheric effect
                    const peakLight = new THREE.PointLight(0xffb36d, 260, 0, 1.4);
                    peakLight.position.set(0, 12, 0);  // Position at the peak
                    peakLight.castShadow = false;
                    peakLight.userData.isCosmosGlobalLight = true;
                    model.add(peakLight);

                    resolve(model);
                },
                undefined,
                (error) => {
                    console.error('Failed to load home planet model:', error);
                    reject(error);
                }
            );
        });
    }
}

// Export
if (typeof window !== 'undefined') {
    window.HomePlanetGenerator = HomePlanetGenerator;
}

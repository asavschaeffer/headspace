/**
 * Home Planet - Special entry point with fog and mountain peak
 * A mystical white foggy sphere with a crystalline mountain peak
 */

class HomePlanetGenerator {
    constructor() {
        this.planetRadius = 8;
        this.peakRadius = 3;
        this.peakHeight = 6;
        this.snowCapHeight = 2;
        this.atmosphereRadius = 12;
    }

    /**
     * Generate the home planet mesh
     */
    generateHomePlanet() {
        const group = new THREE.Group();

        const planetGeometry = new THREE.SphereGeometry(this.planetRadius, 48, 48);
        const planetMaterial = new THREE.MeshPhongMaterial({
            color: 0x4a5568,
            emissive: 0x1a202c,
            emissiveIntensity: 0.28,
            shininess: 12
        });
        const planet = new THREE.Mesh(planetGeometry, planetMaterial);
        group.add(planet);

        const peakGeometry = new THREE.ConeGeometry(this.peakRadius, this.peakHeight, 8);
        const peakMaterial = new THREE.MeshPhongMaterial({
            color: 0xffffff,
            emissive: 0xe0e0e0,
            emissiveIntensity: 0.18,
            shininess: 90
        });
        const peak = new THREE.Mesh(peakGeometry, peakMaterial);
        peak.position.y = this.planetRadius + this.peakHeight * 0.5;
        group.add(peak);

        const snowCapGeometry = new THREE.ConeGeometry(this.peakRadius + 0.5, this.snowCapHeight, 8);
        const snowCapMaterial = new THREE.MeshPhongMaterial({
            color: 0xf5f6ff,
            emissive: 0xffffff,
            emissiveIntensity: 0.12,
            opacity: 0.85,
            transparent: true,
            shininess: 40
        });
        const snowCap = new THREE.Mesh(snowCapGeometry, snowCapMaterial);
        snowCap.position.y = this.planetRadius + this.peakHeight * 0.38;
        group.add(snowCap);

        const sparkGeometry = new THREE.SphereGeometry(0.18, 16, 16);
        const sparkMaterial = new THREE.MeshPhongMaterial({
            color: 0xffcfa3,
            emissive: 0xffcfa3,
            emissiveIntensity: 1.4,
            shininess: 240
        });
        const spark = new THREE.Mesh(sparkGeometry, sparkMaterial);
        spark.position.y = this.planetRadius + this.peakHeight * 0.9;
        group.add(spark);

        const peakLight = new THREE.PointLight(0xffb36d, 260, 0, 1.4);
        peakLight.position.copy(spark.position);
        peakLight.castShadow = false;
        peakLight.userData.isCosmosGlobalLight = true;
        group.add(peakLight);

        const atmosphereGeometry = new THREE.SphereGeometry(this.atmosphereRadius, 48, 48);
        const atmosphereMaterial = new THREE.MeshBasicMaterial({
            color: 0x88ccff,
            transparent: true,
            opacity: 0.08,
            side: THREE.BackSide
        });
        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        group.add(atmosphere);

        spark.userData.animate = (time) => {
            const pulse = Math.sin(time * 3.2) * 0.05 + 1;
            spark.scale.setScalar(pulse);
        };

        group.userData.animate = (time) => {
            planet.rotation.y += 0.00018;
            peak.rotation.y += 0.00018;
            snowCap.rotation.y += 0.00022;
            atmosphere.rotation.y += 0.00012;
            if (spark.userData.animate) {
                spark.userData.animate(time);
            }
        };

        group.userData.isHomePlanet = true;
        group.userData.clickHandler = () => {
            const target = window.HOME_PLANET_TARGET_URL || '/index.html';
            window.location.href = target;
        };

        return group;
    }

    /**
     * Create a spiky mountain peak
     */
    createMountainPeak() {
        const peakGeometry = new THREE.ConeGeometry(0.4, this.peakHeight, 8);

        // Add some jaggedness to make it look crystalline
        const positions = peakGeometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            const z = positions.getZ(i);

            // Add random jitter to vertices for crystalline look
            const jitter = Math.random() * 0.1;
            positions.setXYZ(
                i,
                x + (Math.random() - 0.5) * jitter,
                y,
                z + (Math.random() - 0.5) * jitter
            );
        }
        positions.needsUpdate = true;

        return peakGeometry;
    }

    /**
     * Create animated fog particles
     */
    createFogParticles() {
        return new THREE.Group();
    }

    /**
     * Update home planet animation
     */
    updateAnimation(mesh, time) {
        if (mesh && mesh.userData.animate) {
            mesh.userData.animate(time);
        }
    }
}

// Export
if (typeof window !== 'undefined') {
    window.HomePlanetGenerator = HomePlanetGenerator;
}

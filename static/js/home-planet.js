/**
 * Home Planet - Special entry point with fog and mountain peak
 * A mystical white foggy sphere with a crystalline mountain peak
 */

class HomePlanetGenerator {
    constructor() {
        this.fogDensity = 0.25;
        this.peakHeight = 3.8;
        this.sphereRadius = 2.5;
    }

    /**
     * Generate the home planet mesh
     */
    generateHomePlanet() {
        const group = new THREE.Group();

        // Create foggy sphere base
        const sphereGeometry = new THREE.IcosahedronGeometry(this.sphereRadius, 6);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0xfafaff,
            emissive: 0x141a2b,
            emissiveIntensity: 0.18,
            shininess: 70
        });

        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        group.add(sphere);

        // Create mountain peak
        const peakGeometry = this.createMountainPeak();
        const peakMaterial = new THREE.MeshPhongMaterial({
            color: 0xd4daff,
            emissive: 0x445b9c,
            emissiveIntensity: 0.28,
            shininess: 200
        });

        const peak = new THREE.Mesh(peakGeometry, peakMaterial);
        peak.position.y = this.sphereRadius + this.peakHeight * 0.42;
        group.add(peak);

        // Create shining spark at peak tip
        const sparkGeometry = new THREE.SphereGeometry(0.08, 12, 12);
        const sparkMaterial = new THREE.MeshPhongMaterial({
            color: 0xffcfa3,
            emissive: 0xffcfa3,
            emissiveIntensity: 1.2,
            shininess: 280
        });

        const spark = new THREE.Mesh(sparkGeometry, sparkMaterial);
        spark.position.y = this.sphereRadius + this.peakHeight * 0.47;
        group.add(spark);

        // Animate spark pulse
        spark.userData.animate = (time) => {
            const pulse = Math.sin(time * 3) * 0.15 + 0.85;
            spark.scale.setScalar(pulse);
        };

        // Add a global light sourced from the spark that reaches the entire cosmos
        const peakLight = new THREE.PointLight(0xffffff, 2.2, 0, 0);
        peakLight.position.copy(spark.position);
        peakLight.castShadow = false;
        peakLight.userData.isCosmosGlobalLight = true;
        group.add(peakLight);

        // Create animated fog particles around planet
        const fogParticles = this.createFogParticles();
        group.add(fogParticles);

        // Add glow effect
        group.userData.glowIntensity = 0.5;
        group.userData.animate = (time) => {
            // Animate spark
            if (spark.userData.animate) {
                spark.userData.animate(time);
            }

            // Subtle planet rotation
            sphere.rotation.y += 0.0002;

            // Animate fog particles
            fogParticles.children.forEach((particle, i) => {
                particle.position.y += Math.sin(time * 0.65 + i) * 0.0025;
                particle.rotation.z += 0.001;
            });

            // Pulsing glow
            const glowPulse = Math.sin(time * 0.3) * 0.2 + 0.3;
            sphereMaterial.emissiveIntensity = glowPulse;
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
        const fogGroup = new THREE.Group();

        const particleCount = 280;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);

        for (let i = 0; i < particleCount; i++) {
            // Random position on/near sphere surface
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            const r = this.sphereRadius + Math.random() * 1.9;

            positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = r * Math.cos(phi);
            positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);

            sizes[i] = Math.random() * 0.24 + 0.12;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        const material = new THREE.PointsMaterial({
            color: 0xf3f7ff,
            size: 0.28,
            sizeAttenuation: true,
            transparent: true,
            opacity: this.fogDensity,
            depthWrite: false,
            fog: false
        });

        const particles = new THREE.Points(geometry, material);
        fogGroup.add(particles);

        return fogGroup;
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

import React, { useState, useEffect, useRef, useMemo } from 'react';
import Globe from 'react-globe.gl';
import * as THREE from 'three';
import './GlobeComponent.css';

function SatelliteGlobe() {
    const globeEl = useRef();
    const SAT_SIZE = 1; // Adjusted satellite size
  
    // Initial satellite data
    const [satData, setSatData] = useState([
      { lat: 0, lng: 0, alt: 0.3, name: 'Satellite 1' },
      { lat: 20, lng: 20, alt: 0.3, name: 'Satellite 2' },
      { lat: -20, lng: -20, alt: 0.3, name: 'Satellite 3' },
    ]);
  
    // Satellite geometry and material
    const satGeometry = useMemo(
      () => new THREE.SphereGeometry(SAT_SIZE, 16, 16),
      []
    );
    const satMaterial = useMemo(
      () =>
        new THREE.MeshLambertMaterial({
          color: 'green', // Changed color for visibility
          transparent: true,
          opacity: 0.8,
        }),
      []
    );
  
    // Set up the globe
    useEffect(() => {
      globeEl.current.pointOfView({ altitude: 4 });
      globeEl.current.controls().autoRotate = true;
      globeEl.current.controls().autoRotateSpeed = 0.3;
      globeEl.current.scene().add(new THREE.AmbientLight(0xbbbbbb));
      globeEl.current.scene().add(new THREE.DirectionalLight(0xffffff, 0.6));
    }, []);
  
    // Update satellite positions to simulate orbiting
    useEffect(() => {
      const angularSpeed = 0.1; // radians per second
      const interval = setInterval(() => {
        setSatData((satData) =>
          satData.map((sat, idx) => {
            const time = Date.now() * 0.001; // time in seconds
            const angle = time * angularSpeed + idx * (Math.PI / 1.5);
            const lat = 50 * Math.sin(angle);
            const lng = 50 * Math.cos(angle);
            return { ...sat, lat, lng };
          })
        );
      }, 50); // Update every 50ms
  
      return () => clearInterval(interval);
    }, []);
  
    return (
      <div className="globe-container">
        <Globe
          ref={globeEl}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          backgroundColor="#000011"
          objectsData={satData}
          objectLat="lat"
          objectLng="lng"
          objectAltitude="alt"
          objectLabel="name"
          objectFacesSurface={false} // Ensure satellites don't face the globe surface
          objectThreeObject={(d) => new THREE.Mesh(satGeometry, satMaterial)}
        />
      </div>
    );
  }
export default SatelliteGlobe;

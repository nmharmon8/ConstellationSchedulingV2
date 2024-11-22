import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import Globe from 'react-globe.gl';
import * as THREE from 'three';
import { useAgent } from '../store/AgentStore';
import './GlobeComponent.css';
import TaskModal from './tasks/TaskModal';

function SatelliteGlobe() {
    const globeEl = useRef();
    const {
      tasks,
      loading,
      interpolatedPositions,
      currentActionsAndObs,
    } = useAgent();
    
    const SAT_SIZE = 2;
    const TASK_SIZE = 0.3;
    
    // Add currentPositions state
    const [currentPositions, setCurrentPositions] = useState({});
    const [satData, setSatData] = useState([]);
    const [taskData, setTaskData] = useState([]);
    const [pathsData, setPathsData] = useState([]);
    
    // Add task modal state
    const [selectedTask, setSelectedTask] = useState(null);
    
    // Task colors matching MapChart
    const taskColors = {
      RF: "#FF5722",
      IMAGING: "#2196F3",
      DATA_DOWNLINK: "#4CAF50"
    };
    
    // Satellite geometry and material
    const satGeometry = useMemo(
      () => new THREE.SphereGeometry(SAT_SIZE, 16, 16),
      []
    );
    const satMaterial = useMemo(
      () => new THREE.MeshLambertMaterial({
        color: '#ffffff',
        transparent: true,
        opacity: 0.8,
      }),
      []
    );
    
    const STEP_DURATION = 5000; // Match MapChart animation duration
    const animationRef = useRef(null);
    
    // Update the interpolatedPositions useEffect
    useEffect(() => {
      if (interpolatedPositions && Object.keys(interpolatedPositions).length > 0) {
        let startTime;
        
        const animate = (timestamp) => {
          if (!startTime) startTime = timestamp;
          const progress = (timestamp - startTime) / STEP_DURATION;

          if (progress < 1) {
            const newPositions = {};
            // Update satellite positions
            Object.entries(interpolatedPositions).forEach(([satId, positions]) => {
              const index = Math.min(
                Math.floor(progress * positions.length),
                positions.length - 1
              );
              const [lat, lon] = positions[index];
              newPositions[satId] = { lat, lon };
            });
            
            // Update both positions and satellite data atomically
            setCurrentPositions(newPositions);
            const newSatData = Object.entries(newPositions).map(([satId, pos]) => ({
              id: satId,
              lat: pos.lat,
              lng: pos.lon,
              alt: 0.3,
              name: `Satellite ${satId}`
            }));
            setSatData(newSatData);
            
            animationRef.current = requestAnimationFrame(animate);
          } else {
            // Set final positions
            const finalPositions = {};
            Object.entries(interpolatedPositions).forEach(([satId, positions]) => {
              const [lat, lon] = positions[positions.length - 1];
              finalPositions[satId] = { lat, lon };
            });
            
            // Update final positions and satellite data
            setCurrentPositions(finalPositions);
            const finalSatData = Object.entries(finalPositions).map(([satId, pos]) => ({
              id: satId,
              lat: pos.lat,
              lng: pos.lon,
              alt: 0.3,
              name: `Satellite ${satId}`
            }));
            setSatData(finalSatData);
          }
        };

        animationRef.current = requestAnimationFrame(animate);
      }

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };
    }, [interpolatedPositions]);

    // Separate useEffect for paths to reduce unnecessary updates
    useEffect(() => {
      if (currentActionsAndObs && currentPositions) {
        const newPaths = [];
        
        Object.entries(currentActionsAndObs.sat_to_tasks || {}).forEach(([satId, tasks]) => {
          const satPosition = currentPositions[satId];
          if (!satPosition) return;

          const activeTaskIndex = currentActionsAndObs.sat_to_act[satId];
          const activeTask = tasks[activeTaskIndex];
          
          if (activeTask && (activeTask.is_collection || activeTask.is_data_downlink)) {
            const [adjustedStartLng, adjustedEndLng] = adjustLongitude(
              satPosition.lon, 
              activeTask.longitude
            );
            
            newPaths.push({
              id: `${satId}-${activeTask.id}`,
              points: [
                { lat: satPosition.lat, lng: adjustedStartLng, alt: 0.3 },
                { lat: activeTask.latitude, lng: adjustedEndLng, alt: 0 }
              ],
              color: taskColors[activeTask.task_type_str]
            });
          }
        });
        
        setPathsData(newPaths);
      }
    }, [currentActionsAndObs, currentPositions]);

    // Convert tasks to globe format
    useEffect(() => {
      if (tasks) {
        const validTasks = tasks.filter(task => task.is_access_task);
        const newTaskData = validTasks.map(task => ({
          id: task.id,
          lat: task.latitude,
          lng: task.longitude,
          alt: 0.1,
          type: task.task_type_str,
          priority: task.priority,
          failCount: task.task_fail_count,
          task: task // Store full task object for click handling
        }));
        setTaskData(newTaskData);
      }
    }, [tasks]);

    // Add this helper function
    const adjustLongitude = (startLng, endLng) => {
      // If crossing the anti-meridian
      if (Math.abs(startLng - endLng) > 180) {
        if (startLng < 0) {
          // Adjust end longitude
          return [startLng, endLng - 360];
        } else {
          // Adjust end longitude
          return [startLng, endLng + 360];
        }
      }
      return [startLng, endLng];
    };

    // Set up the globe
    useEffect(() => {
      if (globeEl.current) {
        globeEl.current.pointOfView({ altitude: 2.5 });
        globeEl.current.controls().autoRotate = true;
        globeEl.current.controls().autoRotateSpeed = 0.3;
      }
    }, []);

    // Handle task click
    const handleTaskClick = useCallback((task) => {
      setSelectedTask(task.task);
    }, []);

    if (loading) {
      return <div>Loading globe...</div>;
    }

    return (
      <div className="globe-container">
        <Globe
          ref={globeEl}
          width={window.innerWidth}
          height={window.innerHeight}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
          backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
          
          // Center the globe
          centerAt={[0, 0, 0]}
          
          // Adjust camera position
          pointOfView={{
            lat: 0,
            lng: 0,
            altitude: 2.5
          }}
          
          // Remove all label-related props and replace with custom points
          pointsData={taskData}
          pointLat="lat"
          pointLng="lng"
          pointAltitude="alt"
          pointRadius={TASK_SIZE}
          pointColor={d => taskColors[d.type]}
          pointResolution={12}
          onPointClick={handleTaskClick}
          
          // Keep existing satellite objects
          objectsData={satData}
          objectLabel="name"
          objectLat="lat"
          objectLng="lng"
          objectAltitude="alt"
          objectThreeObject={() => new THREE.Mesh(satGeometry, satMaterial)}
          
          // Keep existing paths (arcs)
          pathsData={pathsData}
          pathPoints="points"
          pathPointLat="lat"
          pathPointLng="lng"
          pathPointAlt="alt"
          pathColor="color"
          pathStroke={2}
          pathDashLength={0.1}
          pathDashGap={0.05}
          pathDashAnimateTime={5000}
          pathTransitionDuration={0}
          
          // Globe settings
          atmosphereColor="rgba(100,149,237)"
          atmosphereAltitude={0.1}
        />
        
        <TaskModal 
          task={selectedTask}
          open={!!selectedTask}
          onClose={() => setSelectedTask(null)}
        />
      </div>
    );
}

export default SatelliteGlobe;

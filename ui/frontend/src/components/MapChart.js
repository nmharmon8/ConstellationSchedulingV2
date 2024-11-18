import React, { useEffect, useState, useRef } from "react";
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  Line,
} from "react-simple-maps";
import { Tooltip } from "react-tooltip";
import "react-tooltip/dist/react-tooltip.css";
import worldCountries from "../data/world-countries.json";
import './MapChart.css';
import { useAgent } from '../store/AgentStore';
import CircularProgress from '@mui/material/CircularProgress';
import SatelliteModal from './SatelliteModal';
import DownlinkStation from './DownlinkStation';
import SatelliteMarker from './SatelliteMarker';
import TaskModal from './TaskModal';
import CreateTaskModal from './CreateTaskModal';
import { geoEqualEarth } from "d3-geo";
import ObservationState from './ObservationState';

const STEP_DURATION = 4000; // 4 seconds

const MapChart = () => {
  const {
    tasks,
    loading,
    isResetting,
    interpolatedPositions,
    currentTasksBeingExecuted,
  } = useAgent();

  // Filter out CHARGE, DESAT, and NOOP tasks since they don't have valid coordinates
  const validTasks = tasks.filter(task => 
    task.is_access_task
  );

  const [currentPositions, setCurrentPositions] = useState({});
  const animationRef = useRef(null);
  const [selectedSatellite, setSelectedSatellite] = useState(null);
  const [selectedTask, setSelectedTask] = useState(null);
  const [createTaskCoordinates, setCreateTaskCoordinates] = useState(null);

  const [mapScale, setMapScale] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth > 2000 ? 190 : 140;
    }
    return 190; // Default scale if window is undefined
  });

  const taskColors = {
    RF: "#FF5722",
    IMAGING: "#2196F3",
    DATA_DOWNLINK: "#4CAF50"
  };

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 2000) {
        setMapScale(190);
      } else {
        setMapScale(140);
      }
    };

    // Initial check
    handleResize();

    // Add event listener
    window.addEventListener('resize', handleResize);

    // Cleanup event listener on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    if (interpolatedPositions && Object.keys(interpolatedPositions).length > 0) {
      let startTime;
      const animate = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const progress = (timestamp - startTime) / STEP_DURATION;

        if (progress < 1) {
          const newPositions = {};
          Object.entries(interpolatedPositions).forEach(([satId, positions]) => {
            const index = Math.min(
              Math.floor(progress * positions.length),
              positions.length - 1
            );
            const [lat, lon] = positions[index];
            newPositions[satId] = { lat, lon };
          });
          setCurrentPositions(newPositions);
          animationRef.current = requestAnimationFrame(animate);
        } else {
          // Set final positions
          const finalPositions = {};
          Object.entries(interpolatedPositions).forEach(([satId, positions]) => {
            const [lat, lon] = positions[positions.length - 1];
            finalPositions[satId] = { lat, lon };
          });
          setCurrentPositions(finalPositions);
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

  const handleSatelliteClick = (satId) => {
    // If clicking the same satellite, do nothing
    if (selectedSatellite === satId) return;
    
    // If clicking a different satellite, update to the new one
    setSelectedSatellite(satId);
  };

  const handleModalClose = () => {
    // Only close when explicitly called (e.g., from a close button in the modal)
    setSelectedSatellite(null);
  };

  const handleTaskClick = (task) => {
    setSelectedTask(task);
  };

  const handleTaskModalClose = () => {
    setSelectedTask(null);
  };

  const handleMapClick = (event) => {
    // Ignore clicks on markers
    if (event.target.closest('.satellite-marker') || event.target.closest('.marker-circle')) {
      return;
    }

    const svg = event.target.closest('svg');
    if (!svg) return;

    // Get SVG's dimensions and bounds
    const svgBounds = svg.getBoundingClientRect();
    
    // Calculate click position relative to SVG
    const x = event.clientX - svgBounds.left;
    const y = event.clientY - svgBounds.top;

    // Create projection with same parameters as ComposableMap
    const projection = geoEqualEarth()
      .scale(mapScale - 50) // Adjust scale if necessary
      .center([0, 0])
      .rotate([-10, 0, 0])
      .translate([svgBounds.width / 2, svgBounds.height / 2]);

    // Convert screen coordinates to geographic coordinates
    const [longitude, latitude] = projection.invert([x, y]);
    
    setCreateTaskCoordinates([longitude, latitude]);
  };

  const handleCreateTaskClose = () => {
    setCreateTaskCoordinates(null);
  };

  if (loading) {
    return <div>Loading map...</div>;
  }

  return (
    <div className="map-container">
      <div className="legend">
        <div className="legend-items">
          {Object.entries(taskColors).map(([taskType, color]) => (
            <div key={taskType} className="legend-item">
              {taskType === 'DATA_DOWNLINK' ? (
                <svg className="legend-station" viewBox="-15 -12 30 24">
                  <g>
                    <circle
                      r={5}
                      className="downlink-base"
                      style={{ 
                        stroke: color,
                        strokeWidth: "1.5",
                        fill: "rgba(255, 255, 255, 0.3)" 
                      }}
                    />
                    <rect
                      x="-1.5"
                      y="-4"
                      width="3"
                      height="4"
                      style={{
                        fill: color,
                        opacity: "0.8"
                      }}
                    />
                    <g transform="translate(0, -4) rotate(-30)">
                      <path
                        d="M 10 0 Q 0 12 -14 0"
                        className="dish-main"
                        style={{ 
                          stroke: color,
                          strokeWidth: "1.5",
                          fill: "rgba(255, 255, 255, 0.2)"
                        }}
                      />
                      <circle
                        cx="0"
                        cy="4"
                        r="1.5"
                        style={{
                          fill: color,
                          opacity: "0.9"
                        }}
                      />
                      <line 
                        x1="-6" 
                        y1="1" 
                        x2="0" 
                        y2="4" 
                        style={{ 
                          stroke: color,
                          strokeWidth: "1"
                        }}
                      />
                      <line 
                        x1="6" 
                        y1="1" 
                        x2="0" 
                        y2="4" 
                        style={{ 
                          stroke: color,
                          strokeWidth: "1"
                        }}
                      />
                    </g>
                  </g>
                </svg>
              ) : (
                <div
                  className="legend-dot"
                  style={{
                    backgroundColor: color,
                    boxShadow: `0 0 10px ${color}40`
                  }}
                />
              )}
              <span className="legend-text">{taskType}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="map-wrapper">
        <ComposableMap
          projection="geoEqualEarth"
          projectionConfig={{
            scale: mapScale, // Use the responsive scale here
            center: [0, 0],
            rotate: [-10, 0, 0],
          }}
          className="composable-map"
          onClick={handleMapClick}
          width={undefined}
          height={undefined}
        >
          <Geographies geography={worldCountries}>
            {({ geographies, projection }) =>
              geographies.map((geo) => (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  className="geography"
                />
              ))
            }
          </Geographies>

          {/* Render Task Markers */}
          {validTasks.map((task) => (
            task.is_data_downlink ? (
              <DownlinkStation 
                key={task.id}
                color={taskColors[task.task_type_str]}
                task={task}
                onClick={() => handleTaskClick(task)}
              />
            ) : (
              <Marker
                key={task.id}
                coordinates={[task.longitude, task.latitude]}
                data-tooltip-id="task-tooltip"
                data-tooltip-content={`${task.task_type_str} (Priority: ${task.priority.toFixed(2)})${
                  task.task_fail_count > 0 ? ` - Failed ${task.task_fail_count} times` : ''
                }`}
                onClick={() => handleTaskClick(task)}
              >
                <g>
                  {task.task_fail_count > 0 ? (
                    // Failed task markers remain the same
                    <>
                      <circle r={4} className="failed-task-pulse" />
                      <circle r={2} className="failed-task-marker" />
                      <line x1="-1.5" y1="-1.5" x2="1.5" y2="1.5" className="failed-task-x" />
                      <line x1="1.5" y1="-1.5" x2="-1.5" y2="1.5" className="failed-task-x" />
                    </>
                  ) : (
                    // Regular task marker
                    <circle 
                      r={2}
                      className="marker-circle"
                      fill={taskColors[task.task_type_str]} 
                    />
                  )}
                </g>
              </Marker>
            )
          ))}

          {/* Render Lines from Satellites to Current Tasks Being Executed */}
          {Object.entries(currentTasksBeingExecuted).map(([satId, task]) => {
            if (!task.is_access_task) return null;

            const satPosition = currentPositions[satId];
            if (!satPosition) return null;

            return (
              <Line
                key={`line-${satId}-${task.id}`}
                from={[satPosition.lon, satPosition.lat]}
                to={[task.longitude, task.latitude]}
                stroke="#FFD700"
                strokeWidth={2}
                strokeDasharray="4,4"
              />
            );
          })}

          {/* Satellite Markers */}
          {Object.entries(currentPositions).map(([satId, position]) => (
            <SatelliteMarker
              key={satId}
              satId={satId}
              position={position}
              isSelected={selectedSatellite === satId}
              onClick={handleSatelliteClick}
            />
          ))}
        </ComposableMap>

        <Tooltip 
          id="task-tooltip" 
          place="top" 
          effect="solid"
          className="tooltip"
        />

        {/* Spinner Overlay */}
        {isResetting && (
          <div className="spinner-overlay">
            <CircularProgress color="inherit" />
          </div>
        )}
        
      </div>

      <div className="observation-state-wrapper">
        <ObservationState />
      </div>

      <SatelliteModal
        open={!!selectedSatellite}
        onClose={handleModalClose}
        satelliteId={selectedSatellite}
      />

      <TaskModal 
        task={selectedTask}
        open={!!selectedTask}
        onClose={handleTaskModalClose}
      />

      <CreateTaskModal
        open={!!createTaskCoordinates}
        onClose={handleCreateTaskClose}
        coordinates={createTaskCoordinates}
      />
    </div>
  );
};

export default MapChart;

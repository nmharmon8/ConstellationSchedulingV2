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

const STEP_DURATION = 3000; // 3 seconds

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

  const taskColors = {
    RF: "#FF5722",
    IMAGING: "#2196F3",
    DATA_DOWNLINK: "#4CAF50"
  };

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

  if (loading) {
    return <div>Loading map...</div>;
  }

  return (
    <div className="map-container">
      <div className="legend">
        <div className="legend-items">
          {Object.entries(taskColors).map(([taskType, color]) => (
            <div key={taskType} className="legend-item">
              <div
                className="legend-dot"
                style={{
                  backgroundColor: color,
                  boxShadow: `0 0 10px ${color}40`
                }}
              ></div>
              <span className="legend-text">{taskType}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="map-wrapper">
        <ComposableMap
          projection="geoEqualEarth"
          projectionConfig={{
            scale: 140,
            center: [0, 0],
            rotate: [-10, 0, 0],
          }}
          className="composable-map"
        >
          <Geographies geography={worldCountries}>
            {({ geographies }) =>
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
              />
            ) : (
              <Marker
                key={task.id}
                coordinates={[task.longitude, task.latitude]}
                data-tooltip-id="task-tooltip"
                data-tooltip-content={`${task.task_type_str} (Priority: ${task.priority.toFixed(2)})${
                  task.task_fail_count > 0 ? ` - Failed ${task.task_fail_count} times` : ''
                }`}
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

      <SatelliteModal
        open={!!selectedSatellite}
        onClose={handleModalClose}
        satelliteId={selectedSatellite}
      />
    </div>
  );
};

export default MapChart;

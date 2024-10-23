import React, { useState, useEffect } from "react";
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
} from "react-simple-maps";
import { Tooltip } from "react-tooltip";
import "react-tooltip/dist/react-tooltip.css";
import worldCountries from "../data/world-countries.json";
import './MapChart.css';
import AgentControlBar from './AgentControlBar';

const MapChart = () => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);

  const taskColors = {
    RF: "#FF5722",
    IMAGING: "#2196F3",
    DATA_DOWNLINK: "#4CAF50"
  };

  useEffect(() => {
    fetch('/api/tasks')
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        // Filter out noop, desat, and charge tasks
        const filteredTasks = data.filter(task => 
          !task.is_noop && !task.is_desat && !task.is_charge
        );
        setTasks(filteredTasks);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching tasks:', error);
        setLoading(false);
      });
  }, []);

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
            scale: 160,
            center: [0, 0],
            rotate: [-10, 0, 0],
          }}
          className="composable-map"
          width={1000}
          height={300}
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

          {tasks.map((task) => (
            <Marker
              key={task.id}
              coordinates={[task.longitude, task.latitude]}
              data-tooltip-id="task-tooltip"
              data-tooltip-content={`${task.task_type} (Priority: ${task.priority.toFixed(2)})`}
            >
              <circle 
                r={2}  /* reduced from 5 */
                className="marker-circle"
                fill={taskColors[task.task_type]} 
              />
              {task.is_data_downlink && (
                <text
                  textAnchor="middle"
                  y={-10}
                  className="marker-text"
                >
                  {task.task_type}
                </text>
              )}
            </Marker>
          ))}
        </ComposableMap>

        <Tooltip 
          id="task-tooltip" 
          place="top" 
          effect="solid"
          className="tooltip"
        />
      </div>
      <AgentControlBar />
    </div>
  );
};

export default MapChart;

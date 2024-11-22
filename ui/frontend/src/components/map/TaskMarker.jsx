import React from 'react';
import { Marker } from "react-simple-maps";
import './TaskMarker.css';

const TaskMarker = React.memo(({ color, isFailed, coordinates, tooltipContent, isMultiSat }) => {
  const MarkerContent = () => {
    if (isFailed) {
      return (
        <g>
          <circle r={3} className="failed-task-marker" />
          <line x1="-1.5" y1="-1.5" x2="1.5" y2="1.5" className="failed-task-x" />
          <line x1="1.5" y1="-1.5" x2="-1.5" y2="1.5" className="failed-task-x" />
        </g>
      );
    }

    if (isMultiSat) {
      return (
        <g>
          <rect 
            x="-2.5" 
            y="-2.5" 
            width="5" 
            height="5" 
            className="multi-sat-marker"
            style={{ fill: color }}
            transform="rotate(45)"
          />
          <rect 
            x="-1" 
            y="-1" 
            width="2" 
            height="2" 
            className="multi-sat-inner"
            transform="rotate(45)"
          />
        </g>
      );
    }

    return (
      <circle 
        r={2}
        className="task-marker"
        style={{ fill: color }}
      />
    );
  };

  return (
    <Marker 
      coordinates={coordinates}
      data-tooltip-id="task-tooltip"
      data-tooltip-content={tooltipContent}
    >
      <MarkerContent />
    </Marker>
  );
});

export default TaskMarker; 
import React from 'react';
import { Marker } from "react-simple-maps";
import './TaskMarker.css';

const TaskMarker = React.memo(({ color, isFailed, coordinates, tooltipContent }) => {
  const MarkerContent = () => {
    if (isFailed) {
      return (
        <g>
          <circle r={4} className="failed-task-pulse" />
          <circle r={2} className="failed-task-marker" />
          <line x1="-1.5" y1="-1.5" x2="1.5" y2="1.5" className="failed-task-x" />
          <line x1="1.5" y1="-1.5" x2="-1.5" y2="1.5" className="failed-task-x" />
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
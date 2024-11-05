import React from 'react';
import { Marker } from "react-simple-maps";
import { Tooltip } from "react-tooltip";
import './SatelliteMarker.css';

const SatelliteMarker = ({ satId, position, isSelected, onClick }) => {
  const displayName = satId.split('_')[0];
  
  return (
    <Marker
      coordinates={[position.lon, position.lat]}
      data-tooltip-id={`sat-tooltip-${satId}`}
      data-tooltip-content={`Satellite: ${satId}`}
      onClick={() => onClick(satId)}
    >
      <g>
        {/* Highlight box for selected satellite */}
        {isSelected && (
          <rect
            x="-15"
            y="-15"
            width="30"
            height="30"
            className="satellite-highlight"
          />
        )}
        
        {/* Satellite body and panels */}
        <g className="satellite-marker">
          {/* Solar panels */}
          <rect x="-12" y="-3" width="4" height="6" className="satellite-panel-left" />
          <rect x="8" y="-3" width="4" height="6" className="satellite-panel-right" />
          
          {/* Main satellite body */}
          <rect x="-4" y="-4" width="8" height="8" className="satellite-body" />
          
          {/* Antenna */}
          <line x1="0" y1="-4" x2="0" y2="-8" className="satellite-antenna" />
          <circle r="1" cy="-8" className="satellite-antenna-tip" />
          
          {/* Communication dish */}
          <path
            d="M 4,-2 A 3,3 0 0 1 4,2"
            className="satellite-dish"
            fill="none"
          />
        </g>
        
        {/* Satellite label */}
        <text
          textAnchor="middle"
          y={16}
          className="marker-text"
        >
          {displayName}
        </text>
      </g>
      <Tooltip id={`sat-tooltip-${satId}`} place="top" effect="solid" className="tooltip" />
    </Marker>
  );
};

export default SatelliteMarker; 
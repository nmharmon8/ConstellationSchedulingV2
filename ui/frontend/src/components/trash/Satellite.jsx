import React from 'react';
import { Marker } from "react-simple-maps";
import { Tooltip } from "react-tooltip";
import './Satellite.css';

const Satellite = ({ satId, position, onClick }) => {
  return (
    <>
      <Marker
        coordinates={[position.lon, position.lat]}
        data-tooltip-id={`sat-tooltip-${satId}`}
        data-tooltip-content={`Satellite: ${satId}`}
        onClick={() => onClick(satId)}
      >
        <g className="satellite-marker">
          {/* Invisible larger clickable area */}
          <circle 
            r={12}
            fill="transparent"
            className="satellite-click-area"
          />
          
          {/* Pulsing effect */}
          <circle 
            r={8}
            className="satellite-pulse"
          />
          
          {/* Main satellite body */}
          <g transform="translate(-8, -8) scale(1)">
            {/* Core */}
            <circle 
              cx="8"
              cy="8"
              r="4"
              className="satellite-core"
            />
            
            {/* Orbital ring */}
            <circle
              cx="8"
              cy="8"
              r="7"
              className="satellite-ring"
            />
            
            {/* Solar panels */}
            <rect
              x="2"
              y="6"
              width="4"
              height="4"
              className="satellite-panels"
            />
            <rect
              x="10"
              y="6"
              width="4"
              height="4"
              className="satellite-panels"
            />
          </g>
        </g>
      </Marker>
      <Tooltip id={`sat-tooltip-${satId}`} place="top" effect="solid" className="tooltip" />
    </>
  );
};

export default Satellite; 
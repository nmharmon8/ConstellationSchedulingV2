import React from 'react';
import './SatList.css';
import { useAgent } from '../store/AgentStore';
import { Box, LinearProgress, Typography } from '@mui/material';

function SatList() {
  const { satellites, loading, error } = useAgent();

  if (loading) {
    return <div className="sat-list">Loading satellites...</div>;
  }

  if (error) {
    return <div className="sat-list">Error: {error}</div>;
  }

  return (
    <div className="sat-list">
      <div className="sat-list-header">
        <h2>Satellites</h2>
        <div className="sat-count">
          Total: {satellites.ids.length}
        </div>
      </div>
      <div className="sat-list-content">
        {satellites.ids.map((satId) => {
          const sat = satellites.currentPositions[satId];
          return (
            <div key={satId} className="sat-item">
              <div className="sat-header">
                <h3>{satId}</h3>
                <span className={`eclipse-status ${sat.in_eclipse ? 'in-eclipse' : ''}`}>
                  {sat.in_eclipse ? '🌑 Eclipse' : '☀️ Sunlight'}
                </span>
              </div>
              
              <div className="sat-metrics">
                <div className="metric">
                  <div className="metric-header">
                    <span>Storage</span>
                    <span>{(sat.storage_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={sat.storage_percentage * 100}
                    className="storage-progress"
                  />
                  <Typography variant="caption" className="metric-detail">
                    {(sat.storage_level / 1e9).toFixed(2)}GB / {(sat.storage_capacity / 1e9).toFixed(2)}GB
                  </Typography>
                </div>

                <div className="metric">
                  <div className="metric-header">
                    <span>Power</span>
                    <span>{(sat.power_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={sat.power_percentage * 100}
                    className="power-progress"
                  />
                  <Typography variant="caption" className="metric-detail">
                    {(sat.power_level / 1e3).toFixed(2)}kW / {(sat.power_capacity / 1e3).toFixed(2)}kW
                  </Typography>
                </div>
              </div>

              <div className="sat-location">
                <Typography variant="body2">
                  Lat: {sat.position.lat.toFixed(2)}°
                </Typography>
                <Typography variant="body2">
                  Lon: {sat.position.lon.toFixed(2)}°
                </Typography>
                <Typography variant="body2">
                  Alt: {(sat.position.alt / 1000).toFixed(2)} km
                </Typography>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default SatList;

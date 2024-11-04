import React from 'react';
import './SatList.css';
import { useAgent } from '../store/AgentStore';
import { Box, LinearProgress, Typography } from '@mui/material';

function SatList() {
  const { 
    currentSatState, 
    loading, 
  } = useAgent();

  if (loading) {
    return <div className="sat-list">Loading satellites...</div>;
  }

  const satelliteIds = Object.keys(currentSatState || {});
  const totalSatellites = satelliteIds.length;

  return (
    <div className="sat-list">
      <div className="sat-list-header">
        <h2>Satellites</h2>
        <div className="sat-count">
          Total: {totalSatellites}
        </div>
      </div>
      <div className="sat-list-content">
        {satelliteIds.map((satId) => {
          const sat = currentSatState[satId];

          if (!sat) {
            return (
              <div key={satId} className="sat-item">
                <div className="sat-header">
                  <h3>{satId}</h3>
                  <span className={`eclipse-status`}>
                    Info Unavailable
                  </span>
                </div>
              </div>
            );
          }

          const eclipseStatus = sat.observation.in_eclipse;

          return (
            <div key={satId} className="sat-item">
              <div className="sat-header">
                <h3>{satId}</h3>
                <span className={`eclipse-status ${eclipseStatus ? 'in-eclipse' : ''}`}>
                  {eclipseStatus ? '🌑 Eclipse' : '☀️ Sunlight'}
                </span>
              </div>
              
              <div className="sat-metrics">
                <div className="metric">
                  <div className="metric-header">
                    <span>Storage</span>
                    <span>{(sat.observation.storage_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={sat.observation.storage_percentage * 100}
                    className="storage-progress"
                  />
                  <Typography variant="caption" className="metric-detail">
                    {(sat.observation.storage_level / 1e9).toFixed(2)}GB / {(sat.observation.storage_capacity / 1e9).toFixed(2)}GB
                  </Typography>
                </div>

                <div className="metric">
                  <div className="metric-header">
                    <span>Power</span>
                    <span>{(sat.observation.power_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={sat.observation.power_percentage * 100}
                    className="power-progress"
                  />
                  <Typography variant="caption" className="metric-detail">
                    {(sat.observation.power_level / 1e3).toFixed(2)}kW / {(sat.observation.power_capacity / 1e3).toFixed(2)}kW
                  </Typography>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default SatList;

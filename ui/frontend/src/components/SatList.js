import React, { useState } from 'react';
import './SatList.css';
import { useAgent } from '../store/AgentStore';
import { Box, LinearProgress, Typography } from '@mui/material';
import SatelliteModal from './SatelliteModal';

function SatList() {
  const { 
    currentSatState, 
    loading, 
  } = useAgent();
  const [selectedSatellite, setSelectedSatellite] = useState(null);

  const handleSatelliteClick = (satId) => {
    setSelectedSatellite(satId);
  };

  const handleModalClose = () => {
    setSelectedSatellite(null);
  };

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
              <div 
                key={satId} 
                className="sat-item"
                onClick={() => handleSatelliteClick(satId)}
                style={{ cursor: 'pointer' }}
              >
                <div className="sat-header">
                  <h3>{satId}</h3>
                  <span className={`eclipse-status`}>
                    Info Unavailable
                  </span>
                </div>
              </div>
            );
          }

          const { observation } = sat;
          const { sat_task } = observation;
          const eclipseStatus = observation.in_eclipse;
          const isAlive = observation.is_alive;
          const taskRequested = sat_task?.task_type || 'N/A';
          const expectedAction = sat_task?.expected_action || 'N/A';
          const actualAction = sat_task?.actual_action || 'N/A';

          const shortSatName = satId.split('_')[0];

          return (
            <div 
              key={satId} 
              className="sat-item"
              onClick={() => handleSatelliteClick(satId)}
              style={{ cursor: 'pointer' }}
            >
              <div className="sat-header">
                <h3>{shortSatName}</h3>
                <div className="sat-status-container">
                  <span className={`eclipse-status ${eclipseStatus ? 'in-eclipse' : ''}`}>
                    {eclipseStatus ? '🌑 Eclipse' : '☀️ Sunlight'}
                  </span>
                  <span className={`sat-status ${isAlive ? 'status-alive' : 'status-dead'}`}>
                    {isAlive ? '🟢 Active' : '🔴 Fault'}
                  </span>
                </div>
              </div>
              
              <div className="sat-metrics">
                <div className="metric task-info">
                  {taskRequested} → {actualAction}
                </div>

                <div className="metric">
                  <div className="metric-header">
                    <span>Storage</span>
                    <span>{(observation.storage_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={observation.storage_percentage * 100}
                    className="storage-progress"
                  />
                  <Typography variant="caption" className="metric-detail">
                    {(observation.storage_level / 1e9).toFixed(2)}GB / {(observation.storage_capacity / 1e9).toFixed(2)}GB
                  </Typography>
                </div>

                <div className="metric">
                  <div className="metric-header">
                    <span>Power</span>
                    <span>{(observation.power_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={observation.power_percentage * 100}
                    className="power-progress"
                  />
                  <Typography variant="caption" className="metric-detail">
                    {(observation.power_level / 1e3).toFixed(2)}kW / {(observation.power_capacity / 1e3).toFixed(2)}kW
                  </Typography>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <SatelliteModal
        open={!!selectedSatellite}
        onClose={handleModalClose}
        satelliteId={selectedSatellite}
      />
    </div>
  );
}

export default SatList;

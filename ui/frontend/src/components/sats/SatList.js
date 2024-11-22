import React, { useState } from 'react';
import { useAgent } from '../../store/AgentStore';
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
    return <div className="h-screen w-full bg-space-dark text-white p-4">Loading satellites...</div>;
  }

  const satelliteIds = Object.keys(currentSatState || {});
  const totalSatellites = satelliteIds.length;

  return (
    <div className="relative w-full h-screen bg-space-dark text-white p-4 shadow-lg flex flex-col overflow-hidden font-sans">
      <div className="flex-none mb-5 border-b border-gray-600 pb-3">
        <h2 className="text-center text-space-cyan m-0">Satellites</h2>
        <div className="text-center">
          Total: {totalSatellites}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto pr-1 w-full 
        [&::-webkit-scrollbar]:w-2
        [&::-webkit-scrollbar-track]:bg-space-darker
        [&::-webkit-scrollbar-thumb]:bg-gray-600
        [&::-webkit-scrollbar-thumb]:rounded-full
        [&::-webkit-scrollbar-thumb]:hover:bg-gray-500">
        {satelliteIds.map((satId) => {
          const sat = currentSatState[satId];

          if (!sat) {
            return (
              <div 
                key={satId} 
                className="bg-space-darker border border-gray-600 rounded-lg p-4 mb-3 mr-1 cursor-pointer hover:bg-opacity-90 transition-colors"
                onClick={() => handleSatelliteClick(satId)}
              >
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-space-cyan m-0">{satId}</h3>
                  <span className={`flex items-center text-sm px-2 py-1 rounded-full bg-space-dark ${eclipseStatus ? 'bg-opacity-50' : ''}`}>
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
              className="bg-space-darker border border-gray-600 rounded-lg p-4 mb-3 mr-1 cursor-pointer hover:bg-opacity-90 transition-colors"
              onClick={() => handleSatelliteClick(satId)}
            >
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-space-cyan m-0">{shortSatName}</h3>
                <div className="flex gap-2">
                  <span className={`flex items-center text-sm px-2 py-1 rounded-full bg-space-dark ${eclipseStatus ? 'bg-opacity-50' : ''}`}>
                    {eclipseStatus ? '🌑 Eclipse' : '☀️ Sunlight'}
                  </span>
                  <span className={`flex items-center text-sm px-2 py-1 rounded-full bg-space-dark ${isAlive ? 'text-green-500' : 'text-red-500'}`}>
                    {isAlive ? '🟢 Active' : '🔴 Fault'}
                  </span>
                </div>
              </div>
              
              <div className="flex flex-col gap-4">
                <div className="bg-gradient-to-r from-space-cyan/10 to-space-cyan/5 border border-space-cyan/20 rounded p-3 text-space-cyan/90 font-mono text-sm relative">
                  {taskRequested} → {actualAction}
                </div>

                {/* Storage Metric */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-sm text-gray-300">
                    <span>Storage</span>
                    <span>{(observation.storage_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={observation.storage_percentage * 100}
                    className="h-1.5 bg-space-cyan/20"
                    sx={{
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#00e6e6'
                      }
                    }}
                  />
                  <Typography variant="caption" className="text-gray-500 text-xs">
                    {(observation.storage_level / 1e9).toFixed(2)}GB / {(observation.storage_capacity / 1e9).toFixed(2)}GB
                  </Typography>
                </div>

                {/* Power Metric */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-sm text-gray-300">
                    <span>Power</span>
                    <span>{(observation.power_percentage * 100).toFixed(1)}%</span>
                  </div>
                  <LinearProgress 
                    variant="determinate" 
                    value={observation.power_percentage * 100}
                    className="h-1.5 bg-yellow-500/20"
                    sx={{
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#ffc400'
                      }
                    }}
                  />
                  <Typography variant="caption" className="text-gray-500 text-xs">
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


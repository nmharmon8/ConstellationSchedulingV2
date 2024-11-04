import React, { useEffect, useState } from 'react';
import { useAgent } from '../store/AgentStore';
import { Box, Typography, Paper, Divider, Button } from '@mui/material';
import './InspectorView.css';

const InspectorView = () => {
  const { observationInspector, fetchObservationInspector } = useAgent();
  const [selectedSatellite, setSelectedSatellite] = useState(null);

  useEffect(() => {
    fetchObservationInspector();
  }, []);

  if (!observationInspector) {
    return (
      <Box className="inspector-container" sx={{ width: '100%' }}>
        <Typography variant="h6" color="primary">
          Loading inspector data...
        </Typography>
      </Box>
    );
  }

  const { debug } = observationInspector;

  return (
    <Box className="inspector-container" sx={{ width: '100%' }}>
      <Paper className="observation-section">
        <Typography variant="h5" className="section-title">
          Observations Configuration
        </Typography>
        
        <Box className="info-grid">
          <Paper className="info-card">
            <Typography variant="h6">Observation Keys</Typography>
            <Box className="key-list">
              {debug.observation_keys.map((key, index) => (
                <Typography key={index} className="key-item">
                  {key}
                </Typography>
              ))}
            </Box>
          </Paper>

          <Paper className="info-card">
            <Typography variant="h6">Normalization Terms</Typography>
            <Box className="terms-grid">
              {Object.entries(debug.normalization_terms).map(([key, value]) => (
                <Box key={key} className="term-item">
                  <Typography className="term-key">{key}:</Typography>
                  <Typography className="term-value">{value}</Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        </Box>
      </Paper>

      <Paper className="satellites-section">
        <Typography variant="h5" className="section-title">
          Satellite Observations
        </Typography>

        <Box className="satellite-buttons">
          {Object.keys(debug.satellite_observation).map((satId) => (
            <Button
              key={satId}
              onClick={() => setSelectedSatellite(satId)}
              className={`satellite-button ${selectedSatellite === satId ? 'active' : ''}`}
              variant="outlined"
            >
              {satId}
            </Button>
          ))}
        </Box>
        
        <Box className="satellites-grid">
          {selectedSatellite && (
            <Paper className="satellite-card">
              <Typography variant="h6" className="satellite-title">
                {selectedSatellite}
              </Typography>
              
              <Divider className="satellite-divider" />
              
              <Box className="observation-data">
                <Typography variant="subtitle1">Numpy Data</Typography>
                <Box className="numpy-grid">
                  {debug.satellite_observation[selectedSatellite].numpy.map((row, rowIndex) => (
                    <Box key={rowIndex} className="numpy-row">
                      {row.map((value, colIndex) => (
                        <Typography key={colIndex} className="numpy-value">
                          {value.toFixed(4)}
                        </Typography>
                      ))}
                    </Box>
                  ))}
                </Box>

                <Typography variant="subtitle1" sx={{ mt: 2 }}>
                  Observations
                </Typography>
                {debug.satellite_observation[selectedSatellite].observations.map((obs, index) => (
                  <Paper key={index} className="observation-item">
                    <Typography className="observation-type">
                      {obs.task_id}
                    </Typography>
                    <Box className="observation-details">
                      {Object.entries(obs).map(([key, value]) => (
                        <Box key={key} className="detail-item">
                          <Typography className="detail-key">{key}:</Typography>
                          <Typography className="detail-value">
                            {typeof value === 'number' ? value.toFixed(4) : value.toString()}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  </Paper>
                ))}
              </Box>
            </Paper>
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default InspectorView; 
import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  LinearProgress
} from '@mui/material';
import { useAgent } from '../store/AgentStore';
import './SatelliteModal.css';

const SatelliteModal = ({ open, onClose, satelliteId }) => {
  const { currentSatState } = useAgent();
  
  if (!satelliteId || !currentSatState || !currentSatState[satelliteId]) {
    return null;
  }

  const sat = currentSatState[satelliteId];
  const {
    lat,
    lon,
    alt,
    observation: {
      storage_level,
      storage_capacity,
      storage_percentage,
      power_level,
      power_capacity,
      power_percentage,
      in_eclipse,
      next_eclipse,
      end_of_eclipse
    }
  } = sat;

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      className="satellite-modal"
      hideBackdrop={true}
      disableScrollLock={true}
      sx={{ 
        position: 'absolute',
        pointerEvents: 'none'
      }}
      PaperProps={{
        sx: {
          pointerEvents: 'auto'
        }
      }}
    >
      <DialogTitle className="satellite-modal-header">
        <Typography className="satellite-modal-title">
          Satellite {satelliteId}
        </Typography>
      </DialogTitle>

      <DialogContent className="satellite-modal-content">
        {/* Location Information */}
        <div className="satellite-info-section">
          <Typography className="satellite-info-title">
            Location
          </Typography>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Latitude</span>
            <span className="satellite-metric-value">{lat.toFixed(4)}°</span>
          </div>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Longitude</span>
            <span className="satellite-metric-value">{lon.toFixed(4)}°</span>
          </div>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Altitude</span>
            <span className="satellite-metric-value">{(alt / 1000).toFixed(2)} km</span>
          </div>
        </div>

        {/* Storage Information */}
        <div className="satellite-info-section">
          <Typography className="satellite-info-title">
            Storage
          </Typography>
          <Box sx={{ mb: 2 }}>
            <LinearProgress 
              variant="determinate" 
              value={storage_percentage * 100}
              className="storage-progress"
            />
            <div className="satellite-metric">
              <span className="satellite-metric-label">Usage</span>
              <span className="satellite-metric-value">
                {(storage_level / 1e9).toFixed(2)}GB / {(storage_capacity / 1e9).toFixed(2)}GB
              </span>
            </div>
          </Box>
        </div>

        {/* Power Information */}
        <div className="satellite-info-section">
          <Typography className="satellite-info-title">
            Power
          </Typography>
          <Box sx={{ mb: 2 }}>
            <LinearProgress 
              variant="determinate" 
              value={power_percentage * 100}
              className="power-progress"
            />
            <div className="satellite-metric">
              <span className="satellite-metric-label">Level</span>
              <span className="satellite-metric-value">
                {(power_level / 1e3).toFixed(2)}kW / {(power_capacity / 1e3).toFixed(2)}kW
              </span>
            </div>
          </Box>
        </div>

        {/* Eclipse Status */}
        <div className="satellite-info-section">
          <Typography className="satellite-info-title">
            Eclipse Status
          </Typography>
          <div className={`satellite-status ${in_eclipse ? 'in-eclipse' : 'in-sunlight'}`}>
            {in_eclipse ? '🌑 In Eclipse' : '☀️ In Sunlight'}
          </div>
          {in_eclipse && (
            <div className="satellite-metric">
              <span className="satellite-metric-label">End of Eclipse</span>
              <span className="satellite-metric-value">{end_of_eclipse.toFixed(0)}s</span>
            </div>
          )}
          {!in_eclipse && (
            <div className="satellite-metric">
              <span className="satellite-metric-label">Next Eclipse</span>
              <span className="satellite-metric-value">{next_eclipse.toFixed(0)}s</span>
            </div>
          )}
        </div>
      </DialogContent>

      <DialogActions className="satellite-modal-actions">
        <Button 
          onClick={onClose}
          className="satellite-modal-button"
          variant="outlined"
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SatelliteModal; 
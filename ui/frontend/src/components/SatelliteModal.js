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
  console.log('Satellite:', sat);
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
      end_of_eclipse,
      is_alive,
      wheel_speed_1,
      wheel_speed_2,
      wheel_speed_3,
      sat_task,
      action,
      reward
    }
  } = sat;

  return (
    <Dialog 
      open={open} 
      maxWidth="sm"
      className="satellite-modal satellite-side-panel"
      hideBackdrop={true}
      disableScrollLock={true}
      onBackdropClick={() => {}}
      onClose={(event, reason) => {
        if (reason !== 'backdropClick') {
          onClose();
        }
      }}
      sx={{ 
        '& .MuiDialog-container': {
          alignItems: 'flex-start',
          justifyContent: 'flex-start'
        }
      }}
      PaperProps={{
        sx: {
          position: 'fixed',
          left: 0,
          top: 0,
          height: '100%',
          maxHeight: '100vh',
          width: '300px',
          maxWidth: '90vw',
          margin: 0,
          borderRadius: '0 12px 12px 0'
        }
      }}
    >
      <DialogTitle className="satellite-modal-header">
        <Typography className="satellite-modal-title">
          Satellite {satelliteId}
        </Typography>
      </DialogTitle>

      <DialogContent className="satellite-modal-content">
        {/* Position Vector Information */}
        <div className="satellite-info-section">
          <Typography className="satellite-info-title">
            Position Vector (r_BP_P)
          </Typography>
          <div className="satellite-metric">
            <span className="satellite-metric-label">X</span>
            <span className="satellite-metric-value">{sat.r_BP_P[0].toFixed(2)} m</span>
          </div>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Y</span>
            <span className="satellite-metric-value">{sat.r_BP_P[1].toFixed(2)} m</span>
          </div>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Z</span>
            <span className="satellite-metric-value">{sat.r_BP_P[2].toFixed(2)} m</span>
          </div>
        </div>

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

        {/* Satellite Status */}
        <div className="satellite-info-section">
          <Typography className="satellite-info-title">
            Satellite Status
          </Typography>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Status</span>
            <span className={`satellite-metric-value ${is_alive ? 'status-alive' : 'status-dead'}`}>
              {is_alive ? '🟢 Active' : '🔴 Inactive'}
            </span>
          </div>
        </div>

        {/* Reaction Wheels */}
        <div className="satellite-info-section">
          <Typography className="satellite-info-title">
            Reaction Wheels
          </Typography>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Wheel 1</span>
            <span className="satellite-metric-value">{wheel_speed_1.toFixed(4)} rad/s</span>
          </div>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Wheel 2</span>
            <span className="satellite-metric-value">{wheel_speed_2.toFixed(4)} rad/s</span>
          </div>
          <div className="satellite-metric">
            <span className="satellite-metric-label">Wheel 3</span>
            <span className="satellite-metric-value">{wheel_speed_3.toFixed(4)} rad/s</span>
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

        {/* Current Action */}
        {(action || sat_task || reward !== undefined) && (
          <div className="satellite-info-section">
            <Typography className="satellite-info-title">
              Current Action
            </Typography>
            {sat_task && (
              <div className="satellite-metric">
                <span className="satellite-metric-label">Requested</span>
                <span className="satellite-metric-value">{sat_task.task_type}</span>
              </div>
            )}
            {action && (
              <div className="satellite-metric">
                <span className="satellite-metric-label">Action</span>
                <span className="satellite-metric-value">{action}</span>
              </div>
            )}
            {reward !== undefined && (
              <div className="satellite-metric">
                <span className="satellite-metric-label">Reward</span>
                <span className="satellite-metric-value">{reward.toFixed(4)}</span>
              </div>
            )}
          </div>
        )}

        {/* Current Task */}
        {sat_task && (
          <div className="satellite-info-section">
            <Typography className="satellite-info-title">
              Current Task
            </Typography>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Type</span>
              <span className="satellite-metric-value">{sat_task.task_type}</span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Task Valid</span>
              <span className={`satellite-metric-value ${sat_task.sat_task_valid ? 'status-alive' : 'status-dead'}`}>
                {sat_task.sat_task_valid ? 'Valid' : 'Invalid'}
              </span>
            </div>
            
            {/* Add new fields here */}
            <div className="satellite-metric">
              <span className="satellite-metric-label">Expected Completion</span>
              <span className={`satellite-metric-value ${sat_task.expect_task_to_complete ? 'status-alive' : 'status-dead'}`}>
                {sat_task.expect_task_to_complete ? 'Yes' : 'No'}
              </span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Power/Storage Valid</span>
              <span className={`satellite-metric-value ${sat_task.power_storage_valid ? 'status-alive' : 'status-dead'}`}>
                {sat_task.power_storage_valid ? 'Valid' : 'Invalid'}
              </span>
            </div>

            {/* Initial States */}
            <div className="satellite-metric">
              <span className="satellite-metric-label">Initial Storage</span>
              <span className="satellite-metric-value">
                {(sat_task.init_storage / 1e9).toFixed(2)} GB
              </span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Initial Power</span>
              <span className="satellite-metric-value">
                {sat_task.init_power.toFixed(2)} W
              </span>
            </div>

            {/* Changes */}
            <div className="satellite-metric">
              <span className="satellite-metric-label">Storage Change</span>
              <span className="satellite-metric-value">
                {(sat_task.predicted_storage_change / 1e9).toFixed(2)} GB
              </span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Power Change</span>
              <span className="satellite-metric-value">
                {sat_task.predicted_power_change.toFixed(2)} W
              </span>
            </div>

            {/* Final States */}
            <div className="satellite-metric">
              <span className="satellite-metric-label">Final Storage</span>
              <span className="satellite-metric-value">
                {(sat_task.final_storage / 1e9).toFixed(2)} GB
              </span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Final Power</span>
              <span className="satellite-metric-value">
                {sat_task.final_power.toFixed(2)} W
              </span>
            </div>

            {/* Final Changes */}
            <div className="satellite-metric">
              <span className="satellite-metric-label">Δ Storage</span>
              <span className="satellite-metric-value">
                {(sat_task.final_storage_change / 1e9).toFixed(2)} GB
              </span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Δ Power</span>
              <span className="satellite-metric-value">
                {sat_task.final_power_change.toFixed(2)} W
              </span>
            </div>

            {/* Add percentage metrics */}
            <div className="satellite-metric">
              <span className="satellite-metric-label">Power %</span>
              <span className="satellite-metric-value">
                {(sat_task.pct_power * 100).toFixed(1)}%
              </span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Storage %</span>
              <span className="satellite-metric-value">
                {(sat_task.pct_storage * 100).toFixed(1)}%
              </span>
            </div>

            {/* Add new action fields */}
            <div className="satellite-metric">
              <span className="satellite-metric-label">Expected Action</span>
              <span className="satellite-metric-value">
                {sat_task.expected_action}
              </span>
            </div>
            <div className="satellite-metric">
              <span className="satellite-metric-label">Actual Action</span>
              <span className="satellite-metric-value">
                {sat_task.actual_action}
              </span>
            </div>
          </div>
        )}
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
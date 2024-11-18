import React from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, Typography, Box, Chip } from '@mui/material';
import './TaskModal.css';

const TaskModal = ({ task, open, onClose }) => {
   
  if (!task) {
    return null;
  }

  const formatNumber = (num) => {
    return num !== undefined ? Number(num).toFixed(3) : '0.000';
  };

  const formatStorageSize = (size) => {
    return size !== undefined ? (Number(size) / 1e6).toFixed(3) : '0.000';
  };

  const getTaskTypeColor = () => {
    switch (task.task_type_str) {
      case 'IMAGING':
        return '#2196F3';
      case 'RF':
        return '#FF5722';
      case 'DATA_DOWNLINK':
        return '#4CAF50';
      case 'CHARGE':
        return '#FFC107';
      case 'DESAT':
        return '#9C27B0';
      default:
        return '#64FFDA';
    }
  };

  return (
    <Dialog 
      open={open} 
      maxWidth="sm"
      className="task-modal task-side-panel"
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
          justifyContent: 'flex-end'
        }
      }}
      PaperProps={{
        sx: {
          position: 'fixed',
          right: 0,
          top: 0,
          height: '100%',
          maxHeight: '100vh',
          width: '300px',
          maxWidth: '90vw',
          margin: 0,
          borderRadius: '12px 0 0 12px',
          backgroundColor: '#1e1e2f',
          color: '#ffffff',
          border: '1px solid rgba(100, 255, 218, 0.1)'
        }
      }}
    >
      <DialogTitle className="task-modal-header">
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h6" color="#00e6e6">
            Task Details
          </Typography>
          <Chip 
            label={task.task_type_str}
            sx={{ 
              backgroundColor: getTaskTypeColor(),
              color: 'white',
            }}
          />
        </Box>
      </DialogTitle>
      <DialogContent>
        {/* Task Status Section */}
        <div className="task-info-section">
          <Typography variant="subtitle1" color="#00e6e6" gutterBottom>
            Status
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip 
              label={task.task_complete ? 'Complete' : 'Pending'}
              color={task.task_complete ? 'success' : 'warning'}
              size="small"
            />
            {task.is_collection && <Chip label="Collection Task" size="small" />}
            {task.is_data_downlink && <Chip label="Data Downlink" size="small" />}
            {task.is_charge && <Chip label="Charging" size="small" />}
            {task.is_desat && <Chip label="Desaturation" size="small" />}
          </Box>
        </div>

        {/* Task Metrics Section */}
        <div className="task-info-section">
          <Typography variant="subtitle1" color="#00e6e6" gutterBottom>
            Task Metrics
          </Typography>
          <div className="task-metric">
            <span className="task-metric-label">Priority</span>
            <span className="task-metric-value">{formatNumber(task.priority)}</span>
          </div>
          <div className="task-metric">
            <span className="task-metric-label">Duration</span>
            <span className="task-metric-value">{formatNumber(task.task_duration)}s</span>
          </div>
          {task.task_reward !== undefined && (
            <div className="task-metric">
              <span className="task-metric-label">Reward</span>
              <span className="task-metric-value">{formatNumber(task.task_reward)}</span>
            </div>
          )}
        </div>

        {/* Resource Requirements */}
        <div className="task-info-section">
          <Typography variant="subtitle1" color="#00e6e6" gutterBottom>
            Resource Requirements
          </Typography>
          {task.storage_size > 0 && (
            <Typography variant="body1" color="#cccccc" gutterBottom>
              Storage Required: {formatStorageSize(task.storage_size)} MB
            </Typography>
          )}
          {task.simultaneous_collects_required > 1 && (
            <Typography variant="body1" color="#cccccc" gutterBottom>
              Required Satellites: {task.simultaneous_collects_required}
            </Typography>
          )}
        </div>

        {/* Location Information */}
        {(task.latitude !== undefined && task.longitude !== undefined) && (
          <div className="task-info-section">
            <Typography variant="subtitle1" color="#00e6e6" gutterBottom>
              Location
            </Typography>
            <Typography variant="body1" color="#cccccc" gutterBottom>
              Latitude: {formatNumber(task.latitude)}°
            </Typography>
            <Typography variant="body1" color="#cccccc" gutterBottom>
              Longitude: {formatNumber(task.longitude)}°
            </Typography>
            <Typography variant="body1" color="#cccccc" gutterBottom>
              Altitude: {formatNumber(task.altitude / 1000)} km
            </Typography>
            {task.min_elev && (
              <Typography variant="body1" color="#cccccc" gutterBottom>
                Minimum Elevation: {formatNumber(task.min_elev * (180/Math.PI))}°
              </Typography>
            )}
          </div>
        )}

        {/* Satellites Collection Info */}
        {task.sats_collecting && task.sats_collecting.length > 0 && (
          <div className="task-info-section">
            <Typography variant="subtitle1" color="#00e6e6" gutterBottom>
              Collecting Satellites
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {task.sats_collecting.map((satId, index) => (
                <Chip 
                  key={index}
                  label={satId}
                  size="small"
                  sx={{ backgroundColor: 'rgba(0, 230, 230, 0.1)' }}
                />
              ))}
            </Box>
          </div>
        )}
      </DialogContent>
      <DialogActions sx={{ 
        borderTop: '1px solid rgba(100, 255, 218, 0.1)', 
        p: 2,
        background: 'linear-gradient(90deg, rgba(100, 255, 218, 0.05), transparent)'
      }}>
        <Button 
          onClick={onClose}
          sx={{
            color: '#00e6e6',
            borderColor: '#00e6e6',
            '&:hover': {
              backgroundColor: 'rgba(0, 230, 230, 0.1)',
            }
          }}
          variant="outlined"
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default TaskModal;

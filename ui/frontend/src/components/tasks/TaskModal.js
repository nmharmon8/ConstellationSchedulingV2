import React from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, Typography, Box, Chip } from '@mui/material';
import './TaskModal.css';

const TaskModal = ({ task, open, onClose }) => {
   
  if (!task) {
    return null;
  }

  const getTaskTypeColor = () => {
    const colorMap = {
      'COLLECT': 'rgba(0, 230, 230, 0.5)',
      'DOWNLINK': 'rgba(255, 193, 7, 0.5)',
      'CHARGE': 'rgba(76, 175, 80, 0.5)',
      'DESAT': 'rgba(244, 67, 54, 0.5)'
    };
    return colorMap[task.task_type_str] || 'rgba(158, 158, 158, 0.5)'; // Default gray if type not found
  };

  const formatValue = (value, key) => {
    if (value === undefined || value === null) return 'N/A';
    
    // Add special case for task ID
    if (key === 'id') {
      return value.toString().substring(0, 12);
    }
    
    // Handle different types
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    } else if (typeof value === 'number') {
      // Special cases for known numeric fields
      if (key.includes('size')) return `${(value / 1e6).toFixed(3)} MB`;
      if (key.includes('altitude')) return `${(value / 1000).toFixed(3)} km`;
      if (key.includes('elev')) return `${(value * (180/Math.PI)).toFixed(3)}°`;
      return value.toFixed(3);
    } else if (Array.isArray(value)) {
      return (
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {value.map((item, idx) => (
            <Chip 
              key={idx}
              label={item}
              size="small"
              sx={{ backgroundColor: 'rgba(0, 230, 230, 0.1)' }}
            />
          ))}
        </Box>
      );
    }
    return value.toString();
  };

  const formatFieldName = (key) => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const shouldSkipField = (key, value) => {
    const skipFields = ['task_type_str']; // Add fields to skip
    return skipFields.includes(key) || value === undefined;
  };

  const groupFields = (task) => {
    const groups = {
      status: ['task_complete', 'is_collection', 'is_data_downlink', 'is_charge', 'is_desat'],
      metrics: ['priority', 'task_duration', 'task_reward'],
      location: ['latitude', 'longitude', 'altitude', 'min_elev'],
      resources: ['storage_size', 'simultaneous_collects_required'],
      satellites: ['sats_collecting'],
      other: []
    };

    const groupedFields = {};
    Object.keys(task).forEach(key => {
      if (shouldSkipField(key, task[key])) return;
      
      let assigned = false;
      for (const [group, fields] of Object.entries(groups)) {
        if (fields.includes(key)) {
          if (!groupedFields[group]) groupedFields[group] = [];
          groupedFields[group].push(key);
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        if (!groupedFields.other) groupedFields.other = [];
        groupedFields.other.push(key);
      }
    });

    return groupedFields;
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
        {Object.entries(groupFields(task)).map(([group, fields]) => (
          fields.length > 0 && (
            <div key={group} className="task-info-section">
              <Typography variant="subtitle1" color="#00e6e6" gutterBottom>
                {formatFieldName(group)}
              </Typography>
              {fields.map(field => (
                <div key={field} className="task-metric">
                  <span className="task-metric-label">{formatFieldName(field)}</span>
                  <span className="task-metric-value">
                    {formatValue(task[field], field)}
                  </span>
                </div>
              ))}
            </div>
          )
        ))}
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

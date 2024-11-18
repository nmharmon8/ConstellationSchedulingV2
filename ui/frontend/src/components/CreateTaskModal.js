import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack
} from '@mui/material';
import { useAgent } from '../store/AgentStore';

const CreateTaskModal = ({ open, onClose, coordinates }) => {
  const { createTask } = useAgent();
  const [formData, setFormData] = useState({
    name: 'New Task',
    priority: 1,
    taskType: 0,
    minElev: 30,
    duration: 60,
    lat: coordinates ? coordinates[1] : 0,
    lon: coordinates ? coordinates[0] : 0
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const taskData = {
      ...formData
    };

    const result = await createTask(taskData);
    if (result) {
      onClose();
      // Reset form
      setFormData({
        name: '',
        priority: 1,
        taskType: 0,
        minElev: 30,
        duration: 60,
        lat: 0,
        lon: 0
      });
    }
  };

  const commonTextFieldStyles = {
    '& .MuiOutlinedInput-root': {
      backgroundColor: 'var(--modal-section)',
      '& fieldset': {
        borderColor: 'rgba(100, 255, 218, 0.23)',
      },
      '&:hover fieldset': {
        borderColor: 'var(--modal-border)',
      },
      '&.Mui-focused fieldset': {
        borderColor: 'var(--modal-border)',
      }
    },
    '& .MuiInputLabel-root': {
      color: 'var(--modal-text)',
      '&.Mui-focused': {
        color: 'var(--modal-border)'
      }
    },
    '& .MuiInputBase-input': {
      color: 'var(--modal-text)',
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="sm" 
      fullWidth
      className="task-modal"
      PaperProps={{
        style: {
          background: 'var(--modal-bg)',
          border: '1px solid var(--modal-border)',
          borderRadius: '12px',
          color: 'var(--modal-text)',
          overflow: 'hidden'
        }
      }}
    >
      <DialogTitle sx={{ 
        background: 'linear-gradient(90deg, rgba(100, 255, 218, 0.1), transparent)',
        borderBottom: '1px solid var(--modal-border)',
        typography: 'h5',
        fontWeight: 'bold',
        color: 'var(--modal-header)'
      }}>
        Create New Task
      </DialogTitle>
      <form onSubmit={handleSubmit}>
        <DialogContent>
          <Stack spacing={2.5} sx={{ mt: 2 }}>
            <TextField
              name="name"
              label="Task Name"
              value={formData.name}
              onChange={handleChange}
              fullWidth
              required
              sx={commonTextFieldStyles}
            />
            
            <FormControl fullWidth>
              <InputLabel sx={{
                color: 'var(--modal-text)',
                '&.Mui-focused': {
                  color: 'var(--modal-border)'
                }
              }}>
                Task Type
              </InputLabel>
              <Select
                name="taskType"
                value={formData.taskType}
                onChange={handleChange}
                label="Task Type"
                required
                sx={{
                  ...commonTextFieldStyles,
                  '& .MuiSelect-icon': {
                    color: 'var(--modal-text)',
                  }
                }}
              >
                <MenuItem value={0} sx={{
                  backgroundColor: '#1e1e2f',
                  '&:hover': {
                    backgroundColor: 'rgba(100, 255, 218, 0.1)',
                  }
                }}>RF</MenuItem>
                <MenuItem value={1} sx={{
                  backgroundColor: '#1e1e2f',
                  '&:hover': {
                    backgroundColor: 'rgba(100, 255, 218, 0.1)',
                  }
                }}>Imaging</MenuItem>
              </Select>
            </FormControl>

            <TextField
              name="priority"
              label="Priority"
              type="number"
              value={formData.priority}
              onChange={handleChange}
              inputProps={{ min: 0, max: 10, step: 0.1 }}
              fullWidth
              required
              sx={commonTextFieldStyles}
            />

            <TextField
              name="minElev"
              label="Minimum Elevation (degrees)"
              type="number"
              value={formData.minElev}
              onChange={handleChange}
              inputProps={{ min: 0, max: 90, step: 1 }}
              fullWidth
              required
              sx={commonTextFieldStyles}
            />

            <TextField
              name="duration"
              label="Duration (seconds)"
              type="number"
              value={formData.duration}
              onChange={handleChange}
              inputProps={{ min: 0, max: 200, step: 1 }}
              fullWidth
              required
              sx={commonTextFieldStyles}
            />

            <Stack direction="row" spacing={2}>
              <TextField
                name="lat"
                label="Latitude"
                type="number"
                value={formData.lat}
                onChange={handleChange}
                inputProps={{ 
                  step: 0.0001,
                  min: -90,
                  max: 90
                }}
                fullWidth
                required
                sx={commonTextFieldStyles}
              />
              <TextField
                name="lon"
                label="Longitude"
                type="number"
                value={formData.lon}
                onChange={handleChange}
                inputProps={{ 
                  step: 0.0001,
                  min: -180,
                  max: 180
                }}
                fullWidth
                required
                sx={commonTextFieldStyles}
              />
            </Stack>
          </Stack>
        </DialogContent>
        <DialogActions sx={{ 
          borderTop: '1px solid var(--modal-border)', 
          p: 2,
          background: 'linear-gradient(90deg, rgba(100, 255, 218, 0.05), transparent)'
        }}>
          <Button 
            onClick={onClose}
            variant="outlined"
            sx={{
              color: 'var(--modal-border)',
              borderColor: 'var(--modal-border)',
              '&:hover': {
                borderColor: 'var(--modal-border)',
                backgroundColor: 'rgba(100, 255, 218, 0.1)',
              }
            }}
          >
            Cancel
          </Button>
          <Button 
            type="submit" 
            variant="contained" 
            sx={{
              backgroundColor: 'var(--modal-border)',
              color: '#0A192F',
              '&:hover': {
                backgroundColor: 'var(--storage-color)',
              }
            }}
          >
            Create Task
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default CreateTaskModal; 
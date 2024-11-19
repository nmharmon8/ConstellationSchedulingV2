import React, { useState, useEffect } from 'react';
import { useAgent } from '../../../store/AgentStore';
import { Typography, Select, MenuItem, FormControl, InputLabel, IconButton } from '@mui/material';
import TaskModal from '../TaskModal';
import { ArrowUpward, ArrowDownward } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Custom styled Select component to match dark theme
const DarkSelect = styled(Select)(({ theme }) => ({
  '& .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: '#00e6e6',
  },
  '& .MuiSelect-icon': {
    color: 'rgba(255, 255, 255, 0.5)',
  },
  // Style for the menu items when opened
  '& .MuiSelect-select': {
    color: 'white',
  },
}));

function TaskList() {
  const { tasks = [], loading, isAutoRunning, pauseAutoStep, startAutoStep } = useAgent();
  const [selectedTask, setSelectedTask] = useState(null);
  const [wasRunning, setWasRunning] = useState(false);

  // State for sorting
  const [sortField, setSortField] = useState('');
  const [sortOrder, setSortOrder] = useState('asc'); // 'asc' or 'desc'

  useEffect(() => {
    console.log('Selected Task changed:', selectedTask);
  }, [selectedTask]);

  const handleTaskClick = (task) => {
    console.log('Task clicked:', task);
    setSelectedTask(task);
    if (isAutoRunning) {
      setWasRunning(true);
      pauseAutoStep();
    }
  };

  const handleCloseModal = () => {
    console.log('Modal closed');
    setSelectedTask(null);
    if (wasRunning) {
      startAutoStep();
      setWasRunning(false);
    }
  };

  // Compute task type statistics
  const taskTypeStats = tasks.reduce((acc, task) => {
    if (task?.task_type_str) {
      acc[task.task_type_str] = (acc[task.task_type_str] || 0) + 1;
    }
    return acc;
  }, {});

  // Sorting logic
  const sortedTasks = [...tasks];
  if (sortField) {
    sortedTasks.sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];

      if (aValue === undefined) return 1;
      if (bValue === undefined) return -1;

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }

      // For string comparison
      const aStr = String(aValue).toLowerCase();
      const bStr = String(bValue).toLowerCase();
      if (aStr < bStr) return sortOrder === 'asc' ? -1 : 1;
      if (aStr > bStr) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });
  }

  if (loading) {
    return <div className="h-screen w-full bg-space-dark text-white p-4">Loading tasks...</div>;
  }

  const formatNumber = (num) => {
    return num !== undefined ? Number(num).toFixed(2) : '0.00';
  };

  const formatStorageSize = (size) => {
    return size !== undefined ? (Number(size) / 1e6).toFixed(2) : '0.00';
  };

  // Handler for changing sort field
  const handleSortFieldChange = (event) => {
    setSortField(event.target.value);
  };

  // Handler for toggling sort order
  const toggleSortOrder = () => {
    setSortOrder((prevOrder) => (prevOrder === 'asc' ? 'desc' : 'asc'));
  };

  // Available fields for sorting
  const sortableFields = [
    { value: 'task_type', label: 'Task Type' },
    { value: 'priority', label: 'Priority' },
    { value: 'task_duration', label: 'Duration' },
    { value: 'storage_size', label: 'Storage Size' },
    { value: 'simultaneous_collects_required', label: 'Satellites Required' },
    { value: 'task_fail_count', label: 'Fail Count' },
    { value: 'latitude', label: 'Latitude' },
    { value: 'longitude', label: 'Longitude' },
    { value: 'altitude', label: 'Altitude' },
  ];

  return (
    <div className="relative w-full h-screen bg-space-dark text-white p-4 shadow-lg flex flex-col overflow-hidden font-sans">
      {/* Header - Updated styling */}
      <div className="flex-none mb-5 border-b border-gray-600 pb-3">
        <h2 className="text-center text-space-cyan m-0">Tasks</h2>
        <div className="flex flex-wrap justify-center gap-2 mt-2">
          {Object.entries(taskTypeStats).map(([type, count]) => (
            <span 
              key={type} 
              className="text-sm px-3 py-1 bg-space-darker rounded-full border border-gray-600"
            >
              {type}: {count}
            </span>
          ))}
        </div>
      </div>

      {/* Updated Sorting Controls with inline arrow */}
      <div className="flex flex-col gap-2 mb-4">
        <div className="flex items-center gap-2">
          <FormControl 
            variant="outlined" 
            size="small" 
            className="w-full"
          >
            <InputLabel 
              id="sort-field-label" 
              className="text-gray-400"
              sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
            >
              Sort By
            </InputLabel>
            <DarkSelect
              labelId="sort-field-label"
              value={sortField}
              onChange={handleSortFieldChange}
              label="Sort By"
              className="bg-space-darker"
              MenuProps={{
                PaperProps: {
                  sx: {
                    bgcolor: '#1e1e2f',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    '& .MuiMenuItem-root': {
                      color: 'white',
                      '&:hover': {
                        bgcolor: 'rgba(0, 230, 230, 0.1)',
                      },
                      '&.Mui-selected': {
                        bgcolor: 'rgba(0, 230, 230, 0.2)',
                        '&:hover': {
                          bgcolor: 'rgba(0, 230, 230, 0.3)',
                        }
                      }
                    }
                  }
                }
              }}
            >
              <MenuItem value="" className="text-white">
                <em>None</em>
              </MenuItem>
              {sortableFields.map((field) => (
                <MenuItem key={field.value} value={field.value}>
                  {field.label}
                </MenuItem>
              ))}
            </DarkSelect>
          </FormControl>
          
          {sortField && (
            <IconButton 
              onClick={toggleSortOrder} 
              aria-label="toggle sort order"
              sx={{
                color: 'rgba(255, 255, 255, 0.7)',
                '&:hover': {
                  color: '#00e6e6',
                  backgroundColor: 'rgba(0, 230, 230, 0.1)',
                }
              }}
              size="small"
            >
              {sortOrder === 'asc' ? <ArrowUpward /> : <ArrowDownward />}
            </IconButton>
          )}
        </div>
      </div>

      {/* Task List Content */}
      <div className="flex-1 overflow-y-auto pr-1 w-full scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-space-dark">
        {!sortedTasks?.length ? (
          <Typography variant="body1" className="text-center text-gray-400">
            No tasks available.
          </Typography>
        ) : (
          <div className="space-y-3">
            {sortedTasks.map((task, index) => {
              if (!task || !task.id) {
                console.warn(`Task at index ${index} is missing an 'id' property.`);
                return null;
              }
              return (
                <div 
                  key={task.id}
                  className={`bg-space-darker border border-gray-600 rounded-lg p-4 mr-1 cursor-pointer hover:bg-opacity-90 transition-colors
                    ${task.task_fail_count > 0 ? 'border-red-500/50' : ''}`}
                  onClick={() => handleTaskClick(task)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      handleTaskClick(task);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  {/* Task Header */}
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-space-cyan font-medium">{task.task_type_str}</span>
                    <span className="text-sm px-2 py-1 rounded-full bg-space-dark">
                      Priority: {formatNumber(task.priority)}
                    </span>
                  </div>

                  {/* Task Details */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm text-gray-300">
                      <span>Duration</span>
                      <span>{formatNumber(task.task_duration)}s</span>
                    </div>

                    {task.storage_size && (
                      <div className="flex justify-between text-sm text-gray-300">
                        <span>Storage Required</span>
                        <span>{formatStorageSize(task.storage_size)} MB</span>
                      </div>
                    )}

                    {task.simultaneous_collects_required > 1 && (
                      <div className="flex justify-between text-sm text-gray-300">
                        <span>Required Satellites</span>
                        <span>{task.simultaneous_collects_required}</span>
                      </div>
                    )}

                    {task.task_fail_count > 0 && (
                      <div className="flex justify-between text-sm text-red-400">
                        <span>Failed Attempts</span>
                        <span>{task.task_fail_count}</span>
                      </div>
                    )}
                  </div>

                  {/* Task Location */}
                  <div className="mt-3 pt-3 border-t border-gray-600 grid grid-cols-3 gap-2 text-sm text-gray-300">
                    <Typography variant="body2">
                      Lat: {formatNumber(task.latitude)}°
                    </Typography>
                    <Typography variant="body2">
                      Lon: {formatNumber(task.longitude)}°
                    </Typography>
                    <Typography variant="body2">
                      Alt: {formatNumber(task.altitude / 1000)} km
                    </Typography>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Modal remains unchanged */}
      {selectedTask && (
        <TaskModal 
          task={selectedTask}
          open={!!selectedTask}
          onClose={handleCloseModal}
        />
      )}
    </div>
  );
}

export default TaskList;
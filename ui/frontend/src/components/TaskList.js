import React, { useState, useEffect } from 'react';
import './TaskList.css';
import { useAgent } from '../store/AgentStore';
import { Typography, Select, MenuItem, FormControl, InputLabel, IconButton } from '@mui/material';
import TaskModal from './TaskModal';
import { ArrowUpward, ArrowDownward } from '@mui/icons-material';

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
    return <div className="task-list">Loading tasks...</div>;
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
    <div className="task-list">
      <div className="task-list-header">
        <h2>Tasks</h2>
        <div className="task-stats">
          {Object.entries(taskTypeStats).map(([type, count]) => (
            <span key={type} className="task-stat">
              {type}: {count}
            </span>
          ))}
        </div>
      </div>

      {/* Sorting Controls */}
      <div className="sorting-controls" style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
        <FormControl variant="outlined" size="small" style={{ minWidth: 200, marginRight: '16px' }}>
          <InputLabel id="sort-field-label">Sort By</InputLabel>
          <Select
            labelId="sort-field-label"
            value={sortField}
            onChange={handleSortFieldChange}
            label="Sort By"
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            {sortableFields.map((field) => (
              <MenuItem key={field.value} value={field.value}>
                {field.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        {sortField && (
          <IconButton onClick={toggleSortOrder} aria-label="toggle sort order">
            {sortOrder === 'asc' ? <ArrowUpward /> : <ArrowDownward />}
          </IconButton>
        )}
      </div>

      <div className="task-list-content">
        {!sortedTasks?.length ? (
          <Typography variant="body1" sx={{ textAlign: 'center', color: '#cccccc' }}>
            No tasks available.
          </Typography>
        ) : (
          <div className="task-items">
            {sortedTasks.map((task, index) => {
              if (!task || !task.id) {
                console.warn(`Task at index ${index} is missing an 'id' property.`);
                return null;
              }
              return (
                <div 
                  key={task.id}
                  className={`task-item ${task.task_fail_count > 0 ? 'failed' : ''}`}
                  onClick={() => handleTaskClick(task)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      handleTaskClick(task);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <div className="task-header">
                    <span className="task-type">{task.task_type_str}</span>
                    <span className="task-priority">
                      Priority: {formatNumber(task.priority)}
                    </span>
                  </div>
                  <div className="task-details">
                    <div className="task-metric">
                      <span>Duration</span>
                      <span>{formatNumber(task.task_duration)}s</span>
                    </div>
                    {task.storage_size && (
                      <div className="task-metric">
                        <span>Storage Required</span>
                        <span>{formatStorageSize(task.storage_size)} MB</span>
                      </div>
                    )}
                    {task.simultaneous_collects_required > 1 && (
                      <div className="task-metric">
                        <span>Required Satellites</span>
                        <span>{task.simultaneous_collects_required}</span>
                      </div>
                    )}
                    {task.task_fail_count > 0 && (
                      <div className="task-metric fail-count">
                        <span>Failed Attempts</span>
                        <span>{task.task_fail_count}</span>
                      </div>
                    )}
                  </div>
                  <div className="task-location">
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
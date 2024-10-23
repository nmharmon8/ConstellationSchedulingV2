import React from 'react';
import './TaskList.css';
import { useAgent } from '../store/AgentStore';
import { Typography } from '@mui/material';

function TaskList() {
  const { tasks = [], loading, error } = useAgent();

  // Compute task type statistics
  const taskTypeStats = tasks.reduce((acc, task) => {
    if (task?.task_type) {
      acc[task.task_type] = (acc[task.task_type] || 0) + 1;
    }
    return acc;
  }, {});

  if (loading) {
    return <div className="task-list">Loading tasks...</div>;
  }

  if (error) {
    return <div className="task-list">Error: {error}</div>;
  }

  const formatNumber = (num) => {
    return num !== undefined ? Number(num).toFixed(2) : '0.00';
  };

  const formatStorageSize = (size) => {
    return size !== undefined ? (Number(size) / 1e6).toFixed(2) : '0.00';
  };

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
      <div className="task-list-content">
        {!tasks?.length ? (
          <Typography variant="body1" sx={{ textAlign: 'center', color: '#cccccc' }}>
            No tasks available.
          </Typography>
        ) : (
          <div className="task-items">
            {tasks.map((task) => task && (
              <div key={task.id || Math.random()} className="task-item">
                <div className="task-header">
                  <span className="task-type">{task.task_type}</span>
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
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default TaskList;

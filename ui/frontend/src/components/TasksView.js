import React, { useState, useMemo } from 'react';
import { 
  Box, 
  List,
  ListItem,
  ListItemText,
  Typography,
  Chip,
} from '@mui/material';
import { useAgent } from '../store/AgentStore';
import TaskSearchBar from './TaskSearchBar';

const TasksView = () => {
  const { tasks, completedTasks } = useAgent();
  const [searchQuery, setSearchQuery] = useState('');

  // Create search indexes using useMemo
  const taskIndexes = useMemo(() => {
    const buildIndexes = (taskList) => {
      if (!taskList) return {};
      
      const indexes = {
        id: new Map(),
        task_type_str: new Map(),
        priority: new Map(),
        storage_size: new Map(),
        task_duration: new Map(),
        latitude: new Map(),
        longitude: new Map(),
        min_elev: new Map()
      };

      taskList.forEach(task => {
        // Index each property
        Object.entries(indexes).forEach(([prop, map]) => {
          if (task[prop] !== undefined) {
            const value = task[prop].toString().toLowerCase();
            if (!map.has(value)) {
              map.set(value, new Set());
            }
            map.get(value).add(task);
          }
        });
      });

      return indexes;
    };

    return {
      active: buildIndexes(tasks),
      completed: buildIndexes(completedTasks)
    };
  }, [tasks, completedTasks]);

  const filterTasks = (taskList, isCompleted = false) => {
    if (!taskList) return [];
    
    const indexes = isCompleted ? taskIndexes.completed : taskIndexes.active;
    const searchParts = searchQuery.trim().toLowerCase().split(' ');
    
    if (!searchQuery.trim()) return taskList;

    // Get matching tasks for each search part
    const matchingSets = searchParts.map(part => {
      if (part.startsWith('@')) {
        const [propKey, value] = part.slice(1).split(':');
        if (!value || !indexes[propKey]) return new Set(taskList);
        
        // Search in the appropriate index
        const matchingTasks = new Set();
        indexes[propKey].forEach((tasks, indexValue) => {
          if (indexValue.includes(value.toLowerCase())) {
            tasks.forEach(task => matchingTasks.add(task));
          }
        });
        return matchingTasks;
      }
      
      // General search across id and task_type_str
      const matchingTasks = new Set();
      indexes.id.forEach((tasks, indexValue) => {
        if (indexValue.includes(part)) {
          tasks.forEach(task => matchingTasks.add(task));
        }
      });
      indexes.task_type_str.forEach((tasks, indexValue) => {
        if (indexValue.includes(part)) {
          tasks.forEach(task => matchingTasks.add(task));
        }
      });
      return matchingTasks;
    });

    // Intersect all matching sets to get final results
    const intersection = [...matchingSets.reduce((acc, set) => {
      if (!acc) return set;
      return new Set([...acc].filter(x => set.has(x)));
    })];

    return intersection;
  };

  const renderTask = (task, isCompleted = false) => {
    if (!task) return null;

    // Helper function to determine task type badges
    const getTaskTypeBadges = () => {
      const badges = [];
      if (task.is_data_downlink) badges.push('Downlink');
      if (task.is_access_task) badges.push('Access');
      if (task.is_noop) badges.push('No-Op');
      if (task.is_charge) badges.push('Charge');
      if (task.is_collection) badges.push('Collection');
      if (task.is_desat) badges.push('Desat');
      return badges;
    };

    return (
      <ListItem
        key={task.id}
        sx={{
          borderRadius: '8px',
          mb: 1,
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
          }
        }}
      >
        <ListItemText
          primary={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              <Typography color="#00FFD1">{task.id}</Typography>
              <Chip 
                label={task.task_type_str}
                size="small"
                sx={{
                  backgroundColor: 'rgba(0, 255, 209, 0.1)',
                  color: '#00FFD1',
                  borderRadius: '4px'
                }}
              />
              <Chip 
                label={`User: ${task.user_id ?? 'N/A'}`}
                size="small"
                sx={{
                  backgroundColor: 'rgba(0, 255, 209, 0.1)',
                  color: '#00FFD1',
                  borderRadius: '4px'
                }}
              />
              {getTaskTypeBadges().map((badge, index) => (
                <Chip 
                  key={index}
                  label={badge}
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(0, 255, 209, 0.1)',
                    color: '#00FFD1',
                    borderRadius: '4px'
                  }}
                />
              ))}
              {isCompleted && (
                <Chip 
                  label="Completed"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(0, 255, 209, 0.1)',
                    color: '#00FFD1',
                    borderRadius: '4px'
                  }}
                />
              )}
            </Box>
          }
          secondary={
            <Box>
              <Typography color="#8892B0" variant="body2">
                Priority: {task.priority?.toFixed(2) ?? 'N/A'} | Location: {task.latitude?.toFixed(2) ?? 'N/A'}°, {task.longitude?.toFixed(2) ?? 'N/A'}°
              </Typography>
              <Typography color="#8892B0" variant="body2">
                Duration: {task.task_duration?.toFixed(2) ?? 'N/A'}s | Storage: {task.storage_size?.toFixed(2) ?? 'N/A'} | Min Elevation: {(task.min_elev ? (task.min_elev * 180 / Math.PI).toFixed(2) : 'N/A')}°
              </Typography>
              <Typography color="#8892B0" variant="body2">
                Reward: {task.task_reward?.toFixed(2) ?? 'N/A'} | Fail Count: {task.task_fail_count ?? 0} | Simultaneous Collects: {task.simultaneous_collects_required ?? 'N/A'}
              </Typography>
              {task.sats_collecting?.length > 0 && (
                <Typography color="#8892B0" variant="body2">
                  Collecting Sats: {task.sats_collecting.join(', ')}
                </Typography>
              )}
            </Box>
          }
        />
      </ListItem>
    );
  };

  return (
    <Box sx={{ height: '100%', p: 2, backgroundColor: '#0A192F', color: '#CCD6F6', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <TaskSearchBar searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
      
      <Box sx={{ display: 'flex', gap: 2, flex: 1, overflow: 'hidden' }}>
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', borderRight: '1px solid rgba(0, 255, 209, 0.1)' }}>
          <Typography variant="h6" color="#00FFD1" sx={{ mb: 2 }}>
            Active Tasks ({filterTasks(tasks).length})
          </Typography>
          <Box sx={{ overflow: 'auto', flex: 1 }}>
            <List>
              {filterTasks(tasks).map(task => renderTask(task))}
            </List>
          </Box>
        </Box>

        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Typography variant="h6" color="#00FFD1" sx={{ mb: 2 }}>
            Completed Tasks ({filterTasks(completedTasks, true).length})
          </Typography>
          <Box sx={{ overflow: 'auto', flex: 1 }}>
            <List>
              {filterTasks(completedTasks, true).map(task => renderTask(task, true))}
            </List>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default TasksView; 
import React, { useState, useMemo, useCallback } from 'react';
import { 
  Box, 
  List,
  Typography,
  Chip,
} from '@mui/material';
import { useAgent } from '../../../store/AgentStore';
import TaskSearchBar from './TaskSearchBar';
import TaskModal from '../TaskModal';
import TaskCard from './TaskCard';

const TasksView = () => {
  const { tasks, completedTasks } = useAgent();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTask, setSelectedTask] = useState(null);

  // Parse search query into filters
  const parsedFilters = useMemo(() => {
    const filters = {};
    const generalSearch = [];

    const parts = searchQuery.trim().toLowerCase().split(' ');
    parts.forEach(part => {
      if (part.startsWith('@')) {
        const [key, value] = part.slice(1).split(':');
        if (key) {
          // Collect all parts after the colon until the next @ or end
          const valueIndex = parts.indexOf(part) + 1;
          let combinedValue = value || '';
          
          while (valueIndex < parts.length && !parts[valueIndex].startsWith('@')) {
            combinedValue += ' ' + parts[valueIndex];
            parts.splice(valueIndex, 1);
          }
          
          if (combinedValue.trim()) {
            filters[key] = combinedValue.trim();
          }
        }
      } else if (!parts.some(p => p.startsWith('@') && p.includes(':'))) {
        generalSearch.push(part);
      }
    });

    return { filters, generalSearch };
  }, [searchQuery]);

  // Filter tasks based on parsed filters
  const filterTasks = useCallback((taskList, isCompleted = false) => {
    if (!taskList) return [];

    return taskList.filter(task => {
      // Apply property filters
      for (const [key, value] of Object.entries(parsedFilters.filters)) {
        const taskValue = task[key];
        if (taskValue === undefined || !taskValue.toString().toLowerCase().includes(value)) {
          return false;
        }
      }

      // Apply general search
      return parsedFilters.generalSearch.every(part => {
        const id = task.id?.toString().toLowerCase() || '';
        const type = task.task_type_str?.toString().toLowerCase() || '';
        return id.includes(part) || type.includes(part);
      });
    });
  }, [parsedFilters]);

  // Memoize filtered tasks
  const filteredActiveTasks = useMemo(() => filterTasks(tasks, false), [tasks, filterTasks]);
  const filteredCompletedTasks = useMemo(() => filterTasks(completedTasks, true), [completedTasks, filterTasks]);

  // Add handler for task clicks
  const handleTaskClick = (task) => {
    setSelectedTask(task);
  };

  return (
    <div className="relative w-full h-screen bg-space-dark text-white p-4 shadow-lg flex flex-col overflow-hidden font-sans">
      <TaskSearchBar searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
      
      <div className="flex gap-4 flex-1 overflow-hidden mt-4">
        {/* Active Tasks Column */}
        <div className="flex-1 flex flex-col overflow-hidden border-r border-gray-600">
          <div className="flex-none mb-5 pb-3">
            <h2 className="text-space-cyan">
              Active Tasks ({filteredActiveTasks.length})
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto pr-1
            [&::-webkit-scrollbar]:w-2
            [&::-webkit-scrollbar-track]:bg-space-darker
            [&::-webkit-scrollbar-thumb]:bg-gray-600
            [&::-webkit-scrollbar-thumb]:rounded-full
            [&::-webkit-scrollbar-thumb]:hover:bg-gray-500">
            {filteredActiveTasks.map(task => (
              <TaskCard 
                key={task.id}
                task={task}
                isCompleted={false}
                onClick={handleTaskClick}
              />
            ))}
          </div>
        </div>

        {/* Completed Tasks Column */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-none mb-5 pb-3">
            <h2 className="text-space-cyan">
              Completed Tasks ({filteredCompletedTasks.length})
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto pr-1
            [&::-webkit-scrollbar]:w-2
            [&::-webkit-scrollbar-track]:bg-space-darker
            [&::-webkit-scrollbar-thumb]:bg-gray-600
            [&::-webkit-scrollbar-thumb]:rounded-full
            [&::-webkit-scrollbar-thumb]:hover:bg-gray-500">
            {filteredCompletedTasks.map(task => (
              <TaskCard
                key={task.id}
                task={task}
                isCompleted={true}
                onClick={handleTaskClick}
              />
            ))}
          </div>
        </div>
      </div>

      <TaskModal
        task={selectedTask}
        open={selectedTask !== null}
        onClose={() => setSelectedTask(null)}
      />
    </div>
  );
};

export default TasksView; 
import React, { useEffect, useState } from 'react';
import { useAgent } from '../../store/AgentStore';
import './ObservationState.css';
import TaskModal from '../tasks/TaskModal';

const ObservationState = () => {
  const { 
    currentActionsAndObs, 
    isAutoRunning, 
    pauseAutoStep, 
    startAutoStep 
  } = useAgent();
  const [selectedTask, setSelectedTask] = useState(null);

  const handleTaskClick = (task) => {
    console.log('Task clicked:', task);
    setSelectedTask(task);
  };

  const handleCloseModal = () => {
    console.log('Modal closed');
    setSelectedTask(null);
  };

  const getDisplayTaskType = (taskType) => {
    switch (taskType) {
      case 'IMAGING':
        return 'EO';
      case 'DATA_DOWNLINK':
        return 'Downlink';
      default:
        return taskType;
    }
  };

  if (!currentActionsAndObs) {
    return null;
  }

  const satIds = Object.keys(currentActionsAndObs.sat_to_tasks || {});
  const maxTasks = Math.max(...satIds.map(satId => 
    currentActionsAndObs.sat_to_tasks[satId]?.length || 0
  ));

  return (
    <div className="observation-table-container flex-1 overflow-auto pr-1 w-full
      [&::-webkit-scrollbar]:w-2
      [&::-webkit-scrollbar-track]:bg-space-darker
      [&::-webkit-scrollbar-thumb]:bg-gray-600
      [&::-webkit-scrollbar-thumb]:rounded-full
      [&::-webkit-scrollbar-thumb]:hover:bg-gray-500
      [&::-webkit-scrollbar:horizontal]:h-2"
    >
      <table className="observation-table">
        <thead>
          <tr>
            <th>Sat ID</th>
            {[...Array(maxTasks)].map((_, index) => (
              <th key={index}>Act {index}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {satIds.map(satId => {
            const tasks = currentActionsAndObs.sat_to_tasks[satId] || [];
            const activeTaskIndex = currentActionsAndObs.sat_to_act[satId];

            // Annotate each task with its original index
            const annotatedTasks = tasks.map((task, index) => ({
              ...task,
              originalIndex: index
            }));
            
            // Separate COLLECTION tasks
            const collectionTasks = annotatedTasks.filter(task => task.is_collection);
            const otherTasks = annotatedTasks.filter(task => !task.is_collection);

            // Reorder tasks: COLLECTION first, then others
            const reorderedTasks = [...collectionTasks, ...otherTasks];

            return (
              <tr key={satId}>
                <td className="sat-id">
                  {satId.split('_')[0]}
                </td>
                {[...Array(maxTasks)].map((_, index) => {
                  const task = reorderedTasks[index];
                  if (!task) {
                    return (
                      <td
                        key={index}
                        className="task-cell empty"
                        style={{ cursor: 'default' }}
                      >
                        -
                      </td>
                    );
                  }

                  const isActive = task.originalIndex === activeTaskIndex;

                  return (
                    <td
                      key={index}
                      className={`task-cell ${isActive ? 'active' : ''}`}
                      onClick={() => task && handleTaskClick(task)}
                      style={{ cursor: task ? 'pointer' : 'default' }}
                    >
                      {task ? `${getDisplayTaskType(task.task_type_str)}(${task.window_index_offset})` : '-'} 
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      {selectedTask && (
        <TaskModal 
          task={selectedTask}
          open={!!selectedTask}
          onClose={handleCloseModal}
        />
      )}
    </div>
  );
};

export default ObservationState;

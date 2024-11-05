import React, { useEffect, useState } from 'react';
import { useAgent } from '../store/AgentStore';
import '../styles/ObservationState.css';
import TaskModal from './TaskModal';

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

  const handleWheel = (event) => {
    if (event.deltaY !== 0) {
      event.preventDefault();
      const container = event.currentTarget;
      container.scrollLeft += event.deltaY;
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
    <div 
      className="observation-table-container"
      onWheel={handleWheel}
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
          {
          
          
          satIds.map(satId => {

            const tasks = currentActionsAndObs.sat_to_tasks[satId] || [];
            const activeTaskIndex = currentActionsAndObs.sat_to_act[satId];
            return (
              <tr key={satId}>
                <td className="sat-id">
                  {satId.slice(0, 5)}...
                </td>
                {[...Array(maxTasks)].map((_, index) => {
                  const task = tasks[index];
                  const isActive = index === activeTaskIndex;

                  return (
                    <td
                      key={index}
                      className={`task-cell ${isActive ? 'active' : ''} ${!task ? 'empty' : ''}`}
                      onClick={() => task && handleTaskClick(task)}
                      style={{ cursor: task ? 'pointer' : 'default' }}
                    >
                      {task ? `${task.task_type_str}(${task.window_index_offset})` : '-'} 
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

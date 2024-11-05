import React, { createContext, useContext, useState, useEffect, useRef } from 'react';

import io from 'socket.io-client';

const AgentContext = createContext(null);

export const AgentProvider = ({ children }) => {

  // Has the current state of the satellites this is the state that resulted from the last step
  const [currentSatState, setCurrentSatState] = useState(null);

  const [isRunning, setIsRunning] = useState(false);
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [error, setError] = useState(null);
  const [observationInspector, setObservationInspector] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentTasksBeingExecuted, setCurrentTasksBeingExecuted] = useState({});
  // Interpolated positions is a dictionary with satellite ids as keys and list of lat/lon tuples
  const [interpolatedPositions, setInterpolatedPositions] = useState({});
  const [currentActionsAndObs, setCurrentActionsAndObs] = useState(null);

  const autoStepTimer = useRef(null);

  // Initialize Socket.IO
  const socketRef = useRef(null);

  useEffect(() => {
    // Connect to Socket.IO server
    socketRef.current = io('http://localhost:5000');

    // Listen for connection
    socketRef.current.on('connect', () => {
      console.log('Connected to Socket.IO server');
    });

    // Listen for step updates
    socketRef.current.on('step_update', (data) => {
      setInterpolatedPositions(data.interpolated_sat_positions);
      setCurrentSatState(data.current_sat_state);
    });

    socketRef.current.on('tasks', (data) => {
      setTasks(data);
    });

    // Listen for agent reset
    socketRef.current.on('agent_reset', (data) => {
      setCurrentSatState(null);
      setTasks([]);
      setInterpolatedPositions({});
      setError(null);
      setLoading(true);
    });

    // Add new listener for current actions and observations
    socketRef.current.on('current_acts_obs', (data) => {
      setCurrentActionsAndObs(data);
      // for each satellite id data.sat_to_act[sat_id] is the index of the task that the satellite is currently executing
      // data.sat_to_tasks[sat_id] is the list of tasks that the satellite can execute
      // we can use this to update the current tasks being executed
      const newCurrentTasksBeingExecuted = {};
      Object.entries(data.sat_to_act).forEach(([satId, taskIndex]) => {
        newCurrentTasksBeingExecuted[satId] = data.sat_to_tasks[satId][taskIndex];
      });
      setCurrentTasksBeingExecuted(newCurrentTasksBeingExecuted);
    });

    fetchTasks();

    // Cleanup on unmount
    return () => {
      socketRef.current.disconnect();
    };
  }, []);

  const takeStep = async () => {
    try {
      setIsRunning(true);
      const response = await fetch('/api/step', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        setError(`Failed to take step: ${response.statusText}`);
        console.log(`Failed to take step: ${response.statusText}`);
        setIsRunning(false);
        return;
      }
    } catch (err) {
      setError(`Failed to take step: ${err.message}`);
      console.log(`Failed to take step: ${err.message}`);
      setIsRunning(false);
    }
  };

  const fetchTasks = async () => {
    try {
      const response = await fetch('/api/tasks');
      if (!response.ok) {
        setError(`Failed to fetch tasks: ${response.statusText}`);
        console.log(`Failed to fetch tasks: ${response.statusText}`);
        return;
      }
      const data = await response.json();
      setTasks(data);
    } catch (err) {
      setError(`Failed to fetch tasks: ${err.message}`);
      console.log(`Failed to fetch tasks: ${err.message}`);
    }
  };

  const fetchObservationInspector = async () => {
    try {
      const response = await fetch('/api/observation/inspector');
      if (!response.ok) {
        setError(`Failed to fetch observation inspector: ${response.statusText}`);
        console.log(`Failed to fetch observation inspector: ${response.statusText}`);
        return;
      }
      const data = await response.json();
      console.log(`Fetched observation inspector: ${data}`);
      console.log(data);
      setObservationInspector(data);
    } catch (err) {
      setError(`Failed to fetch observation inspector: ${err.message}`);
      console.log(`Failed to fetch observation inspector: ${err.message}`);
    }
  }



  // Start Automatic Stepping
  const startAutoStep = () => {
    if (isAutoRunning) return;
    setIsAutoRunning(true);
    takeStep();
    autoStepTimer.current = setInterval(() => {
      takeStep();
    }, 3000);
  };

  // Pause Automatic Stepping
  const pauseAutoStep = () => {
    if (autoStepTimer.current) {
      clearInterval(autoStepTimer.current);
      autoStepTimer.current = null;
    }
    setIsAutoRunning(false);
  };

  // Updated Reset Agent Function
  const resetAgent = async () => {
    try {
      setIsResetting(true);
      setIsRunning(false);
      setIsAutoRunning(false);
      setCurrentSatState(null);
      setTasks([]);
      setInterpolatedPositions({});
      setError(null);
      setLoading(true);
      setObservationInspector(null);
      setCurrentTasksBeingExecuted({});
      setCurrentActionsAndObs(null);

      if (isAutoRunning) {
        pauseAutoStep();
      }

      const response = await fetch('/api/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        setError(`Failed to reset agent: ${errorData.error || response.statusText}`);
        console.log(`Failed to reset agent: ${errorData.error || response.statusText}`);
        return;
      }

      const data = await response.json();
      console.log(data.message);

      await fetchTasks();
      setIsResetting(false);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setIsResetting(false);
      setLoading(false);
      setError(`Failed to reset agent: ${err.message}`);
      console.log(`Failed to reset agent: ${err.message}`);
    }
  };

  const clearError = () => {
    setError(null);
  };

  const value = {
    isRunning,
    isAutoRunning,
    isResetting,
    error,
    clearError,
    tasks,
    loading,
    interpolatedPositions,
    currentActionsAndObs,
    takeStep,
    resetAgent,
    startAutoStep,
    pauseAutoStep,
    fetchTasks,
    setIsRunning,
    currentSatState,
    fetchObservationInspector,
    observationInspector,
    currentTasksBeingExecuted,
  };

  return (
    <AgentContext.Provider value={value}>
      {children}
    </AgentContext.Provider>
  );
};

// Custom hook to use the agent store
export const useAgent = () => {
  const context = useContext(AgentContext);
  if (context === null) {
    console.error('useAgent must be used within an AgentProvider');
    throw new Error('useAgent must be used within an AgentProvider');
  }
  return context;
};

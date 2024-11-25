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
  const [cumulativeReward, setCumulativeReward] = useState(0);
  const [stepReward, setStepReward] = useState(0);
  const [completedTasks, setCompletedTasks] = useState([]);
  const [taskGPTMessages, setTaskGPTMessages] = useState([]);
  const [isTaskGPTModalOpen, setIsTaskGPTModalOpen] = useState(false);
  const [isTaskGPTProcessing, setIsTaskGPTProcessing] = useState(false);

  // Initialize Socket.IO
  const socketRef = useRef(null);

  useEffect(() => {
    // Configure Socket.IO with reconnection options
    socketRef.current = io('http://localhost:4000', {
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });

    // Listen for connection
    socketRef.current.on('connect', () => {
      console.log('Connected to Socket.IO server');
      // Call get initial state
      fetch('/api/initial_state', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }).then(response => {
        if (response.ok) {
          console.log('Initial state fetched');
        }
      });
    });

    // Add connection error handling
    socketRef.current.on('connect_error', (error) => {
      console.log('Connection error:', error);
      setError('Socket connection error. Attempting to reconnect...');
    });

    // Add reconnect listeners
    socketRef.current.on('reconnect', (attemptNumber) => {
      console.log('Reconnected on attempt:', attemptNumber);
      setError(null);
    });

    socketRef.current.on('reconnect_attempt', (attemptNumber) => {
      console.log('Attempting to reconnect:', attemptNumber);
    });

    socketRef.current.on('disconnect', (reason) => {
      console.log('Disconnected:', reason);
      if (reason === 'io server disconnect') {
        // the disconnection was initiated by the server, reconnect manually
        socketRef.current.connect();
      }
      // else the socket will automatically try to reconnect
    });

    // Listen for step updates
    socketRef.current.on('step_update', (data) => {
      if (isResetting) return;

      // Clear previous state first
      setCurrentActionsAndObs(null);  // This will trigger planning lines cleanup
      
      // Then set the new state
      setInterpolatedPositions(data.interpolated_sat_positions);
      setCurrentSatState(data.current_sat_state);
      setTasks(data.tasks);
      setCompletedTasks(data.completed_tasks);
      setCurrentActionsAndObs(data.current_acts_obs);
      setStepReward(data.reward);
      setCumulativeReward(prev => prev + data.reward);
      const newCurrentTasksBeingExecuted = {};
      Object.entries(data.current_acts_obs.sat_to_act).forEach(([satId, taskIndex]) => {
        newCurrentTasksBeingExecuted[satId] = data.current_acts_obs.sat_to_tasks[satId][taskIndex];
      });
      setCurrentTasksBeingExecuted(newCurrentTasksBeingExecuted);
    });

    // Listen for agent reset
    socketRef.current.on('agent_reset', (data) => {
      setIsResetting(false);
      setIsRunning(false);
      setIsAutoRunning(false);
      setCurrentSatState(null);
      setTasks(data.tasks);
      setInterpolatedPositions({});
      setError(null);
      setLoading(false);
      setObservationInspector(null);
      setCurrentTasksBeingExecuted({});
      setCurrentActionsAndObs(null);
      setCumulativeReward(0)
    });

    // Add new tasks_updated listener
    socketRef.current.on('tasks_updated', (updatedTasks) => {
      setTasks(updatedTasks);
    });

    // Add TaskGPT message listener
    socketRef.current.on('task_gpt_info', (data) => {
      setTaskGPTMessages(prev => [...prev, data]);
      if (data.done) {
        setIsTaskGPTProcessing(false);
      }
    });

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
      setObservationInspector(data);
    } catch (err) {
      setError(`Failed to fetch observation inspector: ${err.message}`);
      console.log(`Failed to fetch observation inspector: ${err.message}`);
    }
  }



  // Start Automatic Stepping
  const startAutoStep = async () => {
    if (isAutoRunning) return;
    try {
      const response = await fetch('/api/run/5000', { // 3000ms interval
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        setError(`Failed to start auto-stepping: ${response.statusText}`);
        return;
      }
      
      setIsAutoRunning(true);
    } catch (err) {
      setError(`Failed to start auto-stepping: ${err.message}`);
      console.log(`Failed to start auto-stepping: ${err.message}`);
    }
  };

  // Pause Automatic Stepping
  const pauseAutoStep = async () => {
    try {
      const response = await fetch('/api/pause', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        setError(`Failed to pause auto-stepping: ${response.statusText}`);
        return;
      }
      
      setIsAutoRunning(false);
    } catch (err) {
      setError(`Failed to pause auto-stepping: ${err.message}`);
      console.log(`Failed to pause auto-stepping: ${err.message}`);
    }
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
      setCumulativeReward(0);
      
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

  // Add new createTask function
  const createTask = async (taskData) => {
    try {
      const response = await fetch('/api/create_task', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: taskData.name,
          lat: taskData.lat,
          lon: taskData.lon,
          priority: taskData.priority,
          task_type: taskData.taskType,
          min_elev: taskData.minElev,
          duration: taskData.duration
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        setError(`Failed to create task: ${errorData.error || response.statusText}`);
        return null;
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(`Failed to create task: ${err.message}`);
      console.error(`Failed to create task: ${err.message}`);
      return null;
    }
  };

  const generateTask = async (prompt) => {
    try {
      setTaskGPTMessages([]); // Clear previous messages
      setIsTaskGPTModalOpen(true);
      setIsTaskGPTProcessing(true);

      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        setError(`Failed to generate task: ${response.statusText}`);
        setIsTaskGPTProcessing(false);
        return null;
      }
    } catch (err) {
      setError(`Failed to generate task: ${err.message}`);
      setIsTaskGPTProcessing(false);
      return null;
    }
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
    cumulativeReward,
    setCumulativeReward,
    createTask,
    generateTask,
    taskGPTMessages,
    isTaskGPTModalOpen,
    setIsTaskGPTModalOpen,
    isTaskGPTProcessing,
    completedTasks,
    stepReward,
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

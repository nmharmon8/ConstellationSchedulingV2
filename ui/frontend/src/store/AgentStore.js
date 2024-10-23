import React, { createContext, useContext, useState, useEffect } from 'react';

const AgentContext = createContext(null);

export const AgentProvider = ({ children }) => {
  const [stepData, setStepData] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Satellite state tracking
  const [satellites, setSatellites] = useState({
    ids: [],
    currentPositions: {},
    lastPositions: {},
    observations: {}
  });

  const fetchSatInfo = async () => {
    try {
      const response = await fetch('/api/get_sat_info');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      // Transform the data into a more usable structure
      const transformedData = data.reduce((acc, sat, index) => {
        const id = `SAT-${index}`;
        acc[id] = {
          position: {
            lat: sat.lat,
            lon: sat.lon,
            alt: sat.alt,
            r_BP_P: sat.r_BP_P
          },
          ...sat.observation
        };
        return acc;
      }, {});

      // Update satellite tracking data
      setSatellites(prev => ({
        ids: Object.keys(transformedData),
        currentPositions: transformedData,
        lastPositions: prev.currentPositions,
        observations: transformedData
      }));
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const takeStep = async () => {
    try {
      // Store current positions as last positions before taking new step
      setSatellites(prev => ({
        ...prev,
        lastPositions: { ...prev.currentPositions }
      }));
      
      const response = await fetch('/api/step');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setStepData(data);
      
      // Update current positions with new sat info
      await fetchSatInfo();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const fetchTasks = async () => {
    try {
      const response = await fetch('/api/tasks');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setTasks(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
      throw err;
    }
  };

  // Initialize data on mount
  useEffect(() => {
    fetchSatInfo();
    fetchTasks();
  }, []);

  const value = {
    stepData,
    isRunning,
    error,
    satellites,
    tasks,
    loading,
    takeStep,
    fetchSatInfo,
    fetchTasks,
    setIsRunning,
    setStepData
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
    throw new Error('useAgent must be used within an AgentProvider');
  }
  return context;
};

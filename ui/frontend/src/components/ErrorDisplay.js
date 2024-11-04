import React from 'react';
import { useAgent } from '../store/AgentStore';

const ErrorDisplay = () => {
  const { error, clearError } = useAgent();

  if (!error) return null;

  return (
    <div className="error-display">
      <div className="error-content">
        <h3>Error</h3>
        <p>{error}</p>
        <button onClick={clearError}>Dismiss</button>
      </div>
    </div>
  );
};

export default ErrorDisplay; 
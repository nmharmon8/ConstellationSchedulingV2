import React from 'react';
import { useAgent } from '../store/AgentStore';
import './AgentControlBar.css'; // Import the updated CSS file for styling

const AgentControlBar = () => {
  const {
    resetAgent,
    isRunning,
    isAutoRunning,
    startAutoStep,
    pauseAutoStep,
  } = useAgent();

  const handleStartPause = () => {
    if (isAutoRunning) {
      pauseAutoStep();
    } else {
      startAutoStep();
    }
  };

  const handleReset = async () => {
    try {
      await resetAgent();
      // Optionally, add any additional logic after reset
    } catch (err) {
      console.error('Reset failed:', err);
    }
  };

  return (
    <div className="agent-control-bar">
      <div className="button-group">
        <button
          className="start-pause-button small-button"
          onClick={handleStartPause}
          // Removed disabled={isRunning} to keep the button always enabled
        >
          {isAutoRunning ? 'Pause' : 'Start'}
        </button>
        <button
          className="reset-button small-button"
          onClick={handleReset}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default AgentControlBar;

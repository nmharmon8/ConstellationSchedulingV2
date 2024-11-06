import React from 'react';
import { useAgent } from '../store/AgentStore';
import './AgentControlBar.css'; // Import the updated CSS file for styling
import { Box, Button } from '@mui/material';

const AgentControlBar = () => {
  const { isRunning, isAutoRunning, isResetting, takeStep, resetAgent, startAutoStep, pauseAutoStep } = useAgent();

  return (
    <Box sx={{ display: 'flex', gap: '8px' }}>
      {/* Step Button */}
      <Button
        onClick={takeStep}
        disabled={isAutoRunning || isResetting}
        sx={{
          color: '#64FFDA',
          borderColor: '#64FFDA',
          '&:hover': {
            backgroundColor: 'rgba(100, 255, 218, 0.1)',
            borderColor: '#64FFDA',
          },
          '&.Mui-disabled': {
            borderColor: 'rgba(100, 255, 218, 0.3)',
            color: 'rgba(100, 255, 218, 0.3)',
          },
        }}
        variant="outlined"
        size="small"
      >
        Step
      </Button>

      {/* Run/Pause Button */}
      <Button
        onClick={isAutoRunning ? pauseAutoStep : startAutoStep}
        disabled={isResetting}
        sx={{
          color: '#64FFDA',
          borderColor: '#64FFDA',
          '&:hover': {
            backgroundColor: 'rgba(100, 255, 218, 0.1)',
            borderColor: '#64FFDA',
          },
          '&.Mui-disabled': {
            borderColor: 'rgba(100, 255, 218, 0.3)',
            color: 'rgba(100, 255, 218, 0.3)',
          },
        }}
        variant="outlined"
        size="small"
      >
        {isAutoRunning ? 'Pause' : 'Run'}
      </Button>

      {/* Reset Button */}
      <Button
        onClick={resetAgent}
        disabled={isResetting}
        sx={{
          color: '#64FFDA',
          borderColor: '#64FFDA',
          '&:hover': {
            backgroundColor: 'rgba(100, 255, 218, 0.1)',
            borderColor: '#64FFDA',
          },
          '&.Mui-disabled': {
            borderColor: 'rgba(100, 255, 218, 0.3)',
            color: 'rgba(100, 255, 218, 0.3)',
          },
        }}
        variant="outlined"
        size="small"
      >
        Reset
      </Button>
    </Box>
  );
};

export default AgentControlBar;

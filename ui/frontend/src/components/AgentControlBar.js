import React, { useEffect } from 'react';
import { Box, Button, Typography } from '@mui/material';
import { useAgent } from '../store/AgentStore';

const AgentControlBar = () => {
  const { isRunning, takeStep, startAgent, stopAgent, error } = useAgent();

  useEffect(() => {
    let intervalId;
    if (isRunning) {
      intervalId = setInterval(async () => {
        try {
          const stepResult = await takeStep();
          if (stepResult.current_step.done || stepResult.current_step.truncated) {
            stopAgent();
          }
        } catch (error) {
          console.error('Error during step:', error);
          stopAgent();
        }
      }, 1000); // Take a step every second
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isRunning, takeStep, stopAgent]);

  return (
    <Box
      sx={{
        padding: 2,
        backgroundColor: '#1A2F4C',
        borderTop: '1px solid #64FFDA',
        display: 'flex',
        justifyContent: 'center',
        gap: 2,
        alignItems: 'center'
      }}
    >
      <Button
        variant="contained"
        onClick={takeStep}
        disabled={isRunning}
        sx={{
          backgroundColor: '#64FFDA',
          color: '#0A192F',
          '&:hover': {
            backgroundColor: '#4DB5BD'
          }
        }}
      >
        Take Step
      </Button>
      <Button
        variant="contained"
        onClick={isRunning ? stopAgent : startAgent}
        sx={{
          backgroundColor: isRunning ? '#FF5555' : '#64FFDA',
          color: '#0A192F',
          '&:hover': {
            backgroundColor: isRunning ? '#FF3333' : '#4DB5BD'
          }
        }}
      >
        {isRunning ? 'Stop' : 'Start'}
      </Button>
      {error && (
        <Typography color="error" sx={{ ml: 2 }}>
          Error: {error}
        </Typography>
      )}
    </Box>
  );
};

export default AgentControlBar;

import React, { useState, useEffect } from 'react';
import { useAgent } from '../store/AgentStore';
import { Box, Typography } from '@mui/material';

const GlobalStats = () => {
  const { currentActionsAndObs, cumulativeReward, setCumulativeReward } = useAgent();
  const [currentReward, setCurrentReward] = useState(0);

  useEffect(() => {
    if (!currentActionsAndObs?.sat_to_tasks) return;

    // Calculate current reward from all active tasks
    const newCurrentReward = Object.entries(currentActionsAndObs.sat_to_tasks).reduce((total, [satId, tasks]) => {
      const activeTaskIndex = currentActionsAndObs.sat_to_act[satId];
      if (activeTaskIndex !== undefined && tasks[activeTaskIndex]) {
        return total + (tasks[activeTaskIndex].reward || 0);
      }
      return total;
    }, 0);

    setCurrentReward(newCurrentReward);
    setCumulativeReward(prev => prev + newCurrentReward);
  }, [currentActionsAndObs, setCumulativeReward]);

  return (
    <Box
      sx={{
        width: '100%',
        bgcolor: 'rgba(10, 25, 47, 0.8)',
        padding: '8px 16px',
        display: 'flex',
        gap: '2rem',
        borderBottom: '1px solid rgba(100, 255, 218, 0.1)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Typography sx={{ color: '#8892B0', fontWeight: 600 }}>
          Current Reward:
        </Typography>
        <Typography sx={{ color: '#00FFD1', fontFamily: 'monospace', fontSize: '1.1em' }}>
          {currentReward.toFixed(3)}
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Typography sx={{ color: '#8892B0', fontWeight: 600 }}>
          Cumulative Reward:
        </Typography>
        <Typography sx={{ color: '#00FFD1', fontFamily: 'monospace', fontSize: '1.1em' }}>
          {cumulativeReward.toFixed(3)}
        </Typography>
      </Box>
    </Box>
  );
};

export default GlobalStats; 
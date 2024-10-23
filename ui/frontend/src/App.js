import React from 'react';
import TaskList from './components/TaskList';
import SatList from './components/SatList';
import MapView from './components/MapView';
import { Box } from '@mui/material';
import { AgentProvider } from './store/AgentStore';

function App() {
  return (
    <AgentProvider>
      <Box sx={{ display: 'flex', height: '100vh', width: '100vw' }}>
        {/* Left Side: SatList */}
        <Box sx={{ 
          width: '300px', 
          minWidth: '300px',  /* Added to prevent shrinking */
          overflow: 'hidden' /* Added to contain the shadow */
        }}>
          <SatList />
        </Box>

        {/* Middle: MapView */}
        {/* Setting width to 3px make this work not sure why it displays at full size, but if we don't have 3px it is way to big */}
        <Box sx={{ flex: 1, position: 'relative', width: '3px', height: '100%' }}> 
          <MapView />
        </Box>

        {/* Right Side: TaskList */}
        <Box sx={{ width: '300px' }}>
          <TaskList />
        </Box>
      </Box>
    </AgentProvider>
  );
}

export default App;

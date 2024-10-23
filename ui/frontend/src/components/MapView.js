import React, { useState } from 'react';
import GlobeComponent from './GlobeComponent';
import MapChart from './MapChart';
import { Box, Tabs, Tab } from '@mui/material';

const MapView = () => {
  const [currentTab, setCurrentTab] = useState(0);

  const handleChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%', // Change from '100vh' to '100%' to fill parent
        display: 'flex',
        flexDirection: 'column',
        bgcolor: '#0A192F',
      }}
    >
      <Tabs
        value={currentTab}
        onChange={handleChange}
        centered
        sx={{
          '& .MuiTabs-indicator': {
            backgroundColor: '#64FFDA',
          },
          '& .MuiTab-root': {
            color: '#8892B0',
            '&.Mui-selected': {
              color: '#64FFDA',
            },
          },
          borderBottom: '1px solid rgba(100, 255, 218, 0.1)',
          bgcolor: '#0A192F',
        }}
      >
        <Tab label="Map" />
        <Tab label="Globe" />
      </Tabs>
      <Box
        sx={{
          flexGrow: 1,
          display: 'flex',
        }}
      >
        {currentTab === 0 && <MapChart />}
        {currentTab === 1 && <GlobeComponent />}
      </Box>
    </Box>
  );
};

export default MapView;

import React, { useState, useRef } from 'react';
import GlobeComponent from './GlobeComponent';
import MapChart from './MapChart';
import InspectorView from './InspectorView';
import { Box, Tabs, Tab, IconButton, TextField, LinearProgress } from '@mui/material';
import AgentControlBar from './AgentControlBar';
import GlobalStats from './GlobalStats';
import { useAgent } from '../store/AgentStore';
import TaskGPTModal from './TaskGPTModal';

const MainPanel = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [taskInput, setTaskInput] = useState('');
  const { generateTask, isTaskGPTProcessing } = useAgent();
  const containerRef = useRef(null);

  const handleChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const handleTaskSubmit = async (event) => {
    if (event.key === 'Enter' && taskInput.trim()) {
      await generateTask(taskInput.trim());
      setTaskInput('');
    }
  };

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: '#0A192F',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px',
          borderBottom: '1px solid rgba(100, 255, 218, 0.1)',
          bgcolor: '#0A192F',
          flexWrap: 'wrap',
          gap: '8px',
          position: 'sticky',
          top: 0,
          zIndex: 1,
        }}
      >
        <Box
          sx={{
            display: 'flex',
            minWidth: 'fit-content',
            order: 1,
          }}
        >
          <AgentControlBar />
        </Box>

        {/* TaskGPT Input/Loading Bar */}
        {isTaskGPTProcessing ? (
          <Box
            sx={{
              order: 2,
              flex: '1 1 auto',
              maxWidth: '300px',
            }}
          >
            <LinearProgress 
              sx={{
                backgroundColor: 'rgba(0, 255, 209, 0.2)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: '#00FFD1',
                },
                height: 40,
                borderRadius: '8px',
                boxShadow: '0 0 8px #00FFD1',
              }}
            />
          </Box>
        ) : (
          <TextField
            value={taskInput}
            onChange={(e) => setTaskInput(e.target.value)}
            onKeyPress={handleTaskSubmit}
            placeholder="TaskGPT"
            variant="outlined"
            size="small"
            autoComplete="off"
            inputProps={{
              'aria-label': 'task input',
            }}
            sx={{
              order: 2,
              flex: '1 1 auto',
              maxWidth: '300px',
              background: 'rgba(10, 25, 47, 0.8)',
              borderRadius: '8px',
              position: 'relative',
              zIndex: 1001,
              pointerEvents: 'auto',
              '& .MuiOutlinedInput-root': {
                color: '#00FFD1',
                background: 'rgba(255, 255, 255, 0.05)',
                '& fieldset': {
                  borderColor: '#00FFD1',
                },
                '&:hover fieldset': {
                  borderColor: '#00E5C4',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#00FFD1',
                  boxShadow: '0 0 10px #00FFD1',
                },
                // Make sure the input is clickable
                '& input': {
                  cursor: 'text',
                  zIndex: 1,
                },
              },
              '& .MuiInputBase-input::placeholder': {
                color: '#00FFD1',
                opacity: 0.7,
                fontStyle: 'italic',
              },
              boxShadow: '0 0 8px #00FFD1',
              transition: 'box-shadow 0.3s ease-in-out',
              '&:hover': {
                boxShadow: '0 0 12px #00FFD1',
              },
            }}
          />
        )}

        <Tabs
          value={currentTab}
          onChange={handleChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTabs-indicator': {
              backgroundColor: '#00FFD1',
            },
            '& .MuiTab-root': {
              color: '#8892B0',
              minWidth: '80px',
              padding: '6px 12px',
              '&.Mui-selected': {
                color: '#00FFD1',
              },
            },
            order: 3,
            flex: '0 1 auto',
          }}
        >
          <Tab label="Map" />
          <Tab label="Globe" />
          <Tab label="Inspect" />
        </Tabs>
      </Box>

      {/* Add GlobalStats here */}
      <GlobalStats />

      {/* Content Area */}
      <Box
        ref={containerRef}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          flexGrow: 1,
          overflow: 'hidden',
          position: 'relative',
          minHeight: 0,
        }}
      >
        {currentTab === 0 ? (
          <Box
            sx={{
              width: '100%',
              height: '100%',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <MapChart />
          </Box>
        ) : (
          <Box
            sx={{
              width: '100%',
              height: '100%',
            }}
          >
            {currentTab === 1 && <GlobeComponent />}
            {currentTab === 2 && <InspectorView />}
          </Box>
        )}
      </Box>

      {/* Add TaskGPT Modal */}
      <TaskGPTModal />
    </Box>
  );
};

export default MainPanel;

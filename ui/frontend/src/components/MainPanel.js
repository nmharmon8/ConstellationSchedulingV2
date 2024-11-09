import React, { useState, useRef, useEffect } from 'react';
import GlobeComponent from './GlobeComponent';
import MapChart from './MapChart';
import InspectorView from './InspectorView';
import { Box, Tabs, Tab, IconButton, TextField } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import AgentControlBar from './AgentControlBar';
import ObservationState from './ObservationState';

const MainPanel = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [zoom, setZoom] = useState(1);
  const containerRef = useRef(null);

  // Panning State
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const [taskInput, setTaskInput] = useState('');

  const handleChange = (event, newValue) => {
    setCurrentTab(newValue);
    // Reset offset and zoom when switching tabs
    if (newValue !== 0) {
      setOffset({ x: 0, y: 0 });
    }
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.2, 3)); // Max zoom 3x
    // Reset panning on zoom
    setOffset({ x: 0, y: 0 });
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.2, 0.5)); // Min zoom 0.5x
    // Reset panning on zoom
    setOffset({ x: 0, y: 0 });
  };

  // Define base dimensions for the map
  const baseWidth = 800; // in pixels
  const baseHeight = 600; // in pixels

  // Calculate scaled dimensions based on zoom
  const scaledWidth = baseWidth * zoom;
  const scaledHeight = baseHeight * zoom;

  // Calculate initial zoom to fit the parent container's width
  const calculateInitialZoom = () => {
    if (!containerRef.current) return 1;
    const containerWidth = containerRef.current.clientWidth * 0.9 - 32; // subtract padding (16px * 2)
    return containerWidth / baseWidth;
  };

  // Set initial zoom based on container size
  useEffect(() => {
    const initialZoom = calculateInitialZoom();
    setZoom(initialZoom);
  }, []);

  // Update zoom when window resizes (only for Map tab)
  useEffect(() => {
    const handleResize = () => {
      if (currentTab !== 0) return; // Only adjust zoom for Map tab
      const newZoom = calculateInitialZoom();
      setZoom(newZoom);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [currentTab]);

  // Panning Handlers
  const handleMouseDown = (e) => {
    if (currentTab !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    if (currentTab !== 0) return;
    setIsDragging(false);
  };

  const handleMouseLeave = () => {
    if (currentTab !== 0) return;
    setIsDragging(false);
  };

  const handleTaskInputChange = (event) => {
    setTaskInput(event.target.value);
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
          zIndex: 10,
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

        <Box
          sx={{
            order: 2,
            flex: '1 1 auto',
            maxWidth: '300px',
            position: 'relative',
            zIndex: 11,
          }}
        >
          <TextField
            value={taskInput}
            onChange={handleTaskInputChange}
            placeholder="TaskGPT"
            variant="outlined"
            size="small"
            autoComplete="off"
            inputProps={{
              'aria-label': 'task input',
              style: {
                cursor: 'text',
                caretColor: '#00FFD1',
              }
            }}
            sx={{
              width: '100%',
              background: 'rgba(10, 25, 47, 0.8)',
              borderRadius: '8px',
              pointerEvents: 'auto',
              '& .MuiOutlinedInput-root': {
                color: '#00FFD1',
                background: 'rgba(255, 255, 255, 0.05)',
                '& fieldset': {
                  borderColor: '#00FFD1',
                  zIndex: 0,
                },
                '&:hover fieldset': {
                  borderColor: '#00E5C4',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#00FFD1',
                  boxShadow: '0 0 10px #00FFD1',
                },
                '& input': {
                  position: 'relative',
                  zIndex: 12,
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
        </Box>

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

      {/* Content Area */}
      <Box
        ref={containerRef}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          flexGrow: 1,
          overflow: 'auto',
          padding: '16px',
          position: 'relative',
          // Hide scrollbar for Chrome, Safari, and Opera
          '&::-webkit-scrollbar': {
            display: 'none',
          },
          // Hide scrollbar for IE, Edge, and Firefox
          msOverflowStyle: 'none',  // IE and Edge
          scrollbarWidth: 'none',    // Firefox
        }}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        {/* Zoom Controls - Fixed Position on Top Right */}
        {currentTab === 0 && (
          <Box
            sx={{
              position: 'fixed',
              right: '32px',
              top: '80px', // Adjust based on your header height
              zIndex: 2,
              backgroundColor: 'rgba(10, 25, 47, 0.7)',
              borderRadius: '4px',
              padding: '4px',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
            }}
          >
            <IconButton
              onClick={handleZoomIn}
              size="small"
              sx={{
                color: '#00FFD1',
                '&:hover': {
                  backgroundColor: 'rgba(0, 255, 209, 0.1)',
                },
              }}
              aria-label="Zoom In"
            >
              <AddIcon />
            </IconButton>
            <IconButton
              onClick={handleZoomOut}
              size="small"
              sx={{
                color: '#00FFD1',
                '&:hover': {
                  backgroundColor: 'rgba(0, 255, 209, 0.1)',
                },
              }}
              aria-label="Zoom Out"
            >
              <RemoveIcon />
            </IconButton>
          </Box>
        )}

        {/* Map or Other Tabs Content */}
        {currentTab === 0 ? (
          /* Map Container with Zoom and Panning */
          <Box
            sx={{
              position: 'relative',
              width: `${scaledWidth}px`,
              height: `${scaledHeight}px`,
              transition: 'width 0.2s ease-out, height 0.2s ease-out',
              margin: '0 auto',
              transform: `translate(${offset.x}px, ${offset.y}px)`,
              cursor: isDragging ? 'grabbing' : 'grab',
              userSelect: 'none',
            }}
            onMouseDown={handleMouseDown}
          >
            {/* Render the MapChart */}
            <Box
              sx={{
                width: '100%',
                height: '100%',
              }}
            >
              <MapChart />
            </Box>
          </Box>
        ) : (
          /* Globe and Inspect Tabs Filling the Container */
          <Box
            sx={{
              width: '100%',
              height: 'auto',
              margin: '0 auto',
            }}
          >
            {currentTab === 1 && <GlobeComponent />}
            {currentTab === 2 && <InspectorView />}
          </Box>
        )}

        {/* Observation State */}
        <Box
          sx={{
            width: '100%',
            maxWidth: '100%',
            marginTop: '16px', // Space between map and observation
            transition: 'width 0.2s ease-out',
          }}
        >
          <ObservationState />
        </Box>
      </Box>
    </Box>
  );
};

export default MainPanel;

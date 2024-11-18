import React, { useState, useMemo, useCallback } from 'react';
import { 
  Box, 
  TextField,
  Typography,
  Popper,
  Paper,
  MenuList,
  MenuItem
} from '@mui/material';
import debounce from 'lodash/debounce';

const searchableProps = [
  { key: 'id', display: 'ID' },
  { key: 'task_type_str', display: 'Task Type' },
  { key: 'priority', display: 'Priority' },
  { key: 'storage_size', display: 'Storage Size' },
  { key: 'task_duration', display: 'Duration' },
  { key: 'latitude', display: 'Latitude' },
  { key: 'longitude', display: 'Longitude' },
  { key: 'min_elev', display: 'Min Elevation' },
  { key: 'user_id', display: 'User ID' }
];

const TaskSearchBar = ({ searchQuery, setSearchQuery }) => {
  const [localSearchQuery, setLocalSearchQuery] = useState(searchQuery);
  const [anchorEl, setAnchorEl] = useState(null);
  const [cursorPosition, setCursorPosition] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Debounce the search query update
  const debouncedSetSearchQuery = useCallback(
    debounce((value) => {
      setSearchQuery(value);
    }, 300),
    []
  );

  const showPropertyMenu = useMemo(() => {
    const lastAtSymbol = localSearchQuery.lastIndexOf('@', cursorPosition);
    if (lastAtSymbol === -1) return false;
    
    const nextSpace = localSearchQuery.indexOf(' ', lastAtSymbol);
    if (nextSpace !== -1 && nextSpace < cursorPosition) return false;
    
    const colonAfterAt = localSearchQuery.indexOf(':', lastAtSymbol);
    if (colonAfterAt !== -1 && colonAfterAt < cursorPosition) return false;
    
    return true;
  }, [localSearchQuery, cursorPosition]);

  const filteredProps = useMemo(() => {
    if (!showPropertyMenu) {
      setSelectedIndex(0);
      return [];
    }
    
    const lastAtSymbol = localSearchQuery.lastIndexOf('@', cursorPosition);
    const searchText = localSearchQuery.slice(lastAtSymbol + 1, cursorPosition).toLowerCase();
    
    if (!searchText) return searchableProps;
    
    return searchableProps.filter(prop => 
      prop.display.toLowerCase().includes(searchText) ||
      prop.key.toLowerCase().includes(searchText)
    );
  }, [localSearchQuery, cursorPosition, showPropertyMenu]);

  const handleSearchChange = (e) => {
    const newValue = e.target.value;
    setLocalSearchQuery(newValue);
    debouncedSetSearchQuery(newValue);
    setCursorPosition(e.target.selectionStart);
    setAnchorEl(e.target);
    setSelectedIndex(0); // Reset selection when typing
  };

  const handlePropertySelect = (propKey) => {
    const lastAtSymbol = localSearchQuery.lastIndexOf('@', cursorPosition);
    const newQuery = localSearchQuery.slice(0, lastAtSymbol) + 
                    `@${propKey}:` + 
                    localSearchQuery.slice(cursorPosition);
    setLocalSearchQuery(newQuery);
    debouncedSetSearchQuery(newQuery);
    setAnchorEl(null);
    setSelectedIndex(0);
  };

  const handleKeyDown = (e) => {
    if (!showPropertyMenu || filteredProps.length === 0) return;

    switch (e.key) {
      case 'Escape':
        e.preventDefault();
        setAnchorEl(null);
        setSelectedIndex(0);
        break;
      
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev < filteredProps.length - 1 ? prev + 1 : 0
        );
        break;
      
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev > 0 ? prev - 1 : filteredProps.length - 1
        );
        break;
      
      case 'Enter':
        e.preventDefault();
        if (filteredProps[selectedIndex]) {
          handlePropertySelect(filteredProps[selectedIndex].key);
        }
        break;

      case 'Tab':
        if (showPropertyMenu && filteredProps.length > 0) {
          e.preventDefault();
          handlePropertySelect(filteredProps[selectedIndex].key);
        }
        break;
    }
  };

  return (
    <Box sx={{ position: 'relative' }}>
      <TextField
        fullWidth
        variant="outlined"
        placeholder="Search tasks... (Use @ to search by property)"
        value={localSearchQuery}
        onChange={handleSearchChange}
        onKeyDown={handleKeyDown}
        sx={{
          mb: 2,
          '& .MuiOutlinedInput-root': {
            color: '#00FFD1',
            '& fieldset': { borderColor: 'rgba(0, 255, 209, 0.3)' },
            '&:hover fieldset': { borderColor: '#00FFD1' },
            '&.Mui-focused fieldset': { borderColor: '#00FFD1' },
          },
          '& .MuiInputBase-input::placeholder': {
            color: 'rgba(0, 255, 209, 0.7)',
          },
        }}
      />
      
      <Popper
        open={showPropertyMenu && filteredProps.length > 0}
        anchorEl={anchorEl}
        placement="bottom-start"
        sx={{ zIndex: 1300 }}
      >
        <Paper sx={{ 
          backgroundColor: '#1A2C4E',
          border: '1px solid rgba(0, 255, 209, 0.3)',
          maxHeight: '300px',
          overflow: 'auto'
        }}>
          <MenuList>
            {filteredProps.map((prop, index) => (
              <MenuItem
                key={prop.key}
                onClick={() => handlePropertySelect(prop.key)}
                selected={index === selectedIndex}
                sx={{
                  color: '#00FFD1',
                  backgroundColor: index === selectedIndex ? 
                    'rgba(0, 255, 209, 0.1)' : 'transparent',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 255, 209, 0.1)',
                  },
                  '&.Mui-selected': {
                    backgroundColor: 'rgba(0, 255, 209, 0.2)',
                    '&:hover': {
                      backgroundColor: 'rgba(0, 255, 209, 0.25)',
                    }
                  }
                }}
              >
                <Typography variant="body2">
                  {prop.display} (@{prop.key})
                </Typography>
              </MenuItem>
            ))}
          </MenuList>
        </Paper>
      </Popper>
    </Box>
  );
};

export default TaskSearchBar; 
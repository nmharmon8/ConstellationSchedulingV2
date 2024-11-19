import React, { useEffect, useRef } from 'react';
import { Dialog, DialogContent, DialogTitle, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useAgent } from '../../store/AgentStore';
import './TaskGPTModal.css';

const TaskGPTModal = () => {
  const { 
    isTaskGPTModalOpen, 
    setIsTaskGPTModalOpen, 
    taskGPTMessages,
    isTaskGPTProcessing 
  } = useAgent();
  
  const contentRef = useRef(null);

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [taskGPTMessages]);

  return (
    <Dialog
      open={isTaskGPTModalOpen}
      maxWidth="sm"
      className="taskgpt-modal taskgpt-side-panel"
      hideBackdrop={true}
      disableScrollLock={true}
      onClose={() => setIsTaskGPTModalOpen(false)}
      sx={{
        position: 'fixed',
        zIndex: 1000,
        '& .MuiDialog-container': {
          alignItems: 'flex-start',
          justifyContent: 'flex-end',
          pointerEvents: 'none'
        },
        '& .MuiBackdrop-root': {
          pointerEvents: 'none'
        }
      }}
      PaperProps={{
        sx: {
          position: 'fixed',
          right: 0,
          top: 0,
          height: '100%',
          maxHeight: '100vh',
          width: '400px',
          margin: 0,
          borderRadius: '12px 0 0 12px',
          color: '#ffffff',
          border: '1px solid rgba(100, 255, 218, 0.1)',
          overflowX: 'hidden',
          pointerEvents: 'auto',
          zIndex: 1000
        }
      }}
    >
      <DialogTitle className="taskgpt-modal-header">
        TaskGPT Terminal
        <IconButton
          onClick={() => setIsTaskGPTModalOpen(false)}
          sx={{
            position: 'absolute',
            right: 8,
            top: 8,
            color: '#00e6e6'
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent ref={contentRef} className="taskgpt-content">
        <div className="terminal">
          <div className="terminal-line">
            <span className="prompt">{'>'}</span>
            <span className="message">
              {taskGPTMessages.map(msg => msg.message).join('')}
            </span>
          </div>
          {isTaskGPTProcessing && (
            <div className="terminal-line">
              <span className="prompt">{'>'}</span>
              <span className="cursor"></span>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TaskGPTModal; 
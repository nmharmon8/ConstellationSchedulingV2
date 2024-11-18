import React from 'react';
import TaskList from './components/TaskList';
import SatList from './components/SatList';
import MainPanel from './components/MainPanel';
import ErrorDisplay from './components/ErrorDisplay';
import { AgentProvider } from './store/AgentStore';
import './styles/App.css';

function App() {
  return (
    <AgentProvider>
      {/* <ErrorDisplay /> */}
      <div className="app-container">
        <div className="panel side-panel">
          <SatList />
        </div>
        <div className="panel map-panel">
          <MainPanel />
        </div>
        <div className="panel side-panel">
          <TaskList />
        </div>
        
      </div>
    </AgentProvider>
  );
}

export default App;

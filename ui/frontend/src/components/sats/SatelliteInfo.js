import React from 'react';
import { useAgent } from '../../store/AgentStore';

const SatelliteInfo = () => {
  const { currentSatState } = useAgent();

  const satelliteIds = Object.keys(currentSatState || {});

  return (
    <div>
      <h2>Satellite Information</h2>
      <ul>
        {satelliteIds.map(id => {
          const sat = currentSatState[id];
          return (
            <li key={id}>
              <h3>{id}</h3>
              <p>Position: Lat: {sat.lat.toFixed(2)}, Lon: {sat.lon.toFixed(2)}, Alt: {(sat.alt / 1000).toFixed(2)} km</p>
              <p>In Eclipse: {sat.observation.in_eclipse ? 'Yes' : 'No'}</p>
              <p>Next Eclipse: {sat.observation.next_eclipse.toFixed(2)} seconds</p>
              <p>End of Eclipse: {sat.observation.end_of_eclipse.toFixed(2)} seconds</p>
            </li>
          );
        })}
      </ul>
    </div>
  );
};

export default SatelliteInfo;

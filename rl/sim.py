import numpy as np  
from datetime import datetime

from rl.sat import Satellite, create_random_satellite
from rl.tasks.task_manager import TaskManager
from bsk_rl.utils.orbital import random_orbit
# from bsk_rl.sim import Simulator as BSKSimulator
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc


CONSTANT_DATETIME = datetime(2023, 1, 1, 0, 0, 0).strftime("%Y %b %d %H:%M:%S.%f (UTC)")  # Year, Month, Day, Hour, Minute, Second

from bsk_rl.utils.orbital import random_epoch

def get_default_world_args():
    return {'planetRadius': 6378136.6, 'baseDensity': 1.22, 'scaleHeight': 8000.0, 'utc_init': random_epoch(), 'groundStationsData': [{'name': 'Boulder', 'lat': 40.009971, 'long': -105.243895, 'elev': 1624}, {'name': 'Merritt', 'lat': 28.3181, 'long': -80.666, 'elev': 0.9144}, {'name': 'Singapore', 'lat': 1.3521, 'long': 103.8198, 'elev': 15}, {'name': 'Weilheim', 'lat': 47.8407, 'long': 11.1421, 'elev': 563}, {'name': 'Santiago', 'lat': -33.4489, 'long': -70.6693, 'elev': 570}, {'name': 'Dongara', 'lat': -29.2452, 'long': 114.9326, 'elev': 34}, {'name': 'Hawaii', 'lat': 19.8968, 'long': -155.5828, 'elev': 9}], 'groundLocationPlanetRadius': 6378136.6, 'gsMinimumElevation': 0.17453292519943295, 'gsMaximumRange': -1}

from bsk_rl.sim.world import GroundStationWorldModel

# def sec2nano(t):
#     return int(t * 1e9)

# def nano2sec(t):
#     return t * 1e-9

class Simulator(SimulationBaseClass.SimBaseClass):

    def __init__(self, config, action_def):
        super().__init__()
        self.config = config
        self.action_def = action_def
        self.sim_rate = config['sim_rate']
        self.max_step_duration_sec = config['max_step_duration']
        self.max_sat_coordination = config['max_sat_coordination']
        self.time_limit = config['time_limit']
        self.n_access_windows = config['n_access_windows']
        self.min_tasks = config['min_tasks']
        self.max_tasks = config['max_tasks']
        self.n_sats = config['n_sats']

        

        self.fsw_list = {}
        self.dynamics_list = {}
        self.world = GroundStationWorldModel(self, self.sim_rate, **get_default_world_args())
        self.utc_init = datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)")
        self.satellites = [create_random_satellite(f"EO-{i}", simulator=self, utc_init=self.utc_init) for i in range(self.n_sats)]
        self.task_manager = TaskManager(self.satellites, self.config, self.action_def)
        self.cum_reward = 0

        self.InitializeSimulation()
        self.ConfigureStopTime(0)
        self.ExecuteSimulation()


    # def get_sat_info(self):
    #     sat_info = {}
    #     #  r_BP_P = sat.trajectory.r_BP_P(current_time)
    #     #     lat, lon = ecef_to_latlon(r_BP_P[0], r_BP_P[1], r_BP_P[2])
    #     for sat in self.satellites:
    #         sat_info[sat.id] = sat.get_info()
    #     return sat_info
        
  

    def reset(self):
        assert self.sim_time_ns == 0, "Simulation cannot be reset, you must create a new instance"

        observation, info = self.task_manager.reset()

        info['observation'] = observation.get_info_observation_dict()
        self.last_obs = observation
        
        return observation.get_observation_numpy(), info
    

    def is_alive(self):
        return all(sat.is_alive() for sat in self.satellites)


    def step(self, actions):

        # Simulation time
        end_time = self.sim_time_ns + mc.sec2nano(self.max_step_duration_sec)

        tasks_observations, reward, tasks_info = self.task_manager.step(actions, self.sim_time_ns * mc.NANO2SEC, end_time * mc.NANO2SEC)
        self.cum_reward += reward

        tasks_info['observation'] = self.last_obs.get_info_observation_dict()
        self.last_obs = tasks_observations


        print(f"FSW: Going to run simulation from {self.sim_time_ns} to {end_time} ")
        self.ConfigureStopTime(end_time)
        self.ExecuteSimulation()
      
        return tasks_observations.get_observation_numpy(), reward, tasks_info
    

    @property
    def sim_time_ns(self) -> int:
        """Simulation time in ns, tied to SimBase integrator."""
        return self.TotalSim.CurrentNanos

    @property
    def sim_time(self) -> float:
        """Simulation time in seconds, tied to SimBase integrator."""
        return self.sim_time_ns * mc.NANO2SEC
    
    @property
    def done(self):
        return self.sim_time >= self.time_limit
    
    def __del__(self):
        try:
            # Delete the task manager
            del self.task_manager
        except:
            print("Task manager not deleted")
        # Delete the satellites
        try:
            del self.satellites 
        except:
            print("Satellites not deleted")
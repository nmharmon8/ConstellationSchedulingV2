import numpy as np  
from datetime import datetime, timedelta
import random

from rl.sat import Satellite, create_random_satellite
from rl.tasks.task_manager import TaskManager
from Basilisk.utilities import SimulationBaseClass, orbitalMotion, macros as mc

from bsk_rl.sim.world import BasicWorldModel

def get_random_utc_init():
    # Get the current time plus a random interval between 0 and 24 hours
    now = datetime.now()
    random_time = now + timedelta(hours=random.randint(0, 24))
    return random_time.strftime("%Y %b %d %H:%M:%S.%f (UTC)")

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

        self.utc_init = get_random_utc_init()
        # Build the args for the world model
        world_args = {
            'planetRadius': orbitalMotion.REQ_EARTH * 1e3,
            'baseDensity': 1.22,
            'scaleHeight': 8000.0,
        } 

        self.fsw_list = {}
        self.dynamics_list = {}
        self.world = BasicWorldModel(self, world_rate=self.sim_rate, utc_init=self.utc_init, **world_args)
        self.task_manager = TaskManager(self.config, self.action_def)
 
        self.satellites = [create_random_satellite(f"EO-{i}", simulator=self, utc_init=self.utc_init) for i in range(self.n_sats)]       
        self.cum_reward = 0

        self.InitializeSimulation()
        self.ConfigureStopTime(0)
        self.ExecuteSimulation()

    def get_sat_info(self):
        sat_info = {}
        for sat in self.satellites:
            sat_info[sat.id] = sat.get_info()
        return sat_info
        
    def reset(self):
        assert self.sim_time_ns == 0, "Simulation cannot be reset, you must create a new instance"
        self.task_manager.reset()
        # Get observations
        end_time = self.sim_time_ns + mc.sec2nano(self.max_step_duration_sec)

        info = {
            'actions': [],
            'start_time': self.sim_time_ns * mc.NANO2SEC,
            'end_time': end_time * mc.NANO2SEC,
            'satellite_in_order': [sat.id for sat in self.satellites],
            'init_observation': {},
            'init_satellites': {},
            'init_action_tasks': {},
            'end_observation': {},
            'end_satellites': {},
            'end_action_tasks': {},
            'reward': 0,
        }

         # Collect Info
        for sat in self.satellites:
            observation = self.task_manager.get_observations(sat, self.sim_time)
            info['init_observation'][sat.id] = observation.get_observations_info()
            info['init_satellites'][sat.id] = sat.get_info()
            info['end_observation'][sat.id] = observation.get_observations_info()
            info['end_satellites'][sat.id] = sat.get_info()

        observations = []
        for sat in self.satellites:
            observations.append(self.task_manager.get_observations(sat, end_time * mc.NANO2SEC).get_observations_numpy())
        observations = np.stack(observations, axis=0)

        return observations, info

    
    def is_alive(self):
        return all(sat.is_alive() for sat in self.satellites)

    def step(self, actions, get_info=False):

        print(f"Sim time: {self.sim_time}")

        # Simulation time
        end_time = self.sim_time_ns + mc.sec2nano(self.max_step_duration_sec)

    
        info = {
            'actions': actions,
            'start_time': self.sim_time_ns * mc.NANO2SEC,
            'end_time': end_time * mc.NANO2SEC,
            'satellite_in_order': [sat.id for sat in self.satellites],
            'init_observation': {},
            'init_satellites': {},
            'init_action_tasks': {},
            'end_observation': {},
            'end_satellites': {},
            'end_action_tasks': {},
        }
            # Collect Info
        for sat, action_idx in zip(self.satellites, actions):
            observation = self.task_manager.get_observations(sat, self.sim_time)
            info['init_observation'][sat.id] = observation.get_observations_info()
            info['init_satellites'][sat.id] = sat.get_info()
            task, _ = observation.action_to_task(action_idx)
            info['init_action_tasks'][sat.id] = task.task_info()
      

       

        # Start actions
        sat_tasks = []
        for sat, action_idx in zip(self.satellites, actions):
            observation = self.task_manager.get_observations(sat, self.sim_time)
            task, window_offset = observation.action_to_task(action_idx)
            sat.start_action(task, window_offset, self.sim_time, end_time * mc.NANO2SEC)
            sat_tasks.append((sat, task))

        # Run simulation to take the actions
        print(f"FSW: Going to run simulation from {self.sim_time_ns}ns to {end_time}ns")
        self.ConfigureStopTime(end_time)
        self.ExecuteSimulation() # self.sim_time will now be the end time

        # Complete the actions
        for sat, task in sat_tasks:
            sat.complete_action(task, self.sim_time)

        # Step the task manager to calculate the reward
        reward = self.task_manager.step()

    
        # Collect Info
        for sat, task in sat_tasks:
            observation = self.task_manager.get_observations(sat, self.sim_time)
            info['end_observation'][sat.id] = observation.get_observations_info()
            info['end_satellites'][sat.id] = sat.get_info()
            info['end_action_tasks'][sat.id] = task.task_info()
        info['reward'] = reward

        # Get observations
        observations = []
        for sat in self.satellites:
            observations.append(self.task_manager.get_observations(sat, self.sim_time).get_observations_numpy())
        observations = np.stack(observations, axis=0)
        
        return observations, reward, info
    
    def get_debug_observation(self):
        sat_observations = {
            'satellite_observation': {},
            'normalization_terms': self.config['observation_normalization_terms'],
            'observation_keys': self.config['observation_keys'],
        }
        for sat in self.satellites:
            observations = self.task_manager.get_observations(sat, self.sim_time)
            sat_observations['satellite_observation'][sat.id] = observations.get_debug_observation()
        return sat_observations

    

    

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
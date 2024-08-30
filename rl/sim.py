import numpy as np  
from datetime import datetime

from rl.sat import Satellite
from rl.task import TaskManager
from rl.noop_manger import NoopManager
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.utils.orbital import TrajectorySimulator

CONSTANT_DATETIME = datetime(2023, 1, 1, 0, 0, 0).strftime("%Y %b %d %H:%M:%S.%f (UTC)")  # Year, Month, Day, Hour, Minute, Second


class Simulator():

    # def __init__(self, sim_rate=1.0, max_step_duration=600.0, time_limit=3000, max_sat_coordination=3, n_access_windows=10, n_sats=2, min_tasks=200, max_tasks=3000, n_trajectory_factor=20, **kwargs):
    def __init__(self, config, action_def):
        self.config = config
        self.action_def = action_def
        self.sim_rate = config['sim_rate']
        self.max_step_duration = config['max_step_duration']
        self.max_sat_coordination = config['max_sat_coordination']
        self.time_limit = config['time_limit']
        self.n_access_windows = config['n_access_windows']
        self.min_tasks = config['min_tasks']
        self.max_tasks = config['max_tasks']
        self.n_sats = config['n_sats']

        print(f"Creating trajectories")
        n_trajectory_factor = config['n_trajectory_factor']
        # datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)")
        self.trajectories = [TrajectorySimulator(utc_init=CONSTANT_DATETIME, rN=None, vN=None, oe=random_orbit(alt=800), mu=398600436000000.0) for _ in range(n_trajectory_factor * self.n_sats)]





          

    def reset(self):
        self.sim_time = 0.0
        self.cum_reward = 0
 
        random_trajectories = np.random.choice(self.trajectories, self.n_sats, replace=False)
        self.satellites = [Satellite(f"EO-{i}", trajectory=trajectory) for i, trajectory in enumerate(random_trajectories)]
        self.task_manager = TaskManager(self.satellites, self.config, self.action_def)
        self.noop_manager = NoopManager(self.config, self.action_def)
        observation, info = self.task_manager.reset()

        info['observation'] = observation.get_observations_dict()
        self.last_obs = observation
  
        return observation.get_observation_numpy(), info


    def step(self, actions):
        # Simulation time
        start_time = self.sim_time
        end_time = self.sim_time + self.max_step_duration

        tasks_observations, tasks_reward, tasks_info = self.task_manager.step(actions, start_time, end_time)
        _, noop_reward, _ = self.noop_manager.step(actions, start_time, end_time)
        reward = tasks_reward + noop_reward
        self.cum_reward += reward

        tasks_info['observation'] = self.last_obs.get_observations_dict()
        self.last_obs = tasks_observations
      
        # Update the simulation time
        self.sim_time = end_time
        return tasks_observations.get_observation_numpy(), reward, tasks_info
    

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
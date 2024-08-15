import numpy as np  
from datetime import datetime

# from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion

from bsk_rl.sim.world import GroundStationWorldModel

from rl.sat import Satellite, sat_args
from rl.task import TaskManager
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.utils.orbital import TrajectorySimulator


class Simulator():

    def __init__(self, sim_rate=1.0, max_step_duration=600.0, time_limit=3000, max_sat_coordination=3, n_access_windows=10, n_sats=2, min_tasks=200, max_tasks=3000, n_trajectory_factor=20, **kwargs):

        self.sim_rate = sim_rate
        self.max_step_duration = max_step_duration
        self.max_sat_coordination = max_sat_coordination
        self.time_limit = time_limit
        self.n_access_windows = n_access_windows
        self.min_tasks = min_tasks
        self.max_tasks = max_tasks
        self.n_sats = n_sats

        self.world = None

        print(f"Creating trajectories")
        self.trajectories = [TrajectorySimulator(utc_init=datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)"), rN=None, vN=None, oe=random_orbit(alt=800), mu=398600436000000.0) for _ in range(n_trajectory_factor * n_sats)]


    def reset(self):
        self.sim_time = 0.0
        self.cum_reward = 0
        # self.task_being_collected = {}
        self.task_manager = TaskManager(max_step_duration=self.max_step_duration, max_sat_coordination=self.max_sat_coordination, min_tasks=self.min_tasks, max_tasks=self.max_tasks)
        random_trajectories = np.random.choice(self.trajectories, self.n_sats, replace=False)
        self.satellites = [Satellite(f"EO-{i}", trajectory=trajectory) for i, trajectory in enumerate(random_trajectories)]
        return self.get_obs(), {}


    def step(self, actions):
        # Simulation time
        start_time = self.sim_time
        end_time = self.sim_time + self.max_step_duration

        info = {sat.id: {} for sat in self.satellites}
        task_being_collected = {}

        # Now take actions
        for satellite, action in zip(self.satellites, actions):
            # First n actions are for collecting tasks
            if action < self.n_access_windows and action < len(self.current_tasks_by_sat[satellite.id]):
                _, task = self.current_tasks_by_sat[satellite.id][action]
                task.collect(satellite, start_time, end_time)
                task_being_collected[satellite.id] = task


        # Build info for debugging and plotting        
        for sat_id, task in task_being_collected.items():
            task_reward = task.get_reward()
            info[sat_id]['task_reward'] = task_reward
            info[sat_id]['task'] = task


        # Now get the reward based on the action taken from start to end time
        reward = self.task_manager.step()
        self.cum_reward += reward
        # Update the simulation time
        self.sim_time = end_time

        print(f"Reward: {reward}")

        # Get the next observations
        observations = self.get_obs()
      

        return observations, reward, info
    


    @property
    def done(self):
        return self.sim_time >= self.time_limit

    def get_obs(self):

        # Update task windows
        for satellite in self.satellites:
            self.task_manager.calculate_access_windows(satellite,  calculation_start=self.sim_time, duration=self.max_step_duration * self.n_access_windows)

        self.current_tasks_by_sat = {}
        for satellite in self.satellites:
            self.current_tasks_by_sat[satellite.id] = self.task_manager.get_upcoming_tasks(satellite, self.sim_time)[:self.n_access_windows]

        observations = []
        for satellite in self.satellites:
            sat_observations = []
            for task_start_time, task in self.current_tasks_by_sat[satellite.id]:
                sat_observations.append(task.get_observation(satellite, self.sim_time, task_start_time))

            while len(sat_observations) < self.n_access_windows:
                sat_observations.append(np.array([0, 0, 0, 0, 0]))


            sat_observations = np.concatenate(sat_observations)
            observations.append(sat_observations)

        observations = np.stack(observations, axis=0)

        return observations
    
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
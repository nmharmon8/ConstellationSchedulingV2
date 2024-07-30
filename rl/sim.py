import numpy as np  
import random
from datetime import datetime

from bsk_rl.utils.orbital import TrajectorySimulator
from bsk_rl.utils.orbital import random_orbit

from tqdm import tqdm

from rl.sat import Satellite, sat_args
from rl.task import TaskManager

class Simulator():

    def __init__(self, sim_rate=1.0, max_step_duration=600.0, time_limit=3000, n_access_windows=10, n_sats=2, min_tasks=200, max_tasks=3000, n_trajectory_factor=20, **kwargs):

        self.n_sats = n_sats
        self.min_tasks = min_tasks
        self.max_tasks = max_tasks
        self.sim_rate = sim_rate
        self.max_step_duration = max_step_duration
        self.time_limit = time_limit
        self.n_access_windows = n_access_windows


        self.task_manager = None

        # self.trajectories = []
        # for _ in tqdm(range(n_trajectory_factor * n_sats)):
        #     traj = TrajectorySimulator(
        #         utc_init=datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)"), rN=None, vN=None, oe=random_orbit(alt=800), mu=398600436000000.0)
        #     traj.extend_to(self.time_limit)
        #     self.trajectories.append(traj)

        # self.satellites = []
        # for trj in random.sample(self.trajectories, self.n_sats):
        #     self.satellites.append(Satellite(f"EO-{len(self.satellites)}", trj))

    def reset(self):
        # self.satellites = []
        # for trj in random.sample(self.trajectories, self.n_sats):
        #     self.satellites.append(Satellite(f"EO-{len(self.satellites)}", trj))

        self.satell

        self.task_manager = TaskManager(max_step_duration=self.max_step_duration, min_tasks=self.min_tasks, max_tasks=self.max_tasks)

        self.cum_reward = 0
        self.sim_time = 0.0
        self.task_being_collected = {}
        observations = self.get_obs()
        return observations, {}

    def step(self, actions):
        # Simulation time
        start_time = self.sim_time
        end_time = self.sim_time + self.max_step_duration

        # Now take actions
        for satellite, action in zip(self.satellites, actions):
            # First n actions are for collecting tasks
            if action < self.n_access_windows and action < len(self.current_tasks_by_sat[satellite.id]):
                task = self.current_tasks_by_sat[satellite.id][action]
                task.collect(satellite, start_time, end_time)
                self.task_being_collected[satellite.id] = task
            else:
                self.task_being_collected[satellite.id] = None

        # Now get the reward based on the action taken from start to end time
        reward = self.task_manager.step()
        self.cum_reward += reward
        # Update the simulation time
        self.sim_time = end_time
        # Get the next observations
        observations = self.get_obs()
    
        return observations, reward, self.task_being_collected
    


    @property
    def done(self):
        return self.sim_time >= self.time_limit

    def get_obs(self):

        # Update task windows
        for satellite in self.satellites:
            self.task_manager.calculate_access_windows(satellite,  calculation_start=self.sim_time, duration=self.max_step_duration)

        self.current_tasks_by_sat = {}
        for satellite in self.satellites:
            self.current_tasks_by_sat[satellite.id] = self.task_manager.get_upcoming_tasks(satellite, self.sim_time)[:self.n_access_windows]

        observations = []
        for satellite in self.satellites:
            sat_observations = []
            for task in self.current_tasks_by_sat[satellite.id]:
                sat_observations.append(task.get_observation(satellite.id, self.sim_time))

            while len(sat_observations) < self.n_access_windows:
                sat_observations.append(np.array([0, 0, 0]))

            sat_observations = np.concatenate(sat_observations)
            observations.append(sat_observations)

        observations = np.stack(observations, axis=0)

        return observations
    
    def __del__(self):
        # Delete the task manager
        del self.task_manager
        # Delete the satellites
        del self.satellites 


if __name__ == "__main__":
    import time

    n_sats = 5

    sim = Simulator(sim_rate=1.0, max_step_duration=180.0, time_limit=900, n_access_windows=10, n_sats=n_sats, min_tasks=5000, max_tasks=5000, n_trajectory_factor=2)

    obs, info = sim.reset()

    print(f"Observations shape: {obs.shape}")


    while True:

        while not sim.done:
            next_obs, reward, task_being_collected = sim.step(tuple([0] * n_sats))

            # print(f"Obs: {next_obs.shape}, Reward: {reward} Done: {task_being_collected}")
            print(f"Reward: {reward}")

        obs, info = sim.reset()

   


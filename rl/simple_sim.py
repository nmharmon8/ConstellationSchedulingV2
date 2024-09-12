import numpy as np  
import random
from datetime import datetime

from bsk_rl.utils.orbital import TrajectorySimulator
from bsk_rl.utils.orbital import random_orbit

from tqdm import tqdm

from rl.sat import Satellite, sat_args
from rl.tasks.task_manager import TaskManager

class Simulator():

    def __init__(self, sim_rate=1.0, max_step_duration=600.0, time_limit=3000, n_access_windows=10, n_sats=2, min_tasks=200, max_tasks=3000, n_trajectory_factor=20, **kwargs):

        self.n_sats = n_sats
        self.n_access_windows = n_access_windows
        self.last_obs = None
        self.rewards = []

        self._step = 0
        self.cum_reward = 0
        self.max_possible_reward = 0

        self.reset()


    def make_observation(self):
        observations = []
        rewards = []
        for i in range(self.n_sats):
            obs = []
            rs = []
            for _ in range(self.n_access_windows):
                r = np.random.rand()
                obs.extend([r]*3)
                rs.append(r)

            observations.append(obs)
            rewards.append(rs)

        self.rewards = rewards
        
        obs = np.array(observations)
        return obs

    def reset(self):
        # print(f"Resetting simulator after cumulative reward: {self.cum_reward} and max possible reward: {self.max_possible_reward}")
        self._step = 0
        self.cum_reward = 0
        self.max_possible_reward = 0
        self.last_obs = self.make_observation()
        return self.last_obs, {}
    
    def get_obs(self):
        return self.last_obs
    
    def step(self, actions):
        self._step += 1
        reward = sum([self.last_obs[i][actions[i] * 3] for i in range(self.n_sats)])
        self.cum_reward += reward
        for rs in self.last_obs:
            self.max_possible_reward += max(rs)

        self.last_obs = self.make_observation()
        return self.last_obs, reward, {}

    @property
    def done(self):
        return self._step > 100


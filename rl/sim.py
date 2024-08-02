import numpy as np  

# from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion

from bsk_rl.sim.world import GroundStationWorldModel

from rl.sat import Satellite, sat_args
from rl.task import TaskManager

class Simulator():

    def __init__(self, sim_rate=1.0, max_step_duration=600.0, time_limit=3000, n_access_windows=10, n_sats=2, min_tasks=200, max_tasks=3000, n_trajectory_factor=20, **kwargs):

        self.n_sats = n_sats
        self.n_access_windows = n_access_windows
        self.last_obs = None
        self.rewards = []

        self._step = 0
        self.cum_reward = 0
        self.max_possible_reward = 0


    def make_observation(self):
        observations = []
        rewards = []
        for i in range(self.n_sats):
            obs = []
            rs = []
            for j in range(self.n_access_windows):
                r = np.random.rand()
                # if j == 0:
                #     r = 1
                # else:
                #     r = 0
                obs.append(r)
                rs.append(r)

            observations.append(obs)
            rewards.append(rs)

        self.rewards = rewards
        
        return np.array(observations)

    def reset(self):
        print(f"Resetting simulator after cumulative reward: {self.cum_reward} and max possible reward: {self.max_possible_reward}")
        self._step = 0
        self.cum_reward = 0
        self.max_possible_reward = 0
        self.last_obs = self.make_observation()
        return self.last_obs, {}
    
    def step(self, actions):
        self._step += 1
        reward = sum([self.last_obs[i][actions[i]] for i in range(self.n_sats)])

        
        self.cum_reward += reward
        max_reward = 0
        for rs in self.last_obs:
            max_reward += max(rs)

        self.max_possible_reward += max_reward

        # reward = reward / self.max_possible_reward

        self.last_obs = self.make_observation()
        return self.last_obs, reward, {}

    @property
    def done(self):
        return self._step > 100
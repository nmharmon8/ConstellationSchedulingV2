from time import time_ns

import numpy as np
from gymnasium import Env, spaces

from rl.sim import Simulator


class SatelliteTasking(Env):


    """
    Rest
    Loop:
        Action
        Step
    
    """




    @property
    def dtype(self):
        return self.space.dtype


    def __init__(self, config):

        self.config = config

        self.simulator = None
        self.latest_step_duration = 0.0

        # self.action_space = spaces.Box(low=0, high=1, shape=(3,))
        # self.action_space = spaces.Tuple((spaces.Discrete(10), spaces.Discrete(10)))
        # self.action_space = spaces.Sequence(spaces.Discrete(10))
        # self.action_space = SequenceWithShape(10)
        self.action_space = spaces.Tuple([spaces.Discrete(10) for _ in range(config['n_sats'])])
        # self.action_space = spaces.Tuple((spaces.Discrete(10), spaces.Discrete(10)))

        # self.observation_space = spaces.Box(low=0, high=1, shape=(3,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(config['n_sats'], 30))

        


    @property
    def cum_reward(self):
        return self.simulator.cum_reward


    def reset(self, seed=None, options=None):

        if self.simulator is not None:
            del self.simulator
        
        seed = None
        if seed is None:
            seed = time_ns() % 2**32

        super().reset(seed=seed)
        np.random.seed(seed)

        self.simulator = Simulator(**self.config)

        self.latest_step_duration = 0.0

        observations = self.simulator.get_obs()

        return observations, {}

    def step(self, actions):

        next_obs, reward, task_being_collected = self.simulator.step(actions)

        return next_obs, reward, self.simulator.done, False, {'task_being_collected': task_being_collected} 


    def render(self) -> None:  # pragma: no cover
        """No rendering implemented."""
        return None

    def close(self) -> None:
        """Try to cleanly delete everything."""
        if self.simulator is not None:
            del self.simulator



if __name__ == "__main__":
    import random
    env = SatelliteTasking()
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0

    step = 0
    while not done and not truncated:
        action = [0, 0]
        next_obs, reward, done, truncated, _ = env.step(action)
        print(f"Obs: {obs}, Action: {action}, Reward: {reward} Done: {done} Truncated: {truncated}")
        obs = next_obs
        total_reward += reward
        step += 1
        if step == 3:
            break
from time import time_ns
import time

import numpy as np
from gymnasium import Env, spaces

from rl.sim import Simulator
# from rl.simple_sim import Simulator


class SatelliteTasking(Env):

    @property
    def dtype(self):
        return self.space.dtype


    def __init__(self, config):

        self.config = config

        self.simulator = Simulator(**self.config)
        self.latest_step_duration = 0.0

        self.action_space = spaces.Tuple([spaces.Discrete(10) for _ in range(config['n_sats'])])
        self.observation_space = spaces.Box(low=-1, high=1, shape=(config['n_sats'], 5 * config['n_access_windows']))

    @property
    def cum_reward(self):
        return self.simulator.cum_reward


    def reset(self, seed=None, options=None):

       
        
        seed = None
        if seed is None:
            seed = time_ns() % 2**32

        super().reset(seed=seed)
        np.random.seed(seed)

        observations, info = self.simulator.reset()

        self.latest_step_duration = 0.0

        return observations, info

    def step(self, actions):

        next_obs, reward, info = self.simulator.step(actions)

        return next_obs, reward, self.simulator.done, False, info


    def render(self) -> None:  # pragma: no cover
        """No rendering implemented."""
        return None

    def close(self) -> None:
        """Try to cleanly delete everything."""
        if self.simulator is not None:
            del self.simulator


def main(config):


    import random
    env = SatelliteTasking(config['env'])
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0

    step = 0

    start_time = time.time()   
    time_steps = 2 
    while not done and not truncated:
        action = [0, 0, 0, 0]
        next_obs, reward, done, truncated, _ = env.step(action)
        # print(f"Obs: {obs}, Action: {action}, Reward: {reward} Done: {done} Truncated: {truncated}")
        obs = next_obs
        total_reward += reward
        step += 1
        if (step + 1) % time_steps == 0:
            stop_time = time.time()
            print(f"Avg Step time taken: {(stop_time - start_time) / time_steps}")
            start_time = time.time()
            next_obs, info = env.reset()
            


if __name__ == "__main__":
    from rl.config import parse_args, load_config
    args = parse_args()
    config = load_config(args.config)
    main(config)
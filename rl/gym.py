from time import time_ns
import time
import torch
import numpy as np
import random
from gymnasium import Env, spaces

from rl.sim import Simulator
from rl.action_def import ActionDef

def set_seeds(seed):
    from numpy import random as np_random
    np_random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)


from bsk_rl.utils.orbital import random_epoch

class SatelliteTasking(Env):

    @property
    def dtype(self):
        return self.space.dtype


    def __init__(self, config):

        try:
            self.worker_idx = config.worker_index  # <- but you can also use the worker index
            self.num_workers = config.num_workers  # <- or the total number of workers
            self.vector_env_index = config.vector_index
        except:
            self.worker_idx = 0
            self.num_workers = 1
            self.vector_env_index = 0

        self.seed = self.worker_idx + self.vector_env_index * self.num_workers

        print(f"Seed: {self.seed}")

        set_seeds(self.seed)

        self.config = config
        self.action_def = ActionDef(config)
        self.action_space = self.action_def.action_space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(config['n_sats'], config['n_access_windows'], len(config['observation_keys'])))

    @property
    def cum_reward(self):
        return self.simulator.cum_reward

    def reset(self, seed=None, options=None):
        self.simulator = Simulator(self.config, self.action_def)
        observations, info = self.simulator.reset()
        return observations, info

    def step(self, actions):
        next_obs, reward, info = self.simulator.step(actions)
        print(f"reward: {reward} for actions: {actions}")
        return next_obs, reward, self.simulator.done, not self.simulator.is_alive(), info

    def render(self) -> None:  # pragma: no cover
        """No rendering implemented."""
        return None

    def close(self) -> None:
        """Try to cleanly delete everything."""
        if self.simulator is not None:
            del self.simulator

    def get_debug_observation(self):
        return self.simulator.get_debug_observation()



def main(config):
    from pprint import pprint
    import random
    import cProfile
    import pstats
    import io

    env = SatelliteTasking(config['env'])
    obs, info = env.reset()
    done = False
    terminated = False
    total_reward = 0

    step = 0

    # # Initialize the profiler
    # profiler = cProfile.Profile()
    # profiler.enable()

    while True:  # not done and not terminated:
        action = [step % 4] * config['env']['n_sats']
        obs, reward, done, terminated, info = env.step(action)
        total_reward += reward
        step += 1

        # # Optional: Add a condition to break the loop after a certain number of steps
        # if step >= 10:
        #     break

        print(f"Step {step} reward: {reward}")

    # profiler.disable()

    # # Create a stream to hold the profiling results
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    # ps.print_stats()

    # # Print the profiling results
    # print(s.getvalue())
            


if __name__ == "__main__":
    from rl.config import parse_args, load_config
    args = parse_args()
    config = load_config(args.config)

    # Inject worker_index, num_workers, and vector_index into the config as properties

    main(config)

"""
python -m rl.gym --config rl/configs/basic_config.yaml
"""
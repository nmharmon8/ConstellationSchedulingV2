
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from rl.config import load_config



import os
ray.init(local_mode=True)
from rl.gym import SatelliteTasking

def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

class SatelliteGuard:
    def __init__(self, env):
        self.env = env


    def find_next_downlink_task(self, observations):
        for i, (window_offset, task) in enumerate(observations.upcoming_tasks):
            if task.is_data_downlink:
                return task, window_offset, i
        return None, None, None


    CHARGE_ACTION = 17
    DESAT_ACTION = 18


    def guard_actions(self, actions):
        actions = list(actions)
        sim = self.env.simulator
        sats = sim.satellites
        for i, (sat, action) in enumerate(zip(sats, actions)):
            observations = sim.task_manager.get_observations(sat, sim.sim_time)
            print(f"Sat {sat.name} should charge: {sat.should_charge()} should desat: {sat.should_desat()} should downlink: {sat.should_downlink()}")
            if sat.should_charge():
                actions[i] = self.CHARGE_ACTION
            elif sat.should_desat():
                actions[i] = self.DESAT_ACTION
            if sat.should_downlink():
                task, window_offset, action_idx = self.find_next_downlink_task(observations)
                if task is not None and window_offset == 0:
                    actions[i] = action_idx
                else:
                    # If storage is full but no downlink is available, then we should just wait
                    actions[i] = len(observations) - 1

            else:

                if sat.pct_power() < 0.2 or sat.pct_storage() > 0.9:
                    actions[i] = len(observations) - 1
                else:
                    task, window_offset = observations.action_to_task(action)
                    if window_offset == 0:
                        print(f"Sat {sat.name} is taking default action {action}")
                        actions[i] = action
                    else:
                        print(f"Sat {sat.name} is taking action {action} because it's not the first collect task")
                        task, offset, idx = observations.get_first_collect_task()
                        if task is not None and offset == 0:
                            print(f"Sat {sat.name} is taking action {idx} because it's the first collect task")
                            actions[i] = idx
                        else:
                            print(f"Sat {sat.name} is taking action {len(observations) - 1} because it's not the first collect task")
                            actions[i] = len(observations) - 1
        return actions

def find_latest_checkpoint(model_dir):
    import glob
    checkpoints = glob.glob(model_dir + "/*/*/checkpoint*")
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")
    # Sort checkpoints by number and get the latest one
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1]))
    return latest_checkpoint

class Agent:

    def __init__(self, config, n_tasks=500, greedy=False, seed=42):

        self.greedy = greedy

        # Determine if GPU should be used
        use_gpu = config.get('use_gpu', False)
        num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

        config['env_runners']['num_env_runners'] = 0

        ppo_config = (
            PPOConfig()
            .env_runners(
                num_env_runners=1,
                num_cpus_per_env_runner=3,
            )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(
                env=SatelliteTasking,
                env_config=config['env'],
            )
            .framework("torch")
            .resources(num_gpus=num_gpus) 
        )

        ppo_config.model.update(
            {
                "custom_model": "simple_model",
                "custom_model_config":config['model']
            }
        )

        self.algo = ppo_config.build()

        checkpoint = find_latest_checkpoint(f"{config['checkpoint_path']}/")
        # Convert to absolute path
        checkpoint = os.path.abspath(checkpoint)
        print(f"Restoring from {checkpoint}")
        self.algo.restore(checkpoint)

        config['env']['time_limit'] = 1000000
        config['env']['min_tasks'] = n_tasks
        config['env']['max_tasks'] = n_tasks
        
        self.data_per_step = []

        self.done = False
        self.truncated = False
        self.step = 0    

        self.env = SatelliteTasking(config['env'])
        self.obs, self.info = self.env.reset(seed=seed)
        self.total_reward = 0
        self.guard = SatelliteGuard(self.env)

    def eval(self, steps=20):

        reward_sum = 0

        for _ in range(steps):

            if self.greedy:
                action = [0] * len(self.env.simulator.satellites)
                action = self.guard.guard_actions(action)
            else:
                print("Computing action without exploration")
                action = self.algo.compute_single_action(self.obs, explore=False)
                action = self.guard.guard_actions(action)

            

            next_obs, reward, done, truncated, info = self.env.step(action)

            reward_sum += reward

        return reward_sum

         
    
    def get_inspector_observation_state(self):
        return {
            'obs': self.obs,
            'debug': self.env.get_debug_observation()
        }


if __name__ == "__main__":
    config = load_config("rl/configs/backend_config.yaml")
    
    n_task_tests = [100, 200, 400, 800, 1600]
    steps = [5, 10, 20]
    # policy_reward_sums = []
    # greedy_reward_sums = []
    results = {
        'greedy': [],
        'policy': []
    }

    for i, n_tasks in enumerate(n_task_tests):
        for s in steps:
            agent = Agent(config, n_tasks=n_tasks, seed=i)
            reward_sum = agent.eval(steps=s)
            print(f"Test {i} policy steps {s} n_tasks {n_tasks} reward: {reward_sum}")
            results['policy'].append({'steps': s, 'n_tasks': n_tasks, 'reward': reward_sum})
            

            agent = Agent(config, n_tasks=n_tasks, greedy=True, seed=i)
            reward_sum = agent.eval(steps=s)
            print(f"Test {i} greedy steps {s} n_tasks {n_tasks} reward: {reward_sum}")
            results['greedy'].append({'steps': s, 'n_tasks': n_tasks, 'reward': reward_sum})

            print(f"Current results: {results}")

    print(f"Final results: {results}")


"""
python eval_agent.py
"""
import sys
sys.path.append('../../')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import argparse
import time
import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from pymap3d import ecef2geodetic
from rl.config import parse_args, load_config
from rl.action_def import ActionDef
from rl.tasks.observation import get_observation_from_numpy
from object_def import StepInfo

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

class Agent:

    def __init__(self, config, model_name, greedy=False):

        self.config = config
        self.model_name = model_name
        self.greedy = greedy

        self.info = {}
        self.action_def = ActionDef(config['env'])

        # Determine if GPU should be used
        use_gpu = config.get('use_gpu', False)
        num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

        config['env_runners']['num_env_runners'] = 0

        ppo_config = (
            PPOConfig()
            .training(**config['training_args'])
            .env_runners(**config['env_runners'])
            .api_stack(enable_rl_module_and_learner=False)
            .environment(
                env=SatelliteTasking,
                env_config=config['env'],
            )
            .framework("torch")
            .checkpointing(export_native_model_files=True)
            .resources(num_gpus=num_gpus) 
        )

        ppo_config.model.update(
            {
                "custom_model": "simple_model",
                # "custom_action_dist": "message_dist",
                "custom_model_config":config['model']
            }
        )

        self.algo = ppo_config.build()

        def find_latest_checkpoint(model_dir):
            import glob
            checkpoints = glob.glob(model_dir + "/*/*/checkpoint*")
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {model_dir}")
            # Sort checkpoints by number and get the latest one
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1]))
            return latest_checkpoint

        checkpoint = find_latest_checkpoint(f"/data/nm/{model_name}/")
        print(f"Restoring from {checkpoint}")
        self.algo.restore(checkpoint)

        # config['env']['time_limit'] = 5700
        config['env']['min_tasks'] = 500
        config['env']['max_tasks'] = 500
        
        self.data_per_step = []

        self.done = False
        self.truncated = False
        self.step = 0    

        self.env = SatelliteTasking(config['env'])
        self.obs, self.info = self.env.reset(seed=42)
        self.total_reward = 0

    def get_task_info(self):
        return [task.task_info() for task in self.env.simulator.task_manager.tasks]
    
    def get_sat_info(self):
        return [sat.get_info() for sat in self.env.simulator.satellites]

    def get_info(self):
        return self.info

    def take_step(self):

        if self.greedy:
            action = self.greedy_action(self.obs)
        else:
            print("Computing action without exploration")
            action = self.algo.compute_single_action(self.obs, explore=False)

        print(f"Action: {action}")
        # action = [self.step % 4] 
        # action = [2] * 10
        # action = [0] * 10

        next_obs, reward, done, truncated, info = self.env.step(action)

        self.obs = next_obs
        self.info = info
        self.step += 1
        return StepInfo(info)
    
    def get_inspector_observation_state(self):
        return {
            'obs': self.obs,
            'debug': self.env.get_debug_observation()
        }

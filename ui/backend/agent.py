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
                "custom_action_dist": "message_dist",
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
        self.action_index_to_sat = self.info['action_index_to_sat']
        self.total_reward = 0

        self.current_step_data = {
            'obs': self.obs,
            'reward': 0,
            'done': False,
            'truncated': False,
            'info': self.info
        }

    def get_task_info(self):
        return [task.task_info() for task in self.env.simulator.task_manager.tasks]
    
    def get_sat_info(self):
        return [sat.get_info() for sat in self.env.simulator.satellites]

    def ecef_to_latlon(self, x, y, z):
        lat, lon, alt = ecef2geodetic(x, y, z)
        return lat, lon

    def greedy_action(self, obs):
        def get_action(sat_info):        
            # First look if we can complete any tasks
            for idx, obs in enumerate(sat_info):
                if obs['window_index_offset'] == 0:
                    if obs['task_storage_size'] > 0:
                        if obs['storage_after_task'] < 1:
                            return idx
                        
             # Next check if we can downlink any data
            for idx, obs in enumerate(sat_info):
                if obs['window_index_offset'] == 0:
                    if obs['is_data_downlink']:
                        return idx

            # Finally do a noop task, always last task
            return len(sat_info) - 1
            

        actions = []
        self.obs = get_observation_from_numpy(self.obs, self.action_index_to_sat, self.config['env'])
        for i in range(len(self.info['action_index_to_sat'])):
            sat_id = self.info['action_index_to_sat'][i]
            sat_info = self.obs[sat_id]
            actions.append(get_action(sat_info))
        return actions

    

    def take_step(self):

        if self.greedy:
            action = self.greedy_action(self.obs)
        else:
            print("Computing action without exploration")
            action = self.algo.compute_single_action(self.obs, explore=False)

        next_obs, reward, done, truncated, info = self.env.step(action)

        last_step = self.current_step_data
        self.current_step_data = {
            'obs': next_obs,
            'reward': reward,
            'done': done,
            'truncated': truncated,
            'info': info
        }

        print({
            'last_step': last_step,
            'current_step': self.current_step_data
        })

        return {
            'last_step': last_step,
            'current_step': self.current_step_data
        }

    
        # print(f"Done {done} Truncated {truncated}")
        # print(f"Cumulative reward: {info['cum_reward']}")
        # print(f"Time {self.env.simulator.sim_time}")

        # satellite_data = {}

        # current_time = self.env.simulator.sim_time
        # for i, sat in enumerate(self.env.simulator.satellites):
        #     r_BP_P = sat.trajectory.r_BP_P(current_time)
        #     lat, lon = self.ecef_to_latlon(r_BP_P[0], r_BP_P[1], r_BP_P[2])
        #     satellite_data[sat.id] = info[sat.id]
        #     satellite_data[sat.id]['time'] = current_time
        #     satellite_data[sat.id]['latitude'] = lat
        #     satellite_data[sat.id]['longitude'] = lon
        #     satellite_data[sat.id]['task_reward'] = info[sat.id]['task_reward'] if 'task_reward' in info[sat.id] else 0
        #     satellite_data[sat.id]['actions'] = int(action[i])
        #     satellite_data[sat.id]['action_type'] = self.action_def.get_action_type(action[i])
        #     satellite_data[sat.id]['reward'] = reward
        #     satellite_data[sat.id]['observation'] = info['observation'][sat.id]

        # step_data = {
        #     'step': self.step,
        #     'cum_reward': info['cum_reward'],
        #     'n_tasks_collected': info['n_tasks_collected'],
        #     'satellites': satellite_data,
        #     'done': done,
        #     'truncated': truncated
        # }

        # return step_data
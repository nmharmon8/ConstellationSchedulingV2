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

class SatelliteGuard:
    def __init__(self, env):
        self.env = env

    
    def guard_actions(self, actions):
        actions = list(actions)
        sim = self.env.simulator
        sats = sim.satellites
        for i, (sat, action) in enumerate(zip(sats, actions)):
            observations = sim.task_manager.get_observations(sat, sim.sim_time)
            print(f"Sat {sat.name} should charge: {sat.should_charge()} should desat: {sat.should_desat()} should downlink: {sat.should_downlink()}")
            if sat.should_charge():
                actions[i] = 0
            elif sat.should_desat():
                actions[i] = 1
            if sat.should_downlink():
                task, window_offset = observations.action_to_task(2)
                if window_offset == 0:
                    actions[i] = 2
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

    def __init__(self, config, greedy=False):

        self.config = config
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
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
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
        config['env']['min_tasks'] = 500
        config['env']['max_tasks'] = 500
        
        self.data_per_step = []

        self.done = False
        self.truncated = False
        self.step = 0    

        self.env = SatelliteTasking(config['env'])
        self.obs, self.info = self.env.reset(seed=42)
        self.total_reward = 0
        self.guard = SatelliteGuard(self.env)

    def get_task_info(self):
        return [task.task_info() for task in self.env.simulator.task_manager.tasks]
    
    def get_completed_tasks(self):
        return [task.task_info() for task in self.env.simulator.task_manager.completed_tasks]
    
    def get_sat_info(self):
        return [sat.get_info() for sat in self.env.simulator.satellites]

    def get_info(self):
        return StepInfo(self.info)
    
    def add_new_task(self, name, lat, lon, priority, task_type, min_elev, duration, user_id):
        from rl.tasks.task import CollectTask
        from bsk_rl.utils.orbital import lla2ecef
        import uuid
        from Basilisk.utilities import orbitalMotion

        # Using Earth radius as altitude since we're dealing with ground targets
        r_LP_P = lla2ecef(lat, lon, orbitalMotion.REQ_EARTH * 1e3)
        # must convert lat, lon to r_LP_P

        task = CollectTask(
            name=f"{name}-{uuid.uuid4()}",
            r_LP_P=r_LP_P,
            priority=priority,
            simultaneous_collects_required=1,  # Default to single satellite collection
            task_duration=duration,
            task_type=task_type,
            storage_size=500,  # Default storage size in MB
            max_step_duration=200,
            n_access_windows=20,
            min_elev=min_elev,
            user_id=user_id
        )

        self.env.simulator.task_manager.insert_new_task(task)


    def take_step(self):

        if self.greedy:
            action = self.greedy_action(self.obs)
        else:
            print("Computing action without exploration")
            action = self.algo.compute_single_action(self.obs, explore=False)

        # action = [3] * len(self.env.simulator.satellites)

        # action = self.guard.guard_actions(action)

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

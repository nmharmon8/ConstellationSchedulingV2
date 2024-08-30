import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
from tqdm import tqdm
import argparse
import time
import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from pymap3d import ecef2geodetic
from rl.config import parse_args, load_config
from rl.action_def import ActionDef

def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

def run_policy(config, model_name, steps, output, greedy=False):
    
    action_def = ActionDef(config['env'])

    # Determine if GPU should be used
    use_gpu = config.get('use_gpu', False)
    num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

    import os
    ray.init(local_mode=True)
    from rl.gym import SatelliteTasking

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

    algo = ppo_config.build()

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
    algo.restore(checkpoint)

    exit()

    # config['env']['time_limit'] = 100000
    config['env']['min_tasks'] = 500
    config['env']['max_tasks'] = 500

    env = SatelliteTasking(config['env'])
    obs, info = env.reset(seed=42)
    total_reward = 0


    def ecef_to_latlon(x, y, z):
        lat, lon, alt = ecef2geodetic(x, y, z)
        return lat, lon

    def map_tasks(task):
        r_LP_P = task.r_LP_P
        lat, lon = ecef_to_latlon(r_LP_P[0], r_LP_P[1], r_LP_P[2])
        return {'id': task.id, 'latitude': lat, 'longitude': lon, 'priority': task.priority, 'min_elev': task.min_elev, 'simultaneous_collects_required':task.simultaneous_collects_required}

    tasks = env.simulator.task_manager.tasks
    tasks = [map_tasks(task) for task in tasks]
    data_per_step = []

    done = False
    truncated = False
    step = 0    

    while not done and not truncated:
        if greedy:
            action = [0] * config['env']['n_sats']
        else:
            print("Computing action without exploration")
            action = algo.compute_single_action(obs, explore=False)
        next_obs, reward, done, truncated, info = env.step(action)

        print(f"Done {done} Truncated {truncated}")
        print(f"Cumulative reward: {info['cum_reward']}")
        print(f"Time {env.simulator.sim_time}")
        if done or truncated:
            time.sleep(10)


        satellite_data = {}

        current_time = env.simulator.sim_time
        for i, sat in enumerate(env.simulator.satellites):
            r_BP_P = sat.trajectory.r_BP_P(current_time)
            lat, lon = ecef_to_latlon(r_BP_P[0], r_BP_P[1], r_BP_P[2])
            satellite_data[sat.id] = {}
            satellite_data[sat.id]['time'] = current_time
            satellite_data[sat.id]['latitude'] = lat
            satellite_data[sat.id]['longitude'] = lon
            satellite_data[sat.id]['task_being_collected'] = map_tasks(info[sat.id]['task']) if info[sat.id]['task'] else None
            satellite_data[sat.id]['task_reward'] = info[sat.id]['task_reward'] if 'task_reward' in info[sat.id] else 0
            satellite_data[sat.id]['actions'] = int(action[i])
            satellite_data[sat.id]['action_type'] = action_def.get_action_type(action[i])
            satellite_data[sat.id]['reward'] = reward
            satellite_data[sat.id]['observation'] = info['observation'][sat.id]

        step_data = {
            'step': step,
            'cum_reward': info['cum_reward'],
            'n_tasks_collected': info['n_tasks_collected'],
            'satellites': satellite_data
        }

        data_per_step.append(step_data)
        obs = next_obs
        total_reward += reward

    json_data = {}
    json_data['targets'] = tasks
    json_data['steps'] = data_per_step

    print(json_data)
    print('Saving data to file')
    with open(output, 'w') as f:
        json.dump(json_data, f, indent=2)

    print('Done, trying to shutdown')
    ray.shutdown()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=True,
                    help='the configuration file')
    p.add_argument('--model', type=str, default="v9",
                    help='the model to load')
    p.add_argument('--greedy', action="store_true", help="Whether to use greedy action selection")
    p.add_argument('--output', type=str, default="data.json",
                    help='the output file')
    
    p.add_argument('--steps', type=int, default=10,
                    help='the number of steps to run')

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    run_policy(config, args.model, args.steps, args.output, args.greedy)



"""
python run_poly.py --config=rl/configs/basic_config.yaml --model=v16_10sat
"""
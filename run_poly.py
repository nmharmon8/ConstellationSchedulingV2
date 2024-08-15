import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json

import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from pymap3d import ecef2geodetic
from rl.config import parse_args, load_config


args = parse_args()
name = args.name
config = load_config(args.config)

# Determine if GPU should be used
use_gpu = config.get('use_gpu', False)
num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

import os
ray.init(local_mode=True, _temp_dir=os.path.abspath(f"./logs/{name}"))
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
    }
)

algo = ppo_config.build()
algo.restore("./logs/v2/")

config['env']['time_limit'] = 100000

env = SatelliteTasking(config['env'])

obs, info = env.reset()

done = False
truncated = False
total_reward = 0
steps = 0

def ecef_to_latlon(x, y, z):
    lat, lon, alt = ecef2geodetic(x, y, z)
    return lat, lon

def map_tasks(task):
    r_LP_P = task.r_LP_P
    lat, lon = ecef_to_latlon(r_LP_P[0], r_LP_P[1], r_LP_P[2])
    return {'id': task.id, 'latitude': lat, 'longitude': lon, 'priority': task.priority, 'min_elev': task.min_elev, 'simultaneous_collects_required':task.simultaneous_collects_required}
    

tasks = env.simulator.task_manager.tasks

print("Number of tasks: ", len(tasks))

tasks = [map_tasks(task) for task in tasks]

data_per_step = []

step = 0

# while not done and not truncated:
while step < 100:
    print("Getting action")
    # action = algo.compute_single_action(obs)
    action = [0] * config['env']['n_sats']
    print("Stepping env")
    next_obs, reward, done, truncated, info = env.step(action)

    print("Recording data")

    satellite_data = {}

    current_time = env.simulator.sim_time
    for i, sat in enumerate(env.simulator.satellites):
        # current_time = sat.trajectory.sim_time
        r_BP_P = sat.trajectory.r_BP_P(current_time)
        lat, lon = ecef_to_latlon(r_BP_P[0], r_BP_P[1], r_BP_P[2])
        satellite_data[sat.id] = {}
        satellite_data[sat.id]['time'] = current_time
        satellite_data[sat.id]['latitude'] = lat
        satellite_data[sat.id]['longitude'] = lon
        satellite_data[sat.id]['task_being_collected'] = map_tasks(info[sat.id]['task']) if 'task' in info[sat.id] else None
        satellite_data[sat.id]['task_reward'] = info[sat.id]['task_reward'] if 'task_reward' in info[sat.id] else 0
        satellite_data[sat.id]['actions'] = int(action[i])
        satellite_data[sat.id]['reward'] = reward

    data_per_step.append(satellite_data)

    # print(f"Obs: {obs}, Action: {action}, Reward: {reward} Done: {done} Truncated: {truncated}")
    obs = next_obs
    total_reward += reward

    step += 1


print(f"Total reward in test episode: {total_reward}")



    

json_data = {}
json_data['targets'] = tasks



# for step in data_per_step:
#     for sat in step:
#         step[sat]['targets'] = [map_tasks(target) for target in step[sat]['targets']]


print("Data per step: ", data_per_step)
print(data_per_step[0])



json_data['steps'] = data_per_step
# json_data['action_description'] = action_description
# json_data['observation_description'] = observation_description

print(json_data)

print('Saving data to file')
with open('data.json', 'w') as f:
    json.dump(json_data, f, indent=2)

print('Done')

# algo.stop()
ray.shutdown()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray

from rl.config import parse_args, load_config
from rl.data_callback import CustomDataCallbacks

from ray import air, tune

args = parse_args()
name = args.name
config = load_config(args.config)

# Determine if GPU should be used
use_gpu = config.get('use_gpu', False)
num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

import os
ray.init(local_mode=config['local_mode'], _temp_dir=os.path.abspath(f"/data/nm/{name}"))
from rl.gym import SatelliteTasking

ppo_config = (
    PPOConfig()
    .training(**config['training_args'])
    .env_runners(**config['env_runners'])
    .api_stack(enable_rl_module_and_learner=False)
    .environment(
        env=SatelliteTasking,
        env_config=config['env'],
    )
    # .callbacks(CustomDataCallbacks)
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

from ray.air.constants import TRAINING_ITERATION
stop = {
    TRAINING_ITERATION: 100000,
}

from ray.train import CheckpointConfig
checkpoint_config = CheckpointConfig(
    num_to_keep=3,
    checkpoint_score_attribute="episode_reward_mean",
    checkpoint_score_order="max",
    checkpoint_frequency=100,
)

storage_path = f"/data/nm/{name}"

results = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=checkpoint_config, storage_path=storage_path),
    param_space=ppo_config,
).fit()
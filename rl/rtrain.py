import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.air.constants import TRAINING_ITERATION
from ray.train import CheckpointConfig

from rl.config import parse_args, load_config

from ray import air, tune

# from torch.utils.tensorboard import SummaryWriter

args = parse_args()
name = args.name
config = load_config(args.config)
log_dir = f"/data/nm/{name}"
# writer = SummaryWriter(log_dir)

# Determine if GPU should be used
use_gpu = config['use_gpu']
num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

# Add this to explicitly tell PyTorch to use CUDA
if use_gpu and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import os
ray.init(local_mode=config['local_mode'], _temp_dir=os.path.abspath(f"/data/nm/{name}"))
from rl.gym import SatelliteTasking

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
        "custom_model_config": config['model'],
    }
)



stop = {
    TRAINING_ITERATION: 1000000,
}

checkpoint_config = CheckpointConfig(
    num_to_keep=3,
    checkpoint_score_attribute="episode_reward_mean",
    checkpoint_score_order="max",
    checkpoint_frequency=10,
)

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop=stop, 
        verbose=1, 
        checkpoint_config=checkpoint_config, 
        storage_path=log_dir,
    ),
    param_space=ppo_config,
)
# Run the training
results = tuner.fit()

"""
python -m rl.rtrain --config=rl/configs/train_config.yaml --name=vtest
"""
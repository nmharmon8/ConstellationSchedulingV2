import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray

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

from ray.air.constants import TRAINING_ITERATION
stop = {
    TRAINING_ITERATION: 1000000,
}
from ray.train import CheckpointConfig

checkpoint_config = CheckpointConfig(
    num_to_keep=3,
    checkpoint_score_attribute="episode_reward_mean",
    checkpoint_score_order="max",
    checkpoint_frequency=10,
)

# # Create tuner based on resume configuration
# if config.get('resume_training', False) and config.get('checkpoint_path'):
#     # Resume training from checkpoint
#     tuner = tune.Tuner.restore(
#         path=config['checkpoint_path'],
#         trainable="PPO",
#         param_space=ppo_config,
#     )
# else:
# Start new training run
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

# algo = ppo_config.build()

# if config['resume']:
#     print(f"Restoring from {config['checkpoint_path']}")
#     algo.restore(config['checkpoint_path'])

# for i in range(config['training']['steps']):
#     results = algo.train()
#     print(f"Step {i}: {results}")

#     # Log metrics to tensorboard
#     # Training metrics
#     writer.add_scalar('Training/Total_Loss', results['info']['learner']['default_policy']['learner_stats']['total_loss'], i)
#     writer.add_scalar('Training/Policy_Loss', results['info']['learner']['default_policy']['learner_stats']['policy_loss'], i)
#     writer.add_scalar('Training/Value_Function_Loss', results['info']['learner']['default_policy']['learner_stats']['vf_loss'], i)
#     writer.add_scalar('Training/KL_Divergence', results['info']['learner']['default_policy']['learner_stats']['kl'], i)
#     writer.add_scalar('Training/Entropy', results['info']['learner']['default_policy']['learner_stats']['entropy'], i)
    
#     # Reward metrics
#     writer.add_scalar('Rewards/Mean', results['env_runners']['episode_reward_mean'], i)
#     writer.add_scalar('Rewards/Max', results['env_runners']['episode_reward_max'], i)
#     writer.add_scalar('Rewards/Min', results['env_runners']['episode_reward_min'], i)
    
#     # Episode length
#     writer.add_scalar('Episodes/Mean_Length', results['env_runners']['episode_len_mean'], i)

#     # Save checkpoint
#     if i % 100 == 0:
#         algo.save(f"{log_dir}/checkpoint_{i}")

# writer.close()

"""
python -m rl.train --config=rl/configs/train_config.yaml --name=v154_geo
"""
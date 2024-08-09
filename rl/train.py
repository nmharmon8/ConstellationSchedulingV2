import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from ray.rllib.algorithms.ppo import PPOConfig
import ray

from rl.config import parse_args, load_config
from rl.data_callback import CustomDataCallbacks

args = parse_args()
name = args.name
config = load_config(args.config)

# Determine if GPU should be used
use_gpu = config.get('use_gpu', False)
num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

import os
ray.init(local_mode=config['local_mode'], _temp_dir=os.path.abspath(f"./logs/{name}"))
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
    .callbacks(CustomDataCallbacks)
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

for i in range(config['training']['steps']):
    result = algo.train()
    
    # print((result))
    print(f"Step {i} done")
    save_result = algo.save(checkpoint_dir=f"./logs/{name}")
    path_to_checkpoint = save_result.checkpoint.path
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )

algo.stop()
ray.shutdown()
# python -m rl.train --config=rl/configs/basic_config.yaml

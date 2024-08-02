import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import ray
from rl.config import parse_args, load_config
import time

args = parse_args()
name = args.name
config = load_config(args.config)

import os

from rl.gym import SatelliteTasking

from ray.rllib.algorithms.ppo.ppo import PPO
ppo_config = PPO.get_default_config()


ppo_config.environment(env=SatelliteTasking, env_config=config['env'])
ppo_config.framework("torch")
ppo_config.env_runners(num_env_runners=0, sample_timeout_s = 1000000.0)
ppo_config.training(num_sgd_iter=5, gamma=0.0,  sgd_minibatch_size=128, train_batch_size=512)
ppo_config.model.update(
    {
        "custom_model": "simple_model",
    }
)
ppo_config.model["vf_share_layers"] = True
ppo_config.resources(num_gpus=1)

ray.init(local_mode=config['local_mode'], _temp_dir=os.path.abspath(f"./logs/{name}"))

print("Building PPO Algorithm")
algo = ppo_config.build()

print("Done building PPO Algorithm")

print(f"Environment: {SatelliteTasking}")
print(f"Environment Config: {config['env']}")
print(f"Train Batch Size: {ppo_config.train_batch_size}")
print(f"SGD Minibatch Size: {ppo_config.sgd_minibatch_size}")

for i in range(config['training']['steps']):
    result = algo.train()
    print(f"Step {i} done")
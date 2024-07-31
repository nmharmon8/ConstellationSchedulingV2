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
ray.init(local_mode=config['local_mode'], _temp_dir=os.path.abspath(f"./logs/{name}"), num_gpus=num_gpus)



from rl.gym import SatelliteTasking

env_args = dict()

print(config['training_args'])
print(type(config['training_args']))


# training_args = dict(
#     lr=0.003,
#     gamma=0.999,
#     train_batch_size=250,  # In practice, usually a bigger number
#     num_sgd_iter=10,
#     model=dict(fcnet_hiddens=[512, 512], vf_share_layers=False),
#     lambda_=0.95,
#     use_kl_loss=False,
#     clip_param=0.1,
#     grad_clip=0.5,
# )

# Generic config.
ppo_config = (
    PPOConfig()
    .training(**config['training_args'])
    # .training(**training_args)
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

# ppo_config.model.update(
#     {
#         "custom_model": "sat_model",
#         # "custom_action_dist": "autoregressive_dist",
#     }
# )

algo = ppo_config.build()


for i in range(config['training']['steps']):
    result = algo.train()
    print((result))
    save_result = algo.save(checkpoint_dir=f"./logs/{name}")
    path_to_checkpoint = save_result.checkpoint.path
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )

# run manual test loop: 1 iteration until done
print("Finished training. Running manual test/inference loop.")

algo.stop()
ray.shutdown()

"""
python -m rl.train --config=rl/configs/basic_config.yaml --name=v3
"""
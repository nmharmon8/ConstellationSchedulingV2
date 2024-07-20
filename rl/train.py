import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ray.rllib.algorithms.ppo import PPOConfig
import ray

from rl.config import parse_args, load_config
from rl.data_callback import CustomDataCallbacks

args = parse_args()
name = args.name
config = load_config(args.config)

import os
ray.init(local_mode=config['local_mode'], _temp_dir=os.path.abspath(f"./logs/{name}"))



from rl.gym import SatelliteTasking

env_args = dict()

print(config['training_args'])
print(type(config['training_args']))


# Generic config.
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

)

ppo_config.model.update(
    {
        "custom_model": "simple_model",
        # "custom_action_dist": "sat_dist",
    }
)

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



env = SatelliteTasking()
obs, info = env.reset()
done = False
truncated = False
total_reward = 0
steps = 0


while not done and not truncated:
    action = algo.compute_single_action(obs)
    next_obs, reward, done, truncated, _ = env.step(action)
    print(f"Obs: {obs}, Action: {action}, Reward: {reward} Done: {done} Truncated: {truncated}")
    obs = next_obs
    total_reward += reward



print(f"Total reward in test episode: {total_reward}")
algo.stop()

ray.shutdown()


# python -m rl.train --config=rl/configs/basic_config.yaml

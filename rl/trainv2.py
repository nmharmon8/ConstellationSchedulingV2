# @OldAPIStack

# TODO (sven): Move this script to `examples/rl_modules/...`

import argparse
import os

from ray.air.constants import TRAINING_ITERATION
from ray.rllib.examples.envs.classes.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
# )
# parser.add_argument("--num-cpus", type=int, default=0)
# parser.add_argument(
#     "--framework",
#     choices=["tf", "tf2", "torch"],
#     default="torch",
#     help="The DL framework specifier.",
# )
# parser.add_argument("--use-prev-action", action="store_true")
# parser.add_argument("--use-prev-reward", action="store_true")
# parser.add_argument(
#     "--as-test",
#     action="store_true",
#     help="Whether this script should be run as a test: --stop-reward must "
#     "be achieved within --stop-timesteps AND --stop-iters.",
# )
# parser.add_argument(
#     "--stop-iters", type=int, default=200, help="Number of iterations to train."
# )
# parser.add_argument(
#     "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
# )
# parser.add_argument(
#     "--stop-reward", type=float, default=150.0, help="Reward at which we stop training."
# )

import torch 
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
class PendModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(obs_space.shape[0], 512)
        self.fc2 = nn.Linear(512, 512)
        
        self.action = nn.Linear(512, action_space.n)

        self.value = nn.Linear(512, 1)

        self._features = None

    def forward(self, input_dict, state, seq_lens):

        print(f"Pend Model forward obs: {input_dict['obs'].shape}")
        
        x = nn.functional.relu(self.fc1(input_dict['obs']))
        x = nn.functional.relu(self.fc2(x))
        self._features = x.clone()

        actions = self.action(x)

        return actions, []
    
    
    def value_function(self):
        return self.value(self._features).squeeze(1)


ModelCatalog.register_custom_model("pend_model", PendModel)

import ray
from ray import air, tune

ray.init(local_mode=True)

from ray.rllib.algorithms.ppo.ppo import PPO
config = PPO.get_default_config()

config.env_runners(num_env_runners=0, sample_timeout_s = 1000000.0)

config.environment(env=StatelessCartPole).resources(
    num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
).framework("torch").reporting(min_time_s_per_iteration=0.1)




config.training(num_sgd_iter=5, sgd_minibatch_size=64, vf_loss_coeff=0.0001, train_batch_size=512)
config.model["vf_share_layers"] = True
config.model.update(
    {
        "custom_model": "pend_model",
    }
)

stop = {
    TRAINING_ITERATION: 200,
    NUM_ENV_STEPS_SAMPLED_LIFETIME: 100000,
    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": 150,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop=stop,
    ),
)
results = tuner.fit()


ray.shutdown()
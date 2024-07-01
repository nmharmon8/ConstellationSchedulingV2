import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule

env = gym.make("CartPole-v1")

spec = SingleAgentRLModuleSpec(
    module_class=DiscreteBCTorchModule,
    observation_space=env.observation_space,
    action_space=env.action_space,
    model_config_dict={"fcnet_hiddens": [64]},
)

module = spec.build()
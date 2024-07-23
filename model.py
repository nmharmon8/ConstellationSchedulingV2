import torch 
import torch.nn as nn

import numpy as np

from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer

from ray.rllib.models import ModelCatalog



class SatModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # print(f"Sat Model Observation space: {obs_space}")
        # print(f"Sat Model Action space: {action_space}")
        # (RolloutWorker pid=330260) Sat Model Observation space: Box(-1.0, 1.0, (33,), float32)
        # (RolloutWorker pid=330260) Sat Model Action space: Tuple(Discrete(4), Discrete(4), Discrete(4))

        print(f"Sat Model action space: {action_space}")

        self.n_actions = sum([a.n for a in action_space.spaces])
        # self.n_actions = action_space.n
        self.n_sats = len(action_space.spaces)
        # self.n_sats = 1

        self.obs_dim = obs_space.shape[0]

        print(f"Sat Model observation space: {obs_space}")
        print(f"Sat Model action space: {action_space}")

        # Shared layers
        self.shared_layers = nn.Sequential(
            SlimFC(self.obs_dim, 512, activation_fn="tanh", initializer=normc_initializer(1.0)),
            SlimFC(512, 512, activation_fn="tanh", initializer=normc_initializer(1.0))
        )

        # Action branch
        self.action_branch = SlimFC(512, self.n_actions, activation_fn=None, initializer=normc_initializer(1.0))

        # Value function
        self.value_branch = SlimFC(512, 1, activation_fn=None, initializer=normc_initializer(1.0))

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, input_dict, state, seq_lens):

        print(f"Sat Model forward obs: {input_dict['obs'].shape}")

        obs = torch.cat(input_dict["obs"], dim=1)

        # obs = input_dict["obs"]

        # print(f"Sat Model forward obs: {obs.shape}")

        # obs = torch.ones_like(obs)

        # Check for nan or inf in obs
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"Observation contains nan or inf: {obs}")
            print(f"Input dict: {input_dict}")

        x = self.shared_layers(obs)
        self._features = x.clone()
        action_out = self.action_branch(x)
        # action_out = action_out.view(-1, self.n_sats, self.n_actions // self.n_sats,) #[batch, n_sats, n_actions]
        return action_out, []
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        return self.value_branch(self._features).squeeze(1)

ModelCatalog.register_custom_model("sat_model", SatModel)

from rl.trans import TransformerConfig, CrossBlock, LayerNorm

class SimpleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # print(f"Sat Model Observation space: {obs_space}")
        # print(f"Sat Model Action space: {action_space}")
        # (RolloutWorker pid=330260) Sat Model Observation space: Box(-1.0, 1.0, (33,), float32)
        # (RolloutWorker pid=330260) Sat Model Action space: Tuple(Discrete(4), Discrete(4), Discrete(4))

        # print(f"Sat Model action space: {action_space.feature_space}")
        # Sequence(Discrete(10), stack=False)

        # Get discrete size from sequence
        # self.n_actions = action_space.feature_space.n
        self.n_actions = 10

        # print(f"Sat Model action space: {self.n_actions}")


        # self.n_actions = sum([a.n for a in action_space.spaces])
        # self.n_actions = action_space.n
        # self.n_sats = len(action_space.spaces)
        # self.n_sats = 1

        self.config = TransformerConfig(
            n_head=4,
            n_embd=512,
            dropout=0.0,
            bias=False
        )

        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([CrossBlock(self.config, causal=False) for _ in range(3)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))

        self.obs_dim = obs_space.shape[0] * obs_space.shape[1]


        self.input_proj = nn.Linear(self.obs_dim, self.config.n_embd, bias=self.config.bias)

        self.start_tokens = nn.Parameter(torch.randn(1, 1, self.config.n_embd))
        self.action_embed = nn.Embedding(self.n_actions, self.config.n_embd)

        self.action_branch = nn.Linear(self.config.n_embd, self.n_actions, bias=self.config.bias)

        self.value_branch = nn.Linear(self.config.n_embd, 1, bias=self.config.bias)

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, input_dict, state, seq_lens):

        obs = input_dict['obs'].view(-1, self.obs_dim).float()

        x = self.input_proj(obs).unsqueeze(1)
        self._features = x.clone()[:, -1, :]

        return x, []
    
    def action_module(self, x, state):

        start_idx = self.start_tokens.expand(x.shape[0], -1, -1)
        a_n_idx = self.action_embed(state.long())

        def get_action(idx):

            for i, block in enumerate(self.transformer.h):
                idx = block(idx, x)

            idx = self.transformer.ln_f(idx)
            action_out = self.action_branch(idx)[:, -1, :]

            return action_out
        
        a1 = get_action(start_idx)
        a2 = get_action(a_n_idx)

        return a1, a2
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        return self.value_branch(self._features).squeeze(1)


ModelCatalog.register_custom_model("simple_model", SimpleModel)



from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)

class SatDistribution(TorchDistributionWrapper):
    """Action distribution for multiple satellites, each with 4 discrete actions."""

    def __init__(self, inputs, model):
        super().__init__(inputs, model)

    def deterministic_sample(self):
        dist = self._distribution()

        print(f"Deterministic sample: {self.inputs.shape}")

        # Deterministic sample and sample do not sample on the same dimension
        actions = TorchCategorical(self.inputs.permute(0, 2, 1)).deterministic_sample()
        self._action_logp = dist.logp(actions).sum(dim=-1)
        return tuple(actions.unbind(dim=-1))

    def sample(self):
        dist = self._distribution()
        actions = dist.sample()
        self._action_logp = dist.logp(actions).sum(dim=-1)
        return tuple(actions.unbind(dim=-1))

    def logp(self, actions):
        dist = self._distribution()
        return dist.logp(actions).sum(dim=-1)

    def sampled_action_logp(self):
        return self._action_logp

    def entropy(self):
        return self._distribution().entropy().sum(dim=-1)

    def kl(self, other):
        return self._distribution().kl(other._distribution()).sum(dim=-1)

    def _distribution(self):
        return TorchCategorical(self.inputs)

    @staticmethod
    def required_model_output_shape(action_space, model_config):

        print(action_space)

        num_sats = len(action_space.spaces)
        return num_sats * 4  # 4 actions per satellite
    
ModelCatalog.register_custom_action_dist(
    "sat_dist",
    SatDistribution
)
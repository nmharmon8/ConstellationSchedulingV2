import torch 
import torch.nn as nn

import numpy as np

from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models import ModelCatalog

from rl.trans import TransformerConfig, CrossBlock, LayerNorm, Block

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

        self.obs_dim = obs_space.shape[0] * obs_space.shape[1]

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

        # obs = torch.cat(input_dict["obs"], dim=1)
        obs = input_dict["obs"].view(-1, self.obs_dim).float()

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

        self.input_encoder_proj = nn.Linear(obs_space.shape[1], self.config.n_embd, bias=self.config.bias)
        self.input_encoder = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(self.config, causal=False) for _ in range(3)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))

        
        self.transformer_proj = nn.Linear(obs_space.shape[1], self.config.n_embd, bias=self.config.bias)
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([CrossBlock(self.config, causal=False) for _ in range(3)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))

        self.feature_token = nn.Parameter(torch.randn(1, 1, self.config.n_embd))

        self.feature_transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([CrossBlock(self.config, causal=False) for _ in range(3)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))


        # self.start_tokens = nn.Parameter(torch.randn(1, 1, self.config.n_embd))
        # self.action_embed = nn.Embedding(self.n_actions, self.config.n_embd)

        self.action_branch = nn.Linear(self.config.n_embd, self.n_actions, bias=self.config.bias)

        self.value_branch = nn.Linear(self.config.n_embd, 1, bias=self.config.bias)

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, input_dict, state, seq_lens):

        # obs = input_dict['obs'].view(-1, self.obs_dim).float()
        obs = input_dict['obs'].float()
        b, n_sats, n_features = obs.shape

        obs_encoding = self.input_encoder_proj(obs)
        for i, block in enumerate(self.input_encoder.h):
            obs_encoding = block(obs_encoding)

        feature_x = self.feature_token.expand(b, -1, -1)
        for i, block in enumerate(self.feature_transformer.h):
            feature_x = block(feature_x, obs_encoding)

        self._features = feature_x[:, 0, :]


        sat_encodings = self.transformer_proj(obs)
        for j, block in enumerate(self.transformer.h):
            sat_encodings = block(sat_encodings, obs_encoding)

        actions = self.action_branch(sat_encodings)

        actions = actions.view(actions.shape[0], -1)

        return actions, []
    
    def action_module(self, x, prev_actions):

        idx = self.start_tokens.expand(x.shape[0], -1, -1)

        if len(prev_actions) > 0:
            prev_actions = torch.stack(prev_actions, dim=1).long()
            idx = torch.cat([idx, self.action_embed(prev_actions)], dim=1)

        def get_action(idx):

            for i, block in enumerate(self.transformer.h):
                idx = block(idx, x)

            idx = self.transformer.ln_f(idx)
            action_out = self.action_branch(idx)[:, -1, :]

            return action_out
        
        action  = get_action(idx)

        return action
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        return self.value_branch(self._features).squeeze(1)


ModelCatalog.register_custom_model("simple_model", SimpleModel)

import torch 
import torch.nn as nn

import numpy as np

from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models import ModelCatalog


from rl.trans import TransformerConfig, CrossBlock, LayerNorm, Block

class SimpleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.n_actions = action_space.spaces[0].n

        self.n_sats  = obs_space.shape[0]
        self.n_access_windows = obs_space.shape[1]
        self.observation_features = obs_space.shape[2]
        self.f_dim = self.n_access_windows * self.observation_features

        config =  model_config['custom_model_config']

        self.task_encoder = nn.Sequential(
            nn.Linear(self.observation_features, 128),
            nn.ReLU(),
        )

        self.planning = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(128, 2, False, causal=False, time_emd=False, dropout=0.0) for _ in range(3)]),
            ln_f = LayerNorm(128, bias=False)
        ))

        self.observation_encoder = nn.Sequential(
            nn.Linear(256 * self.n_access_windows, 512),
            nn.ReLU(),
        )



        self.action_branch = nn.Linear(512, self.n_actions, bias=True)
        self.value_branch = nn.Linear(512, 1, bias=True)
    
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float()
        b, n_sats, n_access_windows, n_features = obs.shape

        tasks = self.task_encoder(obs)

        plans = tasks.reshape(b, n_sats * n_access_windows, -1)

        for block in self.planning.h:
            plans = block(plans)
        plans = self.planning.ln_f(plans)

        plans = plans.reshape(b, n_sats, n_access_windows, -1)

        task_plans = torch.cat([plans, tasks], dim=-1)

        observations = self.observation_encoder(task_plans.reshape(b, n_sats, -1))

        self._features = observations.clone()
        actions = self.action_branch(observations)
        actions = actions.reshape(b, -1)
        return actions, []
    
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        value = torch.mean(self.value_branch(self._features), dim=1).squeeze(1)
        return value


ModelCatalog.register_custom_model("simple_model", SimpleModel)

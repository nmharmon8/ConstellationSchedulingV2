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

        # self.n_actions = 11
        print(f"Action space: {action_space}")
        self.n_actions = action_space.spaces[0].n
        print(f"Number of actions: {self.n_actions}")

        
        self.n_sats  = obs_space.shape[0]
        self.n_access_windows = obs_space.shape[1]
        self.observation_features = obs_space.shape[2]
        self.f_dim = self.n_access_windows * self.observation_features

        config =  model_config['custom_model_config']

        # # First do attention over each satellites observations
        # self.obs_featuer_proj = nn.Linear(self.observation_features, config['feature_model']['n_embd'], bias=config['feature_model']['bias'])
        # self.obs_feature_encoder = nn.ModuleDict(dict(
        #     h = nn.ModuleList([Block(config['feature_model']['n_embd'], config['feature_model']['n_head'], config['feature_model']['bias'], causal=False, time_emd=False, dropout=config['feature_model']['dropout']) for _ in range(config['feature_model']['layers'])]),
        #     ln_f = LayerNorm(config['feature_model']['n_embd'], bias=config['feature_model']['bias'])
        # ))


        # # Then do attention over the concatenated observations, e.g satellite level features
        # self.obs_sat_proj = nn.Linear(config['feature_model']['n_embd'] * self.n_access_windows, config['satellite_model']['n_embd'], bias=config['satellite_model']['bias'])
        # self.obs_sat_encoder = nn.ModuleDict(dict(
        #     h = nn.ModuleList([Block(config['satellite_model']['n_embd'], config['satellite_model']['n_head'], config['satellite_model']['bias'], causal=False, time_emd=False, dropout=config['satellite_model']['dropout']) for _ in range(config['satellite_model']['layers'])]),
        #     ln_f = LayerNorm(config['satellite_model']['n_embd'], bias=config['satellite_model']['bias'])
        # ))

        self.task_encoder = nn.Sequential(
            nn.Linear(self.observation_features, 128),
            nn.ReLU(),
        )

        self.observation_encoder = nn.Sequential(
            nn.Linear(128 * self.n_access_windows, 512),
            nn.ReLU(),
        )

        self.planning = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(512, 8, False, causal=False, time_emd=False, dropout=0.0) for _ in range(3)]),
            ln_f = LayerNorm(512, bias=False)
        ))
        # self.planning = nn.Sequential(
        #     nn.Linear(512 * self.n_sats, 1024),
        #     nn.ReLU(),
        # )

        self.dropout = nn.Dropout(0.5)

        self.action_branch = nn.Linear(512, self.n_actions, bias=True)
        self.value_branch = nn.Linear(512 * self.n_sats, 1, bias=True)
    

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float() # (batch, n_sats, access_windows, observation_features)
        b, n_sats, n_access_windows, n_features = obs.shape

        tasks = self.task_encoder(obs)
        observations = self.observation_encoder(tasks.reshape(b, n_sats, -1))

        print(f"Observations shape: {observations.shape}")

        for block in self.planning.h:
            observations = block(observations)
        observations = self.planning.ln_f(observations)
        print(f"Observations shape: {observations.shape}")
        # observations = self.planning(observations.reshape(b, -1))

        self._features = observations.clone()
        actions = self.action_branch(observations)
        print(f"Actions shape: {actions.shape}")
        # actions = actions.reshape(b, -1)
        return actions, []
    
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        value = self.value_branch(self._features.view(self._features.size(0), -1))
        return value.squeeze(1)


ModelCatalog.register_custom_model("simple_model", SimpleModel)

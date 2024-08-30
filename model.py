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

        self.f_dim = obs_space.shape[0] * obs_space.shape[1]
        self.n_sats  = obs_space.shape[0]
        self.n_access_windows = obs_space.shape[1]
        self.observation_features = obs_space.shape[2]

        config =  model_config['custom_model_config']

        # First do attention over each satellites observations
        self.obs_featuer_proj = nn.Linear(self.observation_features, config['feature_model']['n_embd'], bias=config['feature_model']['bias'])
        self.obs_feature_encoder = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config['feature_model']['n_embd'], config['feature_model']['n_head'], config['feature_model']['bias'], causal=False, time_emd=False, dropout=config['feature_model']['dropout']) for _ in range(config['feature_model']['layers'])]),
            ln_f = LayerNorm(config['feature_model']['n_embd'], bias=config['feature_model']['bias'])
        ))


        # Then do attention over the concatenated observations, e.g satellite level features
        self.obs_sat_proj = nn.Linear(config['feature_model']['n_embd'] * self.n_access_windows, config['satellite_model']['n_embd'], bias=config['satellite_model']['bias'])
        self.obs_sat_encoder = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config['satellite_model']['n_embd'], config['satellite_model']['n_head'], config['satellite_model']['bias'], causal=False, time_emd=False, dropout=config['satellite_model']['dropout']) for _ in range(config['satellite_model']['layers'])]),
            ln_f = LayerNorm(config['satellite_model']['n_embd'], bias=config['satellite_model']['bias'])
        ))

        self.action_branch = nn.Linear(config['satellite_model']['n_embd'], self.n_actions, bias=config['satellite_model']['bias'])
        
        self.value_branch_proj = nn.Linear(config['satellite_model']['n_embd'], 16, bias=config['satellite_model']['bias'])
        self.value_branch = nn.Linear(16 * self.n_sats, 1, bias=config['satellite_model']['bias'])
    

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float() # (batch, n_sats, access_windows, observation_features)
        b, n_sats, n_access_windows, n_features = obs.shape

        # Reshape to (batch * n_sats, access_windows, observation_features)
        obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3])
        obs = self.obs_featuer_proj(obs)
        for i, block in enumerate(self.obs_feature_encoder.h):
            obs = block(obs)
        obs = self.obs_feature_encoder.ln_f(obs)

        # Reshape to (batch, n_sats, access_windows, observation_features)
        obs = obs.reshape(b, n_sats, n_access_windows, obs.shape[2])
        obs = obs.reshape(b, n_sats, n_access_windows * obs.shape[3])
        obs = self.obs_sat_proj(obs)
        for i, block in enumerate(self.obs_sat_encoder.h):
            obs = block(obs)
        obs = self.obs_sat_encoder.ln_f(obs)

        self._features = obs.clone()
        actions = self.action_branch(obs)
        return actions, []
    
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        value_proj = self.value_branch_proj(self._features)
        value_proj = torch.nn.functional.relu(value_proj)
        value_proj = value_proj.reshape(value_proj.shape[0], value_proj.shape[1] * value_proj.shape[2])
        value = self.value_branch(value_proj)
        return value.squeeze(1)


ModelCatalog.register_custom_model("simple_model", SimpleModel)

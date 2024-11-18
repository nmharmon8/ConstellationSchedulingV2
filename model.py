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

        # self.f_dim = obs_space.shape[0] * obs_space.shape[1]
        # self.n_sats  = obs_space.shape[0]
        # self.n_access_windows = obs_space.shape[1]
        # self.observation_features = obs_space.shape[2]

        config =  model_config['custom_model_config']


        self.task_encoder = nn.Sequential(
            nn.Linear(self.observation_features, config['satellite_model']['task_embd']),
            nn.ReLU(),
            nn.Linear(config['satellite_model']['task_embd'], config['satellite_model']['task_embd']),
        )

        self.observation_encoder = nn.Sequential(
            nn.Linear(config['satellite_model']['task_embd'] * self.n_access_windows, config['satellite_model']['n_embd']),
            nn.ReLU(),
            nn.Linear(config['satellite_model']['n_embd'], config['satellite_model']['n_embd']),
        )

        self.satellite_encoder = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(
                config['satellite_model']['n_embd'], 
                config['satellite_model']['n_head'], 
                config['satellite_model']['bias'], 
                causal=False, 
                time_emd=False, 
                dropout=config['satellite_model']['dropout']
            ) for _ in range(config['satellite_model']['layers'])]),
            ln_f = LayerNorm(config['satellite_model']['n_embd'], bias=config['satellite_model']['bias'])
        ))

        self.action_branch = nn.Linear(config['satellite_model']['n_embd'], self.n_actions, bias=config['satellite_model']['bias'])
        

        self.value_branch_proj = nn.Linear(config['satellite_model']['n_embd'], 128, bias=config['satellite_model']['bias'])
        self.value_encoder = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(
                128, 
                2, 
                False, 
                causal=False, 
                time_emd=True, 
                dropout=config['satellite_model']['dropout']
            ) for _ in range(2)]),
            ln_f = LayerNorm(128, bias=False)
        ))

        self.value_branch = nn.Linear(128, 1)
    

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float() # (batch, n_sats, access_windows, observation_features)
        b, n_sats, n_access_windows, n_features = obs.shape

        tasks = self.task_encoder(obs)
        obs = self.observation_encoder(tasks.reshape(b, n_sats, -1)) # shape: torch.Size([32, 10, 1024])

        
        for i, block in enumerate(self.satellite_encoder.h):
            obs = block(obs)
        obs = self.satellite_encoder.ln_f(obs) # shape: torch.Size([32, 10, 1024])

        
        self._features = obs.clone()
        actions = self.action_branch(obs).reshape(b, -1)
        return actions, []
    
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        value_proj = self.value_branch_proj(self._features)
        print(f"Value proj shape: {value_proj.shape}")
        for i, block in enumerate(self.value_encoder.h):
            value_proj = block(value_proj)
        print(f"Value proj shape after encoder: {value_proj.shape}")
        value_proj = self.value_encoder.ln_f(value_proj)
        value = self.value_branch(value_proj[:, 0, :])
        print(f"Value shape: {value.shape}")
        return value.squeeze(1)


ModelCatalog.register_custom_model("simple_model", SimpleModel)

import torch 
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

from rl.trans import TransformerConfig, CrossBlock, LayerNorm, Block

class SimpleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.n_actions = 10

        self.config = TransformerConfig(
            n_head=4,
            n_embd=512,
            dropout=0.2,
            bias=False
        )

        self.input_encoder_proj = nn.Linear(obs_space.shape[1], self.config.n_embd, bias=self.config.bias)
        self.input_encoder = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(self.config, causal=False) for _ in range(3)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))

        self.feature_token = nn.Parameter(torch.randn(1, 1, self.config.n_embd))

        self.feature_transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([CrossBlock(self.config, causal=False) for _ in range(3)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))

        self.action_branch = nn.Linear(self.config.n_embd, self.n_actions, bias=self.config.bias)
        self.value_branch = nn.Linear(self.config.n_embd, 1, bias=self.config.bias)

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float()
        b, n_sats, n_features = obs.shape

        obs_encoding = self.input_encoder_proj(obs)
        for i, block in enumerate(self.input_encoder.h):
            obs_encoding = block(obs_encoding)

        feature_x = self.feature_token.expand(b, -1, -1)
        for i, block in enumerate(self.feature_transformer.h):
            feature_x = block(feature_x, obs_encoding)

        self._features = feature_x[:, 0, :]

        actions = self.action_branch(obs_encoding)
        actions = actions.view(b, -1)

        return actions, []
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        return self.value_branch(self._features).squeeze(1)


ModelCatalog.register_custom_model("simple_model", SimpleModel)

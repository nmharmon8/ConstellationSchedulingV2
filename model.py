import torch 
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

class SimpleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.n_actions = 10

    
        self.f_dim = obs_space.shape[0] * obs_space.shape[1]
        self.n_sats  = obs_space.shape[0]

        self.fc1 = nn.Linear(self.f_dim, 512)
        # self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        # self.fc2_drop = nn.Dropout(0.5)

        self.action_branch = nn.Linear(512, self.n_actions * self.n_sats)

        self.value_branch = nn.Linear(512, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].view(-1, self.f_dim).float()

        x = nn.functional.relu(self.fc1(obs))
        # x = self.fc1_drop(x)
        x = nn.functional.relu(self.fc2(x))
        # x = self.fc2_drop(x)
        self._features = x
        actions = self.action_branch(x)

        return actions, []
    
    
    def value_function(self):
        assert self._features is not None, "Must call forward() first"
        value = self.value_branch(self._features)
        return value.squeeze(1)


ModelCatalog.register_custom_model("simple_model", SimpleModel)

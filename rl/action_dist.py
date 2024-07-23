import torch 
import torch.nn as nn

import numpy as np

from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer

from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)

    
class TorchBinaryAutoregressiveDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # Return the action tuple.
        return (a1, a2)

    def logp(self, actions):
        a1, a2 = actions[:, 0], actions[:, 1]
        a1_vec = torch.unsqueeze(a1.float(), 1)
        a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
        return TorchCategorical(a1_logits).logp(a1) + TorchCategorical(a2_logits).logp(
            a2
        )

    def sampled_action_logp(self):
        return self._action_logp

    def entropy(self):
        a1_dist = self._a1_distribution()
        a2_dist = self._a2_distribution(a1_dist.sample())
        return a1_dist.entropy() + a2_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        return a1_terms + a2_terms

    def _a1_distribution(self):
        BATCH = self.inputs.shape[0]
        zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)
        a1_logits, _ = self.model.action_module(self.inputs, zeros)
        a1_dist = TorchCategorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = torch.unsqueeze(a1.float(), 1)
        _, a2_logits = self.model.action_module(self.inputs, a1_vec)
        a2_dist = TorchCategorical(a2_logits)
        return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size

ModelCatalog.register_custom_action_dist("binary_autoregressive_dist", TorchBinaryAutoregressiveDistribution)
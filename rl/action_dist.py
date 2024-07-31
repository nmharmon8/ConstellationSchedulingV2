import torch 
import torch.nn as nn

from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)

class TorchMessageDistribution(TorchDistributionWrapper):

    def _distribution(self, action):
        dist = TorchCategorical(action)
        return dist

    def _distributions(self, deterministic=False):
        dists = []
        actions = []
        n_sats = self.inputs.shape[1]

        for i in range(n_sats):
            logits = self.inputs[:, i, :]
            dist = TorchCategorical(logits)
            dists.append(dist)
            if deterministic:
                actions.append(dist.deterministic_sample())
            else:
                actions.append(dist.sample())
        return dists, actions

    def deterministic_sample(self):
        dists, actions = self._distributions(deterministic=True)
        self._action_logp = sum([dist.logp(action) for dist, action in zip(dists, actions)])
        return tuple(actions)

    def sample(self):
        dists, actions = self._distributions()
        self._action_logp = sum([dist.logp(action) for dist, action in zip(dists, actions)])
        return tuple(actions)

    def logp(self, actions):
        dists, _ = self._distributions()
        logp = sum([dist.logp(actions[..., i]) for i, dist in enumerate(dists)])
        return logp

    def sampled_action_logp(self):
        return self._action_logp

    def entropy(self):
        dists, _ = self._distributions()
        return sum([dist.entropy() for dist in dists])
        
    def kl(self, other):
        dists, _ = self._distributions()
        other_dists, _ = other._distributions()
        return sum([dist.kl(other_dist) for dist, other_dist in zip(dists, other_dists)])

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 10 # Maybe the size of a signle action logits?  # controls model output feature vector size

ModelCatalog.register_custom_action_dist("message_dist", TorchMessageDistribution)
import torch 
import torch.nn as nn
import functools

from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)

# def _actions_to_hashable(actions):
#     return tuple(int(a.item()) if isinstance(a, torch.Tensor) else int(a) for a in actions)

def cache_distribution(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(self, prev_actions=[]):
        if not prev_actions:
            return func(self, prev_actions)
        else:
            # Handle batched inputs
            key = tuple(
                tuple(int(a.detach().cpu().item()) if isinstance(a, torch.Tensor) else int(a) 
                      for a in batch_item)
                for batch_item in zip(*prev_actions)
            )
        
        if key not in cache:
            cache[key] = func(self, prev_actions)
        else:
            print(f"Cache hit")
        return cache[key]
    
    return wrapper
    
class TorchAutoregressiveDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

    # @cache_distribution #Not working? Gradient issues?
    def _distribution(self, prev_actions=[]):
        logits = self.model.action_module(self.inputs, prev_actions)
        dist = TorchCategorical(logits)
        return dist
    
    def _distributions(self):
        dists = []
        actions = []
        n_sats = self.inputs.shape[1]

        for i in range(n_sats):
            dist = self._distribution(actions)
            dists.append(dist)
            actions.append(dist.sample())

        return dists, actions

    def deterministic_sample(self):
        actions = []
        dists = []
        n_sats = self.inputs.shape[1]
        for i in range(n_sats):
            dist = self._distribution(actions)
            dists.append(dist)
            actions.append(dist.deterministic_sample())

        self._action_logp = sum([dist.logp(action) for dist, action in zip(dists, actions)])
        return tuple(actions)

    def sample(self):
        dists, actions = self._distributions()
        self._action_logp = sum([dist.logp(action) for dist, action in zip(dists, actions)])
        return tuple(actions)

    def logp(self, actions):
        actions = [actions[:, i] for i in range(actions.shape[1])]

        dists = []
        for i in range(len(actions)):
            dist = self._distribution(actions[:i])
            dists.append(dist)

        logp = sum([dist.logp(actions[i]) for i, dist in enumerate(dists)])
        return logp

    def sampled_action_logp(self):
        return self._action_logp

    def entropy(self):
        dists, _ = self._distributions()
        return sum([dist.entropy() for dist in dists])
        
    def kl(self, other):
        # a1_dist = self._a1_distribution()
        # a1_terms = a1_dist.kl(other._a1_distribution())

        # a1 = a1_dist.sample()
        # a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
        # return a1_terms + a2_terms

        dists, _ = self._distributions()
        other_dists, _ = other._distributions()

        return sum([dist.kl(other_dist) for dist, other_dist in zip(dists, other_dists)])

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 10 # Maybe the size of a signle action logits?  # controls model output feature vector size

ModelCatalog.register_custom_action_dist("autoregressive_dist", TorchAutoregressiveDistribution)
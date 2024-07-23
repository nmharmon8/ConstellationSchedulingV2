import torch 
import torch.nn as nn


from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)

    
class TorchAutoregressiveDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1)"""

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

        print(f"Deterministic sample: {self.inputs.shape}")

        actions = []
        dists = []
        n_sats = self.inputs.shape[1]

        for i in range(n_sats):
            dist = self._distribution(actions)
            dists.append(dist)
            actions.append(dist.deterministic_sample())

        self._action_logp = sum([dist.logp(action) for dist, action in zip(dists, actions)])

        # # First, sample a1.
        # a1_dist = self._a1_distribution()
        # a1 = a1_dist.deterministic_sample()

        # # Sample a2 conditioned on a1.
        # a2_dist = self._a2_distribution(a1)
        # a2 = a2_dist.deterministic_sample()
        # self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # # Return the action tuple.
        # return (a1, a2)
        return actions

    def sample(self):
        # # First, sample a1.
        # a1_dist = self._a1_distribution()
        # a1 = a1_dist.sample()

        # # Sample a2 conditioned on a1.
        # a2_dist = self._a2_distribution(a1)
        # a2 = a2_dist.sample()
        # self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # # Return the action tuple.
        # return (a1, a2)

        # actions = []
        # n_sats = self.inputs.shape[1]

        # for i in range(n_sats):
        #     dist = self._distribution(actions)
        #     actions.append(dist.sample())

        

        dists, actions = self._distributions()

        self._action_logp = sum([dist.logp(action) for dist, action in zip(dists, actions)])

        print(f"Sampled actions: {actions}")

        return actions

    def logp(self, actions):
        # a1, a2 = actions[:, 0], actions[:, 1]
        # a1_vec = torch.unsqueeze(a1.float(), 1)
        # a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
        # return TorchCategorical(a1_logits).logp(a1) + TorchCategorical(a2_logits).logp(
        #     a2
        # )

        print(f"Logp: {actions.shape}")
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
        # a1_dist = self._a1_distribution()
        # a2_dist = self._a2_distribution(a1_dist.sample())
        # return a1_dist.entropy() + a2_dist.entropy()

        # dists = []
        # n_sats = self.inputs.shape[1]

        # for i in range(n_sats):
        #     # Should I sample a2 the same way each time?
        #     dist = self._distribution([dist.sample() for dist in dists])
        #     dists.append(dist)

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

    # def _a1_distribution(self):
    #     BATCH = self.inputs.shape[0]
    #     zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)
    #     a1_logits, _ = self.model.action_module(self.inputs, zeros)
    #     a1_dist = TorchCategorical(a1_logits)
    #     return a1_dist

    # def _a2_distribution(self, a1):
    #     a1_vec = torch.unsqueeze(a1.float(), 1)
    #     _, a2_logits = self.model.action_module(self.inputs, a1_vec)
    #     a2_dist = TorchCategorical(a2_logits)
    #     return a2_dist

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size

ModelCatalog.register_custom_action_dist("autoregressive_dist", TorchAutoregressiveDistribution)
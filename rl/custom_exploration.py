from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
import numpy as np

torch, nn = try_import_torch()

class CustomActionExploration(Exploration):
    def __init__(self, action_space, *, framework, policy_config, model, num_workers, worker_index):
        super().__init__(
            action_space=action_space,
            framework=framework,
            policy_config=policy_config,
            model=model,
            num_workers=num_workers,
            worker_index=worker_index,
        )
        
        # Parameters for probability modification
        self.actions_per_agent = 20
        self.temperature = 1.0  # Controls how strong the bias towards early actions is
        
        # Create probability weights that decrease for later actions
        weights = np.exp(-np.arange(self.actions_per_agent) / self.temperature)
        self.action_weights = torch.FloatTensor(weights).to(self.device)

    @override
    def get_exploration_action(self, *, action_distribution, timestep, explore):
        if not explore:
            # During evaluation, use the original probabilities
            action = action_distribution.deterministic_sample()
            logp = torch.zeros_like(action)
            return action, logp

        # Get the original logits from the distribution
        logits = action_distribution.inputs
        batch_size = logits.shape[0]
        n_agents = logits.shape[1] // self.actions_per_agent

        # Reshape logits to [batch_size, n_agents, actions_per_agent]
        logits = logits.view(batch_size, n_agents, self.actions_per_agent)

        # Apply the action weights to modify probabilities
        modified_logits = logits + torch.log(self.action_weights)
        
        # Reshape back to original shape
        modified_logits = modified_logits.view(batch_size, -1)

        # Create new distribution with modified logits
        new_dist = type(action_distribution)(modified_logits, action_distribution.model)
        
        # Sample action from modified distribution
        action = new_dist.sample()
        logp = new_dist.logp(action)

        return action, logp
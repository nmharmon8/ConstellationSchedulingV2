
class NoopManager:
    def __init__(self, config, action_def, **kwargs):
        self.config = config
        self.action_def = action_def


    def step(self, actions, start_time, end_time):
        reward = 0.0
        for action in actions:
            if self.action_def.get_action_type(action) == "noop":
                reward += 0.1
        return {}, reward, {}
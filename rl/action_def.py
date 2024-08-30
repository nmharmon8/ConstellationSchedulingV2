from gymnasium import spaces

class ActionDef:

    def __init__(self, config):

        self.n_access_windows = config['n_access_windows']
        self.n_sats = config['n_sats']

        self.action_types = {
            i: f"collect" for i in range(self.n_access_windows)
        }

        self.action_types[len(self.action_types)] = "noop"


        self.n_actions = len(self.action_types)
        self.action_space = spaces.Tuple([spaces.Discrete(self.n_actions) for _ in range(self.n_sats)])

    def get_action_type(self, action):
        return self.action_types[int(action)]
    

    def get_action_type_indexs(self, action_type):
        return [i for i, action in self.action_types.items() if action == action_type]

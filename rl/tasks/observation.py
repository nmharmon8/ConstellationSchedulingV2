import numpy as np

from rl.tasks.task import TaskType

class Observation:

    def __init__(self, config, satellite, task, current_time, window_index):
        self.config = config
        self.current_time = current_time
        self.satellite = satellite
        self.task = task
        self.window_index = window_index
        assert self.task.is_task_possible_in_window(self.satellite, self.window_index)
        self.observation = self._get_observation()

    def __lt__(self, other):
        # First, check if either task is a NOOP task
         # NOOP tasks always come last
        if self.task.is_noop and not other.task.is_noop:
            return False
        if other.task.is_noop and not self.task.is_noop:
            return True
        
        # Charge tasks always come first
        if self.task.is_charge and not other.task.is_charge:
            return True
        if other.task.is_charge and not self.task.is_charge:
            return False
        
        # Desat tasks always come second after charge tasks
        if self.task.is_desat and not other.task.is_desat:
            return True
        if other.task.is_desat and not self.task.is_desat:
            return False
  
        # Downlink tasks always come third after desat tasks
        if self.task.is_data_downlink and not other.task.is_data_downlink:
            return True
        if other.task.is_data_downlink and not self.task.is_data_downlink:
            return False
        
        self_window_offset = self.observation['window_index_offset']
        other_window_offset = other.observation['window_index_offset']
        
        # 2. Sort by window offset (earlier windows first)
        if self_window_offset != other_window_offset:
            return self_window_offset < other_window_offset
        
        # 3. Sort by priority (higher priority first)
        return self.observation['priority'] > other.observation['priority']

    def _get_observation(self):
        current_index = int(self.current_time // self.config['max_step_duration'])
        window_index_offset = self.window_index - current_index
        obs = {}        
        obs.update(self.task.task_info())
        obs.update(self.satellite.get_observation())
        obs['window_index_offset'] = window_index_offset

        if obs['window_index_offset'] > 20:
            print("window index offset is too high", obs)
            raise Exception(f"Window index offset is too high {obs}")


        return obs
    
    def get_window_offset(self):
        return self.observation['window_index_offset']
    
    def get_normalized_observation(self):
        obs = self.observation
        obs_norm_terms = self.config['observation_normalization_terms']
        normalized_obs = {k:obs[k] / obs_norm_terms[k] for k in self.config['observation_keys']}
        for k, v in normalized_obs.items():
            if np.abs(v) > 1:
                print(f"Observation out of bounds: {k} - {v}")
                print(self)
                raise Exception(f"Observation out of bounds: {k} - {v}")
        return normalized_obs
    
    def get_normalized_observation_numpy(self):
        obs = self.get_normalized_observation()
        return np.array([obs[k] for k in self.config['observation_keys']])
    
    def __str__(self):
        obs_str = f"Task: {str(self.task.id)} -- "
        for k, v in self.observation.items():
            obs_str += f"{k}: {v}, "
        obs_str += "\n"
        return obs_str

    def get_info(self):
        return self.observation

class Observations:

    def __init__(self, current_time, upcoming_tasks, satellite, config):
        self.current_time = current_time
        self.satellite = satellite
        self.upcoming_tasks = upcoming_tasks
        self.config = config
        self.n_access_windows = config['n_access_windows']
        self._observations = self._create_observations()

    def __len__(self):
        return len(self._observations)

    def _create_observations(self):
        observations = []
        for window_index, task in self.upcoming_tasks:
            observations.append(Observation(self.config, self.satellite, task, self.current_time, window_index))
        assert len(observations) >= self.n_access_windows
        observations = sorted(observations)
        # The last task is guaranteed to be a noop task and we want to insure a noop task is always available
        observations = observations[:self.n_access_windows-1] + observations[-1:]
        return observations
    
    def action_to_task(self, action_idx):
        return self._observations[action_idx].task, self._observations[action_idx].get_window_offset()
    
    def get_observations_info(self):
        return [x.get_info() for x in self._observations]
    
    def get_observations_numpy(self):
        satellite_observations = [obs.get_normalized_observation_numpy() for obs in self._observations]
        return np.stack(satellite_observations, axis=0)
    
    def get_debug_observation(self):
        debug_observation = {
            'observations': [obs.observation for obs in self._observations],
            'numpy': [obs.get_normalized_observation_numpy() for obs in self._observations]
        }
        return debug_observation
    
    def get_first_collect_task(self):
        for idx, obs in enumerate(self._observations):
            if obs.task.is_collection:
                return obs.task, self._observations[idx].get_window_offset(), idx
        return None, None, None

    

def get_key_from_observation(key, action, observation, sat_index, config):
    """
        Get the value from a numpy array observation given the key, this requires
        figuring out the correct index depending on the key
        observation shape [n_sats, n_access_windows, n_features]
    """
    key_index = config['observation_keys'].index(key)
    norm_value = config['observation_normalization_terms'][key]
    return observation[sat_index, action, key_index] * norm_value

def get_observation_from_numpy(observation, action_index_to_sat, config):
    """
    Convert the numpy array observation back to a dictionary observation
    """
    obs = {}
    for sat_idx, sat_id in action_index_to_sat.items():
        obs[sat_id] = []
        for action in range(observation.shape[1]):
            action_obs = {}
            for key in config['observation_keys']:
                action_obs[key] = get_key_from_observation(key, action, observation, sat_idx, config)
            obs[sat_id].append(action_obs)
    return obs
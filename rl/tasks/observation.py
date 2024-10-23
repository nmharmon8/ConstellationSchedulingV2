import numpy as np

# from rl.tasks.task import EmptyTask

class Observation:

    def __init__(self, config, satellite, task, current_time, window_index):
        self.config = config
        self.current_time = current_time
        self.satellite = satellite
        self.task = task
        self.window_index = window_index
        # ??? 
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
        x, y, z = self.task.r_LP_P
        x = np.cos(x) * np.cos(y)
        y = np.cos(x) * np.sin(y)
        z = np.sin(x)
        obs['x'] = x
        obs['y'] = y
        obs['z'] = z
        obs['window_index'] = self.window_index
        obs['window_index_offset'] = window_index_offset
        obs['priority'] = self.task.priority
        obs['n_required_collects'] = self.task.simultaneous_collects_required
        obs['task_id'] = self.task.id
        obs['task_storage_size'] = self.task.storage_size
        obs['is_data_downlink'] = int(self.task.is_data_downlink)
        obs['is_charge_task'] = int(self.task.is_charge)
        obs['storage_after_task'] = self.satellite.storage_after_task(self.task)
        obs = {**obs, **self.satellite.get_observation()}
        return obs
    
    def get_normalized_observation(self):
        obs = self.observation
        obs_norm_terms = self.config['observation_normalization_terms']
        normalized_obs = {k:obs[k] / obs_norm_terms[k] for k in self.config['observation_keys']}

        # Check if any of the observations are out of bounds -1 to 1
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
        info = {}
        for k, v in self.observation.items():
            info[k] = v
        return info


# class EmptyObservation(Observation):
#     def __init__(self, config, satellite, current_time):
#         super().__init__(config, satellite, EmptyTask(priority=config['noop_task_priority']), current_time, current_time // config['max_step_duration'])


class Observations:

    def __init__(self, current_time, current_tasks_by_sat, satellites, action_def, config, action_index_to_sat):
        self.current_time = current_time
        self.satellites = {sat.id: sat for sat in satellites}
        self.current_tasks_by_sat = current_tasks_by_sat
        self.action_def = action_def
        self.config = config
        self.n_access_windows = config['n_access_windows']
        self._observation_dict = None
        self._tasks = {}
        self.action_index_to_sat = action_index_to_sat
        self.satellite_ordered_ids = [self.action_index_to_sat[i].id for i in range(len(self.action_index_to_sat))]

    def get_observations_dict(self):
        observation_by_sat = {}
        for sat_id in self.satellite_ordered_ids:
            satellite = self.satellites[sat_id]
            observation_by_sat[sat_id] = []
            for window_index, task in self.current_tasks_by_sat[sat_id]:
                self._tasks[task.id] = task 
                observation_by_sat[sat_id].append(Observation(self.config, satellite, task, self.current_time, window_index))
           
            # # Add a charge task  
            # observation_by_sat[sat_id].append(Observation(self.config, satellite, ChargeTask(), self.current_time, 0))
           
            assert len(observation_by_sat[sat_id]) >= self.n_access_windows
            observation_by_sat[sat_id] = sorted(observation_by_sat[sat_id])

            # The last task is guaranteed to be a noop task and we want to insure a noop task is always available
            observation_by_sat[sat_id] = observation_by_sat[sat_id][:self.n_access_windows-1] + observation_by_sat[sat_id][-1:]
            # # Always set the final task in the observation to be the noop task / empty task
            # observation_by_sat[sat_id].append(EmptyObservation(self.config, satellite, self.current_time))

        return observation_by_sat
    
    def get_info_observation_dict(self):
        obs_dict = self.get_observations_dict()
        obs_dict = {k:[x.get_info() for x in v] for k, v in obs_dict.items()}
        return obs_dict
    
    def get_observation_numpy(self):
        all_observations = []
        obs = self.get_observations_dict()
        for sat_id in self.satellite_ordered_ids:
            satellite_observations = []
            for sat_obs in obs[sat_id]:
                satellite_observations.append(sat_obs.get_normalized_observation_numpy())
            all_observations.append(satellite_observations)
        obs = np.stack(all_observations, axis=0)
        return obs
    
    def action_to_task(self):
        # sat_id -> action -> task
        action_to_task = {}
        obs_dict = self.get_observations_dict()
        for sat_id in self.satellite_ordered_ids:
            action_to_task[sat_id] = {}
            for action in self.action_def.get_action_type_indexs('collect'):
                obs = obs_dict[sat_id][action]
                action_to_task[sat_id][action] = obs.task
        return action_to_task
    

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
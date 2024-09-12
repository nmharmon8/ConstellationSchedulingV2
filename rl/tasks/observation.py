import numpy as np

from rl.tasks.task import EmptyTask

class Observation:

    def __init__(self, config, satellite, task, current_time, window_index):
        self.config = config
        self.current_time = current_time
        self.satellite = satellite
        self.task = task
        self.window_index = window_index
        self.observation = self._get_observation()

    def __lt__(self, other):
        self_window_offset = self.observation['window_index_offset']
        other_window_offset = other.observation['window_index_offset']
        
        if self_window_offset != other_window_offset:
            return self_window_offset < other_window_offset
        
        if self.observation['is_data_downlink'] != other.observation['is_data_downlink']:
            return self.observation['is_data_downlink'] > other.observation['is_data_downlink']
        
        return self.observation['priority'] > other.observation['priority']

    def _get_empty_observation(self):
        return {k:self.config['observation_defaults'][k] for k in self.config['observation_keys']}

    def _get_observation(self):
        current_index = int(self.current_time // self.config['max_step_duration'])
        window_index_offset = self.window_index - current_index
        obs = {}
        if self.task.is_task_possible_in_window(self.satellite, self.window_index):
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
            obs = {**obs, **self.satellite.get_observation()}
        else:
            obs = self._get_empty_observation()
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


class EmptyObservation(Observation):
    def __init__(self, config):
        self.config = config
        self.task = EmptyTask()
        self.observation = self._get_empty_observation()

    def get_normalized_observation_numpy(self):
        return np.zeros(len(self.config['observation_keys']))

class Observations:

    def __init__(self, current_time, current_tasks_by_sat, satellites, action_def, config):
        self.current_time = current_time
        self.satellite_ordered_ids = [sat.id for sat in satellites]
        self.satellites = {sat.id: sat for sat in satellites}
        self.current_tasks_by_sat = current_tasks_by_sat
        self.action_def = action_def
        self.config = config
        self.n_access_windows = config['n_access_windows']
        self._observation_dict = None
        self._tasks = {}

    def get_observations_dict(self):
        observation_by_sat = {}
        for sat_id in self.satellite_ordered_ids:
            satellite = self.satellites[sat_id]
            observation_by_sat[sat_id] = []
            for window_index, task in self.current_tasks_by_sat[sat_id]:
                self._tasks[task.id] = task 
                observation_by_sat[sat_id].append(Observation(self.config, satellite, task, self.current_time, window_index))
            while len(observation_by_sat[sat_id]) < self.n_access_windows:
                observation_by_sat[sat_id].append(EmptyObservation(self.config))
            observation_by_sat[sat_id] = sorted(observation_by_sat[sat_id])[:self.n_access_windows]

        # # Print observation by sat
        # for sat_id in self.satellite_ordered_ids:
        #     print(f"Satellite {sat_id}:")
        #     for obs in observation_by_sat[sat_id]:
        #         print(obs)

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
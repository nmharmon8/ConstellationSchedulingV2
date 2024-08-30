from bisect import insort, bisect_left
from collections import defaultdict
import random
import numpy as np

from scipy.optimize import minimize_scalar, root_scalar
from Basilisk.utilities import orbitalMotion

from bsk_rl.utils.orbital import elevation

from pymap3d import ecef2geodetic

def ecef_to_latlon(x, y, z):
    lat, lon, alt = ecef2geodetic(x, y, z)
    return lat, lon, alt

class Task:

    def __init__(self, name, r_LP_P, priority, simultaneous_collects_required=1, task_duration=30.0, max_step_duration=600.0, min_elev=np.radians(45.0)):
        self.name = name
        self.r_LP_P = r_LP_P
        self.priority = priority
        self.min_elev = min_elev
        self.task_duration = task_duration
        self.simultaneous_collects_required = simultaneous_collects_required
        self.max_step_duration = max_step_duration
    
        self.collection_windows = defaultdict(list)
        self.sats_collecting = []

    
        self.successful_collected = False

    @property
    def id(self):
        return f"{self.name}_{id(self)}"

    def step(self):
        self.sats_collecting = []
        
    def collect(self, satellite, collect_start_time, collect_end_time):
        self.sats_collecting.append((satellite, collect_start_time, collect_end_time))

    def count_valid_collections(self):
        valid_collections = 0
        for satellite, collect_start_time, collect_end_time in self.sats_collecting:
            valid_collections += self.get_window(satellite, time=collect_start_time)
        return valid_collections
    
    def get_reward(self):
        reward = 0
        if len(self.sats_collecting) > 0:
            valid_collections = self.count_valid_collections()
            if valid_collections >= self.simultaneous_collects_required:
                if self.simultaneous_collects_required > 1:
                    print(f"Successful collection of task {self.id} with {valid_collections} valid collections")
                self.successful_collected = True
                reward = self.priority
        return reward
    
    def step(self):
        reward = self.get_reward()
        self.sats_collecting = []
        return reward
    
    def add_window(self, satellite, new_window):

        window_start = new_window[0]
        window_end = new_window[1]

        while window_start < window_end:
            index = int(window_start // self.max_step_duration)
            while index >= len(self.collection_windows[satellite.id]):
                self.collection_windows[satellite.id].extend([0] * max(1, len(self.collection_windows[satellite.id])))
            index_end = self.max_step_duration * (index + 1)
            duration = min(index_end, window_end) - window_start
            if duration > self.task_duration:
                self.collection_windows[satellite.id][index] = 1
            window_start = index_end

    def get_window(self, satellite, time=None, window_index=None):
        assert time is not None or window_index is not None
        if time is not None:
            window_index = int(time // self.max_step_duration)
        sat_id = satellite
        if not isinstance(satellite, str):
            sat_id = satellite.id
        if sat_id not in self.collection_windows:
            return 0
        if window_index >= len(self.collection_windows[sat_id]):
            return 0
        return self.collection_windows[sat_id][window_index]

    
    def get_collection_count_for_window(self, window_index):
        return sum([self.get_window(sat_id, window_index=window_index) for sat_id in self.collection_windows.keys()])
    
    def is_task_possible_in_window(self, satellite, window_index):
        if self.get_window(satellite, window_index=window_index) == 1:
            if self.get_collection_count_for_window(window_index) >= self.simultaneous_collects_required:
                return True
        return False
            
    def get_upcoming_windows(self, satellite, start_time):
        """
        Returns the index of the next window for a satellite, or None if there are no upcoming windows
        """
        upcoming_windows = []
        for i in range(int(start_time // self.max_step_duration), len(self.collection_windows[satellite.id])):
            if self.is_task_possible_in_window(satellite, i):
                upcoming_windows.append(i)
        return upcoming_windows
    
    
    def get_observation(self, satellite, current_time, window_index):
        current_index = int(current_time // self.max_step_duration)
        window_index_offset = window_index - current_index

        obs = {}

        if self.is_task_possible_in_window(satellite, window_index):

            # TODO: This could be better
            lat, lon, _ = ecef_to_latlon(self.r_LP_P[0], self.r_LP_P[1], self.r_LP_P[2])
            x = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
            y = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
            z = np.sin(np.radians(lat))

            obs['x'] = x
            obs['y'] = y
            obs['z'] = z
            obs['window_index'] = window_index
            obs['window_index_offset'] = window_index_offset
            obs['priority'] = self.priority
            obs['n_required_collects'] = self.simultaneous_collects_required
            obs['task_id'] = self.id
            # obs = np.array([window_index_offset / 100.0, self.priority, x, y, z])
            # return obs
        
        return obs
    

class Observation:

    def __init__(self, current_time, current_tasks_by_sat, satellites, action_def, config):
        self.current_time = current_time
        self.satellites = satellites
        self.current_tasks_by_sat = current_tasks_by_sat
        self.action_def = action_def
        self.config = config
        self.n_access_windows = config['n_access_windows']
        self._observation_dict = None
        self._tasks = {}

    def get_empty_observation(self):
        return {k:self.config['observation_defaults'][k] for k in self.config['observation_keys']}

    def get_observations_dict(self):
        if self._observation_dict is None:
            observation_by_sat = {}
            for satellite in self.satellites:
                observation_by_sat[satellite.id] = []
                for window_index, task in self.current_tasks_by_sat[satellite.id]:
                    self._tasks[task.id] = task 
                    observation_by_sat[satellite.id].append(task.get_observation(satellite, self.current_time, window_index))
                # Remove empty observations, which are empty dicts
                observation_by_sat[satellite.id] = [obs for obs in observation_by_sat[satellite.id] if obs]

                while len(observation_by_sat[satellite.id]) < self.n_access_windows:
                    observation_by_sat[satellite.id].append(self.get_empty_observation())

            for satellite in self.satellites:
                observation_by_sat[satellite.id] = sorted(observation_by_sat[satellite.id], key=lambda x: (x['window_index_offset'], -1 * x['priority']))
                observation_by_sat[satellite.id] = observation_by_sat[satellite.id][:self.n_access_windows]
            self._observation_dict = observation_by_sat

        return self._observation_dict

    def get_observation_numpy(self):
        obs_norm_terms = self.config['observation_normalization_terms']
        all_observations = []
        obs = self.get_observations_dict()
        for satellite in self.satellites:
            satellite_observations = []
            for sat_obs in obs[satellite.id]:
                satellite_observations.append([sat_obs[key] / obs_norm_terms[key] for key in self.config['observation_keys']])
            all_observations.append(satellite_observations)
        obs = np.stack(all_observations, axis=0)
        return obs
    
    def action_to_task(self):
        # sat_id -> action -> task
        action_to_task = {}
        obs_dict = self.get_observations_dict()
        for sat_id in obs_dict:
            action_to_task[sat_id] = {}
            for action in self.action_def.get_action_type_indexs('collect'):
                action_to_task[sat_id][action] = None
                if action < len(obs_dict[sat_id]):
                    if 'task_id' not in obs_dict[sat_id][action]:
                        # Noop action
                        continue
                    task_id = obs_dict[sat_id][action]['task_id']
                    action_to_task[sat_id][action] = self._tasks[task_id]
        return action_to_task




class TaskManager:

    def __init__(self, satellites, config, action_def):
        self.config = config
        self.max_step_duration = config['max_step_duration']
        self.max_sat_coordination = config['max_sat_coordination']
        self.n_access_windows = config['n_access_windows']

        self.window_calculation_time = 0
        
        self.action_def = action_def
        self.current_tasks_by_sat = {}
        self.satellites = satellites
        self.n_tasks_collected = 0
        self.cumlitive_reward = 0


    def get_observations(self, current_time):
        for sat in self.satellites:
            self.calculate_access_windows(sat,  calculation_start=current_time, duration=self.max_step_duration * self.n_access_windows)

        self.current_tasks_by_sat = {}
        for satellite in self.satellites:
            self.current_tasks_by_sat[satellite.id] = self.get_upcoming_tasks(satellite, current_time)
        self.observation = Observation(current_time, self.current_tasks_by_sat, self.satellites, self.action_def, self.config)
        return self.observation
    

    def reset(self):
        self.n_tasks = random.randint(self.config['min_tasks'], self.config['max_tasks'])
        self.tasks = self.get_random_tasks(self.n_tasks)
        self.current_tasks_by_sat = {}
        observations = self.get_observations(0.0)
        self.n_tasks_collected = 0
        self.cumlitive_reward = 0
        return observations, {}

        
    def step(self, actions, start_time, end_time):

        info = {sat.id: {} for sat in self.satellites}
        task_being_collected = {}

        reward = 0

        action_to_task = self.observation.action_to_task()

        for satellite, action in zip(self.satellites, actions):
            if action not in action_to_task[satellite.id]:
                #Noop action
                continue
            task = action_to_task[satellite.id][action]
            if task is None:
                continue
            task.collect(satellite, start_time, end_time)
            task_being_collected[satellite.id] = task


        for satellite, action in zip(self.satellites, actions):
            if action not in action_to_task[satellite.id]:
                info[satellite.id]['task'] = None
                info[satellite.id]['task_reward'] = 0
                continue
            task = action_to_task[satellite.id][action]
            if task is None:
                info[satellite.id]['task'] = None
                info[satellite.id]['task_reward'] = 0
                continue
            info[satellite.id]['task'] = task
            info[satellite.id]['task_reward'] = task.get_reward()

        for task in self.tasks:
            reward += task.step()

        self.cumlitive_reward += reward
        self.n_tasks_collected += sum(int(task.successful_collected) for task in self.tasks)

        # Remove tasks that have been collected
        self.tasks = [task for task in self.tasks if not task.successful_collected]

        observations = self.get_observations(end_time)

        info['cum_reward'] = self.cumlitive_reward
        info['n_tasks_collected'] = self.n_tasks_collected

        return observations, reward, info


    def get_random_tasks(self, n_targets, radius=orbitalMotion.REQ_EARTH * 1e3):
        tasks = []
        for i in range(n_targets):
            x = np.random.normal(size=3)
            x *= radius / np.linalg.norm(x)
            priority = np.random.rand()
            simultaneous_collects_required = np.random.randint(1, self.max_sat_coordination + 1)
            priority *= simultaneous_collects_required + 1

            # Normalize the priority
            priority = priority / (self.max_sat_coordination + 1)
            tasks.append(Task(f"tgt-{i}", x, max_step_duration=self.max_step_duration, simultaneous_collects_required=simultaneous_collects_required, priority=priority))
        return tasks

    def get_upcoming_tasks(self, satellite, current_time):
        upcoming_tasks = []
        for task in self.tasks:
            windows = task.get_upcoming_windows(satellite, current_time)
            for window_index in windows:
                upcoming_tasks.append((window_index, task))
        return upcoming_tasks
            

    def calculate_access_windows(self, satellite, calculation_start=0.0, duration=180.0):

        if duration <= 0:
            return []

        calculation_end = calculation_start + max(
            duration, satellite.get_dt() * 2, self.max_step_duration
        )
        calculation_end = self.max_step_duration * np.ceil(
            calculation_end / self.max_step_duration
        )
        print(f"Calculating windows from {calculation_start} to {calculation_end}")

        r_BP_P_interp = satellite.get_r_BP_P_interp(calculation_end)
        # print(f"Interpolator: {r_BP_P_interp}")
        window_calc_span = np.logical_and(
            r_BP_P_interp.x >= calculation_start - 1e-9,
            r_BP_P_interp.x <= calculation_end + 1e-9,
        )  # Account for floating point error in window_calculation_time
        times = r_BP_P_interp.x[window_calc_span]
        positions = r_BP_P_interp.y[window_calc_span]
        r_max = np.max(np.linalg.norm(positions, axis=-1))
        access_dist_thresh_multiplier = 1.1

        for task in self.tasks:
            alt_est = r_max - np.linalg.norm(task.r_LP_P)
            access_dist_threshold = (
                access_dist_thresh_multiplier * alt_est / np.sin(task.min_elev)
            )
            candidate_windows = self._find_candidate_windows(
                task.r_LP_P, times, positions, access_dist_threshold
            )
            for candidate_window in candidate_windows:
                roots = self._find_elevation_roots(
                    r_BP_P_interp,
                    task.r_LP_P,
                    task.min_elev,
                    candidate_window,
                )
                new_windows = self._refine_window(
                    roots, candidate_window, (times[0], times[-1])
                )
                for new_window in new_windows:
                    task.add_window(satellite, new_window)

    @staticmethod
    def _find_candidate_windows(
        location: np.ndarray, times: np.ndarray, positions: np.ndarray, threshold: float
    ) -> list[tuple[float, float]]:
        """Find `times` where a window is plausible.

        i.e. where a `positions` point is within `threshold` of `location`. Too big of
        a dt in times may miss windows or produce bad results.
        """
        close_times = np.linalg.norm(positions - location, axis=1) < threshold
        close_indices = np.where(close_times)[0]
        groups = np.split(close_indices, np.where(np.diff(close_indices) != 1)[0] + 1)
        groups = [group for group in groups if len(group) > 0]
        candidate_windows = []
        for group in groups:
            t_start = times[max(0, group[0] - 1)]
            t_end = times[min(len(times) - 1, group[-1] + 1)]
            candidate_windows.append((t_start, t_end))
        return candidate_windows
    
    @staticmethod
    def _find_elevation_roots(
        position_interp,
        location: np.ndarray,
        min_elev: float,
        window: tuple[float, float],
        min_duration: float = 0.1,
    ):
        """Find times where the elevation is equal to the minimum elevation.

        Finds exact times where the satellite's elevation relative to a target is
        equal to the minimum elevation.
        """

        def root_fn(t):
            return -(elevation(position_interp(t), location) - min_elev)

        elev_0, elev_1 = root_fn(window[0]), root_fn(window[1])

        if elev_0 < 0 and elev_1 < 0:
            print("initial_max_step_duration is shorter than the maximum window length; some windows may be neglected.")
            return []
        elif elev_0 < 0 or elev_1 < 0:
            return [root_scalar(root_fn, bracket=window).root]
        else:
            res = minimize_scalar(root_fn, bracket=window, tol=1e-4)
            if res.fun < 0:
                window_mid = res.x
                r_open = root_scalar(root_fn, bracket=(window[0], window_mid)).root
                r_close = root_scalar(root_fn, bracket=(window_mid, window[1])).root
                if r_close - r_open > min_duration:
                    return [r_open, r_close]

        return []
    
    @staticmethod
    def _refine_window(
        endpoints,
        candidate_window,
        computation_window,
    ):
        """Detect if an exact window has been truncated by a coarse window."""
        endpoints = list(endpoints)

        # Filter endpoints that are too close
        for i, endpoint in enumerate(endpoints[0:-1]):
            if abs(endpoint - endpoints[i + 1]) < 1e-6:
                endpoints[i] = None
        endpoints = [endpoint for endpoint in endpoints if endpoint is not None]

        # Find pairs
        if len(endpoints) % 2 == 1:
            if candidate_window[0] == computation_window[0]:
                endpoints.insert(0, computation_window[0])
            elif candidate_window[-1] == computation_window[-1]:
                endpoints.append(computation_window[-1])
            else:
                return []  # Temporary fix for rare issue.

        new_windows = []
        for t1, t2 in zip(endpoints[0::2], endpoints[1::2]):
            new_windows.append((t1, t2))

        return new_windows

    
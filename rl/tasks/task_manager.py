import random
import numpy as np

from scipy.optimize import minimize_scalar, root_scalar
from Basilisk.utilities import orbitalMotion

from bsk_rl.utils.orbital import elevation

from rl.tasks.task import Task
from rl.tasks.observation import Observations

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
        self.observation = Observations(current_time, self.current_tasks_by_sat, self.satellites, self.action_def, self.config)
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
        info = {sat.id: sat.get_observation() for sat in self.satellites}
        reward = 0
        action_to_task = self.observation.action_to_task()

        for satellite, action in zip(self.satellites, actions):
            task = action_to_task[satellite.id][action]
            task.collect(satellite, start_time, end_time)

        for satellite, action in zip(self.satellites, actions):
            task = action_to_task[satellite.id][action]
            info[satellite.id]['task'] = task.task_info()
            info[satellite.id]['task_reward'] = task.get_reward()
            self.n_tasks_collected += 1 if task.task_complete else 0

        for task in self.tasks:
            reward += task.step()

        self.cumlitive_reward += reward
        
        # Remove tasks that have been completed
        self.tasks = [task for task in self.tasks if not task.task_complete]
        observations = self.get_observations(end_time)

        info['cum_reward'] = self.cumlitive_reward
        info['n_tasks_collected'] = self.n_tasks_collected
        return observations, reward, info


    def get_random_tasks(self, n_targets, radius=orbitalMotion.REQ_EARTH * 1e3):
        tasks = []
        for i in range(n_targets):
            tasks.append(Task.create_random_task(self.config))
        tasks.extend(Task.create_data_downlink_tasks(self.config))
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

    
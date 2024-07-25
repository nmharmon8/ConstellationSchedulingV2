from bisect import insort, bisect_left
import random
import numpy as np

from scipy.optimize import minimize_scalar, root_scalar
from Basilisk.utilities import orbitalMotion

from bsk_rl.utils.orbital import elevation

class Task:

    def __init__(self, name, r_LP_P, priority, simultaneous_collects_required=1, task_duration=30.0, min_elev=np.radians(45.0)):
        self.name = name
        self.r_LP_P = r_LP_P
        self.priority = priority
        self.min_elev = min_elev
        self.task_duration = task_duration
        self.simultaneous_collects_required = simultaneous_collects_required

        self.sat_windows = {}
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
            for window in self.sat_windows[satellite.id]:
                # Calculate the overlaping time between the collection window and the task window
                overlap_start = max(collect_start_time, window[0])
                overlap_end = min(collect_end_time, window[1])
                overlap_duration = overlap_end - overlap_start
                if overlap_duration >= self.task_duration:
                    # If the overlap duration is greater than the task duration, then the task is valid
                    valid_collections += 1
                    break
        return valid_collections
    
    def step(self):
        reward = 0
        if len(self.sats_collecting) > 0:
            valid_collections = self.count_valid_collections()
            if valid_collections >= self.simultaneous_collects_required:
                self.successful_collected = True
                reward = self.priority
        self.sats_collecting = []
        return reward
    
    def add_window(self, satellite, new_window):
        if satellite.id not in self.sat_windows:
            self.sat_windows[satellite.id] = []
            self.sat_windows[satellite.id].append(new_window)
            return
        windows = self.sat_windows[satellite.id]
        # Find the position where the new window should be inserted
        pos = bisect_left(windows, new_window[0], key=lambda x: x[0])
        # Check for overlap and merge
        to_remove = []
        for i in range(max(0, pos - 1), min(len(windows), pos + 1)):
            if self._windows_overlap(windows[i], new_window):
                new_window = self._merge_windows(windows[i], new_window)
                to_remove.append(i)
        # Remove the overlapped windows
        for i in reversed(to_remove):
            windows.pop(i)
        # Insert the merged window
        insort(windows, new_window, key=lambda x: x[0])

    def _windows_overlap(self, window1, window2):
        return max(window1[0], window2[0]) <= min(window1[1], window2[1])

    def _merge_windows(self, window1, window2):
        return (min(window1[0], window2[0]), max(window1[1], window2[1]))

    def get_upcoming_window(self, sat_id, current_time):
        """
        Returns the next window for a satellite, or None if there are no upcoming windows
        """
        if sat_id in self.sat_windows:
            for window in self.sat_windows[sat_id]:
                if window[0] >= current_time:
                    return window
        return None
    
    def get_observation(self, sat_id, current_time):
        window = self.get_upcoming_window(sat_id, current_time)
        if window is not None:
            start_time, end_time = window
            start_time = start_time - current_time
            end_time = end_time - current_time
            return np.array([start_time / 100000.0, end_time / 100000.0, self.priority])
        return None


class TaskManager:

    def __init__(self, min_tasks=300, max_tasks=3000, max_step_duration=600.0, **kwargs):
        self.n_tasks = random.randint(min_tasks, max_tasks)
        self.tasks = self.get_random_tasks(self.n_tasks)
        self.window_calculation_time = 0
        self.generation_duration = max_step_duration

    def step(self):
        reward = 0
        for task in self.tasks:
            reward += task.step()
        # Remove tasks that have been collected
        self.tasks = [task for task in self.tasks if not task.successful_collected]
        return reward

    def get_random_tasks(self, n_targets, radius=orbitalMotion.REQ_EARTH * 1e3):
        tasks = []
        for i in range(n_targets):
            x = np.random.normal(size=3)
            x *= radius / np.linalg.norm(x)
            tasks.append(Task(f"tgt-{i}", x, np.random.rand()))
        return tasks

    def get_upcoming_tasks(self, satellite, current_time):
        upcoming_tasks = []
        for task in self.tasks:
            window = task.get_upcoming_window(satellite.id, current_time)
            if window is not None:
                upcoming_tasks.append((window[0],task))
        upcoming_tasks.sort(key=lambda x: x[0])
        upcoming_tasks = [task for _, task in upcoming_tasks]
        return upcoming_tasks
            

    def calculate_access_windows(self, satellite, calculation_start=0.0, duration=180.0):

        if duration <= 0:
            return []

        calculation_end = calculation_start + max(
            duration, satellite.get_dt() * 2, self.generation_duration
        )
        calculation_end = self.generation_duration * np.ceil(
            calculation_end / self.generation_duration
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
                    # print(f"New window: {new_window} task_id: {task.id} sat_id: {satellite.id}")
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
            print("initial_generation_duration is shorter than the maximum window length; some windows may be neglected.")
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

    
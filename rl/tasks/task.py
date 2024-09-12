from collections import defaultdict
import uuid
import numpy as np

from Basilisk.utilities import orbitalMotion
from pymap3d import ecef2geodetic
from bsk_rl.utils.orbital import lla2ecef

def ecef_to_latlon(x, y, z):
    lat, lon, alt = ecef2geodetic(x, y, z)
    return lat, lon, alt

class TaskingType:
    RF = 0
    IMAGING = 1
    DATA_DOWNLINK = 2
    NOOP = 3

    @staticmethod
    def to_str(task_type):
        if task_type == TaskingType.RF:
            return "RF"
        elif task_type == TaskingType.IMAGING:
            return "IMAGING"
        elif task_type == TaskingType.DATA_DOWNLINK:
            return "DATA_DOWNLINK"
        elif task_type == TaskingType.NOOP:
            return "NOOP"

class Task:

    def __init__(self, name, r_LP_P, priority, simultaneous_collects_required, task_duration, tasking_type, storage_size, max_step_duration, min_elev):
        self.name = name
        # Task Data
        self.r_LP_P = r_LP_P
        self.priority = priority
        self.min_elev = min_elev
        self.task_duration = task_duration
        self.simultaneous_collects_required = simultaneous_collects_required
        self.tasking_type = tasking_type
        self.storage_size = storage_size # In MB
        self.max_step_duration = max_step_duration
        self.collection_windows = defaultdict(list)
        self.sats_collecting = []
        self._task_complete = False

    def task_info(self):
        info = {
            'id': self.id,
            'r_LP_P': [float(x) for x in self.r_LP_P],
            'priority': self.priority,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'priority': self.priority,
            'min_elev': self.min_elev,
            'tasking_type': TaskingType.to_str(self.tasking_type),
            'simultaneous_collects_required': self.simultaneous_collects_required,
            'storage_size': self.storage_size,
            'is_data_downlink': bool(self.is_data_downlink),
            'task_duration': self.task_duration,
            'sats_collecting': [sat.id for sat, _, _ in self.sats_collecting],
            'task_complete': bool(self.task_complete),
            'task_reward': self.get_reward()
        }
        return info
            
    @staticmethod
    def create_random_task(config, radius=orbitalMotion.REQ_EARTH * 1e3):
        # Location
        x = np.random.normal(size=3)
        x *= radius / np.linalg.norm(x)
        
        # Simultaneous Collects Required
        simultaneous_collects_required = np.random.randint(1, config['max_sat_coordination'] + 1)
        
        # Priority
        priority = np.random.rand()
        priority *= simultaneous_collects_required + 1
        priority = priority / (config['max_sat_coordination'] + 1)

        #Storage Size
        task_min_storage_size = config['task_min_storage_size']
        task_max_storage_size = config['task_max_storage_size']
        storage_size = np.random.rand() * (task_max_storage_size - task_min_storage_size) + task_min_storage_size

        #Task Duration
        task_duration_min = config['task_min_duration']
        task_duration_max = config['task_max_duration']
        task_duration = np.random.rand() * (task_duration_max - task_duration_min) + task_duration_min

        #Tasking Type
        tasking_type = np.random.choice([TaskingType.RF, TaskingType.IMAGING])

        return Task(
            name=f"tgt-{uuid.uuid4()}",
            r_LP_P=x,
            priority=priority,
            simultaneous_collects_required=simultaneous_collects_required,
            task_duration=task_duration,
            tasking_type=tasking_type,
            storage_size=storage_size,
            max_step_duration=config['max_step_duration'],
            min_elev=config['task_min_elev']
        )

    @staticmethod
    def create_data_downlink_tasks(config, radius=orbitalMotion.REQ_EARTH * 1e3):
        ground_stations = config['groundStations']
        ground_station_tasks = []
        for station in ground_stations:
            # Position
            # Elevation is not being used? Part of radius?
            position = lla2ecef(station['lat'], station['long'], radius)
            
            # Priority
            priority = 0
            
            # Storage Size
            storage_size = -1

            # Task Duration
            task_duration = 30

            # Simultaneous Collects Required
            simultaneous_collects_required = 1

            # Tasking Type
            tasking_type = TaskingType.DATA_DOWNLINK

            task = Task(
                name=f"tgt-{station['name']}",
                r_LP_P=position,
                priority=priority,
                simultaneous_collects_required=simultaneous_collects_required,
                task_duration=task_duration,
                tasking_type=tasking_type,
                storage_size=storage_size,
                max_step_duration=config['max_step_duration'],
                min_elev=config['task_min_elev']
            )
            ground_station_tasks.append(task)
        return ground_station_tasks
        

    @property
    def id(self):
        return f"{self.name}_{id(self)}"

    @property
    def is_data_downlink(self):
        return self.tasking_type == TaskingType.DATA_DOWNLINK
    
    @property
    def task_complete(self):
        return self._task_complete
    
    @property
    def latitude(self):
        lat, _, _ = ecef_to_latlon(self.r_LP_P[0], self.r_LP_P[1], self.r_LP_P[2])
        return float(lat)
    
    @property
    def longitude(self):
        _, lon, _ = ecef_to_latlon(self.r_LP_P[0], self.r_LP_P[1], self.r_LP_P[2])
        return float(lon)
    
    @property
    def altitude(self):
        _, _, alt = ecef_to_latlon(self.r_LP_P[0], self.r_LP_P[1], self.r_LP_P[2])
        return float(alt)
    

    def step(self):
        reward = self.get_reward()
        if self.is_collection_valid():
            success_count = 0
            # Let the satellites know they did this task
            for satellite, _, _ in self.sats_collecting:
                if satellite.can_complete_task(self):
                    satellite.task_completed(self)
                    success_count += 1
                else:
                    satellite.task_failed(self)

            if not self.is_data_downlink and success_count >= self.simultaneous_collects_required:
                self._task_complete = True
            elif not self.is_data_downlink:
                print("Failed task due to stats not having storage, this was not expected at this point")
                raise Exception("Failed task due to stats not having storage, this was not expected at this point")
        else:
            # Let the satellites know they failed this task
            for satellite, _, _ in self.sats_collecting:
                satellite.task_failed(self)

        self.sats_collecting = []
        return reward   
        
    def collect(self, satellite, collect_start_time, collect_end_time):
        satellite.task_started(self)
        self.sats_collecting.append((satellite, collect_start_time, collect_end_time))

    def count_valid_collections(self):
        valid_collections = 0
        for satellite, collect_start_time, collect_end_time in self.sats_collecting:
            # Checks things like storage
            if satellite.can_complete_task(self):
                valid_collections += self.get_window(satellite, time=collect_start_time)
        return valid_collections
    
    def is_collection_valid(self):
        # Only used to validate collection, not for filtering task in observation
        if self.count_valid_collections() >= self.simultaneous_collects_required:
            return True
        return False
    
    def get_reward(self):
        reward = 0
        if self.is_collection_valid():
            reward = self.priority
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
        # This will return true even if the sats don't have enough storage
        # That is requried so that AI can plan around storage
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


class EmptyTask(Task):
    def __init__(self):
        super().__init__("EmptyTask", r_LP_P=(0, 0, 0), priority=0, simultaneous_collects_required= 0, task_duration=0, tasking_type=TaskingType.NOOP, storage_size=0, max_step_duration=0, min_elev=0)

    @property
    def is_data_downlink(self):
        return False
    
    @property
    def task_complete(self):
        return True
    
    def get_reward(self):
        return 0
    
    def is_task_possible_in_window(self, satellite, window_index):
        return False
    
    def collect(self, satellite, start_time, end_time):
        pass

    def step(self):
        return 0
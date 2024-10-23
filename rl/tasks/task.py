from collections import defaultdict
import uuid
import numpy as np

from Basilisk.utilities import orbitalMotion
from pymap3d import ecef2geodetic
from bsk_rl.utils.orbital import lla2ecef

def ecef_to_latlon(x, y, z):
    lat, lon, alt = ecef2geodetic(x, y, z)
    return lat, lon, alt

class TaskType:
    RF = 0
    IMAGING = 1
    DATA_DOWNLINK = 2
    NOOP = 3
    CHARGE = 4
    DESAT = 5

    @staticmethod
    def to_str(task_type):
        if task_type == TaskType.RF:
            return "RF"
        elif task_type == TaskType.IMAGING:
            return "IMAGING"
        elif task_type == TaskType.DATA_DOWNLINK:
            return "DATA_DOWNLINK"
        elif task_type == TaskType.CHARGE:
            return "CHARGE"
        elif task_type == TaskType.NOOP:
            return "NOOP"
        elif task_type == TaskType.DESAT:
            return "DESAT"


class Task:

    def __init__(self, name, task_type):
        self.name = name
        self.task_type = task_type
        self.sats_collecting = []

        # Required for observation
        self.simultaneous_collects_required = 0
        self.storage_size = 0
        self.r_LP_P = np.zeros(3)

    @property
    def id(self):
        return f"{self.name}_{id(self)}"
    
    @property
    def is_data_downlink(self):
        return self.task_type == TaskType.DATA_DOWNLINK
    
    @property
    def is_noop(self):
        return self.task_type == TaskType.NOOP
    
    @property
    def is_charge(self):
        return self.task_type == TaskType.CHARGE
    
    @property
    def is_desat(self):
        return self.task_type == TaskType.DESAT
    
    @property
    def is_collection(self):
        return self.task_type == TaskType.RF or self.task_type == TaskType.IMAGING
    
    @property
    def is_access_task(self):
        """
        The task manager will cacluate if the task is acceable
        This is based on the position of the sat and the elevation angle
        Any access task must have a min_elev and a r_LP_P
        """
        return self.is_collection or self.is_data_downlink
    
    def get_window(self, satellite, window_index):
        assert not self.is_access_task, "Must implement get window for access tasks"
        return 1
    
    def collect(self, satellite, collect_start_time, collect_end_time):
        satellite.task_started(self)
        self.sats_collecting.append((satellite, collect_start_time, collect_end_time))

    def reset(self):
        self.sats_collecting = []

    def task_info(self):
        info = {
            'id': self.id,
            'task_type': TaskType.to_str(self.task_type),
            'sats_collecting': [sat.id for sat, _, _ in self.sats_collecting],
            'is_data_downlink': self.is_data_downlink,
            'is_noop': self.is_noop,
            'is_charge': self.is_charge,
            'is_collection': self.is_collection,
            'is_desat': self.is_desat,
            # Default values
            'priority': 0,
            'simultaneous_collects_required': 0,
            'storage_size': 0,
            'is_data_downlink': False,
            'task_duration': 0,
            'task_complete': False,
            'task_reward': 0,
        }
        return info

    


class PositionTask(Task):

    def __init__(self, name, task_type, r_LP_P, task_duration, max_step_duration, min_elev):

        Task.__init__(self, name, task_type)

        self.r_LP_P = r_LP_P

        self.min_elev = min_elev
        self.task_duration = task_duration

        self.max_step_duration = max_step_duration
        self.collection_windows = defaultdict(list)
        self.r_LP_P = r_LP_P

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
            
    def get_upcoming_windows(self, satellite, start_time):
        """
        Returns the index of the next window for a satellite, or None if there are no upcoming windows
        """
        upcoming_windows = []
        for i in range(int(start_time // self.max_step_duration), len(self.collection_windows[satellite.id])):
            if self.is_task_possible_in_window(satellite, i):
                upcoming_windows.append(i)
        return upcoming_windows
    
    def task_info(self):
        info = Task.task_info(self)
        info.update({
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'min_elev': self.min_elev,
        })
        
        return info

class CollectTask(PositionTask):

    def __init__(self, name, r_LP_P, priority, simultaneous_collects_required, task_duration, task_type, storage_size, max_step_duration, min_elev):
        PositionTask.__init__(self, name, task_type, r_LP_P, task_duration, max_step_duration, min_elev)

        self.priority = priority
        self.simultaneous_collects_required = simultaneous_collects_required
        self.storage_size = storage_size # In MB
        self._task_complete = False

    def task_info(self):
        info = PositionTask.task_info(self)
        info.update({
            'priority': self.priority,
            'simultaneous_collects_required': self.simultaneous_collects_required,
            'storage_size': self.storage_size,
            'task_duration': self.task_duration,
            'task_complete': bool(self.task_complete),
            'task_reward': self.get_reward(),
        })
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
        task_type = np.random.choice([TaskType.RF, TaskType.IMAGING])

        return CollectTask(
            name=f"tgt-{uuid.uuid4()}",
            r_LP_P=x,
            priority=priority,
            simultaneous_collects_required=simultaneous_collects_required,
            task_duration=task_duration,
            task_type=task_type,
            storage_size=storage_size,
            max_step_duration=config['max_step_duration'],
            min_elev=config['task_min_elev']
        )
    

    @property
    def task_complete(self):
        return self._task_complete    

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

        self.reset()
        return reward   
        
    def count_valid_collections(self):
        valid_collections = 0
        for satellite, collect_start_time, collect_end_time in self.sats_collecting:
            # Checks things like storage
            if satellite.can_complete_task(self):
                valid_collections += self.get_window(satellite, time=collect_start_time)
        return valid_collections
    
    def is_task_possible_in_window(self, satellite, window_index):
        # This will return true even if the sats don't have enough storage/charge
        # That is requried so that AI can plan around storage
        if self.get_window(satellite, window_index=window_index) == 1:
            if self.get_collection_count_for_window(window_index) >= self.simultaneous_collects_required:
                return True
        return False
    
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
    




    
class DownlinkTask(PositionTask):
    def __init__(self, name, r_LP_P, priority, task_duration, max_step_duration, min_elev):
        PositionTask.__init__(self, name, TaskType.DATA_DOWNLINK, r_LP_P, task_duration, max_step_duration, min_elev)
        self.priority = priority
        self.sats_collecting = []

    @staticmethod
    def create_data_downlink_tasks(config, radius=orbitalMotion.REQ_EARTH * 1e3):
        ground_stations = config['groundStations']
        ground_station_tasks = []
        for station in ground_stations:
            # Position
            # Elevation is not being used? Part of radius?
            position = lla2ecef(station['lat'], station['long'], radius)
            
            # Task Duration
            task_duration = 30

            task = DownlinkTask(
                name=f"tgt-{station['name']}",
                r_LP_P=position,
                priority=config['downlink_task_priority'],
                task_duration=task_duration,
                max_step_duration=config['max_step_duration'],
                min_elev=config['task_min_elev']
            )
            
            ground_station_tasks.append(task)
        return ground_station_tasks
    
    @property
    def task_complete(self):
        return False 
    
    def get_reward(self):
        reward = self.priority * self.count_valid_collections()
        return reward

    def step(self):
        reward = self.get_reward()
        self.reset()
        return reward   

    def count_valid_collections(self):
        valid_collections = 0
        for satellite, collect_start_time, collect_end_time in self.sats_collecting:
            # Checks things like storage
            if satellite.can_complete_task(self):
                valid_collections += self.get_window(satellite, time=collect_start_time)
        print(f"Valid collections for Downlink: {valid_collections}")
        return valid_collections
    
    def is_task_possible_in_window(self, satellite, window_index):
        # This will return true even if the sats don't have enough storage/charge
        # That is requried so that AI can plan around storage
        if self.get_window(satellite, window_index=window_index) == 1:
            return True
        return False
    
    def task_info(self):
        info = PositionTask.task_info(self)
        info.update({
            'priority': self.priority,
            'simultaneous_collects_required': 0,
            'storage_size': -1,
            'task_duration': self.task_duration,
            'task_complete': bool(self.task_complete),
            'task_reward': self.get_reward(),
        })
        return info
    

class ChargeTask(Task):
    def __init__(self, priority=0, task_duration=0):
        super().__init__("ChargeTask", TaskType.CHARGE)
        self.priority = priority
        self.task_duration = task_duration

    @staticmethod
    def create_charge_task(config):
        return ChargeTask(priority=config['charge_task_priority'], task_duration=30)

    @property
    def task_complete(self):
        return False 
    
    def get_reward(self):
        reward = self.priority * self.count_valid_collections()
        return reward

    def step(self):
        reward = self.get_reward()
        self.reset()
        return reward   

    def count_valid_collections(self):
        # TODO: Check if a sat is in eclipse
        return len(self.sats_collecting)
    
    def is_task_possible_in_window(self, satellite, window_index):
        # Why do i have to valid check methods?
        return True
    
    def task_info(self):
        info = super().task_info()
        info.update({
            'priority': self.priority,
            'simultaneous_collects_required': 0,
            'storage_size': 0,
            'task_duration': self.task_duration,
            'task_complete': bool(self.task_complete),
            'task_reward': self.get_reward(),
        })
        return info


class DesatTask(Task):
    def __init__(self, priority=0, task_duration=0):
        super().__init__("DesatTask", TaskType.DESAT)
        self.priority = priority
        self.task_duration = task_duration

    @staticmethod
    def create_desat_task(config):
        return DesatTask(priority=config['desat_task_priority'], task_duration=30)

    @property
    def task_complete(self):
        return False
    
    def get_reward(self):
        return self.priority * len(self.sats_collecting)
    
    def count_valid_collections(self):
        # I don't think this is dependent of the env, just the sat
        return len(self.sats_collecting)
    
    def is_task_possible_in_window(self, satellite, window_index):
        return True
    
    def step(self):
        reward = self.get_reward()
        self.reset()
        return reward
    
    
    def task_info(self):
        info = super().task_info()
        info.update({
            'priority': self.priority,
            'simultaneous_collects_required': 0,
            'storage_size': 0,
            'task_duration': self.task_duration,
            'task_complete': bool(self.task_complete),
            'task_reward': self.get_reward(),
        })
        return info
    

class EmptyTask(Task):
    def __init__(self, priority):
        super().__init__("EmptyTask",  task_type=TaskType.NOOP)
        self.priority = priority
 
    @property
    def task_complete(self):
        return False
    
    def get_reward(self):
        return self.priority * len(self.sats_collecting)
    
    def is_task_possible_in_window(self, satellite, window_index):
        return True

    def step(self):
        reward = self.get_reward()
        self.reset()
        return reward

import numpy as np
from scipy import interpolate

def interpolate_lat_lon(lats, lons, times, num_interpolation_points):
    """
        Interpolate the latitude, longitude and time
    """
    lats = np.array(lats)
    lons = np.array(lons)
    times = np.array(times)

    lon_diff = np.diff(lons)
    lon_diff[lon_diff > 180] -= 360
    lon_diff[lon_diff < -180] += 360
    unwrapped_lons = np.concatenate(([lons[0]], lons[0] + np.cumsum(lon_diff)))

    # Create interpolation functions
    lat_interp = interpolate.interp1d(times, lats, kind='linear')
    lon_interp = interpolate.interp1d(times, unwrapped_lons, kind='linear')

    # Generate interpolated points
    interp_times = np.linspace(times.min(), times.max(), num_interpolation_points)
    interp_lats = lat_interp(interp_times)
    interp_lons = lon_interp(interp_times) % 360
    interp_lons = np.where(interp_lons > 180, interp_lons - 360, interp_lons)
    return list(zip(interp_lats.tolist(), interp_lons.tolist())) #, interp_times.tolist()

def interpolate_value(init_val, end_val, num_interpolation_points):
    return np.linspace(init_val, end_val, num_interpolation_points)


class TaskInfo:
    def __init__(self, task):
        """
            {
                "id":"ChargeTask_140048150397376",
                "task_type":"CHARGE",
                "sats_collecting":[
                
                ],
                "is_data_downlink":false,
                "is_noop":false,
                "is_charge":true,
                "is_collection":false,
                "is_desat":false,
                "priority":0.0,
                "simultaneous_collects_required":0,
                "storage_size":0,
                "task_duration":30,
                "task_complete":false,
                "task_reward":0.0
            },
            {
                "id":"tgt-41ec1f48-5588-45cc-9e5a-5d50c68c2b10_140048150490720",
                "task_type":"IMAGING",
                "sats_collecting":[
                
                ],
                "is_data_downlink":false,
                "is_noop":false,
                "is_charge":false,
                "is_collection":true,
                "is_desat":false,
                "priority":0.8652114629106292,
                "simultaneous_collects_required":1,
                "storage_size":959.5577308362037,
                "task_duration":118.21736536831808,
                "task_complete":false,
                "task_reward":0,
                "latitude":-23.0307336555097,
                "longitude":-171.1124923108761,
                "altitude":3249.478072575009,
                "min_elev":0.7853981633974483
            },
        """
        self.task = task

    @property
    def lat(self):
        if "latitude" in self.task:
            return self.task["latitude"]
        return None
    
    @property
    def lon(self):
        if "longitude" in self.task:
            return self.task["longitude"]
        return None
    
    @property
    def priority(self):
        if "priority" in self.task:
            return self.task["priority"]
        return None
    
    @property
    def task_type(self):
        if "task_type" in self.task:
            return self.task["task_type"]
        return None
    
    @property
    def sats_collecting(self):
        if "sats_collecting" in self.task:
            return self.task["sats_collecting"]
        return None

    @property
    def task_complete(self):
        if "task_complete" in self.task:
            return self.task["task_complete"]
        return None

class SatelliteInfo:
    def __init__(self, satellite):
        """
        {
            "r_BP_P":"array("[
                -6713649.6056056,
                -1204165.62487742,
                -2213624.88263376
            ]")",
            "lat":-18.08106070202639,
            "lon":-169.83150382712105,
            "alt":794908.6799137916,
            "observation":{
                "is_alive":true,
                "storage_level":23046293376.0,
                "storage_capacity":40000000000,
                "storage_percentage":0.5761573344,
                "power_level":349756.30343348044,
                "power_capacity":720000.0,
                "power_percentage":0.48577264365761175,
                "in_eclipse":true,
                "next_eclipse":4380.0,
                "end_of_eclipse":450.0
            }
        }
        """
        self.satellite = satellite

    @property
    def lat(self):
        return self.satellite["lat"]
    
    @property
    def lon(self):
        return self.satellite["lon"]
    
    @property
    def storage_percentage(self):
        return self.satellite["observation"]["storage_percentage"]
    
    @property
    def power_percentage(self):
        return self.satellite["observation"]["power_percentage"]
    
    @property
    def in_eclipse(self):
        return self.satellite["observation"]["in_eclipse"]
    
class ObservationsInfo:
    def __init__(self, observations):
        self.observations = observations
        self.tasks = [TaskInfo(task) for task in observations]

    def get_task_json(self):
        return [task.task for task in self.tasks]


class StepInfo:
    def __init__(self, step):
        self.step = step
        self.actions = self.step["actions"]
        self.start_time = self.step["start_time"]
        self.end_time = self.step["end_time"]
        # This is the order that the actions are listed in
        # so sat_ids[0] is taking actions[0]
        self.sat_ids = self.step["satellite_in_order"]
        self.init_observation = {sat_id: ObservationsInfo(obs) for sat_id, obs in self.step["init_observation"].items()}
        self.init_action_tasks = {sat_id: TaskInfo(task) for sat_id, task in self.step["init_action_tasks"].items()}
        self.init_satellites = {sat_id: SatelliteInfo(sat) for sat_id, sat in self.step["init_satellites"].items()}
        self.end_observation = {sat_id: ObservationsInfo(obs) for sat_id, obs in self.step["end_observation"].items()}
        self.end_action_tasks = {sat_id: TaskInfo(task) for sat_id, task in self.step["end_action_tasks"].items()}
        self.end_satellites = {sat_id: SatelliteInfo(sat) for sat_id, sat in self.step["end_satellites"].items()}
        self.reward = self.step["reward"]


    def get_sat_ids(self):
        return list(self.init_satellites.keys())
    
    def get_interpolated_sat_positions(self, interpolation_factor=80):
        sat_positions = {}
        for sat_id in self.get_sat_ids():
            sat_positions[sat_id] = interpolate_lat_lon(
                (self.init_satellites[sat_id].lat, self.end_satellites[sat_id].lat),
                (self.init_satellites[sat_id].lon, self.end_satellites[sat_id].lon),
                (self.start_time, self.end_time), interpolation_factor)
        return sat_positions
    
    def get_current_sat_state(self):
        return {sat_id: self.end_satellites[sat_id].satellite for sat_id in self.get_sat_ids()}
    
    def get_current_actions_and_observations(self):
        """
        This will use the init observation as that is what was used to create the current actions
        """
        
        return  {
            "sat_to_act": {sat_id: act for sat_id, act in zip(self.sat_ids, self.actions)},
            "sat_to_tasks": {sat_id: v.get_task_json() for sat_id, v in self.init_observation.items()}
        }
    
    def get_reward(self):
        return self.reward

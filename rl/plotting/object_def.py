"""
task = {
    "id":"tgt-dbea8e72-4967-43a9-ac54-506242c2535d_139545844771344",
    "r_LP_P":[
    6004288.904821746,
    598295.0005286315,
    2066684.38036302
    ],
    "priority":0.16857640481509484,
    "latitude":19.02484232142649,
    "longitude":5.690431446207682,
    "altitude":2254.9848913863793,
    "min_elev":0.7853981633974483,
    "tasking_type":"IMAGING",
    "simultaneous_collects_required":3,
    "storage_size":911.3346876820792,
    "is_data_downlink":false,
    "task_duration":86.08906064810631,
    "sats_collecting":[
        "id1", "id2", "id3"
    ],
    "task_complete":false,
    "task_reward":0
}

obs = {
    "x":0.584855081287172,
    "y":0.6727108667560088,
    "z":0.5520785560498054,
    "window_index":7,
    "window_index_offset":7,
    "priority":0,
    "n_required_collects":1,
    "task_id":"tgt-Dongara_139545844802864",
    "task_storage_size":-1,
    "is_data_downlink":1,
    "storage_level":0.0,
    "storage_capacity":1000,
    "storage_percentage":0.0
}

satellite = {
    "time":200.0,
    "latitude":-37.54195407133745,
    "longitude":15.972944683426267,
    "task_being_collected": task,
    "task_reward":0,
    "actions":1,
    "action_type":"collect",
    "reward":1.9197098207057113,
    "observation":[
        obs1, obs2, obs3, ...
    ],
    "storage_level":37655108188.0,
    "storage_capacity":40000000000,
    "storage_percentage":0.9413777047
}

step = {
    "step":0,
    "cum_reward":1.9197098207057113,
    "n_tasks_collected":0,
    "satellites":{
        "id1":satellite1,
        "id2":satellite2,
        "id3":satellite3,
        ...
    }
}
    
Json input format:
{ 
    tasks:[task1, task2, task3, ...],
    steps:[step1, step2, step3, ...]
}
"""

import json
import numpy as np
from scipy import interpolate
from copy import deepcopy

from rl.plotting.task_object_def import Task



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
    return interp_lats, interp_lons, interp_times

def interpolate_value(init_val, end_val, num_interpolation_points):
    return np.linspace(init_val, end_val, num_interpolation_points)
    
class ObjectDef:

    def __init__(self, json_file, interpolation_factor=1):
        self.json_file = json_file
        self.data = json.load(open(json_file))
        # For each step how many steps should be interpolated so if interpolation_factor is 2, then each step is 2 steps
        self.interpolation_factor = interpolation_factor
        self.interpolate_data()

    def interpolate_data(self):
        """
            Interpolate steps, the value that is interpolated is the latitude, longitude and time
        """

        values_to_interpolate = ["storage_percentage"]

        steps = self.data["steps"]
        interpolated_steps = []
        for cur_step, next_step in zip(steps[:-1], steps[1:]):
            new_steps = [deepcopy(cur_step) for _ in range(self.interpolation_factor)]
            for sat_id, satellite in cur_step["satellites"].items():
                inter_lats, inter_lons, inter_times = interpolate_lat_lon(
                    (satellite["latitude"], next_step['satellites'][sat_id]["latitude"]), 
                    (satellite["longitude"], next_step['satellites'][sat_id]["longitude"]), 
                    (satellite["time"], next_step['satellites'][sat_id]["time"]), 
                    self.interpolation_factor)
                inter_values = {}
                for k in values_to_interpolate:
                    inter_values[k] = interpolate_value(satellite[k], next_step['satellites'][sat_id][k], self.interpolation_factor)
                
                for i in range(self.interpolation_factor):
                    new_steps[i]["satellites"][sat_id]["latitude"] = inter_lats[i]
                    new_steps[i]["satellites"][sat_id]["longitude"] = inter_lons[i]
                    new_steps[i]["satellites"][sat_id]["time"] = inter_times[i]
                    for k in values_to_interpolate:
                        new_steps[i]["satellites"][sat_id][k] = inter_values[k][i]

            interpolated_steps.extend(new_steps)
        self.data["steps"] = interpolated_steps


    def get_targets(self):
        return self.data["targets"]

    def get_steps(self):
        return self.data["steps"]
    
    def get_number_of_frames(self):
        return len(self.data["steps"])
    
    def get_satellite_ids(self):
        return list(self.data["steps"][0]["satellites"].keys())
    
    def get_number_of_satellites(self):
        return len(self.data["steps"][0]["satellites"])
    
    def get_number_of_observations(self):
        return len(list(self.data["steps"][0]["satellites"].values())[0]["observation"])
    
    @property
    def tasks(self):
        return [Task(task) for task in self.data["tasks"]]
    
    def get_satellites_at_step(self, frame):
        return self.data["steps"][frame]["satellites"]
    
    def get_action(self, satellite_id, frame):
        return self.data["steps"][frame]["satellites"][satellite_id]["action"]
    
    def get_action_type(self, satellite_id, frame):
        return self.data["steps"][frame]["satellites"][satellite_id]["action_type"]
    
    def get_task_being_collected(self, satellite_id, frame):
        return Task(self.data["steps"][frame]["satellites"][satellite_id]["task_being_collected"])
    
    def get_sat_lon_lat(self, satellite_id, frame):
        # Lon is typically x and lat is typically y
        return self.data["steps"][frame]["satellites"][satellite_id]["longitude"], self.data["steps"][frame]["satellites"][satellite_id]["latitude"]
    
    def get_cum_reward(self, frame):
        return self.data["steps"][frame]["cum_reward"]
    
    def get_n_tasks_collected(self, frame):
        return self.data["steps"][frame]["n_tasks_collected"]
    
    def get_step(self, frame):
        return self.data["steps"][frame]['step']

    def get_storage_percentage(self, satellite_id, frame):
        storage_pct = self.data["steps"][frame]["satellites"][satellite_id]["storage_percentage"]
        return storage_pct
    
if __name__ == "__main__":
    object_def = ObjectDef("data.json", interpolation_factor=50)
    print(f"Number of interpolated steps: {object_def.get_number_of_frames()}")
    print(f"Number of satellites: {object_def.get_number_of_satellites()}")
    print(f"Number of observations: {object_def.get_number_of_observations()}")

    sats_ids = object_def.get_satellite_ids()
    for i in range(object_def.get_number_of_frames()):
        print(object_def.get_sat_lon_lat(sats_ids[0], i))

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
    "task_type":"IMAGING",
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
""" 

class Task:

    COLLECT_TASKING = ['RF', 'IMAGING']

    def __init__(self, task):
        self.task = task

    @property
    def is_collect_tasking(self):
        return self.task["task_type"] in self.COLLECT_TASKING
    
    @property
    def is_downlink_tasking(self):
        return self.task["task_type"] == "DATA_DOWNLINK"
    
    @property
    def task_type(self):
        return self.task["task_type"]
    
    @property
    def storage_size(self):
        return self.task["storage_size"]
    
    @property
    def task_duration(self):
        return self.task["task_duration"]
    
    @property
    def priority(self):
        return self.task["priority"]
    
    @property
    def simultaneous_collects_required(self):
        return self.task["simultaneous_collects_required"]

    @property
    def id(self):
        return self.task["id"]
    
    @property
    def task_complete(self):
        return self.task["task_complete"]
    
    @property
    def task_reward(self):
        return self.task["task_reward"]

    @property
    def sats_collecting(self):
        return self.task["sats_collecting"]
    
    @property
    def r_LP_P(self):
        return self.task["r_LP_P"]
    
    @property
    def min_elev(self):
        return self.task["min_elev"]
    
    @property
    def task_type(self):
        return self.task["task_type"]
    
    @property
    def lat(self):
        return self.task["latitude"]
    
    @property
    def lon(self):
        return self.task["longitude"]
    
    @property
    def altitude(self):
        return self.task["altitude"]

    


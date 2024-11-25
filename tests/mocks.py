import numpy as np
from unittest.mock import Mock
from rl.tasks.task import TaskType

class MockTask:
    def __init__(self, task_type=TaskType.IMAGING):
        self.task_type = task_type
        self.r_LP_P = np.array([1, 2, 3])
        
    @property
    def is_charge(self):
        return self.task_type == TaskType.CHARGE
    
    @property
    def is_desat(self):
        return self.task_type == TaskType.DESAT
    
    @property 
    def is_data_downlink(self):
        return self.task_type == TaskType.DATA_DOWNLINK
    
    @property
    def is_collection(self):
        return self.task_type in [TaskType.RF, TaskType.IMAGING]
    
    def get_task_type_str(self):
        return TaskType.to_str(self.task_type)

class MockDynamics:
    def __init__(self):
        self.battery_charge = 100
        self.battery_charge_fraction = 0.5
        self.storage_level = 1000
        self.storage_level_fraction = 0.4
        self.wheel_speeds_fraction = [0.1, 0.2, 0.3]
        
        self.powerMonitor = Mock(storageCapacity=200)
        self.storageUnit = Mock(storageCapacity=2000)
        self.instrumentPowerSink = Mock(nodePowerOut=50)
        self.transmitter = Mock(nodeBaudRate=100)
        self.instrument = Mock(nodeBaudRate=200)
        
    def is_alive(self, log_failure=False):
        return True
        
    def battery_valid(self):
        return True
        
    def data_storage_valid(self):
        return True
        
    def rw_speeds_valid(self):
        return True
        
    def altitude_valid(self):
        return True

class MockFSW:
    def __init__(self):
        pass
        
    def is_alive(self, log_failure=False):
        return True
        
    def action_downlink(self):
        pass
        
    def action_charge(self):
        pass
        
    def action_drift(self):
        pass
        
    def action_desat(self):
        pass
        
    def action_nadir_scan(self, r_LP_P):
        pass 
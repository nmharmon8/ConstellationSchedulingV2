import logging
from datetime import datetime

import numpy as np
from weakref import proxy
from bsk_rl.utils.orbital import TrajectorySimulator
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.sim import dyn, fsw


class StorageUnit:

    def __init__(self, storage_capacity):
        self.storage_capacity = storage_capacity # in MB
        self.storage_level = 0.0 # in MB
    
    def get_storage_level(self):
        return self.storage_level

    def get_storage_percentage(self):
        return self.storage_level / self.storage_capacity

    def can_complete_task(self, task):
        if task.is_data_downlink:
            return True
        if (self.storage_level + task.storage_size) <= self.storage_capacity:
            return True
        return False
    
    def task_started(self, task):
        pass # Don't think storage needs to do anything here

    def task_completed(self, task):
        if task.is_data_downlink:
            self.storage_level = 0.0 # Data downlink so storage is empty
        else:
            self.storage_level += task.storage_size


    def task_failed(self, task):
        pass # Don't think storage needs to do anything here

    def get_observation(self):
        return {
            'storage_level': self.storage_level,
            'storage_capacity': self.storage_capacity,
            'storage_percentage': self.get_storage_percentage()
        }
    

class Satellite:

    def __init__(self, sat_args, simulator, name, trajectory=None, utc_init=None, sim_rate=1.0):

        self.sim_rate = sim_rate
        self.name = name
        self.logger = logging.getLogger(__name__).getChild(self.name)

        self.utc_init = utc_init
        if self.utc_init is None:
            self.utc_init = datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)")

        # Orbital elements.
        self.oe= random_orbit(alt=800)
        # Gravitational parameter
        self.mu = 398600436000000.0

        self.trajectory = trajectory
        if self.trajectory is None:
            self.trajectory = TrajectorySimulator(
                utc_init=self.utc_init,
                rN=None, #self.sat_args["rN"],
                vN=None, #self.sat_args["vN"],
                oe=self.oe,
                mu=self.mu,
            )

        self.simulator = proxy(simulator)
        self.dyn_type = dyn.FullFeaturedDynModel
        self.dynamics = dyn.FullFeaturedDynModel(self, dyn_rate=self.sim_rate, **sat_args)
        self.fsw_type = fsw.SteeringImagerFSWModel
        self.fsw = fsw.SteeringImagerFSWModel(self, fsw_rate=self.sim_rate, **sat_args)

        self.storage_unit = StorageUnit(1000)

    def can_complete_task(self, task):
        print(f"Satellite {self.name} can complete task: {self.storage_unit.can_complete_task(task)}")
        return self.storage_unit.can_complete_task(task)
    
    def task_started(self, task):
        print(f"Satellite {self.name} task started: {task.id}")
        self.storage_unit.task_started(task)

    def task_completed(self, task):
        print(f"Satellite {self.name} task completed: {task.id}")
        self.storage_unit.task_completed(task)

    def task_failed(self, task):
        print(f"Satellite {self.name} task failed: {task.id}, task size: {task.storage_size}, remaining storage: {self.storage_unit.storage_capacity - self.storage_unit.storage_level}")
        self.storage_unit.task_failed(task)

    def get_observation(self):
        return self.storage_unit.get_observation()
    
    def storage_after_task(self, task):
        if task.is_data_downlink:
            return 0.0
        else:
            return (self.storage_unit.storage_level + task.storage_size) / self.storage_unit.storage_capacity

    @property
    def id(self):
        return f"{self.name}_{id(self)}"

    @property
    def storage_level(self):
        return self.storage_unit.storage_level
    
    @property
    def storage_capacity(self):
        return self.storage_unit.storage_capacity

    def set_action(self, action):
        pass

    def get_dt(self):
        return self.trajectory.dt
    
    def get_r_BP_P_interp(self, end_time):
        self.trajectory.extend_to(end_time)
        return self.trajectory.r_BP_P
    

    def __del__(self):
        del self.trajectory


from bsk_rl.utils.attitude import random_tumble
from bsk_rl.utils.orbital import random_orbit

def create_random_satellite(name, simulator, trajectory=None, utc_init=None):

    sat_args = {
        'hs_min': 0.0, 
        'maxCounterValue': 4, 
        'thrMinFireTime': 0.02, 
        'desatAttitude': 'nadir', 
        'controlAxes_B': [1, 0, 0, 0, 1, 0, 0, 0, 1], 
        'thrForceSign': 1, 
        'K1': 0.25, 
        'K3': 3.0, 
        'omega_max': 0.087, 
        'servo_Ki': 5.0, 
        'servo_P': 30.0, 
        'imageAttErrorRequirement': 0.01, 
        'imageRateErrorRequirement': 0.01, 
        'inst_pHat_B': [0, 0, 1], 
        'batteryStorageCapacity': 1440000, 
        'storedCharge_Init': np.random.uniform(30.0 * 3600.0, 70.0 * 3600.0), 
        'disturbance_vector': np.random.normal(scale=0.0001, size=3), 
        'dragCoeff': 2.2, 
        'groundLocationPlanetRadius': 6378136.6, 
        'imageTargetMinimumElevation': 0.7853981633974483, 
        'imageTargetMaximumRange': -1, 
        'instrumentBaudRate': 500000.0, 
        'instrumentPowerDraw': -30.0, 
        'losMaximumRange': -1.0, 
        'basePowerDraw': -10.0, 
        'wheelSpeeds': np.random.uniform(-3000, 3000, 3), 
        'maxWheelSpeed': 6000.0, 
        'u_max': 0.4, 
        'rwBasePower': 0.4, 
        'rwMechToElecEfficiency': 0.0, 
        'rwElecToMechEfficiency': 0.5, 
        'panelArea': 1.0, 
        'panelEfficiency': 0.2, 
        'nHat_B': np.array([ 0,  0, -1]), 
        'mass': 330, 
        'width': 1.38, 
        'depth': 1.04, 
        'height': 1.58, 
        'sigma_init': random_tumble(maxSpinRate=0.0001)[0], 
        'omega_init': random_tumble(maxSpinRate=0.0001)[1], 
        'rN': None, 
        'vN': None, 
        'oe': random_orbit(alt=800), 
        'mu': 398600436000000.0, 
        'dataStorageCapacity': 40000000000.0, 
        'bufferNames': None, 
        'storageUnitValidCheck': False, 
        'storageInit': np.random.uniform(0, 5000 * 8e6), 
        'thrusterPowerDraw': -80.0, 
        'transmitterBaudRate': -112000000.0, 
        'transmitterNumBuffers': 100, 
        'transmitterPowerDraw': -25.0
    }

    return Satellite(sat_args, simulator=simulator, name=name, trajectory=trajectory, utc_init=utc_init)
        

    
      

if __name__ == "__main__":

    print("Creating satellites")

    # Create 100 satellites
    sats = [Satellite(f"EO-{i}") for i in range(100)]

    print(f"Number of satellites: {len(sats)}")

    # Delete the satellites
    del sats

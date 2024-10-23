import logging
from datetime import datetime

import numpy as np
from weakref import proxy
from bsk_rl.utils.orbital import TrajectorySimulator
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.sim import dyn, fsw

from rl.tasks.task import TaskType

class Satellite:

    def __init__(self, sat_args, simulator, name, utc_init, sim_rate=1.0):

        self.sim_rate = sim_rate
        self.name = name
        self.logger = logging.getLogger(__name__).getChild(self.name)

        self.oe= random_orbit(alt=800)
        self.mu = 0.3986004415e15

        self.trajectory = TrajectorySimulator(
            utc_init=utc_init,
            oe=self.oe,
            mu=self.mu,
        )

        self.simulator = proxy(simulator)
        self.dyn_type = dyn.GroundStationDynModel
        self.dynamics = dyn.GroundStationDynModel(self, dyn_rate=self.sim_rate, **sat_args)
        self.fsw_type = fsw.ContinuousImagingFSWModel
        self.fsw = fsw.ContinuousImagingFSWModel(self, fsw_rate=self.sim_rate, **sat_args)
        # self.fsw_type = fsw.ImagingFSWModel
        # self.fsw = fsw.ImagingFSWModel(self, fsw_rate=self.sim_rate, **sat_args)

    def _get_storage_change(self, task):
        if task.is_data_downlink:
            transmitter_baud_rate = self.dynamics.transmitter.nodeBaudRate
            storage_change = transmitter_baud_rate * self.simulator.max_step_duration_sec #* task.duration
        else:
            # return (self.storage_unit.storage_level + task.storage_size) / self.storage_unit.storage_capacity
            instrument_baud_rate = self.dynamics.instrument.nodeBaudRate
            storage_change = instrument_baud_rate * self.simulator.max_step_duration_sec #* task.duration
        return storage_change
    
    def _get_power_change(self, task):
        if task.is_data_downlink:
            return self.dynamics.transmitterPowerSink.nodePowerOut
        else:
            return self.dynamics.instrumentPowerSink.nodePowerOut

    def can_complete_task(self, task):
        # storage_change = self._get_storage_change(task)
        # power_change = self._get_power_change(task)
        # if storage_change + self.dynamics.storage_level <= 0:
        #     return False
        # if power_change + self.dynamics.battery_charge <= 0:
        #     return False
        # return True
        return True
    
    def in_eclipse(self):
        """
            Start is always the start of the next eclipse so if start is greater then end then we are in eclipse
        """
        eclipse_start, eclipse_end = self.trajectory.next_eclipse(self.simulator.sim_time)
        return eclipse_start > eclipse_end
    
    def next_eclipse(self):
        """
            Start is always the start of the next eclipse so if start is greater then end then we are in eclipse
        """
        eclipse_start, eclipse_end = self.trajectory.next_eclipse(self.simulator.sim_time)
        return eclipse_start-self.simulator.sim_time
    
    def end_of_eclipse(self):
        eclipse_start, eclipse_end = self.trajectory.next_eclipse(self.simulator.sim_time)
        return eclipse_end-self.simulator.sim_time

    def print_storage_level(self):
        print(f"FSW: Storage: Satellite {self.name} storage valid: {self.dynamics.data_storage_valid()}")
        print(f"FSW: Storage: Satellite {self.name} storage level: {self.dynamics.storage_level} storage capacity: {self.dynamics.storageUnit.storageCapacity}")
        print(f"FSW: Storage: Satellite {self.name} storage percentage: {self.dynamics.storage_level_fraction}")

    def print_power_level(self):
        print(f"FSW: Power: Satellite {self.name} power level: {self.dynamics.battery_charge} power capacity: {self.dynamics.powerMonitor.storageCapacity}")
        print(f"FSW: Power: Satellite {self.name} power percentage: {self.dynamics.battery_charge_fraction}")
        print(f"FSW: Power: Satellite {self.name} power valid: {self.dynamics.battery_valid()}")

    def task_started(self, task):
        print(f"Satellite {self.name} task started: {task.id}")
        
        if task.is_data_downlink:
            print(f"FSW: Satellite {self.name} downlinking")
            self.fsw.action_downlink()
        elif task.is_charge:
            print(f"FSW: Satellite {self.name} charging")
            print(f"FSW: Satellite {self.name} in eclipse: {self.in_eclipse()} next eclipse: {self.next_eclipse()} end of eclipse: {self.end_of_eclipse()}")
            if not self.in_eclipse():
                print(f"FSW: Satellite {self.name} not in eclipse, charging")
                self.fsw.action_charge()
            else:
                print(f"FSW: Satellite {self.name} in eclipse, not charging")
                self.fsw.action_drift()
        elif task.is_desat:
            print(f"FSW: Satellite {self.name} desaturating")
            self.fsw.action_desat()
        elif task.is_collection:
            print(f"FSW: Satellite {self.name} collecting data")
            self.fsw.action_nadir_scan(task.r_LP_P)
            # self.fsw.action_image(task.r_LP_P, task.data_name)
        elif task.task_type == TaskType.NOOP:
            self.fsw.action_drift()
        else:
            raise ValueError(f"Invalid task type: {TaskType.to_str(task.task_type)}")

        self.print_storage_level()
        self.print_power_level()

    def task_completed(self, task):
        print(f"Satellite {self.name} task completed: {task.id}")
        # self.storage_unit.task_completed(task)

    def task_failed(self, task):
        pass
        # print(f"Satellite {self.name} task failed: {task.id}, task size: {task.storage_size}, remaining storage: {self.storage_unit.storage_capacity - self.storage_unit.storage_level}")
        # self.storage_unit.task_failed(task)

    def get_observation(self):
        return {
            'storage_level': self.dynamics.storage_level,
            'storage_capacity': self.dynamics.storageUnit.storageCapacity,
            'storage_percentage': self.dynamics.storage_level_fraction,
            'power_level': self.dynamics.battery_charge,
            'power_capacity': self.dynamics.powerMonitor.storageCapacity,
            'power_percentage': self.dynamics.battery_charge_fraction,
            'in_eclipse': self.in_eclipse(),
            'next_eclipse': self.next_eclipse(),
            'end_of_eclipse': self.end_of_eclipse(),
        }
    
    def storage_after_task(self, task):
        storage_change = self._get_storage_change(task)
        if task.is_data_downlink:
            # Downlink baud rate is negative so storage_change is negative
            return min(0, self.dynamics.storage_level + storage_change)
        else:
            return max(self.dynamics.storageUnit.storageCapacity, self.dynamics.storage_level + storage_change)


    def is_alive(self, log_failure=False):
        is_alive = self.dynamics.is_alive(log_failure=log_failure) and self.fsw.is_alive(
            log_failure=log_failure
        )
        print(f"FSW: Satellite {self.name} is alive: {is_alive}")
        return is_alive

    @property
    def id(self):
        return f"{self.name}_{id(self)}"

    @property
    def storage_level(self):
        # return self.storage_unit.storage_level
        return self.dynamics.storage_level
    
    @property
    def storage_capacity(self):
        # return self.storage_unit.storage_capacity
        return self.dynamics.storageUnit.storageCapacity

    def set_action(self, action):
        print(f"FSW: Satellite {self.name} setting action: {action}")
        pass

    def get_dt(self):
        return self.trajectory.dt
    
    def get_r_BP_P_interp(self, end_time):
        self.trajectory.extend_to(end_time)
        return self.trajectory.r_BP_P

    def __del__(self):
        del self.trajectory

    def get_info(self):
        from pymap3d import ecef2geodetic
        r_BP_P = self.trajectory.r_BP_P(self.simulator.sim_time)
        lat, lon, alt = ecef2geodetic(r_BP_P[0], r_BP_P[1], r_BP_P[2])
        return {
            'r_BP_P': r_BP_P,
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'observation': self.get_observation(),
        }


from bsk_rl.utils.attitude import random_tumble
from bsk_rl.utils.orbital import random_orbit

def create_random_satellite(name, simulator, utc_init):

    sat_args = {
        'hs_min': 0.0, 
        'maxCounterValue': 4, 
        'thrMinFireTime': 0.02, 
        'desatAttitude': 'nadir', 
        'controlAxes_B': [1, 0, 0, 0, 1, 0, 0, 0, 1], 
        'thrForceSign': 1, 
        'K1': 0.25, # MRP Steering
        'K3': 3.0, # MRP Steering
        'omega_max': 0.087, # MRP Steering
        'servo_Ki': 5.0, 
        'servo_P': 30.0, 

        'K': 7.0,
        'Ki': -1,
        'P': 35.0,

        'imageAttErrorRequirement': 0.01, 
        'imageRateErrorRequirement': 0.01, 
        'inst_pHat_B': [0, 0, 1], 
        'batteryStorageCapacity': 200 * 3600, 
        'storedCharge_Init': np.random.uniform(0.3, 1.0) * 200 * 3600, 
        'disturbance_vector': np.random.normal(scale=0.0001, size=3), 
        'dragCoeff': 2.2, 
        'groundLocationPlanetRadius': 6378136.6, 
        'imageTargetMinimumElevation': 0.7853981633974483, 
        'imageTargetMaximumRange': -1, 
        'instrumentBaudRate': 20000000.0, # Collection data rate
        'instrumentPowerDraw': -500.0, 
        'losMaximumRange': -1.0, 
        'basePowerDraw': 0.0, 
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
        'dataStorageCapacity': 5000 * 8e6, 
        'bufferNames': None, 
        'storageUnitValidCheck': False, 
        'storageInit': np.random.uniform(0, 5000 * 8e6), 
        'thrusterPowerDraw': 0.0, 
        'transmitterBaudRate': -200000000.0, # Downlink rate
        'transmitterNumBuffers': 100, 
        'transmitterPowerDraw': 0.0
    }

    return Satellite(sat_args, simulator=simulator, name=name, utc_init=utc_init)
        


class SatelliteManager:

    def __init__(self, simulator, config):
        self.n_sats = config['n_sats']
        self.satellites = [create_random_satellite(f"EO-{i}", simulator=simulator) for i in range(self.n_sats)]

    def get_observation(self):
        return {sat.id: sat.get_observation() for sat in self.sats}
      

if __name__ == "__main__":

    print("Creating satellites")

    # Create 100 satellites
    sats = [Satellite(f"EO-{i}") for i in range(100)]

    print(f"Number of satellites: {len(sats)}")

    # Delete the satellites
    del sats

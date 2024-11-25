import logging

import numpy as np
from pymap3d import ecef2geodetic
from weakref import proxy
from bsk_rl.utils.orbital import TrajectorySimulator
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.sim import dyn, fsw

from rl.tasks.task import TaskType

class SatelliteTask:
    def __init__(self, task, sat):
        self.task = task
        self.sat = sat

        self.init_storage = self.sat.storage_level
        self.init_power = self.sat.dynamics.battery_charge
        self.init_alive = self.sat.is_alive()
        self.init_wheel_speeds = self.sat.dynamics.wheel_speeds_fraction

        self.predicted_storage_change = self.sat.get_task_storage_change(self.task)
        self.predicted_power_change = self.sat.get_power_change(self.task)

        self.expect_task_to_complete = True
        if self.predicted_storage_change + self.init_storage > self.sat.storage_capacity:
            self.expect_task_to_complete = False        
        elif self.sat.pct_power() < 0.1:
            # Power is hard to estimate so we will just check if the power is less than 10%
            self.expect_task_to_complete = False

        if self.task.is_charge and self.sat.in_eclipse():
            self.expect_task_to_complete = False

        if self.task.is_desat:
            self.expect_task_to_complete = True


        self.final_storage = None
        self.final_power = None
        self.final_alive = True
        self.final_storage_change = None
        self.final_power_change = None
        self.final_wheel_speeds = None

        self.power_storage_valid = False

        self.expected_action = None
        if self.task.is_charge:
            self.expected_action = Actions.CHARGE
        elif self.task.is_desat:
            self.expected_action = Actions.DESAT
        elif self.task.is_collection:
            self.expected_action = Actions.COLLECTION
        elif self.task.is_data_downlink:
            self.expected_action = Actions.DOWNLINK
        else:
            self.expected_action = Actions.DRIFT

        self.actual_action = None


    def task_complete(self, action):
        self.actual_action = action

        self.final_storage = self.sat.storage_level
        self.final_power = self.sat.dynamics.battery_charge
        self.final_alive = self.sat.is_alive()
        self.final_wheel_speeds = self.sat.dynamics.wheel_speeds_fraction
        if self.sat.pct_power() > 0.03 and self.sat.pct_storage() < 0.97:
            self.power_storage_valid = True

        self.final_storage_change = self.final_storage - self.init_storage
        self.final_power_change = self.final_power - self.init_power

    @property
    def sat_task_valid(self):
        if self.task.is_charge:
            return not self.sat.in_eclipse()
        if self.task.is_desat:
            return True
        if self.task.is_data_downlink:
            return True
        # All collection tasks are valid if the power and storage are valid and the satellite is still alive
        return self.power_storage_valid and self.final_alive and self.actual_action == self.expected_action

    @property
    def observation(self):
        return {
            'init_storage': self.init_storage,
            'init_power': self.init_power,
            'predicted_storage_change': self.predicted_storage_change,
            'predicted_power_change': self.predicted_power_change,
            'final_storage': self.final_storage,
            'final_power': self.final_power,
            'final_storage_change': self.final_storage_change,
            'final_power_change': self.final_power_change,
            'task_type': self.task.get_task_type_str(),
            'sat_task_valid': self.sat_task_valid,
            'expect_task_to_complete': self.expect_task_to_complete,
            'power_storage_valid': self.power_storage_valid,
            'pct_power': self.sat.pct_power(),
            'pct_storage': self.sat.pct_storage(),
            'expected_action': self.expected_action,
            'actual_action': self.actual_action,
            'init_wheel_speeds': self.init_wheel_speeds,
            'final_wheel_speeds': self.final_wheel_speeds,
        }
    
class Actions:
    DOWNLINK = 'DOWNLINK'
    CHARGE = 'CHARGE'
    COLLECTION = 'COLLECTION'
    DESAT = 'DESAT'
    DRIFT = 'DRIFT'



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
        self.dyn_type = dyn.ContinuousImagingDynModel
        self.dynamics = dyn.ContinuousImagingDynModel(self, dyn_rate=self.sim_rate, oe=self.oe, mu=self.mu, **sat_args)
        self.fsw_type = fsw.ContinuousImagingFSWModel
        self.fsw = fsw.ContinuousImagingFSWModel(self, fsw_rate=self.sim_rate, **sat_args)

        self.sat_task = None
        self.action = Actions.DRIFT
        self.last_action_reward = 0

    
    def get_power_change(self, task):
        """
            This is a rough estimate of the power change for a task
            You must run the simulation forward to get the actual power change
        """
        if task.is_charge and not self.in_eclipse():
            # Should fully recharge so the power change is the amount of storage that can be charged
            return self.dynamics.powerMonitor.storageCapacity - self.dynamics.battery_charge
        elif task.is_charge:
            # If we are in eclipse then we can't charge and will just drift
            return 0
        elif task.is_desat:
            # This should result in a power gain as the energy is converted from motion to electrical
            return 10000
        else:
            # We are doing a collection or data downlink so power is being used
            return -self.dynamics.instrumentPowerSink.nodePowerOut * self.simulator.max_step_duration_sec

    def should_charge(self):
        return self.pct_power() < 0.2

    def should_downlink(self):
        return self.pct_storage() > 0.9
    
    def should_desat(self):
        wheel_speeds = self.dynamics.wheel_speeds_fraction
        return np.any(np.abs(wheel_speeds) > 0.7)

    def can_complete_task(self, task):
        sat_task = SatelliteTask(task, self)
        return sat_task.sat_task_valid
    
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

    def in_valid_state(self):
        # Check if the satellite is alive and the FSW is alive and task is valid
        dynamics_valid = self.dynamics.is_alive(log_failure=False) and self.fsw.is_alive(log_failure=False) 
        task_valid = self.sat_task.sat_task_valid if self.sat_task is not None else True
        return dynamics_valid and task_valid
    
    def pct_power(self):
        return self.dynamics.battery_charge_fraction
    
    def pct_storage(self):
        return self.dynamics.storage_level_fraction
    
    def print_stat_stats(self, info=None):
        sat_stats = f"##############################################\n"
        sat_stats += f"Satellite {self.name} stats\n"
        if info is not None:
            sat_stats += info
        sat_stats += f"    Charge: {self.dynamics.battery_charge_fraction}\n"
        sat_stats += f"    Storage: {self.dynamics.storage_level_fraction}\n"
        sat_stats += f"    In eclipse: {self.in_eclipse()}\n"
        sat_stats += f"    Wheel speeds: {self.dynamics.wheel_speeds_fraction}\n"
        sat_stats += f"    Is alive: {self.is_alive()}\n"
        sat_stats += f"        FSW is alive: {self.fsw.is_alive(log_failure=True)}\n"
        sat_stats += f"        Dynamics is alive: {self.dynamics.is_alive(log_failure=False)}\n"
        sat_stats += f"             battery_valid: {self.dynamics.battery_valid()}\n"
        sat_stats += f"             data_storage_valid: {self.dynamics.data_storage_valid()}\n"
        sat_stats += f"             rw_speeds_valid: {self.dynamics.rw_speeds_valid()}\n"
        sat_stats += f"             altitude_valid: {self.dynamics.altitude_valid()}\n"
        sat_stats += f"##############################################\n"
        print(sat_stats)

    def _task_started(self, task, window_offset):
        
        # I think it is always safe to desat
        if task.is_desat or self.should_desat():
            self.fsw.action_desat()
            self.action = Actions.DESAT
            return
        
        # At this point it should be safe to charge
        if task.is_charge or self.should_charge():
            if not self.in_eclipse():
                self.fsw.action_charge()
                self.action = Actions.CHARGE
            else:
                # self.fsw.action_drift()
                self.fsw.action_desat()
                self.action = Actions.DRIFT
            return

        # Now it should be safe to downlink
        if task.is_data_downlink and window_offset == 0:
            self.fsw.action_downlink()
            self.action = Actions.DOWNLINK
            return

        # Check if we need to downlink 
        if self.should_downlink():
            # Drift until we downlink
            # self.fsw.action_drift()
            self.fsw.action_desat()
            self.action = Actions.DRIFT
            return
        

        # At this point it should be safe to collect
        if task.is_collection:
            self.fsw.action_nadir_scan(task.r_LP_P)
            self.action = Actions.COLLECTION
            return
        

        # If Noop task then drift
        # self.fsw.action_drift()
        self.fsw.action_desat()
        self.action = Actions.DRIFT
        

    def start_action(self, task, window_offset, start_time, end_time):
        """
        Called before running the simulation step
        """
        # self.print_stat_stats(info=f"    Pre-task stats {task.get_task_type_str()}\n")
        self.sat_task = SatelliteTask(task, self)
        task.collect(self, start_time, end_time)
        self._task_started(task, window_offset)

    def complete_action(self, task, end_time):
        """
        Called after running the simulation step
        """
        self.sat_task.task_complete(self.action)
        task.complete(self, end_time)
        self.last_action_reward = task.get_reward()
    
    def get_task_storage_change(self, task):
        if task.is_data_downlink:
            return self.dynamics.transmitter.nodeBaudRate * self.simulator.max_step_duration_sec
        else:
            return self.dynamics.instrument.nodeBaudRate * self.simulator.max_step_duration_sec

    def is_alive(self, log_failure=False):
        is_alive = self.dynamics.is_alive(log_failure=log_failure) and self.fsw.is_alive(
            log_failure=log_failure
        )
        return is_alive

    @property
    def id(self):
        return f"{self.name}_{id(self)}"

    @property
    def storage_level(self):
        return self.dynamics.storage_level
    
    @property
    def storage_capacity(self):
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
        r_BP_P = self.trajectory.r_BP_P(self.simulator.sim_time)
        lat, lon, alt = ecef2geodetic(r_BP_P[0], r_BP_P[1], r_BP_P[2])
        return {
            'r_BP_P': r_BP_P,
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'observation': self.get_observation(),
        }
    
    def get_observation(self):
        return {   
                'is_alive': self.is_alive(),
                'storage_level': self.dynamics.storage_level,
                'storage_capacity': self.dynamics.storageUnit.storageCapacity,
                'storage_percentage': self.dynamics.storage_level_fraction,
                'power_level': self.dynamics.battery_charge,
                'power_capacity': self.dynamics.powerMonitor.storageCapacity,
                'power_percentage': self.dynamics.battery_charge_fraction,
                'wheel_speed_1': self.dynamics.wheel_speeds_fraction[0],
                'wheel_speed_2': self.dynamics.wheel_speeds_fraction[1],
                'wheel_speed_3': self.dynamics.wheel_speeds_fraction[2],
                'in_eclipse': self.in_eclipse(),
                'next_eclipse': self.next_eclipse(),
                'end_of_eclipse': self.end_of_eclipse(), 
                'sat_task': self.sat_task.observation if self.sat_task is not None else None,
                'action': self.action,
                'sat_last_action_reward': self.last_action_reward,
                'should_charge': self.should_charge(),
                'should_downlink': self.should_downlink(),
                'should_desat': self.should_desat(),
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
        'panelArea': 1.0, # Charge with 2 panel area and 100% efficiency allows full recharge in 200 seconds
        'panelEfficiency': 1.0, 
        'nHat_B': np.array([ 0,  0, -1]), 
        'mass': 330, 
        'width': 1.38, 
        'depth': 1.04, 
        'height': 1.58, 
        'sigma_init': random_tumble(maxSpinRate=0.0001)[0], 
        'omega_init': random_tumble(maxSpinRate=0.0001)[1], 
        'rN': None, 
        'vN': None, 
        'dataStorageCapacity': 5000 * 8e6, 
        'bufferNames': None, 
        'storageUnitValidCheck': True, # Will fail if storage is full
        'storageInit': np.random.uniform(0, 5000 * 8e6), 
        'thrusterPowerDraw': 0.0, 
        'transmitterBaudRate': -200000000.0, # Downlink rate
        'transmitterNumBuffers': 100, 
        'transmitterPowerDraw': 0.0
    }

    return Satellite(sat_args, simulator=simulator, name=name, utc_init=utc_init)
        


if __name__ == "__main__":

    print("Creating satellites")

    # Create 100 satellites
    sats = [Satellite(f"EO-{i}") for i in range(100)]

    print(f"Number of satellites: {len(sats)}")

    # Delete the satellites
    del sats

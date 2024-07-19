import inspect
import logging
from datetime import datetime

import numpy as np

from bsk_rl.utils.orbital import TrajectorySimulator
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.sim import dyn, fsw

from bsk_rl.utils.functional import (
    AbstractClassProperty,
    collect_default_args,
    safe_dict_merge,
    valid_func_name,
)

sat_args = dict(
    u_max=0.4,
    K1=0.25,
    K3=3.0,
    omega_max=0.087,
    servo_Ki=5.0,
    servo_P=150 / 5,
    oe=lambda: random_orbit(alt=800),

    # Data
    dataStorageCapacity=5000 * 8e6,  # MB to bits
    storageInit=lambda: np.random.uniform(0, 5000 * 8e6),
    instrumentBaudRate=0.5e6,
    transmitterBaudRate=-112e6,

    # Power
    batteryStorageCapacity=400 * 3600,  # Wh to W*s
    storedCharge_Init=lambda: np.random.uniform(400 * 3600 * 0.2, 400 * 3600 * 0.8),
    basePowerDraw=-10.0,
    instrumentPowerDraw=-30.0,
    transmitterPowerDraw=-25.0,
    thrusterPowerDraw=-80.0,

    # Attitude
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,

    # Wheels ? 
    disturbance_vector=lambda: np.random.normal(scale=0.0001, size=3),
    maxWheelSpeed=6000.0,  # RPM
    wheelSpeeds=lambda: np.random.uniform(-3000, 3000, 3),
    desatAttitude="nadir",
    nHat_B=np.array([0, 0, -1]),  # Solar panel orientation
)


def default_sat_args(dyn_type, fsw_type, **kwargs):
    """Compile default arguments for :class:`~bsk_rl.sim.dyn.DynamicsModel` and :class:`~bsk_rl.sim.fsw.FSWModel`, replacing those specified.

    Args:
        **kwargs: Arguments to override in the default arguments.

    Returns:
        Dictionary of arguments for simulation models.
    """
    defaults = collect_default_args(dyn_type)
    defaults = safe_dict_merge(defaults, collect_default_args(fsw_type))
    for name in dir(fsw_type):
        if inspect.isclass(getattr(fsw_type, name)) and issubclass(
            getattr(fsw_type, name), fsw.Task
        ):
            defaults = safe_dict_merge(
                defaults, collect_default_args(getattr(fsw_type, name))
            )

    for k, v in kwargs.items():
        if k not in defaults:
            raise KeyError(f"{k} not a valid key for sat_args")
        defaults[k] = v
    return defaults


class Satellite:

    def __init__(self, name, simulator=None, world=None, sim_rate=1.0, utc_init=None, **kwargs):

        self.name = name
        self.logger = logging.getLogger(__name__).getChild(self.name)

        # self.requires_retasking = True
        # self.trajectory = None
        # self._timed_terminal_event_name = None
        # self.sim_rate = sim_rate
        # # UTC Date string '2008 APR 06 19:39:56.515 (UTC)'
        self.utc_init = utc_init
        if self.utc_init is None:
            self.utc_init = datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)")

        self.dyn_type = dyn.FullFeaturedDynModel
        # self.dynamics = None
        self.fsw_type = fsw.SteeringImagerFSWModel
        # self.fsw = None

        # self.simulator = simulator
        # self.world = world

        self.sat_args_generator = default_sat_args(self.dyn_type, self.fsw_type, **kwargs)
        self.requires_retasking = True
        self.sat_args = {
            k: v if not callable(v) else v() for k, v in self.sat_args_generator.items()
        }
        # print(f"Sat Args: {self.sat_args}")
        # I think the trajectory is just for utility
        self.trajectory = TrajectorySimulator(
            utc_init=self.utc_init,
            rN=self.sat_args["rN"],
            vN=self.sat_args["vN"],
            oe=self.sat_args["oe"],
            mu=self.sat_args["mu"],
        )
        # self._timed_terminal_event_name = None
        # # I think the dynamics is the actual satellite being simulated
        # self.dynamics = dyn.FullFeaturedDynModel(self, self.sim_rate, **self.sat_args)
        # self.fsw = fsw.SteeringImagerFSWModel(self, self.sim_rate, **self.sat_args)
        

    @property
    def id(self):
        return f"{self.name}_{id(self)}"

    def get_obs(self):
        return np.array([1, 1, 1])

    def set_action(self, action):
        pass

    def get_dt(self):
        return self.trajectory.dt
    
    def get_r_BP_P_interp(self, end_time):
        self.trajectory.extend_to(end_time)
        return self.trajectory.r_BP_P



if __name__ == "__main__":

    sat = Satellite("EO-1", None, **sat_args)

    from rl.sim import get_random_tasks
    tasks = get_random_tasks(100)

    sat.calculate_access_windows(tasks)
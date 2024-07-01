import numpy as np
from bsk_rl import act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw

class ScanningDownlinkDynModel(dyn.ContinuousImagingDynModel, dyn.GroundStationDynModel):
    # Define some custom properties to be accessed in the state
    @property
    def instrument_pointing_error(self) -> float:
        r_BN_P_unit = self.r_BN_P/np.linalg.norm(self.r_BN_P) 
        c_hat_P = self.satellite.fsw.c_hat_P
        return np.arccos(np.dot(-r_BN_P_unit, c_hat_P))
    
    @property
    def solar_pointing_error(self) -> float:
        a = self.world.gravFactory.spiceObject.planetStateOutMsgs[
            self.world.sun_index
        ].read().PositionVector
        a_hat_N = a / np.linalg.norm(a)
        nHat_B = self.satellite.sat_args["nHat_B"]
        NB = np.transpose(self.BN)
        nHat_N = NB @ nHat_B
        return np.arccos(np.dot(nHat_N, a_hat_N))

class ScanningSatellite(sats.AccessSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction"),
            dict(prop="instrument_pointing_error", norm=np.pi),
            dict(prop="solar_pointing_error", norm=np.pi)
        ),
        obs.Eclipse(),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm=5700),
            dict(prop="opportunity_close", norm=5700),
            type="ground_station",
            n_ahead_observe=1,
        ),
    ]
    action_spec = [
        act.Scan(duration=180.0),
        act.Charge(duration=180.0),
        act.Downlink(duration=60.0),
        act.Desat(duration=60.0),
    ]
    dyn_type = ScanningDownlinkDynModel
    fsw_type = fsw.ContinuousImagingFSWModel

from bsk_rl.utils.orbital import random_orbit



sat_args=dict(
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
        imageAttErrorRequirement=0.1,
        imageRateErrorRequirement=0.1,
        disturbance_vector=lambda: np.random.normal(scale=0.0001, size=3),
        maxWheelSpeed=6000.0,  # RPM
        wheelSpeeds=lambda: np.random.uniform(-3000, 3000, 3),
        desatAttitude="nadir",
        nHat_B=np.array([0, 0, -1]),  # Solar panel orientation
        # oe=lambda: random_orbit(alt=800),
    )


def get_sats(n_sats):
    sats = [ScanningSatellite(f"Scanner-{i}", sat_args=sat_args) for i in range(n_sats)]
    return sats
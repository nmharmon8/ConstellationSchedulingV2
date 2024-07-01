
import numpy as np
from bsk_rl import act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw

import model

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
        obs.Time(),
    ]
    action_spec = [
        act.Scan(duration=180.0),
        act.Charge(duration=180.0),
        act.Downlink(duration=60.0),
        act.Desat(duration=60.0),
    ]
    dyn_type = ScanningDownlinkDynModel
    fsw_type = fsw.ContinuousImagingFSWModel



sat = ScanningSatellite(
    "Scanner-1",
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
    )
)


duration = 2 * 5700.0  # About 2 orbits
env_args = dict(
    satellite=sat,
    scenario=scene.UniformNadirScanning(value_per_second=1/duration),
    rewarder=data.ScanningTimeReward(),
    time_limit=duration,
    failure_penalty=-1.0,
    terminate_on_time_limit=True,
)



from bsk_rl.utils.rllib import EpisodeDataCallbacks

class CustomDataCallbacks(EpisodeDataCallbacks):
    def pull_env_metrics(self, env):
        reward = env.rewarder.cum_reward
        reward = sum(reward.values()) / len(reward)
        orbits = env.simulator.sim_time / (95 * 60)

        data = dict(
            reward=reward,
            reward_per_orbit=reward / orbits,
            # Are satellites dying, and how and when?
            alive=float(env.satellite.is_alive()),
            rw_status_valid=float(env.satellite.dynamics.rw_speeds_valid()),
            battery_status_valid=float(env.satellite.dynamics.battery_valid()),
            orbits_complete=orbits,
        )
        if not env.satellite.is_alive():
            data["orbits_complete_partial_only"] = orbits
        return  data
    


from bsk_rl import SatelliteTasking
from bsk_rl.utils.rllib import unpack_config
from ray.rllib.algorithms.ppo import PPOConfig

training_args = dict(
    lr=0.00003,
    gamma=0.999,
    train_batch_size=1000,  # In practice, usually a bigger number
    num_sgd_iter=10,
    # model=dict(fcnet_hiddens=[512, 512], vf_share_layers=False),
    model=dict(
    custom_model="sat_model",
    # custom_action_dist="sat_dist",
    ),
    lambda_=0.95,
    use_kl_loss=False,
    clip_param=0.1,
    grad_clip=0.5,
)

config = (
    PPOConfig()
    .training(**training_args)
    .env_runners(num_env_runners=2, sample_timeout_s=100000.0)
    .environment(
        env=unpack_config(SatelliteTasking),
        env_config=env_args,
    )
    .callbacks(CustomDataCallbacks)
    .reporting(
        metrics_num_episodes_for_smoothing=1,
        metrics_episode_collection_timeout_s=180,
    )
    .checkpointing(export_native_model_files=True)
    .framework(framework="torch")
)

# config.model.update(
#         {
#             "custom_model": "sat_model",
#             # "custom_action_dist": "sat_dist",
#         }
#     )



import ray
from ray import tune

ray.init(
    ignore_reinit_error=True,
    num_cpus=3,
    object_store_memory=2_000_000_000,  # 2 GB
)

# Run the training
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 1000},  # Adjust the number of iterations as needed
    checkpoint_freq=10,
    checkpoint_at_end=True,
)

# Shutdown Ray
ray.shutdown()
import argparse
import os

import numpy as np

import ray
from ray.air.constants import TRAINING_ITERATION
# from envs.classes.correlated_actions_env import CorrelatedActionsEnv
# from _old_api_stack.models.autoregressive_action_model import TorchAutoregressiveActionModel
# from _old_api_stack.models.autoregressive_action_dist import TorchBinaryAutoregressiveDistribution

from ray.rllib.models import ModelCatalog
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from torch.utils.tensorboard import SummaryWriter

from bsk_rl import data, scene
from bsk_rl.utils.rllib import unpack_config
from model import SatModel
from bsk_rl import GeneralSatelliteTasking 
from sats import get_sats

from ray.rllib.algorithms.ppo import PPOConfig

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=200, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=100000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=200.0,
        help="Reward at which we stop training.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="satellite_tasking",
        help="Name of the experiment for logging purposes.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args

if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    from bsk_rl import sats, act, obs, scene, data, comm
    from bsk_rl.sim import dyn, fsw


    class ScanningDownlinkDynModel(dyn.FullFeaturedDynModel):
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

    class ImagingSatellite(sats.ImagingSatellite):
        observation_spec = [
            obs.OpportunityProperties(
                dict(prop="priority"), 
                dict(prop="opportunity_open", norm=5700.0),
                n_ahead_observe=10,
            ),
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
            act.Image(n_ahead_image=10),
            # act.Scan(duration=180.0),
            act.Charge(duration=180.0),
            act.Downlink(duration=60.0),
            act.Desat(duration=60.0),
        ]
        # dyn_type = dyn.FullFeaturedDynModel
        dyn_type = ScanningDownlinkDynModel
        fsw_type = fsw.SteeringImagerFSWModel


    from bsk_rl.utils.orbital import random_orbit

    sat_args = dict(
        imageAttErrorRequirement=0.01,
        imageRateErrorRequirement=0.01,
        # batteryStorageCapacity=1e9,
        # storedCharge_Init=1e9,
        # dataStorageCapacity=1e12,
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
        # imageAttErrorRequirement=0.1,
        # imageRateErrorRequirement=0.1,
        disturbance_vector=lambda: np.random.normal(scale=0.0001, size=3),
        maxWheelSpeed=6000.0,  # RPM
        wheelSpeeds=lambda: np.random.uniform(-3000, 3000, 3),
        desatAttitude="nadir",
        nHat_B=np.array([0, 0, -1]),  # Solar panel orientation


    )

    duration = 2 * 5700.0  # About 2 orbits
    env_args = dict(
        satellites=[
        ImagingSatellite("EO-1", sat_args),
        ImagingSatellite("EO-2", sat_args),
        ImagingSatellite("EO-3", sat_args),
        ],
        scenario=scene.UniformTargets(1000),
        rewarder=data.UniqueImageReward(),
        communicator=comm.FreeCommunication(),
        log_level="INFO",
        time_limit=duration,
    )

    # Generic config.
    config = (
        PPOConfig()
        .env_runners(num_env_runners=7, sample_timeout_s=1000000.0)
        # Batch-norm models have not been migrated to the RL Module API yet.
        .api_stack(enable_rl_module_and_learner=False)
        .environment(
            env=unpack_config(GeneralSatelliteTasking),
            env_config=env_args,
        )
        .reporting(
            metrics_num_episodes_for_smoothing=1,
            metrics_episode_collection_timeout_s=180,
        )
        .framework("torch")
        .checkpointing(export_native_model_files=True)
        # .training(gamma=0.5)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    config.model.update(
        {
            "custom_model": "sat_model",
            # "custom_action_dist": "sat_dist",
        }
    )

    # use stop conditions passed via CLI (or defaults)
    stop = {
        TRAINING_ITERATION: args.stop_iters,
        NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    }

    config.algo_class = "PPO"
    algo = config.build()
    algo.restore("./logs/train_v2/")

    # run manual test loop: 1 iteration until done
    print("Finished training. Running manual test/inference loop.")


    env = unpack_config(GeneralSatelliteTasking)(env_args)

    


    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0

    from pymap3d import ecef2geodetic

    def ecef_to_latlon(x, y, z):
        lat, lon, alt = ecef2geodetic(x, y, z)
        return lat, lon

    print("Satellites:")
    for sat in env.satellites:
        print(f"  Satellite: {sat}")
        print(f"    Trajectory (TrajectorySimulator):")
        print(f"      UTC Init: {sat.trajectory.utc_init}")
        print(f"      Simulation Time: {sat.trajectory.sim_time:.2f} s")
        print(f"      Initial Position (r_N_init): {sat.trajectory.rN_init}")
        print(f"      Initial Velocity (v_N_init): {sat.trajectory.vN_init}")
        print(f"      Timestep (dt): {sat.trajectory.dt} s")
        
        # Get current position in ECEF
        current_time = sat.trajectory.sim_time
        r_BP_P = sat.trajectory.r_BP_P(current_time)
        
        # Convert ECEF to lat/lon
        lat, lon = ecef_to_latlon(r_BP_P[0], r_BP_P[1], r_BP_P[2])
        print(f"      Current Position:")
        print(f"        Latitude: {lat:.4f}°")
        print(f"        Longitude: {lon:.4f}°")
        
        # Print interpolator information
        r_BN_N = sat.trajectory.r_BN_N
        r_BP_P = sat.trajectory.r_BP_P
        print(f"      r_BN_N: Interpolator for satellite position in inertial frame")
        print(f"        - x-range: [{r_BN_N.x.min():.2f}, {r_BN_N.x.max():.2f}] s")
        print(f"        - y-shape: {r_BN_N.y.shape}")
        print(f"      r_BP_P: Interpolator for satellite position in planet-fixed frame")
        print(f"        - x-range: [{r_BP_P.x.min():.2f}, {r_BP_P.x.max():.2f}] s")
        print(f"        - y-shape: {r_BP_P.y.shape}")

        # Print eclipse information
        eclipse_start, eclipse_end = sat.trajectory.next_eclipse(sat.trajectory.sim_time)
        print(f"      Next Eclipse:")
        print(f"        Start: {eclipse_start:.2f} s")
        print(f"        End: {eclipse_end:.2f} s")

    print(f"    Window Calculation Time: {sat.window_calculation_time}")
    print(f"    Upcoming Opportunities: {len(sat.upcoming_opportunities)}")
    if isinstance(sat, ImagingSatellite):
        print(f"    Known Targets: {len(sat.known_targets)}")


    for sat in env.satellites:
        print(sat.action_description)
        print(sat.observation_description)
        # print(sat.known_targets)
        next_target = sat.find_next_opportunities(n=10, types="target")
        print(f"Next Target: {next_target}")
        next_targets = sat.find_next_opportunities(n=10, filter=sat.get_access_filter(), types="target")


    def map_targets(target):
        print(target)
        print(type(target)) 
        from bsk_rl.scene.targets import Target
        if isinstance(target, dict):
            lat, lon = ecef_to_latlon(target['r_LP_P'][0], target['r_LP_P'][1], target['r_LP_P'][2])
            return {'id': target['target'].id, 'latitude': lat, 'longitude': lon}
        elif isinstance(target, Target):
            lat, lon = ecef_to_latlon(target.r_LP_P[0], target.r_LP_P[1], target.r_LP_P[2])
            return {'id': target.id, 'latitude': lat, 'longitude': lon}
        else:
            raise TypeError(f"Invalid target type: {type(target)}")
        
    map_targets(env.satellites[0].known_targets[0])
    map_targets(sat.find_next_opportunities(n=10, filter=sat.get_access_filter(), types="target")[0])

    # exit()

    import json
    from collections import defaultdict


    data_per_step = []
    all_possible_targets = env.satellites[0].known_targets
    action_description = env.satellites[0].action_description
    observation_description = env.satellites[0].observation_description
    n_action = 0

    while not done and not truncated:
        action = algo.compute_single_action(obs)
        # Get random action
        # action = [random.randint(0, 9) for _ in range(3)]
        next_obs, reward, done, truncated, _ = env.step(action)

        # Record satellite positions

        targets_this_step = {}
        satellite_data = {}

        current_time = env.simulator.sim_time

        for i, sat in enumerate(env.satellites):
            # current_time = sat.trajectory.sim_time
            r_BP_P = sat.trajectory.r_BP_P(current_time)
            lat, lon = ecef_to_latlon(r_BP_P[0], r_BP_P[1], r_BP_P[2])

            satellite_data[sat.id] = {}

            satellite_data[sat.id]['time'] = current_time
            satellite_data[sat.id]['latitude'] = lat
            satellite_data[sat.id]['longitude'] = lon
            satellite_data[sat.id]['targets'] = sat.find_next_opportunities(n=10, filter=sat.get_access_filter(), types="target")
            satellite_data[sat.id]['actions'] = int(action[i])
            satellite_data[sat.id]['reward'] = reward

        data_per_step.append(satellite_data)

        print(f"Obs: {obs}, Action: {action}, Reward: {reward} Done: {done} Truncated: {truncated}")
        obs = next_obs
        total_reward += reward
        n_action += 1

        # if n_action > 10:
        #     break

    print(f"Total reward in test episode: {total_reward}")
    algo.stop()

    # Write data to JSON file
    def map_targets(target):
        from bsk_rl.scene.targets import Target
        if isinstance(target, dict):
            lat, lon = ecef_to_latlon(target['r_LP_P'][0], target['r_LP_P'][1], target['r_LP_P'][2])
            return {'id': target['target'].id, 'latitude': lat, 'longitude': lon}
        elif isinstance(target, Target):
            lat, lon = ecef_to_latlon(target.r_LP_P[0], target.r_LP_P[1], target.r_LP_P[2])
            return {'id': target.id, 'latitude': lat, 'longitude': lon}
        else:
            raise TypeError(f"Invalid target type: {type(target)}")
        

    json_data = {}
    json_data['targets'] = [map_targets(target) for target in all_possible_targets]
    


    for step in data_per_step:
        for sat in step:
            step[sat]['targets'] = [map_targets(target) for target in step[sat]['targets']]

    
    print("Data per step: ", data_per_step)
    print(data_per_step[0])



    json_data['steps'] = data_per_step
    json_data['action_description'] = action_description
    json_data['observation_description'] = observation_description

    print(json_data)

    with open('data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    

    ray.shutdown()
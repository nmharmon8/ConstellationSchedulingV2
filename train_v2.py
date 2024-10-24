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

    # from bsk_rl.utils.rllib import EpisodeDataCallbacks
    from ray.rllib.algorithms.callbacks import DefaultCallbacks

    from bsk_rl.utils.rllib import EpisodeDataCallbacks

    class CustomDataCallbacks(EpisodeDataCallbacks):

        def __init__(self, *args, name=args.name, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_count = 0
            logdir = f"./logs/{name}"
            self.writer = SummaryWriter(logdir)


        def pull_env_metrics(self, env):
            reward = env.rewarder.cum_reward
            reward = sum(reward.values()) / len(reward)
            orbits = env.simulator.sim_time / (95 * 60)

            self.writer.add_scalar("reward", reward, self.step_count)
            self.writer.add_scalar("reward_per_orbit", reward / orbits, self.step_count)
            self.writer.add_scalar("orbits_complete", orbits, self.step_count)

            data = dict(
                reward=reward,
                reward_per_orbit=reward / orbits,
                orbits_complete=orbits,
            )
            return data

        def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
            self.step_count += 1



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
        scenario=scene.UniformTargets(3000),
        rewarder=data.UniqueImageReward(),
        communicator=comm.FreeCommunication(),
        log_level="INFO",
        time_limit=duration,
    )

    training_args = dict(
        lr=0.00003,
        gamma=0.999,
        train_batch_size=1000,
        num_sgd_iter=10,
        lambda_=0.95,
        use_kl_loss=False,
        clip_param=0.1,
        grad_clip=0.5,
    )

    # Generic config.
    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(num_env_runners=7, sample_timeout_s=1000000.0)
        # Batch-norm models have not been migrated to the RL Module API yet.
        .api_stack(enable_rl_module_and_learner=False)
        .environment(
            env=unpack_config(GeneralSatelliteTasking),
            env_config=env_args,
        )
        .callbacks(CustomDataCallbacks)
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


    print("Starting training loop.")

    for _ in range(args.stop_iters):
        result = algo.train()
        print(pretty_print(result))

    # run manual test loop: 1 iteration until done
    print("Finished training. Running manual test/inference loop.")


    env = unpack_config(GeneralSatelliteTasking)(env_args)
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0

    import random

    while not done and not truncated:
        # action = algo.compute_single_action(obs)
        # Get random action
        action = [random.randint(0, 9) for _ in range(3)]
        next_obs, reward, done, truncated, _ = env.step(action)
        print(f"Obs: {obs}, Action: {action}, Reward: {reward} Done: {done} Truncated: {truncated}")
        obs = next_obs
        total_reward += reward
    print(f"Total reward in test episode: {total_reward}")
    algo.stop()

    ray.shutdown()
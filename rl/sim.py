import numpy as np  

# from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion

from bsk_rl.sim.world import GroundStationWorldModel

from rl.sat import Satellite, sat_args
from rl.task import TaskManager

# SimulationBaseClass.SimBaseClass
class Simulator():

    def __init__(self, sim_rate=1.0, max_step_duration=600.0, time_limit=3000, n_access_windows=10, **kwargs):

        self.sim_time = 0.0

        self.sim_rate = sim_rate
        self.max_step_duration = max_step_duration
        self.time_limit = time_limit
        self.n_access_windows = n_access_windows

        self.world = None

        self.satellites = [Satellite("EO-1", self, self.world, **sat_args), Satellite("EO-2", self, self.world, **sat_args)]
        self.task_manager = TaskManager(500)

        self.cum_reward = 0


    def step(self, actions):
        # Simulation time
        start_time = self.sim_time
        end_time = self.sim_time + min(self.sim_time + self.max_step_duration, self.time_limit)

        # Now take actions
        for satellite, action in zip(self.satellites, actions):
            # First n actions are for collecting tasks
            if action < self.n_access_windows and action < len(self.current_tasks_by_sat[satellite.id]):
                task = self.current_tasks_by_sat[satellite.id][action]
                task.collect(satellite, start_time, end_time)

        # Now get the reward based on the action taken from start to end time
        reward = self.task_manager.step()
        self.cum_reward += reward
        # Update the simulation time
        self.sim_time = end_time

        # Get the next observations
        observations = self.get_obs()
        return observations, reward
    



    @property
    def done(self):
        return self.sim_time >= self.time_limit


    
    # def pre_step(self):
    #     print("Calculating access windows for sim time", self.sim_time)
    #     # Update task windows
    #     # self.task_manager.pre_step(sim_time=self.sim_time)
    #     # for satellite in self.satellites:
    #     #     self.task_manager.calculate_access_windows(satellite, self.sim_time)
        

    # def post_step(self):
    #     self.task_manager.post_step(sim_time=self.sim_time)

    def get_obs(self):

        for satellite in self.satellites:
            self.task_manager.calculate_access_windows(satellite, self.sim_time)

        self.current_tasks_by_sat = {}
        for satellite in self.satellites:
            self.current_tasks_by_sat[satellite.id] = self.task_manager.get_upcoming_tasks(satellite, self.sim_time)[:self.n_access_windows]
        
        observations = []
        for satellite in self.satellites:
            sat_observations = []
            for task in self.current_tasks_by_sat[satellite.id]:
                sat_observations.append(task.get_observation(satellite.id, self.sim_time))

            while len(sat_observations) < self.n_access_windows:
                sat_observations.append(np.array([0, 0, 0]))

            sat_observations = np.concatenate(sat_observations)
            observations.append(sat_observations)

        observations = np.stack(observations, axis=0)
        return observations
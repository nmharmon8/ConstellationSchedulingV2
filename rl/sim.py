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
        # super().__init__()

        self.sim_time = 0.0
        # self.sim_step = 0

        self.sim_rate = sim_rate
        self.max_step_duration = max_step_duration
        self.time_limit = time_limit
        self.n_access_windows = n_access_windows

        # world_args = GroundStationWorldModel.default_world_args()
        # world_args = {
        #     k: v if not callable(v) else v()
        #     for k, v in world_args.items()
        # }
        # self.world = GroundStationWorldModel(self, sim_rate, priority=300, **world_args)

        # self.fsw_list = {}
        # self.dynamics_list = {}

        self.world = None

        self.satellites = [Satellite("EO-1", self, self.world, **sat_args), Satellite("EO-2", self, self.world, **sat_args)]
        # self.satellites = [Satellite(f"EO-{i}", self, self.world, **sat_args) for i in range(1, 300)]

        self.task_manager = TaskManager(500)

        self.reward=0
        self.cum_reward = 0

        self._step()

        # self.InitializeSimulation()
        # self.ConfigureStopTime(0)
        # self.ExecuteSimulation()

    @property
    def done(self):
        return self.sim_time >= self.time_limit

    def take_action(self, actions):
        for satellite, action in zip(self.satellites, actions):
            # First n actions are for collecting tasks
            if action < self.n_access_windows and action < len(self.current_tasks_by_sat[satellite.id]):
                task = self.current_tasks_by_sat[satellite.id][action]
                task.collect(satellite, self.sim_time)
            
    def get_task_reward(self):
        return self.task_manager.get_task_reward()

    def get_obs(self):
        # print("Getting observations")
        observations = []
        for satellite in self.satellites:
            sat_observations = []
            for task in self.task_manager.get_upcoming_tasks(satellite, self.sim_time):
                sat_observations.append(task.get_observation(satellite.id, self.sim_time))
                if len(sat_observations) == self.n_access_windows:
                    break

            while len(sat_observations) < self.n_access_windows:
                sat_observations.append(np.array([0, 0, 0]))

            sat_observations = np.concatenate(sat_observations)
            # print(f"Observations for {satellite.id}: {sat_observations}")
            observations.append(sat_observations)

        observations = np.stack(observations, axis=0)
        # observations = np.concatenate(observations, axis=0)
        # print(f"Observations: {observations.shape}")
        return observations
    
    def _step(self):

        print("Clearing task collections")
        # Clear task collections
        self.task_manager.clear_collecting()

        print("Calculating access windows for sim time", self.sim_time)
        # Update task windows
        for satellite in self.satellites:
            self.task_manager.calculate_access_windows(satellite, self.sim_time)


        print("Getting current tasks")
        self.current_tasks_by_sat = {}
        for satellite in self.satellites:
            self.current_tasks_by_sat[satellite.id] = self.task_manager.get_upcoming_tasks(satellite, self.sim_time)[:self.n_access_windows]


    def run(self) -> None:
        """Propagate the simulator.

        Propagates for a duration up to the ``max_step_duration``, stopping if the
        environment time limit is reached or an event is triggered.
        """
        # simulation_time = mc.sec2nano(
        #     min(self.sim_time + self.max_step_duration, self.time_limit)
        # )

        simulation_time =  min(self.sim_time + self.max_step_duration, self.time_limit)

        # # print(f"Running simulation at most to {simulation_time*mc.NANO2SEC:.2f} seconds")
        # self.ConfigureStopTime(simulation_time)
        # self.ExecuteSimulation()

        self.sim_time += simulation_time



        self.reward = self.get_task_reward()
        self.cum_reward += self.reward

        print(f"Run Sim Cum Reward: {self.cum_reward}")

        self._step()

        
    

    def delete_event(self, event_name) -> None:
        """Remove an event from the event map.

        Makes event checking faster. Due to a performance issue in Basilisk, it is
        necessary to remove created for tasks that are no longer needed (even if it is
        inactive), or else significant time is spent processing the event at each step.
        """
        event = self.eventMap[event_name]
        self.eventList.remove(event)
        del self.eventMap[event_name]

    def __del__(self):
        """Log when simulator is deleted."""
        # logger.debug("Basilisk simulator deleted")
        print("Basilisk simulator deleted")

    # @property
    # def sim_time_ns(self) -> int:
    #     """Simulation time in ns, tied to SimBase integrator."""
    #     return self.TotalSim.CurrentNanos

    # @property
    # def sim_time(self) -> float:
    #     """Simulation time in seconds, tied to SimBase integrator."""
    #     return self.sim_time_ns * mc.NANO2SEC
    
    def _get_reward(self):
        return 1.0
    
    def _get_terminated(self):
        return False


if __name__ == "__main__":
    import time
    from line_profiler import LineProfiler

    # Create a LineProfiler object
    profiler = LineProfiler()

    # Wrap the Simulator.__init__ method with the profiler
    profiler.add_function(Simulator.__init__)

    # Define a function to create simulators
    @profiler
    def create_simulators(n):
        for i in range(n):
            print(f"Creating simulator {i+1}/{n}")
            sim = Simulator()
            print("Get obs")
            sim.get_obs()
            print("Take action")
            sim.take_action([0, 0])
            print("Run")
            sim.run()
            print("Del")
            del sim

    # Run the profiled function
    start = time.time()
    create_simulators(10)  # Reduced to 10 for quicker profiling, adjust as needed
    stop = time.time()

    print(f"Time to create 10 simulators: {stop - start}")

    # Print the profiling results
    profiler.print_stats()
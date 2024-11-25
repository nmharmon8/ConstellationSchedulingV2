import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from rl.sat import SatelliteTask, Satellite, Actions, create_random_satellite
from rl.tasks.task import TaskType
from bsk_rl.sim.world import BasicWorldModel
from tests.mocks import MockTask, MockDynamics, MockFSW

class TestSatelliteTask:
    @pytest.fixture
    def mock_satellite(self):
        sat = Mock()
        sat.storage_level = 1000
        sat.storage_capacity = 2000
        sat.dynamics = MockDynamics()
        sat.fsw = MockFSW()
        sat.pct_power = lambda: 0.5
        sat.pct_storage = lambda: 0.4
        sat.in_eclipse = lambda: False
        sat.get_task_storage_change = lambda x: 100
        sat.get_power_change = lambda x: 50
        sat.is_alive = lambda: True
        return sat

    def test_init(self, mock_satellite):
        """
        Test initialization of SatelliteTask with normal conditions.
        Verifies that initial values are correctly set and task is expected to complete
        under normal operating conditions.
        """
        task = MockTask()
        sat_task = SatelliteTask(task, mock_satellite)
        
        assert sat_task.init_storage == 1000
        assert sat_task.init_power == 100
        assert sat_task.init_alive == True
        assert sat_task.predicted_storage_change == 100
        assert sat_task.predicted_power_change == 50
        assert sat_task.expect_task_to_complete == True

    def test_task_complete(self, mock_satellite):
        """
        Test task completion under normal conditions.
        Verifies that final values are correctly set and power/storage validity is properly checked.
        """
        task = MockTask()
        sat_task = SatelliteTask(task, mock_satellite)
        sat_task.task_complete()
        
        assert sat_task.final_storage == 1000
        assert sat_task.final_power == 100
        assert sat_task.final_alive == True
        assert sat_task.final_storage_change == 0
        assert sat_task.final_power_change == 0
        assert sat_task.power_storage_valid == True

    def test_low_power_validation(self, mock_satellite):
        """
        Test task validation when satellite has low power.
        Verifies that tasks are marked as invalid when power is below threshold.
        """
        mock_satellite.pct_power = lambda: 0.02  # 2% power
        task = MockTask(TaskType.IMAGING)
        sat_task = SatelliteTask(task, mock_satellite)
        sat_task.task_complete()
        
        assert sat_task.expect_task_to_complete == False
        assert sat_task.sat_task_valid == False

    def test_full_storage_validation(self, mock_satellite):
        """
        Test task validation when satellite storage is full.
        Verifies that collection tasks are marked as invalid when storage is near capacity.
        """
        mock_satellite.storage_level = 1900  # Almost full
        mock_satellite.pct_storage = lambda: 0.98
        task = MockTask(TaskType.IMAGING)
        sat_task = SatelliteTask(task, mock_satellite)
        sat_task.task_complete()  # Need to call this to set power_storage_valid
        
        assert sat_task.power_storage_valid == False
        assert sat_task.sat_task_valid == False

    def test_charge_in_eclipse_validation(self, mock_satellite):
        """
        Test charge task validation when satellite is in eclipse.
        Verifies that charge tasks are marked as invalid during eclipse periods.
        """
        mock_satellite.in_eclipse = lambda: True
        task = MockTask(TaskType.CHARGE)
        sat_task = SatelliteTask(task, mock_satellite)
        
        assert sat_task.expect_task_to_complete == False
        assert sat_task.sat_task_valid == False

    def test_desat_always_valid(self, mock_satellite):
        """
        Test that desaturation tasks are always considered valid.
        Verifies that desat tasks are valid regardless of power or storage conditions.
        """
        mock_satellite.pct_power = lambda: 0.05  # Low power
        mock_satellite.pct_storage = lambda: 0.95  # High storage
        task = MockTask(TaskType.DESAT)
        sat_task = SatelliteTask(task, mock_satellite)
        
        assert sat_task.expect_task_to_complete == True
        assert sat_task.sat_task_valid == True

class TestSatellite:
    @pytest.fixture
    def mock_simulator(self):
        sim = Mock()
        sim.sim_time = 0
        sim.max_step_duration_sec = 100
        # Create a proper world mock that's a subclass of BasicWorldModel
        class MockWorld(BasicWorldModel):
            def __init__(self):
                pass
        sim.world = MockWorld()
        return sim

    @patch('rl.sat.dyn.GroundStationDynModel')
    @patch('rl.sat.fsw.ContinuousImagingFSWModel')
    @patch('rl.sat.TrajectorySimulator')
    def test_create_random_satellite(self, mock_traj, mock_fsw, mock_dyn, mock_simulator):
        """
        Test satellite creation with default parameters.
        Verifies that a new satellite is initialized with correct default values.
        """
        utc_init = datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)")
        sat = create_random_satellite("test-sat", mock_simulator, utc_init)
        
        assert sat.name == "test-sat"
        assert sat.action == Actions.DRIFT
        assert sat.last_action_reward == 0
        assert sat.sat_task is None

    @patch('rl.sat.dyn.GroundStationDynModel')
    @patch('rl.sat.fsw.ContinuousImagingFSWModel')
    @patch('rl.sat.TrajectorySimulator')
    def test_get_power_change(self, mock_traj, mock_fsw, mock_dyn, mock_simulator):
        """
        Test power change calculations for different task types.
        Verifies correct power change estimates for charging, desat, and collection tasks.
        """
        utc_init = datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)")
        sat = create_random_satellite("test-sat", mock_simulator, utc_init)
        
        # Mock the dynamics attributes needed for power_change calculations
        sat.dynamics.powerMonitor.storageCapacity = 200
        sat.dynamics.battery_charge = 100
        sat.dynamics.instrumentPowerSink.nodePowerOut = 50
        
        # Test charge task not in eclipse
        task = MockTask(TaskType.CHARGE)
        sat.in_eclipse = lambda: False
        power_change = sat.get_power_change(task)
        assert power_change == 100  # 200 - 100
        
        # Test charge task in eclipse
        sat.in_eclipse = lambda: True
        power_change = sat.get_power_change(task)
        assert power_change == 0
        
        # Test desat task
        task = MockTask(TaskType.DESAT)
        power_change = sat.get_power_change(task)
        assert power_change == 10000
        
        # Test collection task
        task = MockTask(TaskType.IMAGING)
        power_change = sat.get_power_change(task)
        assert power_change == -5000  # -50 * 100

    @patch('rl.sat.dyn.GroundStationDynModel')
    @patch('rl.sat.fsw.ContinuousImagingFSWModel')
    @patch('rl.sat.TrajectorySimulator')
    def test_task_actions(self, mock_traj, mock_fsw, mock_dyn, mock_simulator):
        """
        Test satellite action selection for different task types.
        Verifies that correct actions are set based on task type and conditions.
        """
        utc_init = datetime.now().strftime("%Y %b %d %H:%M:%S.%f (UTC)")
        sat = create_random_satellite("test-sat", mock_simulator, utc_init)
        
        # Test downlink action
        task = MockTask(TaskType.DATA_DOWNLINK)
        sat._task_started(task, 0)
        assert sat.action == Actions.DOWNLINK
        
        # Test charge action (not in eclipse)
        task = MockTask(TaskType.CHARGE)
        sat.in_eclipse = lambda: False
        sat._task_started(task, 0)
        assert sat.action == Actions.CHARGE
        
        # Test charge action (in eclipse)
        sat.in_eclipse = lambda: True
        sat._task_started(task, 0)
        assert sat.action == Actions.DRIFT
        
        # Test desat action  
        task = MockTask(TaskType.DESAT)
        sat._task_started(task, 0)
        assert sat.action == Actions.DESAT
        
        # Test collection action
        task = MockTask(TaskType.IMAGING)
        sat._task_started(task, 0)
        assert sat.action == Actions.COLLECTION

if __name__ == "__main__":
    pytest.main([__file__])

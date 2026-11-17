"""Fused controller tests, runnable without ROS or CUDA.

Load unchanged controller definitions while excluding ROS imports. This keeps
numerical tests usable on simulation-only developer machines.
"""
import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def load_controller():
    path = SCRIPTS / 'mppi_node.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed = {'dataclasses', 'functools', 'typing', 'time', 'jax', 'jax.numpy',
               'numpy', 'utils', 'occupancy_grid', 'controller_config', 'dynamic_bicycle',
               'selected_path_publisher', 'os'}
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 or isinstance(node, ast.ImportFrom) and node.module in allowed
                 or isinstance(node, ast.Import)
                 and all(alias.name in allowed for alias in node.names)]
    module = types.ModuleType('_test_mppi_node')
    module.Node = object
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


controller = load_controller()


class ControlStepTest(unittest.TestCase):
    def setUp(self):
        self.cfg = controller.MPPIConfig(H=5, T=8)
        self.grid = controller.og.OccupancyGrid.create(
            width=1.0, height=2.0, middle=(0.5, 0.3), resolution=0.2,
            num_beams=21, distance_cap=0.2, inflation_radius=0.1)
        x = np.linspace(0, 5, 51, dtype=np.float32)
        self.course = jnp.asarray(np.column_stack((x, x * 0, x * 0, x * 0 + 2)))
        self.state = controller.ControllerState(
            jax.random.PRNGKey(7),
            jnp.full((self.cfg.H, self.cfg.NU), 0.05),
            jnp.broadcast_to(jnp.array(self.cfg.U_STD), (self.cfg.H, self.cfg.NU)))
        self.observation = controller.ControlObservation(
            jnp.array([0., 0., 0.1, 1.]), jnp.full((21,), 3.),
            jnp.float32(0.017), jnp.bool_(False))

    def legacy_step(self, state, obs, cfg):
        # Independent reconstruction of the former callback's compute path.
        U, std = state.U, state.U_std
        if bool(obs.reset):
            U = jnp.zeros_like(U)
            std = jnp.broadcast_to(jnp.array(cfg.U_STD), std.shape)
        U = controller.shift_sequence(U, obs.elapsed, U[-1], cfg.DT)
        if cfg.COV_UPDATE:
            std = controller.shift_sequence(std, obs.elapsed, jnp.array(cfg.U_STD), cfg.DT)
        grid = self.grid.from_scan(obs.ranges)
        ref = controller.calc_interpolated_ref_trajectory_gpu(
            obs.x0[0], obs.x0[1], *self.course.T, cfg.DT, cfg.H,
            v0=obs.x0[3], max_accel=cfg.MAX_ACCEL)
        key, subkey = jax.random.split(state.key)
        if cfg.COV_UPDATE:
            U, std, samples, weights = controller.mppi_step_cov(
                subkey, U, std, obs.x0, ref, cfg, grid)
        else:
            U, samples, weights = controller.mppi_step(
                subkey, U, obs.x0, ref, cfg, grid)
        return controller.ControllerState(key, U, std), grid, ref, samples, weights

    def test_parity_and_visualization(self):
        for covariance in (False, True):
            cfg = controller.replace(self.cfg, COV_UPDATE=covariance)
            state = self.state
            for elapsed, reset in ((0., False), (0.017, False), (0.05, True)):
                obs = self.observation._replace(
                    elapsed=jnp.float32(elapsed), reset=jnp.bool_(reset))
                expected, grid, ref, samples, weights = self.legacy_step(state, obs, cfg)
                for export_path in (False, True):
                    with self.subTest(covariance=covariance, elapsed=elapsed,
                                      reset=reset, export_path=export_path):
                        result = controller.control_step(
                            state, obs, self.course, self.grid, cfg, export_path=export_path)
                        for actual, wanted in zip(result.state, expected):
                            np.testing.assert_allclose(actual, wanted, rtol=2e-5, atol=2e-5)
                        np.testing.assert_allclose(result.action, expected.U[0], atol=2e-5)
                        np.testing.assert_array_equal(result.grid.cells, grid.cells)
                        np.testing.assert_allclose(result.grid.free_distance, grid.free_distance)
                        if export_path:
                            np.testing.assert_allclose(result.path_xy, controller.simulate(
                                obs.x0, result.state.U, cfg)[:, :2], atol=2e-5)
                        else:
                            self.assertIsNone(result.path_xy)
                state = expected

    def test_resident_dispatch_requires_no_host_transfers(self):
        for export_path in (False, True):
            args = (self.state, self.observation, self.course, self.grid, self.cfg)
            jax.block_until_ready(controller.control_step(*args, export_path=export_path))
            with jax.transfer_guard('disallow'):
                result = controller.control_step(*args, export_path=export_path)
                jax.block_until_ready(result)
            self.assertEqual(result.action.devices(), self.state.U.devices())

    def test_callback_dispatches_once(self):
        node = object.__new__(controller.MPPI)
        node.control_state, node.course, node.grid, node.config = (
            self.state, self.course, self.grid, self.cfg)
        node.tracker = None
        node.get_clock = Mock()
        node._prepare_control_observation = Mock(return_value=self.observation)
        node.path_publisher = None
        node._publish_control_result = Mock()
        node._log_control_stats = Mock()
        with patch.object(controller, 'control_step', wraps=controller.control_step) as dispatch:
            node.control_callback()
            self.assertEqual(dispatch.call_count, 1)
        node._publish_control_result.assert_called_once()

    def test_warmup_preserves_live_state(self):
        node = object.__new__(controller.MPPI)
        node.control_state, node.course, node.grid, node.config = (
            self.state, self.course, self.grid, self.cfg)
        node.tracker = None
        node.path_publisher = Mock()
        node.get_logger = Mock()
        with patch.object(controller, 'control_step', wraps=controller.control_step) as dispatch:
            node._warm_up()
            dispatch.assert_called_once()
            self.assertTrue(dispatch.call_args.kwargs['export_path'])
        self.assertIs(node.control_state, self.state)


class DeviceSelectionTest(unittest.TestCase):
    def make_node(self, use_cpu):
        node = object.__new__(controller.MPPI)
        node.declare_parameter = Mock()
        node.get_parameter = Mock(return_value=types.SimpleNamespace(value=use_cpu))
        node.get_logger = Mock()
        return node

    def test_explicit_backend_selection(self):
        for use_cpu, backend in ((True, 'cpu'), (False, 'gpu')):
            with self.subTest(use_cpu=use_cpu):
                node = self.make_node(use_cpu)
                device = object()
                with patch.object(jax, 'devices', return_value=[device]) as devices, \
                        patch.object(jax.config, 'update') as update:
                    node._configure_device()
                node.declare_parameter.assert_called_once_with('use_cpu', False)
                devices.assert_called_once_with(backend)
                update.assert_called_once_with('jax_default_device', device)
                self.assertIs(node.device, device)

    def test_gpu_unavailable_does_not_fall_back(self):
        node = self.make_node(False)
        with patch.object(jax, 'devices', side_effect=RuntimeError('no GPU')) as devices:
            with self.assertRaisesRegex(RuntimeError, 'use_cpu:=true'):
                node._configure_device()
        devices.assert_called_once_with('gpu')

    def test_empty_backend_fails(self):
        node = self.make_node(False)
        with patch.object(jax, 'devices', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'No JAX gpu devices'):
                node._configure_device()


class LaunchDeviceTest(unittest.TestCase):
    def load_launch(self, name):
        launch = types.ModuleType('launch')
        launch.LaunchDescription = list
        actions = types.ModuleType('launch.actions')
        actions.DeclareLaunchArgument = lambda name, **kwargs: (name, kwargs)
        substitutions = types.ModuleType('launch.substitutions')
        substitutions.LaunchConfiguration = lambda name: ('configuration', name)
        ros = types.ModuleType('launch_ros.actions')
        ros.Node = lambda **kwargs: types.SimpleNamespace(**kwargs)
        parameters = types.ModuleType('launch_ros.parameter_descriptions')
        parameters.ParameterValue = lambda value, value_type: (value, value_type)
        namespace = {}
        path = SCRIPTS.parent / 'launch' / name
        with patch.dict(sys.modules, {
                'launch': launch, 'launch.actions': actions,
                'launch.substitutions': substitutions,
                'launch_ros': types.ModuleType('launch_ros'),
                'launch_ros.actions': ros,
                'launch_ros.parameter_descriptions': parameters}):
            exec(compile(path.read_text(), str(path), 'exec'), namespace)
            return namespace['generate_launch_description']()

    def test_simulation_launches_use_cpu(self):
        for name in ('levine_launch.py', 'spielberg_launch.py'):
            node, = self.load_launch(name)
            self.assertIs(node.parameters[0]['use_cpu'], True)
            self.assertEqual(node.parameters[0]['odom_topic'], '/ego_racecar/odom')

    def test_car_requires_gpu_and_explicit_odometry(self):
        *arguments, node = self.load_launch('car_launch.py')
        defaults = dict(arguments)
        self.assertNotIn('default_value', defaults['odom_topic'])
        self.assertEqual(defaults['track']['default_value'], 'levine')
        self.assertEqual(defaults['use_cpu']['default_value'], 'false')
        self.assertEqual(defaults['dynamics']['default_value'], 'kinematic')
        self.assertEqual(node.parameters[0]['use_cpu'],
                         (('configuration', 'use_cpu'), bool))
        self.assertEqual(node.parameters[0]['dynamics'], ('configuration', 'dynamics'))
        self.assertEqual(node.parameters[0]['odom_topic'],
                         ('configuration', 'odom_topic'))


if __name__ == '__main__':
    unittest.main()

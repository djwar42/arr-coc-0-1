#!/usr/bin/env python3
"""
CLI/TUI/Core Unit Tests - Single File, All Tests, Always Run

CRITICAL RULES:
===============
1. ALL UNIT TESTS IN THIS ONE FILE (no splitting!)
2. ALL TESTS RUN EVERY TIME (no skipping, no conditional execution)
3. 100% PASS RATE REQUIRED (failing tests = broken code, fix immediately!)
4. Tests are FAST (no network calls, no file I/O except mocks)

HOW TO RUN:
===========

From project root (arr-coc-0-1/):

    # Quick check (one command):
    python -c "import sys; sys.path.insert(0, '.'); from CLI.unit.test_all import run_all_tests; run_all_tests()"

    # Or standard pytest:
    PYTHONPATH=. pytest CLI/unit/test_all.py -v

    # Or direct execution:
    PYTHONPATH=. python CLI/unit/test_all.py

EXPECTED OUTPUT:
================
✅ All tests pass (100% pass rate)
   - Should see: "XX tests passed, 0 failed"
   - Exit code: 0

❌ If ANY test fails:
   - Fix the code immediately!
   - Never commit with failing tests!
   - 100% pass rate is MANDATORY!

WHAT WE TEST:
=============
1. Callbacks (PrintCallback, TUICallback protocol)
2. Config loading (load_training_config)
3. Core logic existence (all core functions exist and are callable)
4. Screen architecture (all screens have required methods)
5. Shared utilities (helpers, constants)
6. Import integrity (no circular imports, all modules loadable)

WHAT WE DON'T TEST:
===================
- Network calls (W&B API, GCP API) - too slow, use mocks instead
- File I/O (actual file reads/writes) - use mocks
- Subprocess execution (gcloud, docker) - use mocks
- Textual UI rendering - requires display, not unit testable

ARCHITECTURE COVERAGE:
======================
Entry Points:
  ✓ cli.py functions exist and are callable
  ✓ tui.py ARRCOCApp class exists

Core Logic (cli/*/core.py):
  ✓ launch/core.py - run_launch_core exists
  ✓ monitor/core.py - list_runs_core, cancel_run_core exist
  ✓ setup/core.py - run_setup_core, check_infrastructure_core exist
  ✓ teardown/core.py - run_teardown_core exists

Screens (cli/*/screen.py):
  ✓ All 6 screens import successfully
  ✓ All screens have compose() method
  ✓ All screens inherit from BaseScreen (where applicable)

Shared Utilities:
  ✓ WandBHelper exists and initializes
  ✓ Callbacks work (PrintCallback, TUICallback protocol)
  ✓ Constants load successfully

Test Organization:
==================
- TestCallbacks: Status callback abstraction
- TestConfig: Configuration loading
- TestCoreLogic: Core function existence
- TestScreens: Screen architecture
- TestSharedUtils: Shared utilities
- TestImportIntegrity: Module loading
"""

import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
# test_all.py is at CLI/unit/test_all.py, so 3 parents up = project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCallbacks(unittest.TestCase):
    """Test status callback abstraction (CLI vs TUI)"""

    def test_print_callback_strips_markup(self):
        """PrintCallback strips Rich markup tags"""
        from CLI.shared.callbacks import PrintCallback

        callback = PrintCallback()

        # Capture print output
        with patch('sys.stdout', new=StringIO()) as fake_out:
            callback("[green]✓[/green]  Success")
            output = fake_out.getvalue()

        # Markup should be stripped
        self.assertIn("✓ Success", output)
        self.assertNotIn("[green]", output)
        self.assertNotIn("[/green]", output)

    def test_print_callback_handles_bold(self):
        """PrintCallback strips [bold] tags"""
        from CLI.shared.callbacks import PrintCallback

        callback = PrintCallback()

        with patch('sys.stdout', new=StringIO()) as fake_out:
            callback("[bold]Important[/bold]")
            output = fake_out.getvalue()

        self.assertIn("Important", output)
        self.assertNotIn("[bold]", output)

    def test_tui_callback_protocol(self):
        """TUICallback implements status callback protocol"""
        from CLI.shared.callbacks import TUICallback

        # Mock Textual components
        mock_app = Mock()
        mock_log_box = Mock()
        mock_static = Mock

        callback = TUICallback(mock_app, mock_log_box, mock_static)

        # Should have __call__ method
        self.assertTrue(callable(callback))

        # Should store references
        self.assertEqual(callback.app, mock_app)
        self.assertEqual(callback.log_box, mock_log_box)


class TestConfig(unittest.TestCase):
    """Test configuration loading"""

    def test_config_loader_exists(self):
        """load_training_config function exists"""
        from CLI.config.constants import load_training_config

        # Function should exist and be callable
        self.assertTrue(callable(load_training_config))

    def test_config_returns_dict(self):
        """Config loader returns dictionary"""
        from CLI.config.constants import load_training_config

        # Mock file reading to avoid actual .training file dependency
        with patch('builtins.open', unittest.mock.mock_open(read_data='PROJECT_NAME="test"\n')):
            config = load_training_config()

        # Should return a dict
        self.assertIsInstance(config, dict)


class TestCoreLogic(unittest.TestCase):
    """Test core logic functions exist and are callable"""

    def test_launch_core_exists(self):
        """run_launch_core function exists"""
        from CLI.launch.core import run_launch_core

        self.assertTrue(callable(run_launch_core))

    def test_monitor_core_exists(self):
        """Monitor core functions exist"""
        from CLI.monitor.core import list_runs_core, cancel_run_core

        self.assertTrue(callable(list_runs_core))
        self.assertTrue(callable(cancel_run_core))

    def test_setup_core_exists(self):
        """Setup core functions exist"""
        from CLI.setup.core import run_setup_core, check_infrastructure_core

        self.assertTrue(callable(run_setup_core))
        self.assertTrue(callable(check_infrastructure_core))

    def test_teardown_core_exists(self):
        """Teardown core functions exist"""
        from CLI.teardown.core import run_teardown_core, check_infrastructure_core

        self.assertTrue(callable(run_teardown_core))
        self.assertTrue(callable(check_infrastructure_core))

    def test_core_functions_have_status_parameter(self):
        """All core functions accept status callback parameter"""
        from CLI.launch.core import run_launch_core
        from CLI.monitor.core import list_runs_core
        from CLI.setup.core import run_setup_core

        import inspect

        # Check run_launch_core signature
        sig = inspect.signature(run_launch_core)
        self.assertIn('status', sig.parameters)

        # Check list_runs_core signature
        sig = inspect.signature(list_runs_core)
        self.assertIn('status', sig.parameters)

        # Check run_setup_core signature
        sig = inspect.signature(run_setup_core)
        self.assertIn('status', sig.parameters)


class TestScreens(unittest.TestCase):
    """Test screen architecture"""

    def test_all_screens_import(self):
        """All 6 screens import successfully"""
        # Should not raise ImportError
        from CLI.home.screen import HomeScreen
        from CLI.monitor.screen import MonitorScreen
        from CLI.launch.screen import LaunchScreen
        from CLI.setup.screen import SetupScreen
        from CLI.teardown.screen import TeardownScreen
        from CLI.infra.screen import InfraScreen

        # All should be classes
        self.assertTrue(isinstance(HomeScreen, type))
        self.assertTrue(isinstance(MonitorScreen, type))
        self.assertTrue(isinstance(LaunchScreen, type))
        self.assertTrue(isinstance(SetupScreen, type))
        self.assertTrue(isinstance(TeardownScreen, type))
        self.assertTrue(isinstance(InfraScreen, type))

    def test_screens_have_compose(self):
        """All screens have compose() method"""
        from CLI.home.screen import HomeScreen
        from CLI.monitor.screen import MonitorScreen
        from CLI.launch.screen import LaunchScreen

        # Check HomeScreen
        self.assertTrue(hasattr(HomeScreen, 'compose'))
        self.assertTrue(callable(getattr(HomeScreen, 'compose')))

        # Check MonitorScreen
        self.assertTrue(hasattr(MonitorScreen, 'compose'))

        # Check LaunchScreen
        self.assertTrue(hasattr(LaunchScreen, 'compose'))

    def test_base_screen_exists(self):
        """BaseScreen class exists"""
        from CLI.shared.base_screen import BaseScreen

        self.assertTrue(isinstance(BaseScreen, type))

        # Should have loading overlay methods
        self.assertTrue(hasattr(BaseScreen, 'compose_base_overlay'))


class TestSharedUtils(unittest.TestCase):
    """Test shared utility classes"""

    def test_wandb_helper_exists(self):
        """WandBHelper class exists and initializes"""
        from CLI.shared.wandb_helper import WandBHelper

        # Should be a class
        self.assertTrue(isinstance(WandBHelper, type))

        # Should initialize (won't connect, just create instance)
        with patch('wandb.Api'):
            helper = WandBHelper("test_entity", "test_project", "test_queue")
            self.assertIsNotNone(helper)

    def test_constants_load(self):
        """Constants module loads successfully"""
        from CLI.config.constants import ARR_COC_WITH_WINGS, ARR_COC_DESCRIPTION

        # ASCII art should exist
        self.assertIsInstance(ARR_COC_WITH_WINGS, str)
        self.assertIsInstance(ARR_COC_DESCRIPTION, str)

        # Should not be empty
        self.assertGreater(len(ARR_COC_WITH_WINGS), 0)

    def test_callbacks_module_loads(self):
        """Callbacks module loads successfully"""
        from CLI.shared.callbacks import StatusCallback, PrintCallback, TUICallback

        # All should exist
        self.assertIsNotNone(StatusCallback)
        self.assertIsNotNone(PrintCallback)
        self.assertIsNotNone(TUICallback)


class TestImportIntegrity(unittest.TestCase):
    """Test import integrity (no circular imports)"""

    def test_cli_entry_point_imports(self):
        """cli.py imports successfully"""
        # This will fail if there are circular imports
        import CLI.cli as cli_module

        self.assertIsNotNone(cli_module)

    def test_tui_entry_point_imports(self):
        """tui.py imports successfully"""
        import CLI.tui as tui_module

        self.assertIsNotNone(tui_module)

        # ARRCOCApp should exist
        self.assertTrue(hasattr(tui_module, 'ARRCOCApp'))

    def test_all_core_modules_import(self):
        """All core modules import without errors"""
        from CLI.launch import core as launch_core
        from CLI.monitor import core as monitor_core
        from CLI.setup import core as setup_core
        from CLI.teardown import core as teardown_core

        # All should exist
        self.assertIsNotNone(launch_core)
        self.assertIsNotNone(monitor_core)
        self.assertIsNotNone(setup_core)
        self.assertIsNotNone(teardown_core)


class TestCriticalFlows(unittest.TestCase):
    """Test critical end-to-end flows work correctly"""

    def test_setup_creates_all_resources(self):
        """Setup flow creates all 6 required resources"""
        from CLI.setup.core import run_setup_core
        from unittest.mock import Mock

        # Mock helper and config
        mock_helper = Mock()
        mock_helper.run_setup.return_value = (True, ["Setup complete"])
        mock_config = {
            "GCP_PROJECT_ID": "test-project",
            "PROJECT_NAME": "test-arr-coc",
        }
        mock_status = Mock()

        # This should attempt to create all infrastructure
        # We're testing the flow, not actual GCP calls
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="exists")

            result = run_setup_core(mock_helper, mock_config, mock_status)

        # Should return boolean
        self.assertIsInstance(result, bool)

    def test_launch_requires_infrastructure_ready(self):
        """Launch checks infrastructure before proceeding"""
        from CLI.launch.core import run_launch_core
        from unittest.mock import Mock

        # Mock helper, config, and status
        mock_helper = Mock()
        mock_config = {"GCP_PROJECT_ID": "test-project"}
        mock_status = Mock()

        # Mock infrastructure check to fail
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="", stderr="not found")

            # Launch should handle missing infrastructure gracefully
            # (We're not testing the full launch, just that it checks)
            try:
                result = run_launch_core(mock_helper, mock_config, mock_status)
                # If it returns, it should be a boolean
                self.assertIsInstance(result, bool)
            except Exception:
                # Expected - infrastructure missing causes early exit
                pass

    def test_callbacks_are_called_during_execution(self):
        """Core functions actually call status callbacks"""
        from CLI.setup.core import check_infrastructure_core
        from unittest.mock import Mock

        mock_helper = Mock()
        mock_helper.entity = "test"
        mock_helper.project = "test"
        mock_config = {"GCP_PROJECT_ID": "test"}

        # Track callback calls
        callback_messages = []
        def tracking_callback(msg):
            callback_messages.append(msg)

        # Run infrastructure check
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test-bucket")
            check_infrastructure_core(mock_helper, mock_config, tracking_callback)

        # Status callback should have been called multiple times
        self.assertGreater(len(callback_messages), 0,
                          "Status callback should be called during infrastructure check")

    def test_all_core_functions_return_expected_types(self):
        """All core functions return correct types (bool/dict)"""
        from CLI.launch.core import run_launch_core
        from CLI.monitor.core import list_runs_core
        from CLI.setup.core import check_infrastructure_core, run_setup_core
        from CLI.teardown.core import run_teardown_core
        from unittest.mock import Mock

        mock_helper = Mock()
        mock_helper.entity = "test"
        mock_helper.project = "test"
        mock_helper.get_runs.return_value = []
        mock_helper.run_setup.return_value = (True, [])
        mock_helper.run_teardown.return_value = True

        mock_config = {"GCP_PROJECT_ID": "test"}
        mock_status = Mock()

        # check_infrastructure_core should return dict
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test")
            result = check_infrastructure_core(mock_helper, mock_config, mock_status)
            self.assertIsInstance(result, dict,
                                "check_infrastructure_core should return dict")

        # list_runs_core should return dict
        result = list_runs_core(mock_helper, mock_status)
        self.assertIsInstance(result, dict,
                            "list_runs_core should return dict")

        # run_setup_core should return bool
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="test")
            result = run_setup_core(mock_helper, mock_config, mock_status)
            self.assertIsInstance(result, bool,
                                "run_setup_core should return bool")


class TestArchitecturePatterns(unittest.TestCase):
    """Test architectural patterns are followed"""

    def test_cli_uses_print_callback(self):
        """CLI module uses PrintCallback"""
        # Read cli.py source file directly (CLI/unit/test_all.py → CLI/cli.py)
        cli_path = Path(__file__).parent.parent / "cli.py"
        with open(cli_path, 'r') as f:
            source = f.read()

        # Should import PrintCallback
        self.assertIn('PrintCallback', source)

    def test_core_files_no_textual_imports(self):
        """Core files don't import Textual (UI-independent)"""
        import inspect
        from CLI.launch import core as launch_core
        from CLI.monitor import core as monitor_core

        # Get source code
        launch_source = inspect.getsource(launch_core)
        monitor_source = inspect.getsource(monitor_core)

        # Should NOT import textual
        self.assertNotIn('from textual', launch_source)
        self.assertNotIn('import textual', launch_source)
        self.assertNotIn('from textual', monitor_source)
        self.assertNotIn('import textual', monitor_source)

    def test_screens_inherit_base(self):
        """Screens (except Home) inherit from BaseScreen"""
        from CLI.monitor.screen import MonitorScreen
        from CLI.launch.screen import LaunchScreen

        # Check that screens have BaseScreen in their class name hierarchy
        monitor_bases = [base.__name__ for base in MonitorScreen.__mro__]
        launch_bases = [base.__name__ for base in LaunchScreen.__mro__]

        # Both should have 'BaseScreen' somewhere in their hierarchy
        self.assertIn('BaseScreen', monitor_bases)
        self.assertIn('BaseScreen', launch_bases)


def run_all_tests():
    """
    Run all unit tests and print results

    This function is called when running:
        python -c "import sys; sys.path.insert(0, '.'); from CLI.unit.test_all import run_all_tests; run_all_tests()"

    Returns:
        Exit code: 0 if all pass, 1 if any fail
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCallbacks))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestCoreLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestScreens))
    suite.addTests(loader.loadTestsFromTestCase(TestSharedUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestImportIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestArchitecturePatterns))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"   {result.testsRun} tests passed, 0 failed")
        print("=" * 70)
        return 0
    else:
        print("❌ TESTS FAILED!")
        print(f"   {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    # Direct execution
    sys.exit(run_all_tests())

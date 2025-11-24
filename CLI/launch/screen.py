# <claudes_code_comments>
# ** Function List **
# LaunchScreen.__init__(helper, config) - Initialize with W&B helper and .training config
# LaunchScreen.compose() - Build UI with content panel and button bar
# LaunchScreen.on_mount() - Display config preview DataTable
# LaunchScreen.on_button_pressed(event) - Handle back, submit, and monitor button clicks
# LaunchScreen.submit_job() - Entry point: calls core logic via thin wrapper
# LaunchScreen.ensure_gcp_agent_deployed() - Thin TUI wrapper around core.run_launch_core()
# LaunchScreen.submission_complete(success, result) - Handle submission result
# LaunchScreen.update_spinner() - Animate ASCII spinner during submission
# LaunchScreen.action_submit() - Submit button handler (keyboard 's')
# LaunchScreen.action_back() - Return to home screen
# LaunchScreen.action_monitor() - Navigate to monitor screen
# LaunchScreen._create_error_static(text) - Create red error widget without markup
#
# ** Technical Review **
# Job submission screen - thin TUI wrapper around cli/launch/core.py business logic.
# Displays training configuration preview, then calls run_launch_core() from core module.
#
# Flow: on_mount() → load config → display in DataTable → user clicks Submit →
# submit_job() → ensure_gcp_agent_deployed() → Creates TUICallback → Calls run_launch_core() →
# Core logic handles all GCP operations → submission_complete() updates UI state
#
# TUI Responsibilities (ONLY):
# - Display configuration preview in DataTable
# - Show/hide submission log box
# - Animate spinner during submission
# - Enable/disable buttons based on state
# - Mount status updates from TUICallback to log box
# - Handle success/failure UI states
#
# Core Logic (in core.py):
# - Check for existing jobs
# - Submit to W&B queue
# - Setup GCP secrets and service accounts
# - Build/check Docker runner image
# - Create/update Cloud Run Job
# - Execute runner and stream logs
# - All subprocess calls and gcloud commands
#
# Separation: Screen.py is <100 lines of UI code, core.py is 1226 lines of business logic.
# This ensures CLI and TUI can't drift - both use identical core.run_launch_core().
#
# ⚠️ IMPORTANT: ASYNC/THREAD PATTERN (DO NOT REMOVE):
# This screen uses threading for background operations. See monitor/screen.py (lines 89-129)
# for the COMPLETE pattern documentation including:
# - Dedicated worker threads with precise timing
# - Thread-safe UI updates with self.app.call_from_thread() (NOT self.call_from_thread!)
# - GIL yielding in tight loops (time.sleep(0.001) every 5 iterations)
# - Health tracking and comprehensive logging
# Reference pattern when adding any threaded operations to this screen!
# </claudes_code_comments>

import threading
from typing import Dict, Any

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Header, Label, Static

from ..shared.wandb_helper import WandBHelper
from ..shared.base_screen import BaseScreen
from ..shared.callbacks import TUICallback
from .core import run_launch_core
# COOL SPINNER - Import random char generator (42 chars: 4 rotation + 38 special)
from ..shared.cool_spinner import get_next_spinner_char


class LaunchScreen(BaseScreen):
    """Launch new job screen - Submit training configuration"""

    CSS = """
    LaunchScreen {
        height: 100vh;
    }

    /* STANDARD: Page title */
    #page-title {
        height: 1;
        min-height: 1;
        width: 100%;
        padding: 0 0 0 2;
        text-align: left;
        content-align: left middle;
        color: $accent;
        background: $surface;
        text-style: bold;
    }

    /* STANDARD: Scrollable content panel */
    #content-panel {
        height: 1fr;
        max-height: 1fr;
        overflow-y: auto;
        padding: 2;
        background: $surface-lighten-1;
    }

    /* STANDARD: Fixed button bar at bottom */
    #button-bar {
        dock: bottom;
        height: auto;
        width: 100%;
        padding: 1 2 3 2;
        background: $surface;
        layout: horizontal;
    }

    /* Button styles inherited from global CSS in tui.py */

    #config-table {
        width: 100%;
        height: auto;
        max-height: 20;
    }

    #launch-spinner-header {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
        display: none;
    }

    #launch-spinner-header.visible {
        display: block;
    }

    #submission-log {
        width: 100%;
        height: 1fr;  /* Use most available space (was 15 lines - too small!) */
        min-height: 20;  /* Minimum 20 lines */
        border: solid $accent;
        padding: 1;
        background: $surface-darken-1;
        overflow-y: auto;
        display: none;
    }

    #submission-log.visible {
        display: block;
    }
    """

    BINDINGS = [
        ("s", "submit", "Submit"),
        ("m", "monitor", "Monitor"),
        ("escape", "back", "Back"),
        ("q", "back", "Back to Home"),
    ]

    def __init__(self, helper: WandBHelper, config: Dict[str, str]):
        super().__init__(loading_message="Loading configuration...")
        self.helper = helper
        self.config = config
        self.submitting = False
        # COOL SPINNER - Uses random char generator (no need for frames list or index!)
        self.spinner_timer = None
        self.infrastructure_ready = False  # Page flag: 100% infrastructure check
        self.launch_locked = False  # Track if launch was blocked by lock

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("🚀  LAUNCH · Training Configuration", id="page-title")

        # Content panel (pre-mounted, hidden under loading overlay)
        with VerticalScroll(id="content-panel") as content:
            # Launch process description
            yield Static("\n[bold cyan]When You Launch a Training Run:[/bold cyan]\n")
            yield Static("[dim]1.[/dim] Job submitted to W&B Launch queue")
            yield Static("[dim]2.[/dim] Docker images built (if code changed) with comprehensive hashing")
            yield Static("[dim]3.[/dim] Cloud Run runner executes → submits to Vertex AI")
            yield Static("[dim]4.[/dim] Vertex AI pulls image (30s-2min) → Pod starts → Training begins")
            yield Static("")
            yield Static("[dim]💡 Use [bold]Monitor[/bold] screen to track training progress in real-time[/dim]\n")

            # Infrastructure status (updated by background check)
            yield Static("", id="infra-status")
            yield Static("")

            # Header
            yield Static("[bold cyan]Training Configuration:[/bold cyan]\n")

            # Config edit note
            yield Static("[dim]💡 To edit config: Copy [bold].training.starter[/bold] → [bold].training[/bold], then edit values[/dim]\n")

            # Config table
            table = DataTable(id="config-table", zebra_stripes=True, show_cursor=False)
            table.add_columns("Parameter", "Value")
            yield table

            # Spacing
            yield Static("")

            # Spinner header (initially hidden)
            yield Static("", id="launch-spinner-header")

            # Submission log (initially hidden)
            yield VerticalScroll(id="submission-log")

        # Button bar
        with Horizontal(id="button-bar"):
            yield Button("← Back (ESC)", id="back-btn", classes="left-btn pastel-gray")
            yield Static("", classes="spacer")
            monitor_btn = Button("Monitor (m)", id="monitor-btn", classes="action-btn pastel-blue")
            monitor_btn.display = False  # Hidden initially
            yield monitor_btn
            yield Button("Submit (s)", id="submit-btn", classes="action-btn pastel-green")

        yield Footer()

        # Loading overlay (LAST - appears on top)
        yield from self.compose_base_overlay()

    def initialize_content(self) -> Any:
        """Load configuration data (instant - already in memory)"""
        return self.config

    def finish_loading(self, data: Any = None) -> None:
        """Populate config table with loaded data"""
        super().finish_loading(data)  # Hide loading overlay

        # Populate config table
        table = self.query_one("#config-table", DataTable)
        table.clear()

        if data:
            for key, value in data.items():
                # Skip internal keys
                if key.startswith("_"):
                    continue
                table.add_row(key, str(value))

        # Start background infrastructure check (lazy, after page loads)
        self._start_infrastructure_check()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        if event.button.id == "back-btn":
            if not self.submitting:
                self.app.pop_screen()
        elif event.button.id == "submit-btn":
            if not self.submitting:
                self.submit_job()
        elif event.button.id == "monitor-btn":
            self.app.push_screen("monitor")

    def submission_complete(self, success: bool, result: str = ""):
        """Called from worker thread when submission completes"""
        submit_btn = self.query_one("#submit-btn", Button)
        back_btn = self.query_one("#back-btn", Button)
        self.submitting = False

        # Stop and hide spinner
        if self.spinner_timer:
            self.spinner_timer.stop()
            self.spinner_timer = None

        try:
            spinner_header = self.query_one("#launch-spinner-header")
            spinner_header.remove_class("visible")
        except Exception:
            pass

        if success:
            # Success - re-enable and show success
            submit_btn.disabled = False
            submit_btn.label = "Submit Another (s)"
            submit_btn.variant = "success"
            back_btn.disabled = False

            self.notify(
                f"✓ Job submitted successfully!",
                severity="information",
                timeout=30,
            )

            # Show the "View in Monitor" button
            monitor_btn = self.query_one("#monitor-btn", Button)
            monitor_btn.display = True
        else:
            # Error - re-enable with error state
            submit_btn.disabled = False
            submit_btn.label = "Retry (s)"
            submit_btn.variant = "error"
            back_btn.disabled = False

            self.notify(
                f"❌ Submission failed - see log above",
                severity="error",
                timeout=25,
            )

    def _create_error_static(self, text: str) -> Static:
        """Create a red Static widget without markup"""
        return Static(f"[red]{text}[/red]", markup=False)

    def update_spinner(self):
        """Animate ASCII spinner (random special chars between rotations!)"""
        try:
            spinner = self.query_one("#launch-spinner", Static)
            char = get_next_spinner_char()
            spinner.update(char)  # No preceding space
        except Exception:
            pass

    def ensure_gcp_agent_deployed(self) -> None:
        """
        Thin TUI wrapper around core.run_launch_core()

        UI Setup:
        - Show submission log box and spinner
        - Disable buttons during submission

        Core Logic:
        - Create TUICallback for status updates
        - Call run_launch_core() from core.py in background thread
        - Core handles all GCP operations and subprocess calls

        Result:
        - Call submission_complete() with success/failure
        """
        log_box = self.query_one("#submission-log")
        log_box.add_class("visible")
        log_box.remove_children()  # Clear previous logs

        # Scroll to log box
        log_box.scroll_visible(animate=True)

        # Show spinner header
        spinner_header = self.query_one("#launch-spinner-header")

        # Stop any existing spinner timer first
        if self.spinner_timer:
            self.spinner_timer.stop()
            self.spinner_timer = None

        # Check if spinner already exists - update it instead of removing/mounting
        existing_spinners = list(spinner_header.query("#launch-spinner"))
        if existing_spinners:
            # Reuse existing spinner - just update the text
            existing_spinners[0].update("🚀Submitting job...")
        else:
            # Create new spinner only if it doesn't exist
            spinner_header.mount(Static("🚀Submitting job...", id="launch-spinner"))

        spinner_header.add_class("visible")

        # Start spinner animation
        self.spinner_timer = self.set_interval(0.1, self.update_spinner)

        def run_in_thread():
            """Worker thread - calls core logic"""
            try:
                # Create TUICallback that mounts status updates to log_box
                callback = TUICallback(self.app, log_box, Static)

                # Call core logic (all business logic is here!)
                success = run_launch_core(self.helper, self.config, callback)

                # Update UI based on result
                self.app.call_from_thread(self.submission_complete, success)

            except Exception as e:
                # Unexpected error
                callback(f"[red]❌ Unexpected error: {str(e)}[/red]")
                self.app.call_from_thread(self.submission_complete, False)

        # Launch worker thread
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

    def submit_job(self) -> None:
        """Submit training job using core logic"""
        # CRITICAL: Check infrastructure flag before submission
        if not self.infrastructure_ready:
            self.notify(
                "❌ Infrastructure not ready! Run Setup (press 3) to configure GCP/W&B infrastructure.",
                severity="error",
                timeout=10
            )
            return

        # Change button state before starting
        self.submitting = True
        submit_btn = self.query_one("#submit-btn", Button)
        submit_btn.disabled = True
        submit_btn.label = "Submitting..."
        submit_btn.variant = "primary"

        # Disable back button during submission
        back_btn = self.query_one("#back-btn", Button)
        back_btn.disabled = True

        # Run submission (thin wrapper around core.py)
        self.ensure_gcp_agent_deployed()

    def action_submit(self) -> None:
        if not self.submitting:
            self.submit_job()

    def action_back(self) -> None:
        if not self.submitting:
            self.app.pop_screen()

    def on_unmount(self) -> None:
        """Stop spinner timer when leaving screen (avoid background updates!)"""
        if self.spinner_timer:
            self.spinner_timer.stop()
            self.spinner_timer = None

    def action_monitor(self) -> None:
        """Navigate to Monitor screen (keyboard shortcut 'm')"""
        self.app.push_screen("monitor")

    def _start_infrastructure_check(self) -> None:
        """Start background infrastructure check (called after page loads)"""
        import threading

        # Cancel previous check if still running
        if hasattr(self, '_check_cancelled'):
            self._check_cancelled = True

        # Reset cancellation flag
        self._check_cancelled = False

        # Show checking state immediately (direct call - we're in main thread)
        self._update_infra_status({"checking": True})

        def check_infra_async():
            try:
                from CLI.setup.core import check_infrastructure_core

                # Check if cancelled before starting
                if self._check_cancelled:
                    return

                class SilentCallback:
                    def __call__(self, message: str):
                        pass

                # Check infrastructure (always fresh check)
                info = check_infrastructure_core(self.helper, self.config, SilentCallback(), app=self.app)

                # Check if cancelled after check
                if self._check_cancelled:
                    return

                # Extract existence flags
                gcp = info.get("gcp", {})
                wandb_info = info.get("wandb", {})

                registry_exists = gcp.get("registry", {}).get("exists", False)
                persistent_registry_exists = gcp.get("persistent_registry", {}).get("exists", False)
                sa_exists = gcp.get("service_account", {}).get("exists", False)
                queue_exists = wandb_info.get("queue", {}).get("exists", False)

                setup_complete = (registry_exists and persistent_registry_exists and sa_exists and queue_exists)

                # Update UI and flag (only if not cancelled)
                if not self._check_cancelled:
                    self.infrastructure_ready = setup_complete
                    self.app.call_from_thread(self._update_infra_status, {"setup_complete": setup_complete})
            except Exception as e:
                # Silently fail - don't crash background thread
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=check_infra_async, daemon=True)
        thread.start()

    def _update_infra_status(self, status_data: Dict) -> None:
        """Update infrastructure status widget (called from background thread)"""
        try:
            status_widget = self.query_one("#infra-status", Static)
            submit_btn = self.query_one("#submit-btn", Button)

            # Checking state
            if status_data.get("checking", False):
                status_widget.update("[yellow]⏳ Checking infrastructure...[/yellow]")
                submit_btn.label = "Checking..."
                submit_btn.disabled = True
            # Complete state
            elif status_data.get("setup_complete", False):
                status_widget.update("[green]✓ Infrastructure: 100% Ready[/green]")
                submit_btn.label = "Submit (s)"
                submit_btn.disabled = False
            # Incomplete state
            else:
                status_widget.update("[yellow]⚠️  Infrastructure: Incomplete - Run Setup (press 3)[/yellow]")
                submit_btn.label = "Submit (s)"
                submit_btn.disabled = True
        except Exception:
            pass  # Widget might not exist yet

    def on_screen_resume(self) -> None:
        """Called when returning to this screen - re-check infrastructure"""
        self._start_infrastructure_check()

    def on_screen_suspend(self) -> None:
        """Called when leaving this screen - cancel background check"""
        if hasattr(self, '_check_cancelled'):
            self._check_cancelled = True

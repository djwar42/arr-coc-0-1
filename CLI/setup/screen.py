# <claudes_code_comments>
# ** Function List **
# SetupScreen.__init__(config) - Initialize with config and SetupHelper
# SetupScreen.compose() - Build UI skeleton (title, content panel, button bar)
# SetupScreen.initialize_content() - Check prerequisites in background (BaseScreen pattern)
# SetupScreen.finish_loading(data) - Display UI based on prerequisites (BaseScreen pattern)
# SetupScreen.mount_prerequisites_ui(queue_exists) - Show manual queue setup guide
# SetupScreen.mount_normal_ui() - Show infrastructure overview + run button
# SetupScreen.refresh_infrastructure() - Refresh infrastructure display
# SetupScreen.run_setup() - Execute setup via core logic
# SetupScreen.action_run_setup() - Button handler - start setup
# SetupScreen.action_refresh() - Refresh infrastructure details
# SetupScreen.action_back() - Return to home screen
# SetupScreen.on_button_pressed(event) - Handle button clicks
#
# ** Technical Review **
# Infrastructure setup TUI wrapper - uses BaseScreen loading overlay pattern for consistent UX.
# Manages UI state machine for prerequisites, setup execution, and queue verification.
#
# Flow: BaseScreen.on_mount() → initialize_content() (checks prerequisites in background)
# → finish_loading() → mount appropriate UI (prerequisites or normal)
# → user clicks "Run Setup" → run_setup() → calls run_setup_core()
#
# TUI Responsibilities (ONLY):
# - Display infrastructure status (GCP, W&B, HF)
# - Show/hide setup log box
# - Button state management (enable/disable)
# - Loading overlay (inherited from BaseScreen)
# - Prerequisites vs normal UI switching
# - Queue verification countdown display
#
# Core Logic (in core.py):
# - Check W&B queue prerequisites
# - Gather infrastructure info from GCP/W&B/HF
# - Execute GCP setup (via setup_helper.py)
# - All subprocess calls and API queries
#
# Separation: Screen.py manages UI state, core.py handles all infrastructure operations.
# This ensures CLI and TUI can't drift - both use identical run_setup_core().
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
from textual.widgets import Button, Footer, Header, Label, Static

from ..shared.setup_helper import SetupHelper
from ..shared.base_screen import BaseScreen
import threading
from .core import run_setup_core, check_infrastructure_core, _check_prerequisites_core
from ..shared.wandb_helper import WandBHelper


from ..shared.log_paths import get_log_path
class SetupScreen(BaseScreen):
    """Infrastructure setup screen - GCP, W&B, HuggingFace"""

    CSS = """
    SetupScreen {
        height: 100vh;
    }

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

    #content-panel {
        height: 1fr;
        overflow-y: auto;
        padding: 2;
        background: $surface-lighten-1;
    }

    #button-bar {
        dock: bottom;
        height: auto;
        width: 100%;
        padding: 1 2 3 2;
        background: $surface;
        layout: horizontal;
    }

    /* Button styles inherited from global CSS in tui.py */

    #setup-log {
        width: 100%;
        height: 30;
        border: solid $accent;
        padding: 1;
        background: $surface-darken-1;
        overflow-y: auto;
        display: none;
    }

    #setup-log.visible {
        display: block;
    }
    """

    BINDINGS = [
        ("s", "run_setup", "Run Setup"),
        ("r", "refresh", "Refresh"),
        ("escape", "back", "Back"),
        ("q", "back", "Back to Home"),
    ]

    def __init__(self, config: Dict[str, str]):
        super().__init__(loading_message="Checking infrastructure...")
        self.config = config
        self.helper = SetupHelper(config)
        self.wandb_helper = WandBHelper(
            config.get("WANDB_ENTITY", ""),
            config.get("WANDB_PROJECT", "arr-coc-0-1"),
            config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")
        )
        self.setup_running = False
        self.queue_verified = False
        self.queue_check_timer = None  # Track interval timer for cleanup
        self.setup_status_data = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("⚙️  SETUP · Infrastructure Configuration", id="page-title")

        # Content panel
        yield VerticalScroll(id="content-panel")

        # Button bar
        with Horizontal(id="button-bar"):
            yield Button("← Back (ESC)", id="back-btn", classes="left-btn pastel-gray")
            yield Static("", classes="spacer")
            yield Button("Refresh (r)", id="refresh-btn", variant="primary", classes="action-btn")
            yield Button("Run Setup (s)", id="setup-btn", variant="success", classes="action-btn")

        
        # Loading overlay (LAST - appears on top)
        yield from self.compose_base_overlay()

        yield Footer()

    def initialize_content(self):
        """
        Do all the work with 3 BREATHE POINTS to let spinner animate!

        Breathe points give main thread time to process timer callbacks
        and render 3-4 spinner frames. Total work split into phases with
        500ms pauses between.
        """
        from CLI.shared.performance_monitor import get_monitor
        import time
        monitor = get_monitor()

        # Create silent callback (no status updates to UI)
        class SilentCallback:
            def __call__(self, message: str):
                pass

        status = SilentCallback()

        # BREATHE POINT 1: Immediate animation start!
        time.sleep(0.1)  # 100ms - let spinner start

        # BREATHE POINT 2: Early animation
        time.sleep(0.1)  # 100ms - another frame

        # BREATHE POINT 3: More early animation
        time.sleep(0.1)  # 100ms - keep it going

        # BREATHE POINT 4: Final early breathe
        time.sleep(0.1)  # 100ms - establish spinner rhythm

        # PHASE 1: Check prerequisites (W&B API calls)
        op_id = monitor.start_operation("check_wandb_prerequisites", category="API")
        prereq_result = _check_prerequisites_core(self.wandb_helper, self.config, status)
        monitor.end_operation(op_id)

        # BREATHE POINT 5: Between prerequisites and infrastructure
        time.sleep(0.1)  # 100ms - give main thread a chance

        # PHASE 2: Check infrastructure (GCP subprocess calls)
        op_id = monitor.start_operation("check_infrastructure", category="GCP")
        info = check_infrastructure_core(self.wandb_helper, self.config, status, app=self.app)
        monitor.end_operation(op_id)

        # BREATHE POINT 6: After infrastructure check
        time.sleep(0.1)  # 100ms - more animation time

        # BREATHE POINT 7: Final breathe before UI update
        time.sleep(0.1)  # 100ms - last chance for animation

        # Return data - finish_loading() will handle UI update
        return {"prereq_result": prereq_result, "info": info}

    def finish_loading(self, data: Any = None):
        """
        Update UI based on loaded data (called from main thread).

        Hides loading overlay and mounts appropriate UI (setup wizard or status table)
        based on prerequisites check result.
        """
        # Hide loading overlay NOW (data loaded!)
        super().finish_loading(data)

        # Mount appropriate UI based on prerequisites
        if data:
            prereq_result = data.get("prereq_result", {})
            info = data.get("info", {})

            if prereq_result.get("queue_exists"):
                self.queue_verified = True
                self.mount_normal_ui()
                self.display_infrastructure(info)
            else:
                self.mount_prerequisites_ui(prereq_result)

    def mount_prerequisites_ui(self, prereq_result: Dict) -> None:
        """Show manual queue setup guide"""
        content = self.query_one("#content-panel")

        queue_name = prereq_result.get("queue_name", "vertex-ai-queue")
        entity = self.config.get("WANDB_ENTITY", "")
        project = self.config.get("WANDB_PROJECT", "arr-coc-0-1")

        content.mount(Static("\n[bold yellow]⚠️  Prerequisites Missing[/bold yellow]"))
        content.mount(Static(f"\nW&B Launch Queue '{queue_name}' not found.\n"))
        content.mount(Static("[bold]Manual Queue Creation Required:[/bold]"))
        content.mount(Static(f"\n1. Visit: https://wandb.ai/{entity}/{project}/launch"))
        content.mount(Static("2. Click 'Create Queue' button"))
        content.mount(Static(f"3. Name: {queue_name}"))
        content.mount(Static("4. Select 'Vertex AI' as resource"))
        content.mount(Static("5. Save queue\n"))
        content.mount(Static("[dim]This screen will auto-check every 30s for the queue...[/dim]"))

        # Disable setup button (queue not ready yet)
        setup_btn = self.query_one("#setup-btn", Button)
        setup_btn.disabled = True
        setup_btn.label = "Waiting for Queue..."

        # Start auto-checking for queue (store timer for cleanup)
        # 30s interval (not 5s) to reduce expensive W&B API calls!
        self.queue_check_timer = self.set_interval(30, self.check_for_queue_and_update)

    def check_for_queue_and_update(self) -> None:
        """Periodically check if queue was created"""
        def check():
            class SilentCallback:
                def __call__(self, message: str):
                    pass

            status = SilentCallback()
            prereq_result = _check_prerequisites_core(self.wandb_helper, self.config, status)

            if prereq_result["queue_exists"]:
                self.queue_verified = True

                # STOP THE TIMER! Queue detected, no more checking needed!
                if self.queue_check_timer:
                    self.queue_check_timer.stop()
                    self.queue_check_timer = None

                # Clear content and mount normal UI
                self.app.call_from_thread(
                    lambda: self.query_one("#content-panel").remove_children()
                )
                self.app.call_from_thread(self.mount_normal_ui)
                self.app.call_from_thread(
                    lambda: self.notify("✓ Queue detected! You can now run setup.", severity="information")
                )

        thread = threading.Thread(target=check, daemon=True)
        thread.start()

    def mount_normal_ui(self) -> None:
        """Show infrastructure overview + run button"""
        content = self.query_one("#content-panel")

        content.mount(Static("\n[bold cyan]Infrastructure Status[/bold cyan]"))
        content.mount(Static("", id="infrastructure-container"))
        content.mount(VerticalScroll(id="setup-log"))

        # Re-enable setup button (queue is ready!)
        setup_btn = self.query_one("#setup-btn", Button)
        setup_btn.disabled = False
        setup_btn.label = "Run Setup (s)"

        # Load infrastructure details
        self.refresh_infrastructure()

    def refresh_infrastructure(self) -> None:
        """Refresh infrastructure display"""
        # Use Textual's run_worker (replaces threading.Thread)
        self.run_worker(self._fetch_and_display_async, exclusive=True)

    async def _fetch_and_display_async(self):
        """Fetch and display infrastructure info (Textual worker)"""
        try:
            class SilentCallback:
                def __call__(self, message: str):
                    pass

            status = SilentCallback()

            # Get infrastructure info from core
            info = check_infrastructure_core(self.wandb_helper, self.config, status, app=self.app)

            # Display (no call_from_thread needed - we're in a worker!)
            self.display_infrastructure(info)

        except Exception as e:
            self.notify_with_full_error("Error Fetching Infrastructure", str(e))

    def display_infrastructure(self, info: Dict) -> None:
        """Display infrastructure info in UI using unified infra_print"""
        try:
            # Check if container exists (screen might have been navigated away)
            try:
                container = self.query_one("#infrastructure-container")
            except Exception:
                # Container doesn't exist - screen not in normal state or navigated away
                return

            container.remove_children()

            # 🎯 USE UNIFIED DISPLAY FUNCTION!
            from CLI.shared.infra_print import display_infrastructure as format_infra

            # Get formatted output (Rich markup enabled for TUI Static widgets)
            output = format_infra(info, use_rich=True)

            # Mount as single Static widget
            container.mount(Static(output))

        except Exception as e:
            self.notify_with_full_error("Error Displaying Infrastructure", str(e))

    def run_setup(self) -> None:
        """Execute setup via core logic"""
        if self.setup_running:
            return

        # PRINCIPLE: "Core first for CLI principle"
        # Core run_setup() handles all checks and validation
        # TUI just displays core's messages - no duplicate logic
        self.setup_running = True

        # Disable button
        setup_btn = self.query_one("#setup-btn", Button)
        setup_btn.disabled = True
        setup_btn.label = "Running..."

        # Show log box (only exists if normal UI is mounted)
        try:
            log_box = self.query_one("#setup-log")
            log_box.add_class("visible")
            log_box.remove_children()
        except Exception:
            # Log box doesn't exist yet (prerequisites UI is showing)
            # This shouldn't happen if button states are correct, but handle gracefully
            self.notify("⚠️  Setup not available yet - queue verification in progress", severity="warning")
            self.setup_running = False
            setup_btn.disabled = False
            setup_btn.label = "Run Setup (s)"
            return

        def run_in_thread():
            try:
                from CLI.shared.callbacks import TUICallback

                # Create callback that mounts to log box
                status = TUICallback(self.app, log_box, Static)

                # Call core logic
                success = run_setup_core(self.wandb_helper, self.config, status)

                # Update UI
                self.app.call_from_thread(self.setup_complete, success)

            except Exception as e:
                self.app.call_from_thread(
                    lambda: self.notify_with_full_error("Setup Error", str(e))
                )
                self.app.call_from_thread(self.setup_complete, False)

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

    def setup_complete(self, success: bool) -> None:
        """Handle setup completion"""
        self.setup_running = False

        setup_btn = self.query_one("#setup-btn", Button)
        setup_btn.disabled = False

        if success:
            setup_btn.label = "Setup Complete ✓"
            setup_btn.variant = "success"
            self.notify("✓ Infrastructure setup complete!", severity="information")

            # Refresh infrastructure display
            self.refresh_infrastructure()

            # Notify home screen to refresh (removes "Setup not detected" warning)
            try:
                from CLI.home.screen import HomeScreen
                for screen in self.app.screen_stack:
                    if isinstance(screen, HomeScreen):
                        screen.call_later(screen.on_screen_resume)
                        break
            except Exception:
                pass  # Home screen not in stack, no need to refresh
        else:
            setup_btn.label = "Retry Setup (s)"
            setup_btn.variant = "error"
            self.notify("❌ Setup failed - see log above", severity="error")

    def action_run_setup(self) -> None:
        if not self.setup_running:
            self.run_setup()

    def on_screen_resume(self) -> None:
        """Called when returning to this screen - refresh infrastructure status"""
        self.refresh_infrastructure()

    def on_screen_suspend(self) -> None:
        """Called when leaving this screen - stop background timers"""
        # Stop queue auto-check timer if running
        if self.queue_check_timer:
            self.queue_check_timer.stop()
            self.queue_check_timer = None

    def action_refresh(self) -> None:
        self.refresh_infrastructure()
        self.notify("✓ Refreshing infrastructure...", severity="information")

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_unmount(self) -> None:
        """Stop queue check timer when leaving screen (avoid background API calls!)"""
        if self.queue_check_timer:
            self.queue_check_timer.stop()
            self.queue_check_timer = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "refresh-btn":
            self.action_refresh()
        elif event.button.id == "setup-btn":
            self.action_run_setup()
        elif event.button.id == "view-infra-btn":
            # Navigate to infra screen
            from CLI.infra.screen import InfraScreen
            self.app.push_screen(InfraScreen(self.config))

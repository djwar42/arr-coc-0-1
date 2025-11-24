# <claudes_code_comments>
# ** Function List **
# TeardownScreen.__init__(config) - Initialize with config and SetupHelper
# TeardownScreen.compose() - Build UI (title, content panel, button bar)
# TeardownScreen.initialize_content() - Check infrastructure in background (BaseScreen pattern)
# TeardownScreen.finish_loading(data) - Display resources and UI (BaseScreen pattern)
# TeardownScreen.run_dry_run() - Execute dry-run via core logic
# TeardownScreen.run_teardown() - Execute real teardown via core logic
# TeardownScreen.teardown_complete(success, dry_run) - Handle teardown completion
# TeardownScreen.action_back() - Return to home screen
# TeardownScreen.on_button_pressed(event) - Handle button clicks
#
# ** Technical Review **
# Teardown TUI wrapper - uses BaseScreen loading overlay pattern for consistent UX.
# Manages UI state for dry-run preview and resource deletion confirmation.
#
# Flow: BaseScreen.on_mount() → initialize_content() (checks infrastructure in background)
# → finish_loading() → display resources → user clicks "Dry Run" or "Teardown"
# → calls run_teardown_core() → teardown_complete() updates UI based on result
#
# TUI Responsibilities (ONLY):
# - Display infrastructure status
# - Show resource list preview
# - Confirmation input ("DELETE" required)
# - Button state management (enable/disable)
# - Log box for teardown output
# - Success/failure notifications
#
# Core Logic (in core.py):
# - Check infrastructure exists (local file check)
# - List resources that would be deleted
# - Execute teardown.sh (via helper)
# - All business logic
#
# Separation: Screen.py manages UI state, core.py handles all teardown operations.
# This ensures CLI and TUI can't drift - both use identical run_teardown_core().
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

from typing import Dict, Any
import threading

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Checkbox, Footer, Header, Label, Static

from ..shared.setup_helper import SetupHelper
from ..shared.wandb_helper import WandBHelper
from ..shared.base_screen import BaseScreen
from .core import run_teardown_core, list_resources_core
from ..setup.core import check_infrastructure_core  # Use comprehensive check from setup
from ..shared.callbacks import TUICallback


class TeardownScreen(BaseScreen):
    """Interactive infrastructure teardown"""

    CSS = """
    TeardownScreen {
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

    Checkbox {
        margin: 0 1;
    }

    #teardown-log {
        height: 20;
        border: tall $error;
        background: $surface-darken-1;
        padding: 1;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        ("escape", "back", "Back"),
        ("q", "back", "Back to Home"),
    ]

    def __init__(self, helper: WandBHelper, config: Dict[str, str]):
        super().__init__(loading_message="Checking infrastructure...")
        self.wandb_helper = helper
        self.config = config
        self.setup_helper = SetupHelper(config)
        self.running = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("🗑️  TEARDOWN · Infrastructure Deletion", id="page-title")
        yield VerticalScroll(id="content-panel")

        with Horizontal(id="button-bar"):
            yield Button("← Back (ESC)", id="back-btn", classes="left-btn pastel-gray")
            yield Static("", classes="spacer")
            yield Button("Dry Run", id="dry-run-btn", classes="action-btn pastel-yellow")
            yield Button("Teardown (checking...)", id="teardown-btn", classes="action-btn pastel-red", disabled=True)

        
        # Loading overlay (LAST - appears on top)
        yield from self.compose_base_overlay()

        yield Footer()

    def initialize_content(self) -> Any:
        """Do all the work here in the background thread! No second thread needed."""
        import time

        class SilentCallback:
            def __call__(self, message: str):
                pass

        status = SilentCallback()

        try:
            # Check infrastructure (API calls in background)
            info = check_infrastructure_core(self.wandb_helper, self.config, status)

            # Get resource list
            resources = list_resources_core(self.config, status)

            return {"info": info, "resources": resources, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def finish_loading(self, data: Any = None) -> None:
        """Update UI based on loaded data (called from main thread)"""
        # Hide loading overlay NOW (data loaded!)
        super().finish_loading(data)

        # Mount warning first
        content = self.query_one("#content-panel")
        content.mount(Static("\n[bold red]⚠️  WARNING: This PERMANENTLY deletes all GCP resources![/bold red]\n"))

        if data and data.get("success"):
            info = data.get("info")
            resources = data.get("resources")
            # Update UI
            self._display_resources(info, resources)
        elif data:
            # Show error (press 'e' to see full details)
            error = data.get("error", "Unknown error")
            self.notify_with_full_error("Error Checking Infrastructure", error)

    def _display_resources(self, info: dict, resources: list):
        """Display resource information (called from background thread)"""
        content = self.query_one("#content-panel")

        # Display status and resources
        if info:
            # Extract existence flags from comprehensive check (same as setup/home)
            gcp = info.get("gcp", {})
            wandb_info = info.get("wandb", {})

            buckets = gcp.get("buckets", {})
            bucket_count = buckets.get("count", 0)
            registry_exists = gcp.get("registry", {}).get("exists", False)
            sa_exists = gcp.get("service_account", {}).get("exists", False)
            queue_exists = wandb_info.get("queue", {}).get("exists", False)

            # Infrastructure exists if ANY DELETABLE component exists
            # NOTE: W&B Queue is NOT deleted by teardown (manual only)
            # NOTE: Persistent registry is NOT deleted (PyTorch base image preservation)
            infrastructure_exists = (bucket_count > 0 or registry_exists or sa_exists)

            # Status
            if infrastructure_exists:
                content.mount(Static("[bold]Status:[/bold] [green]Infrastructure detected[/green]\n"))

                # Show detailed breakdown (only components that will be deleted)
                components = []
                if bucket_count > 0:
                    components.append(f"{bucket_count} Regional Bucket(s)")
                if registry_exists:
                    components.append("Artifact Registry (deletable)")
                if sa_exists:
                    components.append("Service Account")

                content.mount(Static(f"[dim]Will delete: {', '.join(components)}[/dim]\n"))

                # Show resources that will be deleted
                content.mount(Static("[bold]Resources that will be deleted:[/bold]"))
                resource_text = "\n".join([f"  • {r}" for r in resources])
                content.mount(Static(resource_text + "\n"))

                # Note about manual queue deletion (if queue exists)
                if queue_exists:
                    queue_name = self.config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")
                    entity = self.config.get("WANDB_ENTITY", "")
                    content.mount(Static(f"\n[yellow]Note:[/yellow] W&B Queue '[cyan]{queue_name}[/cyan]' must be deleted manually"))
                    content.mount(Static(f"[dim]→ https://wandb.ai/{entity}/launch[/dim]\n"))

                # Enable teardown button
                teardown_btn = self.query_one("#teardown-btn", Button)
                teardown_btn.disabled = False
                teardown_btn.label = "Teardown"
                teardown_btn.variant = "error"

            else:
                content.mount(Static("[bold]Status:[/bold] [dim]No infrastructure detected[/dim]\n"))
                content.mount(Static("\n[dim]All resources have been cleaned up, or setup was never run.[/dim]\n"))
                content.mount(Static("[dim]Nothing to delete.[/dim]\n\n"))

                # Note about manual queue deletion (if queue exists but nothing else)
                if queue_exists:
                    queue_name = self.config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")
                    entity = self.config.get("WANDB_ENTITY", "")
                    content.mount(Static(f"[yellow]Note:[/yellow] W&B Queue '[cyan]{queue_name}[/cyan]' still exists (manual deletion only)"))
                    content.mount(Static(f"[dim]→ https://wandb.ai/{entity}/launch[/dim]\n\n"))

                # Disable teardown button (nothing to delete)
                teardown_btn = self.query_one("#teardown-btn", Button)
                teardown_btn.disabled = True
                teardown_btn.label = "Teardown (nothing to delete)"
                teardown_btn.variant = "default"

        # Teardown selection with checkboxes
        content.mount(Static("\n[bold]Select what to delete:[/bold]"))
        content.mount(Static(""))

        # Infrastructure checkbox
        content.mount(Static("[bold]Infrastructure (buckets, service accounts, worker pools):[/bold]"))
        content.mount(Checkbox("DELETE infrastructure (keeps all arr- images)", id="checkbox-delete"))
        content.mount(Static(""))

        # Image checkboxes
        content.mount(Static("[bold]Docker Images (select one or more):[/bold]"))
        content.mount(Checkbox("arr-pytorch-base", id="checkbox-pytorch"))
        content.mount(Checkbox("arr-ml-stack", id="checkbox-ml-stack"))
        content.mount(Checkbox("arr-trainer", id="checkbox-trainer"))
        content.mount(Checkbox("arr-vertex-launcher", id="checkbox-launcher"))
        content.mount(Static("\n[dim]Select checkboxes above, then click 'Teardown' or 'Dry Run'[/dim]\n"))

        # Log area
        content.mount(Static("[bold]Teardown Log:[/bold]"))
        content.mount(VerticalScroll(id="teardown-log"))

    def _get_teardown_mode_from_checkboxes(self) -> str:
        """Read checkboxes and build mode string for core logic"""
        keywords = []

        # Check infrastructure checkbox
        if self.query_one("#checkbox-delete", Checkbox).value:
            keywords.append("DELETE")

        # Check image checkboxes
        if self.query_one("#checkbox-pytorch", Checkbox).value:
            keywords.append("ARR-PYTORCH-BASE")
        if self.query_one("#checkbox-ml-stack", Checkbox).value:
            keywords.append("ARR-ML-STACK")
        if self.query_one("#checkbox-trainer", Checkbox).value:
            keywords.append("ARR-TRAINER")
        if self.query_one("#checkbox-launcher", Checkbox).value:
            keywords.append("ARR-VERTEX-LAUNCHER")

        return " ".join(keywords)

    def run_dry_run(self) -> None:
        """Execute dry-run via core logic"""
        if self.running:
            return

        # Get selected items from checkboxes
        teardown_mode = self._get_teardown_mode_from_checkboxes()

        if not teardown_mode:
            self.notify("⚠️ Please select at least one checkbox", severity="warning")
            return

        self.running = True
        self.teardown_mode = teardown_mode  # Store what to dry-run

        # Update UI
        dry_run_btn = self.query_one("#dry-run-btn", Button)
        teardown_btn = self.query_one("#teardown-btn", Button)
        back_btn = self.query_one("#back-btn", Button)

        dry_run_btn.disabled = True
        teardown_btn.disabled = True
        back_btn.disabled = True
        dry_run_btn.label = "Running..."
        dry_run_btn.variant = "primary"

        # Clear log
        log_box = self.query_one("#teardown-log")
        log_box.remove_children()

        # Start worker (manual threading - no @work decorator)
        self.run_dry_run_worker(log_box)

    def run_dry_run_worker(self, log_box) -> None:
        """Worker thread for dry-run (keeps UI responsive)"""
        def worker():
            try:
                # Create callback that mounts to log box
                status = TUICallback(self.app, log_box, Static)

                # Call core logic (blocking subprocess - runs in thread)
                success = run_teardown_core(
                    self.setup_helper,
                    self.config,
                    status,
                    dry_run=True,
                    mode=getattr(self, 'teardown_mode', 'DELETE')  # Pass teardown mode for dry-run too!
                )

                # Update UI (thread-safe)
                self.app.call_from_thread(self.teardown_complete, success, True)

            except Exception as e:
                self.app.call_from_thread(
                    lambda: self.notify_with_full_error("Dry-run Error", str(e))
                )
                self.app.call_from_thread(self.teardown_complete, False, True)

        # Start daemon thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def run_teardown(self) -> None:
        """Execute real teardown via core logic"""
        if self.running:
            return

        # Get selected items from checkboxes
        teardown_mode = self._get_teardown_mode_from_checkboxes()

        if not teardown_mode:
            self.notify("⚠️ Please select at least one checkbox", severity="warning")
            return

        self.running = True
        self.teardown_mode = teardown_mode  # Store what to delete

        # Update UI
        dry_run_btn = self.query_one("#dry-run-btn", Button)
        teardown_btn = self.query_one("#teardown-btn", Button)
        back_btn = self.query_one("#back-btn", Button)

        dry_run_btn.disabled = True
        teardown_btn.disabled = True
        back_btn.disabled = True
        teardown_btn.label = "Deleting..."
        teardown_btn.variant = "primary"

        # Clear log
        log_box = self.query_one("#teardown-log")
        log_box.remove_children()

        # Start worker (manual threading - no @work decorator)
        self.run_teardown_worker(log_box)

    def run_teardown_worker(self, log_box) -> None:
        """Worker thread for teardown (keeps UI responsive)"""
        def worker():
            try:
                # Create callback that mounts to log box
                status = TUICallback(self.app, log_box, Static)

                # Call core logic (blocking subprocess - runs in thread)
                success = run_teardown_core(
                    self.setup_helper,
                    self.config,
                    status,
                    dry_run=False,
                    mode=getattr(self, 'teardown_mode', 'DELETE')  # Pass teardown mode
                )

                # Update UI (thread-safe)
                self.app.call_from_thread(self.teardown_complete, success, False)

            except Exception as e:
                self.app.call_from_thread(
                    lambda: self.notify_with_full_error("Teardown Error", str(e))
                )
                self.app.call_from_thread(self.teardown_complete, False, False)

        # Start daemon thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def teardown_complete(self, success: bool, dry_run: bool) -> None:
        """Handle teardown completion"""
        self.running = False

        dry_run_btn = self.query_one("#dry-run-btn", Button)
        teardown_btn = self.query_one("#teardown-btn", Button)
        back_btn = self.query_one("#back-btn", Button)

        # Re-enable back button
        back_btn.disabled = False

        # Always re-enable dry-run
        dry_run_btn.disabled = False
        dry_run_btn.label = "Dry Run"
        dry_run_btn.variant = "warning"

        if dry_run:
            # Dry-run completed
            teardown_btn.disabled = False
            teardown_btn.variant = "error"
            self.notify("ℹ️ Dry-run complete. No resources deleted.", severity="information")
        elif success:
            # Teardown succeeded
            teardown_btn.disabled = True
            teardown_btn.label = "Teardown Complete ✓"
            teardown_btn.variant = "success"
            self.notify("✓ Teardown completed successfully!", severity="information")

            # Refresh setup screen if exists
            try:
                setup_screen = self.app.get_screen("setup")
                setup_screen.refresh_infrastructure()
            except Exception:
                pass
        else:
            # Teardown failed
            teardown_btn.disabled = False
            teardown_btn.label = "Teardown"
            teardown_btn.variant = "error"
            self.notify("✗ Teardown failed. Check logs.", severity="error")

    def action_back(self) -> None:
        if not self.running:
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn" and not self.running:
            self.action_back()
        elif event.button.id == "dry-run-btn" and not self.running:
            self.run_dry_run()
        elif event.button.id == "teardown-btn" and not self.running:
            self.run_teardown()

# <claudes_code_comments>
# ** Function List **
# InfraScreen.__init__(config) - Initialize with .training config
# InfraScreen.compose() - Build UI with infrastructure DataTables
# InfraScreen.on_mount() - Load and display all GCloud infrastructure
# InfraScreen.refresh_infrastructure() - Refresh all infrastructure data
# InfraScreen.load_buckets() - Load GCS bucket information
# InfraScreen.load_artifact_registry() - Load Artifact Registry repositories
# InfraScreen.load_service_accounts() - Load service account details
# InfraScreen.load_vertex_ai_info() - Load Vertex AI configuration
# InfraScreen.load_wandb_queue() - Load W&B Launch queue info
# InfraScreen.action_refresh() - Manual refresh handler
# InfraScreen.action_back() - Return to home screen
# InfraScreen.on_button_pressed(event) - Handle button clicks
#
# ** Technical Review **
# Comprehensive GCloud infrastructure display screen. Shows all project resources in organized
# DataTables: GCS buckets, Artifact Registry repos, service accounts, Vertex AI config, W&B queues.
#
# Flow: on_mount() → load all infrastructure in parallel → populate DataTables → user can refresh
# or navigate back
#
# Infrastructure sections displayed:
#
# 1. GCS Buckets:
#    - Bucket name
#    - Location (region)
#    - Storage class (STANDARD, NEARLINE, etc.)
#    - Created date
#    - Size estimate (if available)
#
# 2. Artifact Registry:
#    - Repository name
#    - Format (Docker, Python, etc.)
#    - Location
#    - Description
#    - Created date
#
# 3. Service Accounts:
#    - Email
#    - Display name
#    - Roles/permissions
#    - Key count
#    - Status (enabled/disabled)
#
# 4. Vertex AI:
#    - Project ID
#    - Region
#    - Available quotas
#    - Active jobs count
#
# 5. W&B Launch Queue:
#    - Queue name
#    - Entity
#    - Agent status
#    - Jobs pending/running
#
# Loading states:
# - Initial: "⏳ Loading infrastructure..."
# - Refresh: Updates all tables simultaneously
# - Error: Shows "Failed to load [resource]" with retry option
#
# Commands (gcloud):
# - Buckets: gcloud storage buckets list --format=json
# - Artifact Registry: gcloud artifacts repositories list --format=json
# - Service Accounts: gcloud iam service-accounts list --format=json
# - Vertex AI: gcloud ai models list --format=json (or similar)
#
# Navigation:
# - Back button returns to home
# - Refresh button (r) reloads all data
# - Auto-refresh every 60s (optional, can be toggled)
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

import os
import sys
import subprocess
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Static, DataTable, Button, Label, LoadingIndicator
from textual.message import Message

from ..config.constants import ARR_COC_WITH_WINGS
from ..shared.base_screen import BaseScreen
from ..shared.wandb_helper import WandBHelper
from ..setup.core import check_infrastructure_core, display_infrastructure_tree


# 🦡⚡ Custom Messages for Progressive Quota Toasts!
class GpuQuotaProgress(Message):
    """Message sent when a GPU quota check completes"""
    def __init__(self, gpu_display: str, region: str, quota: int, failed: bool = False):
        super().__init__()
        self.gpu_display = gpu_display
        self.region = region
        self.quota = quota
        self.failed = failed


class C3QuotaProgress(Message):
    """Message sent when a C3 quota check completes"""
    def __init__(self, region: str, vcpus: int, failed: bool = False):
        super().__init__()
        self.region = region
        self.vcpus = vcpus
        self.failed = failed

# Cache warming debug flag (from tui.py)
STEVEN_CACHE_WARM_DEBUG = True
from .core import check_image_security_cached, format_security_summary_core, should_show_security_core


class InfraScreen(BaseScreen):
    """Infrastructure overview screen - Comprehensive GCloud resource display"""

    def __init__(self, config: Dict[str, str]):
        super().__init__(loading_message="Loading infrastructure details...")
        self.config = config
        self.project_id = None
        self.loading = False

    CSS = """
    InfraScreen {
        height: 100vh;
    }

    /* STANDARD: Page title */
    #page-title {
        height: auto;
        padding: 1;
        text-align: center;
        color: $accent;
        background: $panel;
    }

    /* STANDARD: Scrollable content panel */
    #content-panel {
        height: 1fr;
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

    .left-btn {
        width: 12;
        height: 3;
    }

    .spacer {
        width: 1fr;
    }

    .action-btn {
        width: 12;
        height: 3;
        margin: 0 1;
    }

    /* Section headers */
    .section-header {
        width: 100%;
        text-style: bold;
        color: $accent;
        padding: 1 0 0 0;
        margin-top: 1;
    }

    /* Infrastructure tables */
    .infra-table {
        width: 100%;
        height: auto;
        max-height: 15;
        margin: 0 0 2 0;
    }

    /* Status indicators */
    .status-indicator {
        width: 100%;
        padding: 1 0;
        text-align: center;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("b", "back", "Back"),
    ]

    def __init__(self, config: dict):
        super().__init__(loading_message="Loading infrastructure...")
        self.config = config

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Page title
        yield Static("Infrastructure Overview", id="page-title")

        # STANDARD STRUCTURE: Content panel
        with VerticalScroll(id="content-panel"):
            # Infrastructure tree will be displayed here
            yield Static("", id="infra-tree", shrink=True)

        # STANDARD STRUCTURE: Fixed button bar
        with Horizontal(id="button-bar"):
            yield Button("Back (b)", id="back-btn", classes="left-btn pastel-gray")
            yield Static("", classes="spacer")
            yield Button("Refresh (r)", id="refresh-btn", classes="action-btn pastel-cyan")


        # Loading overlay (LAST - appears on top)
        yield from self.compose_base_overlay()

        yield Footer()

    def initialize_content(self):
        """Do all the work here in the background thread! No second thread needed."""
        import time
        from CLI.shared.performance_monitor import get_monitor

        # 🦡🔥 STEVEN LOGS: Entering Infra screen
        from CLI.shared.steven_toasts import steven_log_screen_entry
        steven_log_screen_entry(self.app, "Infrastructure", "User pressed '5' or navigated to Infra")

        # 🧹 Cancel background cache warming - STOP TIMERS + CANCEL WORKERS!
        try:
            # 🛑 STEP 1: Stop timers (prevents new workers from spawning!)
            self.app.stop_cache_warming()

            # 🛑 STEP 2: Cancel existing workers (node=self.app because workers created with app.run_worker())
            self.app.workers.cancel_group(self.app, "cache_warming")

            # 🦡💙 STEVEN'S DUAL BLUE HEARTS!
            from CLI.shared.steven_toasts import steven_log_cancellation, steven_notify
            steven_log_cancellation("Background cache warming workers", "Infrastructure")

            # 💙 BLUE HEART #1: Cache warming cancelled!
            steven_notify(self.app, "🦡💙 Cache warming cancelled!", severity="information")

            # 💙 BLUE HEART #2: Welcome to Infra!
            steven_notify(self.app, "🦡💙 Welcome to Infra! Checking your infrastructure...", severity="information")

            if STEVEN_CACHE_WARM_DEBUG:
                from CLI.shared.log_paths import get_log_path
                from datetime import datetime
                log_file = get_log_path("cache_warm.log")
                with open(log_file, "a") as f:
                    f.write(f"{datetime.now().isoformat()} 🛑 INFRA_LOAD: Cancelling background warming (full fetch starting)\n")
        except Exception:
            pass  # Workers might not exist yet

        monitor = get_monitor()

        # Create helper
        entity = self.config.get("WANDB_ENTITY", "")
        project = self.config.get("WANDB_PROJECT", "arr-coc-0-1")
        queue = self.config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

        helper = WandBHelper(entity, project, queue)

        # TUI callback that collects output
        class TUICallback:
            def __init__(self):
                self.lines = []

            def __call__(self, message: str):
                self.lines.append(message)

        status = TUICallback()

        try:
            # Check infrastructure with PROGRESSIVE BLUE HEARTS! 💙
            from CLI.shared.steven_toasts import steven_notify
            import time as time_module

            # Run infrastructure check with blue heart toast!
            from CLI.shared.stevens_dance import stevens_log
            stevens_log("infra", "🍞 BREADCRUMB 1: Starting infra check")

            op_id = monitor.start_operation("check_infrastructure_status", category="GCP")
            start_time = time_module.time()

            # 🦡⚡ Create callbacks that post messages for progressive toasts!
            def gpu_progress_callback(gpu_display: str, region: str, quota: int, failed: bool = False):
                """Called when GPU quota check completes - posts message to main thread"""
                self.post_message(GpuQuotaProgress(gpu_display, region, quota, failed))

            def c3_progress_callback(region: str, vcpus: int, failed: bool = False):
                """Called when C3 quota check completes - posts message to main thread"""
                self.post_message(C3QuotaProgress(region, vcpus, failed))

            info = check_infrastructure_core(
                helper,
                self.config,
                status,
                gpu_progress_callback=gpu_progress_callback,
                c3_progress_callback=c3_progress_callback
            )

            duration = time_module.time() - start_time
            monitor.end_operation(op_id)
            stevens_log("infra", f"🍞 BREADCRUMB 2: Infra check complete, duration = {duration:.2f}s")

            # Show blue heart or angry message based on duration
            # 🦡🎩 ORACLE DISCOVERY: app.notify() uses post_message() internally (thread-safe!)
            # Source: textual/app.py:4511 - notify() calls self.post_message(Notify(notification))
            # post_message() is designed for cross-thread communication - no wrapper needed!
            stevens_log("infra", f"🍞 BREADCRUMB 3: About to notify, duration < 8.0? {duration < 8.0}")

            if duration < 8.0:
                # Fast check! Blue heart! 💙
                stevens_log("infra", "🍞 BREADCRUMB 4: Fast path, calling stevens_log for COMPLETE")
                stevens_log("infra", f"💙 INFRA_CHECK_COMPLETE: {duration:.2f}s (fast!)")
                stevens_log("infra", "🍞 BREADCRUMB 5: stevens_log done, calling steven_notify")
                steven_notify(self.app, f"🦡💙 Infrastructure checked! ({duration:.1f}s)", "information")
                stevens_log("infra", "🍞 BREADCRUMB 6: steven_notify done!")
            else:
                # SLOW CHECK! ANGRY MESSAGE! 🚨
                stevens_log("infra", f"🚨 INFRA_CHECK_SLOW: {duration:.2f}s (FUCK OFF!)")
                steven_notify(self.app, f"🦡🚨 Infrastructure check FUCK OFF TOOK {duration:.1f}s!", "warning")

            # Display as tree (pass config for GPU-specific quota instructions)
            display_infrastructure_tree(info, status, self.config)

            # Add Docker image security section (moved from ../monitor to infra!)
            status("")
            status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            status("                    DOCKER IMAGE SECURITY")
            status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            status("")

            # Check security for all 4 images (with blue heart toast!)
            start_time_security = time_module.time()
            security_data = check_image_security_cached(self.config, status)
            duration_security = time_module.time() - start_time_security

            # Show blue heart or angry message
            # 🦡🎩 ORACLE DISCOVERY: app.notify() uses post_message() internally (thread-safe!)
            if duration_security < 8.0:
                stevens_log("infra", f"💙 SECURITY_CHECK_COMPLETE: {duration_security:.2f}s")
                steven_notify(self.app, f"🦡💙 Security scanned! ({duration_security:.1f}s)", "information")
            else:
                stevens_log("infra", f"🚨 SECURITY_CHECK_SLOW: {duration_security:.2f}s (FUCK OFF!)")
                steven_notify(self.app, f"🦡🚨 Security scan FUCK OFF TOOK {duration_security:.1f}s!", "warning")

            if security_data and should_show_security_core(security_data):
                # Format and display security summary
                security_summary = format_security_summary_core(security_data)
                status(security_summary)
                status("")

                # Show link to Artifact Registry console
                if security_data.get('console_url'):
                    status(f"  [dim]→ View in console: {security_data['console_url']}[/dim]")
                    status("")

            # Return tree output
            tree_output = "\n".join(status.lines)

            # 🦡 Flush Stevens Dance logs immediately (don't wait for 10k lines!)
            from CLI.shared.stevens_dance import stevens_flush_all
            stevens_flush_all()

            return {"tree_output": tree_output, "success": True}
        except Exception as e:
            # 🦡 Flush logs even on error!
            from CLI.shared.stevens_dance import stevens_flush_all
            stevens_flush_all()
            return {"error": str(e), "success": False}

    def finish_loading(self, data=None):
        """Update UI based on loaded data (called from main thread)"""
        # Hide loading overlay NOW (data loaded!)
        super().finish_loading(data)

        if data and data.get("success"):
            tree_output = data.get("tree_output")
            self._update_tree(tree_output)
        elif data:
            # Show error (press 'e' to see full details)
            error = data.get("error", "Unknown error")
            self.notify_with_full_error("Error Loading Infrastructure", error)

    def _update_tree(self, tree_output: str):
        """Update tree widget (called from background thread)"""
        tree_widget = self.query_one("#infra-tree", Static)
        tree_widget.update(tree_output)

    def action_refresh(self) -> None:
        """Refresh infrastructure tree"""
        # Show loading overlay again
        if self.loading_overlay:
            self.loading_overlay.display = True

        # Re-run initialization using Textual's worker system
        self.run_worker(self._load_content, name="content_loader", thread=True, exclusive=True)

    def action_back(self) -> None:
        """Return to home screen"""
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "refresh-btn":
            pass  # Placeholder for refresh

    # 🦡⚡💙 PROGRESSIVE QUOTA TOAST HANDLERS!
    def on_gpu_quota_progress(self, event: GpuQuotaProgress) -> None:
        """Handle GPU quota check completion - show blue toast!"""
        if event.failed:
            # ❤️ RED TOAST - Error!
            self.notify(f"❤️ {event.gpu_display} check failed in {event.region}", severity="error", timeout=3)
        elif event.quota > 0:
            # 💙 BLUE TOAST - Success!
            self.notify(f"💙 {event.gpu_display} quota in {event.region}: {event.quota}", severity="information", timeout=2)
        # If quota is 0, don't show toast (not an error, just no quota)

    def on_c3_quota_progress(self, event: C3QuotaProgress) -> None:
        """Handle C3 quota check completion - show blue toast!"""
        if event.failed:
            # ❤️ RED TOAST - Error!
            self.notify(f"❤️ C3 check failed in {event.region}", severity="error", timeout=3)
        elif event.vcpus > 0:
            # 💙 BLUE TOAST - Success!
            self.notify(f"💙 C3 quota in {event.region}: {event.vcpus} vCPUs", severity="information", timeout=2)
        # If vcpus is 0, don't show toast (not an error, just no quota)

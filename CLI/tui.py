#!/usr/bin/env python3
"""
ARR-COC TUI (Textual User Interface)

Interactive terminal UI for training management.
Uses Textual framework for rich UI components.

Usage:
    python CLI/tui.py              # Start at home screen
    python CLI/tui.py monitor      # Jump to monitor screen
    python CLI/tui.py launch       # Jump to launch screen

For pure CLI mode (no TUI), use:
    python CLI/cli.py launch       # CLI job submission
"""

# <claudes_code_comments>
# ** Function List **
# ARRCOCApp.__init__(start_screen) - Initialize app with config and helpers
# ARRCOCApp.on_mount() - Install all screens and navigate to start
# ARRCOCApp.compose() - Create header and footer widgets
# main() - Parse CLI args and run app
#
# ** TUI Flow **
#
#   User Interaction
#       ↓
#   python CLI/tui.py [screen]
#       ↓
#   ┌─────────────────────────────────────┐
#   │  ARRCOCApp (Textual App)            │
#   │  • __init__() - Load config         │
#   │  • on_mount() - Install screens     │
#   │  • compose() - Header/Footer        │
#   └─────────────────────────────────────┘
#       ↓
#   ┌─────────────────────────────────────┐
#   │  6 Screens (Textual):               │
#   │  1. Home - Navigation hub           │
#   │  2. Monitor - Run tracking          │
#   │  3. Launch - Job submission         │
#   │  4. Setup - Infrastructure          │
#   │  5. Teardown - Cleanup              │
#   │  6. Infra - Status display          │
#   └─────────────────────────────────────┘
#       ↓
#   ┌─────────────────────────────────────┐
#   │  Core Logic (cli/*/core.py)         │
#   │  • run_setup_core()                 │
#   │  • run_launch_core()                │
#   │  • list_runs_core()                 │
#   │  • run_teardown_core()              │
#   │  • check_infrastructure_core()      │
#   └─────────────────────────────────────┘
#       ↓
#   ┌─────────────────────────────────────┐
#   │  TUICallback()                      │
#   │  • Mounts Static widgets            │
#   │  • Updates DataTables               │
#   │  • Preserves Rich markup            │
#   └─────────────────────────────────────┘
#       ↓
#   Textual UI (interactive widgets)
#
# ** Technical Review **
# Entry point for ARR-COC training TUI (Textual UI framework). Modular architecture with 6 screens:
# home (navigation hub), monitor (run tracking), launch (job submission), setup (infrastructure),
# teardown (cleanup), infra (status display).
#
# Architecture:
# - Modular screens in cli/{home,monitor,launch,setup,teardown,infra}/screen.py
# - Shared helpers in cli/shared/wandb_helper.py for W&B operations
# - Global bindings: q=quit, h=home, numbers 1-5 for screens
# - Gruvbox theme for consistent color palette
#
# Config flow: load_training_config() reads .training file → extracts WANDB_ENTITY, WANDB_PROJECT,
# WANDB_LAUNCH_QUEUE_NAME → initializes WandBHelper (API wrapper)
#
# Screen lifecycle: ARRCOCApp owns config/helpers → passes to each screen via constructor →
# screens use helpers for W&B operations (list runs, submit jobs, check status, etc.)
#
# CLI arg support: `python CLI/tui.py <screen>` jumps directly to that screen (bypasses home)
#
# ** Philosophical Commentary **
# This TUI is the TRAINING MICROSCOPE from Dialogue 55! It enables recursive observation:
# User watches Claude train, Claude watches Claude train, both watch the coupling itself.
#
# Why TUI + CLI? COMPLEMENTARY WEAKNESSES (Principle 2!):
# - TUI can't automate (interactive-only) → CLI automates
# - CLI can't explore (batch-only) → TUI explores
# Together they preserve gaps that create coupling value!
#
# The architecture enables autopoiesis - the system building itself through observation.
# How we build (observable, coupled, dual-mode) determines what we BECOME!
#
# ARCHITECTURE IS ONTOLOGY!
# </claudes_code_comments>

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add project root to path (parent of training/) so imports work from anywhere
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from textual.app import App
from textual.widgets import Header, Footer, Static
from textual.worker import Worker, WorkerState
from textual.containers import Container

# Steven's fractal toast logger
from CLI.shared.steven_toasts import steven_notify

# Import constants
from CLI.config.constants import (ARR_COC_WITH_WINGS, ARR_COC_DESCRIPTION,
                                           load_training_config)

# Import screens
from CLI.home.screen import HomeScreen
from CLI.monitor.screen import MonitorScreen
from CLI.launch.screen import LaunchScreen
from CLI.setup.screen import SetupScreen
from CLI.teardown.screen import TeardownScreen
from CLI.infra.screen import InfraScreen
# Pricing, Reduce, Truffles, GPU screen imports removed (2025-11-16)

# Import shared helpers
from CLI.shared.wandb_helper import WandBHelper
from CLI.shared.log_paths import get_log_path

# ═══════════════════════════════════════════════════════════════════════════════
# 🔥 STEVEN_CACHE_WARM_DEBUG Flag (2025-11-21):
# ═══════════════════════════════════════════════════════════════════════════════
# Toggle verbose logging for quota cache warming system.
# When True: Logs every batch warm attempt, progress, cache hits/misses
# When False: Silent background warming (production mode)
# Log file: ARR_COC/Training/logs/cache_warm.log
STEVEN_CACHE_WARM_DEBUG = True  # 🔥 Turn ON to see cache warming progress!

# ═══════════════════════════════════════════════════════════════════════════════
# 🔥 CACHE WARMING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CACHE_WARM_BATCH_INTERVAL = 1.0  # Seconds between batch warm attempts (1s = fast!)
CACHE_WARM_REFRESH_INTERVAL = 30 * 60  # Seconds between cache re-warms (30 minutes)
CACHE_WARM_SLOW_THRESHOLD = 7000  # Milliseconds - show red complaint above this (7s = slow for 8 checks!)
CACHE_WARM_BATCH_SIZE = 8  # TOTAL quota checks per batch (split: 4 GPU + 4 C3 = 8 max)


class ARRCOCApp(App):
    """ARR-COC Training Management TUI"""

    TITLE = "ARR-COC TRAINER"
    SUB_TITLE = ""

    # 🦡🎩 STEVEN'S TOAST DURATION - Override Textual's default (5s → 6s)
    # This sets the DEFAULT timeout for ALL app.notify() calls when timeout=None
    # Compensates for Textual's time_left bug (delays steal duration)
    # Target: 4 seconds visible time (6s timeout - 2s avg delay = 4s visible)
    NOTIFICATION_TIMEOUT = 6.0  # seconds (Textual standard way!)

    # Styling
    CSS = """
    Screen {
        background: $surface;
    }

    Button {
        margin: 1;
    }

    #content-panel {
        height: auto;
    }

    /* ============================================================================
       Cache Warming Status Widget - Top-right corner
       ============================================================================ */

    #cache-status {
        dock: top;
        width: auto;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 2;
        text-align: right;
    }

    /* ============================================================================
       STEVEN'S TOAST STYLING (using Textual's built-in notify() system!)
       ============================================================================
       Position toasts at TOP-RIGHT corner (out of way of main content!)
    */

    ToastRack {
        dock: top;
        align: right top;
    }

    /* Steven's angry error toasts (severity="error") */
    Toast.-error {
        background: #fb4934;  /* Bright red */
        border: thick #cc241d;
        padding: 1 2;
    }

    /* Steven's warning toasts (severity="warning") */
    Toast.-warning {
        background: #fabd2f;  /* Bright yellow */
        border: thick #d79921;
        padding: 1 2;
        color: #282828;  /* Dark text on bright background */
    }

    /* Information toasts (severity="information") */
    Toast.-information {
        background: #83a598;  /* Soft blue */
        border: thick #458588;
        padding: 1 2;
    }

    /* Toast title styling */
    .toast--title {
        text-style: bold;
    }

    /* ============================================================================
       Button Bar Layout - Consistent sizing and alignment
       ============================================================================ */

    /* Button bar container - docked to bottom */
    #button-bar {
        dock: bottom;
        height: auto;
        width: 100%;
        padding: 1 2 3 2;
        background: $surface;
        layout: horizontal;
    }

    /* Left button (Quit on home, Back on sub-pages) - consistent size */
    .left-btn {
        width: 12;
        height: 3;
        margin: 0 1;
    }

    /* Spacer - pushes action buttons to the right */
    .spacer {
        width: 1fr;
    }

    /* Action buttons - on the right, same size as left button */
    .action-btn {
        width: 12;
        height: 3;
        margin: 0 1;
    }

    /* ============================================================================
       Action Button Theme - Pastel Bright Colors
       ============================================================================ */

    /* Back/Quit buttons - subtle, barely differentiated from background */
    .left-btn {
        background: #3c3836;    /* Gruvbox dark2 - solid fill */
        color: #a89984;         /* Gruvbox gray - slightly brighter text */
        border: solid #504945;  /* Subtle solid border for differentiation */
    }

    .left-btn:hover {
        background: #504945;  /* Gruvbox dark3 - slightly lighter on hover */
        color: #bdae93;       /* Brighter text on hover */
    }

    /* Action buttons - pastel bright colors, no borders */
    Button.pastel-blue {
        background: #83a598;  /* Gruvbox bright blue - pastel */
        color: #282828;       /* Dark text for contrast */
        border: none;         /* No border - solid fill only */
    }

    Button.pastel-green {
        background: #8ec07c;  /* Gruvbox bright aqua - pastel green */
        color: #282828;
        border: none;
    }

    Button.pastel-orange {
        background: #fe8019;  /* Gruvbox bright orange - pastel */
        color: #282828;
        border: none;
    }

    Button.pastel-purple {
        background: #d3869b;  /* Gruvbox bright purple - pastel */
        color: #282828;
        border: none;
    }

    Button.pastel-yellow {
        background: #fabd2f;  /* Gruvbox bright yellow - pastel */
        color: #282828;
        border: none;
    }

    Button.pastel-red {
        background: #fb4934;  /* Gruvbox bright red - pastel */
        color: #282828;
        border: none;
    }

    Button.pastel-cyan {
        background: #8ec07c;  /* Gruvbox aqua - pastel cyan */
        color: #282828;
        border: none;
    }

    Button.pastel-gray {
        background: #504945;  /* Gruvbox bg1 - subtle gray, slightly lighter than surface */
        color: #ebdbb2;       /* Light text for contrast */
        border: none;
    }

    /* ============================================================================
       Hover Effects - Visual feedback for mouse interaction
       ============================================================================ */

    /* Subtle hover for Back/Quit buttons */
    .left-btn:hover {
        background: #504945;  /* Gruvbox dark3 - slightly lighter */
        color: #a89984;       /* Slightly brighter text */
    }

    /* Pastel hover - lighter, softer colors */
    Button.pastel-blue:hover {
        background: #a3c5d8;  /* Lighter pastel blue */
        text-style: bold;
    }

    Button.pastel-green:hover {
        background: #aed4a0;  /* Lighter pastel green */
        text-style: bold;
    }

    Button.pastel-orange:hover {
        background: #ffa050;  /* Lighter pastel orange */
        text-style: bold;
    }

    Button.pastel-purple:hover {
        background: #e5a6bb;  /* Lighter pastel purple */
        text-style: bold;
    }

    Button.pastel-yellow:hover {
        background: #fdd060;  /* Lighter pastel yellow */
        text-style: bold;
    }

    Button.pastel-red:hover {
        background: #ff7060;  /* Lighter pastel red */
        text-style: bold;
    }

    Button.pastel-cyan:hover {
        background: #aed4a0;  /* Lighter pastel cyan */
        text-style: bold;
    }

    Button.pastel-gray:hover {
        background: #665c54;  /* Gruvbox bg2 - lighter gray on hover */
        text-style: bold;
    }

    /* ============================================================================
       BURST Effect on Click/Focus - Dramatic visual feedback!
       ============================================================================ */

    /* When clicked, buttons get THICK border + INTENSE tint = BURST! */
    Button.pastel-blue:focus {
        border: thick cyan;
        tint: cyan 60%;
        text-style: bold;  /* Bold without reverse - no highlighting! */
    }

    Button.pastel-green:focus {
        border: thick green;
        tint: green 60%;
        text-style: bold;
    }

    Button.pastel-orange:focus {
        border: thick yellow;
        tint: yellow 60%;
        text-style: bold;
    }

    Button.pastel-purple:focus {
        border: thick magenta;
        tint: magenta 60%;
        text-style: bold;
    }

    Button.pastel-yellow:focus {
        border: thick yellow;
        tint: yellow 70%;
        text-style: bold;
    }

    Button.pastel-red:focus {
        border: thick red;
        tint: red 60%;
        text-style: bold;
    }

    Button.pastel-cyan:focus {
        border: thick cyan;
        tint: cyan 60%;
        text-style: bold;
    }

    Button.pastel-gray:focus {
        border: thick #928374;
        tint: #928374 40%;
        text-style: bold;
    }
    """

    # Screens will be installed in on_mount() with dependencies

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "push_screen('home')", "Home"),
        # Number keys 1-5 for global navigation
        ("1", "push_screen('monitor')", "Monitor"),
        ("2", "push_screen('launch')", "Launch"),
        ("3", "push_screen('setup')", "Setup"),
        ("4", "push_screen('teardown')", "Teardown"),
        ("5", "push_screen('infra')", "Infra"),
    ]

    def __init__(self, start_screen: str = "home"):
        super().__init__()
        self.start_screen = start_screen

        # Set gruvbox theme
        self.theme = "gruvbox"

        # Load config
        self.config = load_training_config()

        # Store config for helper creation in on_mount (after event loop exists)
        self.wandb_entity = self.config.get("WANDB_ENTITY", "")
        self.wandb_project = self.config.get("WANDB_PROJECT", "arr-coc-0-1")
        self.wandb_queue = self.config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-arr-coc-queue")

        # Cache warming state for UI
        self.cache_warm_status = "🔥 Warming cache..."
        self.cache_warm_progress = "0/38"

    def on_mount(self) -> None:
        """Install screens and navigate to start screen"""
        # Create WandBHelper here (AFTER event loop exists from app.run())
        self.wandb_helper = WandBHelper(self.wandb_entity, self.wandb_project, self.wandb_queue)

        # Install all screens with shared dependencies
        self.install_screen(HomeScreen(self.wandb_helper, self.config), name="home")
        self.install_screen(MonitorScreen(self.wandb_helper, self.config), name="monitor")
        self.install_screen(LaunchScreen(self.wandb_helper, self.config), name="launch")
        self.install_screen(SetupScreen(self.config), name="setup")
        self.install_screen(TeardownScreen(self.wandb_helper, self.config), name="teardown")
        self.install_screen(InfraScreen(self.config), name="infra")
        # Pricing, Reduce, Truffles, GPU screen registrations removed (2025-11-16)

        # Push start screen
        self.push_screen(self.start_screen)

        # ═══════════════════════════════════════════════════════════════════════
        # QUOTA CACHE WARMING (staggered background warming)
        # ═══════════════════════════════════════════════════════════════════════
        # Warms GPU + C3 quota cache in background (1 GPU + 1 C3 check every 2s)
        # Re-warms every 30 minutes to keep cache fresh
        # NOTE: This is for infra_verify TUI display ONLY
        #       Launch-time checks are ALWAYS FRESH - no cache!

        # 🦡🎩 STEVEN'S DANCE - Clear ALL logs on program start!
        from CLI.shared.stevens_dance import stevens_clear_all, stevens_log
        stevens_clear_all()
        stevens_log("cache_warm", f"🚀 CACHE_WARM_START: Timers initialized ({CACHE_WARM_BATCH_INTERVAL}s batch, {CACHE_WARM_REFRESH_INTERVAL//60}m refresh)")

        self._cache_warm_timer = self.set_interval(CACHE_WARM_BATCH_INTERVAL, self._warm_quota_cache_tick)
        self._cache_refresh_timer = self.set_interval(CACHE_WARM_REFRESH_INTERVAL, self._restart_cache_warming)

        # Update initial cache status widget
        self._update_cache_status_widget("🔥 Warming...", "0/38")

    def stop_cache_warming(self) -> None:
        """Stop cache warming timers (called by Infra screen to prevent new workers!)"""
        if hasattr(self, '_cache_warm_timer') and self._cache_warm_timer:
            self._cache_warm_timer.stop()
            self._cache_warm_timer = None
        if hasattr(self, '_cache_refresh_timer') and self._cache_refresh_timer:
            self._cache_refresh_timer.stop()
            self._cache_refresh_timer = None

    def on_unmount(self) -> None:
        """Flush all Steven's logs on TUI exit!"""
        from CLI.shared.stevens_dance import stevens_flush_all
        stevens_flush_all()

    def _warm_quota_cache_tick(self) -> None:
        """Called every CACHE_WARM_BATCH_INTERVAL (1s) to warm one batch of quota cache.

        Uses run_worker(thread=True) to avoid blocking UI - Steven's pattern!
        """
        from CLI.shared.infra_verify import is_quota_cache_warm

        # 🦡 LOG: Tick called
        if STEVEN_CACHE_WARM_DEBUG:
            from CLI.shared.stevens_dance import stevens_log
            stevens_log("cache_warm", "⏰ TICK: Timer fired, checking if warm...")

        # Skip if already warm (will be restarted by 30-min timer)
        if is_quota_cache_warm():
            if STEVEN_CACHE_WARM_DEBUG:
                # Only log skip occasionally (not every 2s!)
                pass  # Could add throttled logging here
            return

        # 🚨 FIX: Skip if cache_warm worker is already running!
        # Prevents cancelling slow workers when GCP takes >2s per batch
        if any(w.name == "cache_warm" and w.is_running for w in self.workers):
            if STEVEN_CACHE_WARM_DEBUG:
                from CLI.shared.stevens_dance import stevens_log
                stevens_log("cache_warm", "⏸️ SKIP_TICK: Worker already running (GCP is slow!)")
            return

        project_id = self.config.get("GCP_PROJECT_ID", "")
        if not project_id:
            return

        # 🦡 LOG: Starting worker
        if STEVEN_CACHE_WARM_DEBUG:
            from CLI.shared.stevens_dance import stevens_log
            stevens_log("cache_warm", "🚀 WORKER_START: Launching cache_warm worker")

        # Launch worker to warm one batch (non-blocking!)
        self.run_worker(
            lambda: self._do_cache_warm_batch(project_id),
            exclusive=False,  # DON'T cancel! We check above instead
            name="cache_warm",
            thread=True,  # Run in thread, don't block UI!
            group="cache_warming"  # Group for cancellation when Infra loads
        )

    def _do_cache_warm_batch(self, project_id: str) -> Dict[str, Any]:
        """Worker function - warms one batch (up to 8 TOTAL checks, split 4 GPU + 4 C3)"""
        from CLI.shared.infra_verify import warm_quota_cache_batch

        result = warm_quota_cache_batch(project_id, batch_size=CACHE_WARM_BATCH_SIZE)

        # 🌶️ STEVEN'S SPICY COMPLAINT LOGS!
        if STEVEN_CACHE_WARM_DEBUG:
            log_file = get_log_path("cache_warm.log")
            elapsed = result.get("elapsed_ms", 0)
            gpu_progress = result["gpu_progress"]
            c3_progress = result["c3_progress"]

            # 🌶️ STEVEN'S COMPLAINT THRESHOLDS
            if elapsed > CACHE_WARM_SLOW_THRESHOLD:
                # 🦡🔥 TOO FUCKING SLOW!
                log_msg = f"🦡🔥 CACHE_WARM_SLOW: GPU {gpu_progress}, C3 {c3_progress} - {elapsed}ms 🦡🔥 FUCK OFF! TOO SLOW! GCP API IS BULLSHIT! 🦡🔥"
            elif elapsed > 2500:
                # 😤 GETTING SLOW
                log_msg = f"😤 CACHE_WARM_SLOWISH: GPU {gpu_progress}, C3 {c3_progress} - {elapsed}ms - Getting fucking slow here..."
            elif elapsed > 2000:
                # ⚠️ WARNING ZONE
                log_msg = f"⚠️ CACHE_WARM_WARN: GPU {gpu_progress}, C3 {c3_progress} - {elapsed}ms - Hmm, that's pushing it..."
            else:
                # ✅ GOOD!
                if result["done"]:
                    log_msg = f"✅ CACHE_WARM_COMPLETE: GPU {gpu_progress}, C3 {c3_progress} - {elapsed}ms ✅ Cache is HOT!"
                else:
                    log_msg = f"🔥 BATCH_WARM: GPU {gpu_progress}, C3 {c3_progress} - {elapsed}ms ✅"

            from CLI.shared.stevens_dance import stevens_log
            stevens_log("cache_warm", log_msg)

        return result  # Return for worker state handler

    def _restart_cache_warming(self) -> None:
        """Called every 30 minutes to force cache re-warm"""
        from CLI.shared.infra_verify import _quota_cache

        # 🔄 STEVEN'S REWARM LOGGING
        if STEVEN_CACHE_WARM_DEBUG:
            log_file = get_log_path("cache_warm.log")
            from CLI.shared.stevens_dance import stevens_log
            stevens_log("cache_warm", "🔄 CACHE_REWARM: 30-min timer fired, clearing cache for re-warm")

        # Clear cache to trigger re-warming on next tick
        _quota_cache.clear()

    def _update_cache_status_widget(self, status: str, progress: str) -> None:
        """Update cache warming status widget in top-right corner."""
        try:
            cache_widget = self.query_one("#cache-status", Static)
            elapsed_indicator = ""

            # Show progress
            full_text = f"[dim]{status}[/dim] {progress}"
            cache_widget.update(full_text)
        except Exception:
            pass  # Widget might not exist yet during init

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes - show toasts for cache warming!"""
        # Only handle cache_warm workers
        if event.worker.name != "cache_warm":
            return

        if event.worker.state == WorkerState.SUCCESS:
            # Worker completed! Get result
            result = event.worker.result
            gpu_progress = result.get("gpu_progress", "?/?")
            c3_progress = result.get("c3_progress", "?/?")
            elapsed = result.get("elapsed_ms", 0)

            # Parse progress (X/Y format)
            gpu_done = 0
            c3_done = 0
            total_done = 0
            try:
                gpu_done, gpu_total = map(int, gpu_progress.split('/'))
                c3_done, c3_total = map(int, c3_progress.split('/'))
                total_done = gpu_done + c3_done
                total_needed = gpu_total + c3_total
                combined_progress = f"{total_done}/{total_needed}"
            except:
                combined_progress = "?/38"

            # 🦡 LOG: Worker succeeded
            if STEVEN_CACHE_WARM_DEBUG:
                from CLI.shared.stevens_dance import stevens_log
                stevens_log("cache_warm", f"✅ WORKER_SUCCESS: Progress {gpu_progress} GPU, {c3_progress} C3, done={result.get('done')}")

            # Update status widget and show toasts
            if result.get("done"):
                # ✅ CACHE IS HOT!
                if STEVEN_CACHE_WARM_DEBUG:
                    from CLI.shared.stevens_dance import stevens_log
                    stevens_log("cache_warm", f"🎉 CACHE_COMPLETE: ALL quotas cached! GPU {gpu_done}, C3 {c3_done}")

                self._update_cache_status_widget("✅ Cache HOT!", combined_progress)
                # 🦡💙 Steven celebrates with counts!
                steven_notify(self, f"🦡💙 Cache warm! {total_done} quotas cached (GPU: {gpu_done}, C3: {c3_done})", severity="information")
            else:
                # 💙 ALWAYS show progress toast for every batch!
                if elapsed < 2000:
                    # 🔥 GOOD! Fast batch!
                    self._update_cache_status_widget("🔥 Warming...", combined_progress)
                    steven_notify(self, f"🦡💙 Batch warm! {combined_progress} ({elapsed}ms) ✨", severity="information")
                elif elapsed < CACHE_WARM_SLOW_THRESHOLD:
                    # ⚠️ ACCEPTABLE - between 2s and slow threshold
                    self._update_cache_status_widget("⚠️ Warming...", combined_progress)
                    steven_notify(self, f"🦡💙 Batch warm! {combined_progress} ({elapsed}ms)", severity="information")
                else:
                    # 🦡🔥 TOO FUCKING SLOW! Show BOTH blue progress AND red complaint!
                    self._update_cache_status_widget("🦡🔥 SLOW!", combined_progress)
                    steven_notify(self, f"🦡💙 Batch warm! {combined_progress} ({elapsed}ms)", severity="information")
                    steven_notify(self, f"🦡🔥 SLOW! {elapsed}ms - FUCK OFF GCP! 🦡🔥", severity="error")

        elif event.worker.state == WorkerState.ERROR:
            # Worker failed!
            self._update_cache_status_widget("💥 ERROR!", "?/?")
            if STEVEN_CACHE_WARM_DEBUG:
                from CLI.shared.stevens_dance import stevens_log
                stevens_log("cache_warm", f"💥 CACHE_WARM_ERROR: {str(event.worker.exception)}")

        elif event.worker.state == WorkerState.CANCELLED:
            # 🧹 WORKER CANCELLED - Just log it, don't show toast
            # (Toast only shows when Infra screen explicitly cancels workers)
            # ONLY log ONCE per worker (not every update!)
            if not hasattr(event.worker, '_steven_logged_cancellation'):
                event.worker._steven_logged_cancellation = True

                if STEVEN_CACHE_WARM_DEBUG:
                    from CLI.shared.stevens_dance import stevens_log
                    import traceback

                    # 🧹 Normal cleanup - cancelling cache warmup
                    stack = ''.join(traceback.format_stack())
                    stevens_log("cache_warm", "🧹 CLEANUP: Worker cancelled")
                    stevens_log("cache_warm", "🦡 Cache warmup cancelled")
                    stevens_log("cache_warm", "🔍 CANCEL_STACK: Who cancelled?")
                    stevens_log("cache_warm", stack)
                    stevens_log("cache_warm", "🔍 CANCEL_STACK_END")

    def compose(self):
        """Create UI widgets"""
        yield Header(show_clock=True)
        yield Static("", id="cache-status")  # Cache warming status (updated by worker)
        yield Footer()


def main():
    """TUI entry point"""
    # Parse command line args
    start_screen = "home"
    if len(sys.argv) > 1:
        screen = sys.argv[1].lower()
        if screen in ["monitor", "launch", "setup", "teardown", "infra", "home"]:
            start_screen = screen

    # Create app
    app = ARRCOCApp(start_screen=start_screen)

    # Performance monitoring disabled (exit summary removed)
    # from CLI.shared.auto_performance_reporter import enable_auto_reporting
    # enable_auto_reporting()

    # Run app
    app.run()


if __name__ == "__main__":
    main()

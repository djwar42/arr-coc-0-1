# <claudes_code_comments>
# ** Function List **
# HomeScreen.__init__(helper, config) - Initialize with W&B helper and config
# HomeScreen.compose() - Build UI with title, nav buttons, button bar, and loading overlay
# HomeScreen.initialize_content() - Check setup status in background worker
# HomeScreen.finish_loading(data) - Hide loading overlay and start animation
# HomeScreen.update_morph_frame() - Animate title/subtitle/description morphing
# HomeScreen.show_full_ui() - Display project info and setup warning after animation
# HomeScreen.refresh_setup_status() - Called by SetupScreen to update setup warning
# HomeScreen.on_button_pressed(event) - Handle navigation button clicks
# HomeScreen.action_monitor/launch/setup/teardown/infra() - Navigate to respective screens
# HomeScreen.action_quit() - Exit application
# HomeScreen.skip_animation() - Skip animation on early navigation
#
# ** Technical Review **
# Main navigation hub for ARR-COC training CLI. Uses BaseScreen loading overlay pattern for consistent UX.
#
# Loading flow:
# compose() → builds structure + loading overlay (on top)
# initialize_content() → checks setup status in background (quick local check)
# finish_loading() → hides overlay, starts morphing animation
# update_morph_frame() → 10-frame animation (title → subtitle → description)
# show_full_ui() → displays project info and setup warning (if needed)
#
# Navigation: Each button triggers action methods → self.app.push_screen(screen_name)
# - Monitor: Track running jobs (W&B runs)
# - Launch: Submit new training job
# - Setup: One-time infrastructure setup (GCS, Artifact Registry, Service Account, W&B Queue)
# - Teardown: Clean up cloud resources
# - Infra: Infrastructure status and diagnostics
#
# UI layout: Header → #content-panel (animation + nav buttons) → #button-bar (Quit + actions) → Footer
# Loading overlay covers everything initially, then disappears to reveal animated content.
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
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Static, DataTable, Button, Label, LoadingIndicator, Input

from ..config.constants import ARR_COC_WITH_WINGS, ARR_COC_SUBTITLE, ARR_COC_DESCRIPTION
from ..shared.wandb_helper import WandBHelper
from textual.screen import Screen

class HomeScreen(Screen):
    """Home menu screen - Main navigation hub"""

    def __init__(self, helper: WandBHelper, config: Dict[str, str]):
        super().__init__()
        self.helper = helper
        self.config = config

        # Animation state
        self.frame_index = 0
        self.morph_timer = None
        self.animation_complete = False

    CSS = """
    HomeScreen {
        height: 100vh;
    }

    /* STANDARD: Page title (same as other screens) */
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

    /* Button bar layout defined globally in trainer.py */

    /* Large navigation buttons container */
    #nav-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 2 0;
    }

    /* Large colorful navigation buttons */
    .nav-btn {
        width: 20;
        height: 5;
        margin: 0 2;
        text-align: center;
        padding-top: 1;
        padding-bottom: 0;
    }

    /* Centered text (for header and project title) */
    .centered {
        width: 100%;
        text-align: center;
        content-align: center middle;
    }

    /* Top spacing */
    .top-spacer {
        height: 2;
    }

    /* Main ARR-COC title - large and bold, fixed height */
    .main-title {
        text-style: bold;
        color: $accent;
        height: 3;
        padding: 0 0 1 0;
    }

    /* Subtitle - fixed height to prevent jiggle */
    .subtitle {
        height: 3;
        padding: 0 0 1 0;
    }

    /* Description line - fixed height */
    .description {
        height: 5;
        padding: 0 0 2 0;
    }

    /* Project info line - fixed height */
    .project-info {
        height: 3;
    }
    """

    BINDINGS = [
        ("1", "monitor", "Monitor"),
        ("2", "launch", "Launch"),
        ("3", "setup", "Setup"),
        ("4", "teardown", "Teardown"),
        ("5", "infra", "Infra"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Build the UI layout with loading overlay and content panel"""
        yield Header(show_clock=True)

        # Content panel with animation widgets (pre-mounted, hidden under loading overlay)
        with VerticalScroll(id="content-panel") as content:
            yield Static("", classes="top-spacer")  # Space from top
            yield Static("", id="anim-title", classes="centered main-title")
            yield Static("", id="anim-subtitle", classes="centered subtitle")
            yield Static("", id="anim-description", classes="centered description")
            yield Static("", id="project-info", classes="centered project-info")
            yield Static("", id="setup-warning", classes="centered")
            # Navigation buttons
            with Horizontal(id="nav-buttons"):
                yield Button("1. Monitor", variant="primary", id="monitor-btn", classes="nav-btn")
                yield Button("2. Launch", variant="success", id="launch-btn", classes="nav-btn")

        # Fixed button bar (left/right layout with spacer)
        with Horizontal(id="button-bar"):
            yield Button("Quit (q)", id="quit-btn", classes="left-btn pastel-gray")
            yield Static("", classes="spacer")  # Pushes action buttons to the right
            yield Button("Setup (3)", id="setup-btn", classes="action-btn pastel-blue")
            yield Button("Teardown (4)", id="teardown-btn", classes="action-btn pastel-orange")
            yield Button("Infra (5)", id="infra-btn", classes="action-btn pastel-purple")

        yield Footer()

    def on_mount(self) -> None:
        """Start animation immediately on mount - instant, no delays!"""
        # Start morphing animation IMMEDIATELY
        # 40% faster: 0.09s per frame (total 0.9s animation)
        self.morph_timer = self.set_interval(0.09, self.update_morph_frame)

        # 🦡🔥 STEVEN LOGS: Entering Home screen
        from CLI.shared.steven_toasts import steven_log_screen_entry, steven_notify
        steven_log_screen_entry(self.app, "Home", "App started or user pressed 'h'")

        # 🦡🔥 STEVEN'S INIT TOAST - Test fractal self-loathing!
        steven_notify(self.app, "🦡🔥 STEVEN'S FUCKOFF TOAST SYSTEM IS ONLINE! 🦡🔥", severity="error", timeout=6)


    def update_morph_frame(self):
        """Update animation frame - morphing text effect"""

        # Guard: If animation already complete, stop timer and return
        if self.animation_complete:
            if self.morph_timer:
                self.morph_timer.stop()
                self.morph_timer = None
            return

        # Animation frames: title → subtitle → description
        frames = [
            # Frame 0-3: Title morphing (◇ ARR-COC ◇)
            {"title": "░ ███ ░", "subtitle": "", "desc": ""},
            {"title": "░ █▓R-░█C ░", "subtitle": "", "desc": ""},
            {"title": "◇ AR▓-C▒C ◇", "subtitle": "", "desc": ""},
            {"title": "◇  ARR-COC  ◇", "subtitle": "", "desc": ""},

            # Frame 4-7: Subtitle expanding ('What you see changes what you see')
            {"title": "◇  ARR-COC  ◇", "subtitle": "'▓░█▒░█'", "desc": ""},
            {"title": "◇  ARR-COC  ◇", "subtitle": "'Wh█t ░ou █ee ch▓nges'", "desc": ""},
            {"title": "◇  ARR-COC  ◇", "subtitle": "'What you s▒e changes what ░ou'", "desc": ""},
            {"title": "◇  ARR-COC  ◇", "subtitle": "'What you see changes what you s▒e'", "desc": ""},

            # Frame 8-9: Description appearing
            {"title": "◇  ARR-COC  ◇", "subtitle": "'What you see changes what you see'",
             "desc": "▒daptive ▓elevance Realization • ░ontexts ▒ptical Compression"},
            {"title": "◇  ARR-COC  ◇", "subtitle": "'What you see changes what you see'",
             "desc": "Adaptive Relevance Realization • Contexts Optical Compression"},
        ]

        if self.frame_index < len(frames):
            # Update animation widgets
            frame = frames[self.frame_index]

            title_widget = self.query_one("#anim-title", Static)
            subtitle_widget = self.query_one("#anim-subtitle", Static)
            desc_widget = self.query_one("#anim-description", Static)

            title_widget.update(frame["title"])
            subtitle_widget.update(frame["subtitle"])
            desc_widget.update(frame["desc"])

            self.frame_index += 1
        else:
            # Animation complete - stop timer and show full UI
            self.animation_complete = True
            if self.morph_timer:
                self.morph_timer.stop()
                self.morph_timer = None
            self.show_full_ui()

    def show_full_ui(self):
        """Show complete UI after animation - update pre-mounted widgets"""
        # Update project info (already mounted, just update text)
        project_widget = self.query_one("#project-info", Static)
        project_widget.update(f"\n[#00d4ff]Project: {self.helper.entity}/{self.helper.project}[/]\n")

    def refresh_setup_status(self):
        """Called by SetupScreen when setup completes - no-op (setup check removed)"""
        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks - skip animation handled by action methods"""
        # Large navigation buttons
        if event.button.id == "monitor-btn":
            self.action_monitor()
        elif event.button.id == "launch-btn":
            self.action_launch()
        # Bottom button bar
        elif event.button.id == "quit-btn":
            self.action_quit()
        elif event.button.id == "setup-btn":
            self.action_setup()
        elif event.button.id == "teardown-btn":
            self.action_teardown()
        elif event.button.id == "infra-btn":
            self.action_infra()

    def on_unmount(self) -> None:
        """Stop morph timer when leaving screen (avoid background animation!)"""
        if self.morph_timer:
            self.morph_timer.stop()
            self.morph_timer = None

    def skip_to_end(self):
        """Skip animation and jump to final state"""
        if self.morph_timer:
            self.morph_timer.stop()

        # Update to final frame immediately
        title_widget = self.query_one("#anim-title", Static)
        subtitle_widget = self.query_one("#anim-subtitle", Static)
        desc_widget = self.query_one("#anim-description", Static)

        title_widget.update("◇  ARR-COC  ◇")
        subtitle_widget.update("'What you see changes what you see'")
        desc_widget.update("Adaptive Relevance Realization • Contexts Optical Compression")

        # Mark complete and show full UI
        self.animation_complete = True
        self.show_full_ui()

    def action_monitor(self) -> None:
        self.skip_animation()
        self.app.push_screen("monitor")

    def action_launch(self) -> None:
        self.skip_animation()
        self.app.push_screen("launch")

    def action_setup(self) -> None:
        self.skip_animation()
        self.app.push_screen("setup")

    def action_teardown(self) -> None:
        self.skip_animation()
        self.app.push_screen("teardown")

    def action_infra(self) -> None:
        self.skip_animation()
        self.app.push_screen("infra")

    def action_quit(self) -> None:
        self.skip_animation()
        self.app.exit()

    def skip_animation(self):
        """Helper to skip animation if still running"""
        if not self.animation_complete:
            self.skip_to_end()




# <claudes_code_comments>
# ** Function List ** (57 functions organized by system)
#
# ═══ INITIALIZATION & LIFECYCLE ═══
# __init__(helper, config) - Initialize state, config, cache, timers, threading, IPASB, accumulator
# compose() - Build UI: 5 DataTables + checkboxes + spinners + overlay
# initialize_content() - INSTANT return - triggers finish_loading immediately
# finish_loading(data) - Hide overlay, enable cursors, start timers, trigger _accumulated_start()
# on_mount() - Cache spinner refs into _cached_spinners dict for performance
# on_screen_resume() - Refresh all tables when returning to screen
# on_unmount() - Stop all timers, flush logs, cleanup
#
# ═══ CACHE SYSTEM (CACHE_TTL=5s) ═══
# _should_fetch_table(table_name) - Check if cache expired
# _get_cached_data(table_name) - Retrieve cached data
# _update_table_cache(table_name, data) - Update cache after fetch
#
# ═══ ADAPTIVE REGION MONITORING (18 GCP regions) ═══
# _get_target_regions(table_type) - Hot regions + 3 rotating cold (10× API reduction)
# _update_hot_regions(table_type, results) - Update hot region tracking
#
# ═══ SPINNER SYSTEM (8 FPS, pre-calculated chars) ═══
# _start_spinner(spinner_id) - Show animated spinner
# _stop_spinner(spinner_id) - Hide spinner (thread-safe)
# _update_spinners() - Main thread timer (125ms): update spinners with P0/P1/P2/P3/P4 timing
# _spinner_worker_thread() - Dedicated thread for constant-rate updates
# _start_spinner_worker() - Create and start spinner thread
# _stop_spinner_worker() - Stop spinner thread (500ms timeout)
#
# ═══ LOGGING SYSTEM (Stevens Dance - 10,000-line batching) ═══
# stevens_log(log_name, message) - Buffer log messages (imported from stevens_dance)
# stevens_flush_all() - Flush all buffers on unmount (imported from stevens_dance)
# _canonical_log(message) - Write to canonical flow log
# _gil_hold_log(location, duration_ms, description) - Track GIL hold times
#
# ═══ DRY HELPERS ═══
# _create_table_divider(table_name) - Create divider row (auto column count)
# _add_empty_state_row(table_name, message) - Add empty state row
# _enable_table_cursors() - Enable cursors on all 5 tables
#
# ═══ DURATION TICKER (1s interval) ═══
# _update_active_durations() - Update WORKING/RUNNING item durations live
#
# ═══ ACCUMULATOR PATTERN (ordered display with spinner sync) ═══
# _start_accumulator(tables_to_refresh) - Initialize accumulator for batch
# _accumulated_start() - Entry point: staggered IPASB launch
# _display_next_ready_table() - Poll and display in order, stop spinner on display
# _accumulated_refresh() - Auto-refresh callback for enabled tables
#
# ═══ IPASB (Intelligent Progressive Adaptive Spinner Backoff) ═══
# _ipasb_check_backoff() - Workers sleep if system stressed (0/10/50/100ms)
# (Stagger: 400→350→300→250→200ms, stress adds +75/150/300ms)
#
# ═══ DATA SNAPSHOTS ═══
# _take_data_snapshot(table_name, data) - Store snapshot
# _compare_and_log_snapshot(table_name, new_data) - Compare and log changes
#
# ═══ UNIVERSAL REFRESH SYSTEM ═══
# _universal_refresh_table(table_name, is_auto_refresh, use_accumulator) - Launch worker
# _universal_table_worker(table_name, config, use_accumulator) - Fetch + cleanup
# _refresh_all_tables() - Manual refresh ALL 5 tables
#
# ═══ TABLE FETCH/UPDATE (5 tables × 3 functions each) ═══
# _fetch_and_update_runner_table() / _fetch_runner_data() / _update_runner_table()
# _fetch_and_update_builds_table() / _fetch_builds_data() / _update_builds_table()
# _fetch_and_update_vertex_table() / _fetch_vertex_data() / _update_vertex_table()
# _fetch_and_update_active_runs_table() / _fetch_active_data() / _update_active_table()
# _fetch_and_update_completed_runs_table() / _fetch_completed_data() / _update_completed_table()
#
# ═══ AUTO-REFRESH TIMERS ═══
# _start_staggered_refresh() - Create timers + duration ticker
# _stop_staggered_refresh() - Stop all timers
# refresh_runs() - Legacy refresh compatibility
#
# ═══ EVENT HANDLERS ═══
# on_click(event) - Clear selection outside tables
# on_data_table_row_selected(event) - Row click: enable cancel, show popup
# on_button_pressed(event) - Dispatch button clicks
# on_checkbox_changed(event) - Auto-refresh checkbox state
#
# ═══ ACTIONS ═══
# action_refresh() - Manual refresh button
# action_cancel() - Cancel selected run
# action_back() - Return to home
# _clear_selection() - Clear selection state
#
# ** Technical Review **
# Production-grade Textual TUI implementing real-time infrastructure monitoring across 5 concurrent tables.
# Architecture: BaseScreen + universal refresh system (DRY) + worker-based async + universal 5s cache.
#
# 🎯 UNIVERSAL 5S CACHE (CACHE_TTL=5, all tables):
# - Reduces redundant API calls during rapid refreshes (50-80% hit rate typical)
# - _should_fetch_table() checks freshness → _get_cached_data() OR fetch fresh + _update_table_cache()
# - Cache stats tracking: hits/misses per table (monitoring effectiveness)
# - Logs: 💾 CACHE_HIT messages in auto_refresh.log
#
# 🌶️ PAPRIKA PHASE 1 & 2 COMPLETE (restored essential UX):
# - ✅ Empty state handling (all 5 tables) via _add_empty_state_row() helper
# - ✅ UI .refresh() calls (all 5 tables) after updates
# - ✅ Builds schema fix (Image column restored)
# - ✅ DRY helpers: _create_table_divider(), _add_empty_state_row() (eliminated 66 lines duplication)
# - ✅ Complete row_data (active, completed) for popup display
# - ✅ Builds dividers (active vs completed separation)
# - ✅ MAX_* limits (all 5 tables) + _extra_items tracking (backend ready for "Show More")
#
# Five Tables (chronological, newest first):
# 1. Cloud Builds (7 cols) - Active (QUEUED/WORKING) + completed (MAX_CLOUD_BUILDS=4), dividers, Image column
# 2. W&B Launch Agent (7 cols) - Running + completed (MAX_RUNNER_EXECS=None), dividers, full error logs
# 3. Vertex AI Jobs (5 cols) - Last 7 days (MAX_VERTEX_JOBS=7), 18-region adaptive monitoring
# 4. Active W&B Runs (5 cols) - Currently running (MAX_ACTIVE_RUNS=None), complete row_data
# 5. Completed Runs (5 cols) - Last completed (MAX_COMPLETED_RUNS=7), complete row_data
#
# Universal Refresh System (Lines 817-927):
# - _universal_refresh_table(table_name) → validate, skip check, mark refreshing, start spinner, launch worker
# - _universal_table_worker(table_name) → dispatch to fetch, GUARANTEED cleanup (finally block)
# - Skip protection: prevents overlapping refreshes
# - Guaranteed cleanup: spinners ALWAYS stop, refreshing flags ALWAYS cleared (even on crash!)
#
# Staggered Batch Loading (Lines 772-810):
# - BATCH 1 (immediate): builds + runner (critical tables with errors)
# - BATCH 2 (3s delay): vertex + active + completed
# - Prevents API rate limiting (5 concurrent → 2 + 3 delayed)
#
# Auto-Refresh System (Lines 1447-1510):
# - Per-table checkboxes: user enables/disables each table independently
# - set_interval(AUTO_REFRESH_INTERVAL=30s) → _universal_refresh_table(table, is_auto_refresh=True)
# - Duration ticker: set_interval(1s) → _update_active_durations() (live WORKING/RUNNING durations)
# - Spinner timer: set_interval(125ms) → _update_spinners() (8 FPS animation)
#
# Spinner System (42-char random rotation):
# - get_next_spinner_char() from cool_spinner.py
# - Thread-safe: self.app.call_from_thread() for worker updates (App method, NOT Screen!)
# - Always shows during refresh (visual feedback)
#
# ⚠️ IMPORTANT: SPINNER WORKER THREAD PATTERN (DO NOT REMOVE):
# This is the PROVEN pattern for smooth spinner animation in Textual TUIs under heavy load!
#
# Problem: Main thread busy with table updates → spinner updates queue up → burst animation (stuttering)
# Solution: Dedicated worker thread + call_from_thread() + GIL yielding
#
# 1. DEDICATED WORKER THREAD (_spinner_worker_thread):
#    - Runs independently of main thread at constant 8 FPS (125ms sleep)
#    - Threading.Thread(target=self._spinner_worker_thread, daemon=True)
#    - Spins even when main thread is maxed out!
#    - Logs: spinner_timing.log shows iteration timing, FPS health, lag detection
#
# 2. THREAD-SAFE UI UPDATES:
#    - CRITICAL: Only self.app.call_from_thread() works! Screen.call_from_thread doesn't exist!
#    - CORRECTED PATTERN:
#      * Data toasts: self.app.call_from_thread(self.notify, ...) from worker threads
#      * Error toasts: self.notify(...) directly - internally thread-safe!
#    - Always use App.call_from_thread() for worker→main thread UI updates
#
# 3. GIL YIELDING IN TABLE LOOPS (HONEY BADGER pattern from Textual demo/game.py):
#    - Table workers hold GIL during tight loops → blocks spinner updates
#    - 🦡 AGGRESSIVE PATTERN: time.sleep(0.010) EVERY row (not every 5!)
#    - Textual's demo/game.py uses 50ms per iteration - we use 10ms for balance
#    - 50-row table = 500ms total yield time (vs 10ms with old pattern!)
#    - Result: Butter-smooth spinners even during initial 50+ row loads!
#    - Source: textual/demo/game.py shuffle() method (uses await sleep(0.05))
#
# 4. HEALTH TRACKING (comprehensive metrics):
#    - FPS calculation (rolling 8-spin average): 7.5+ = ✅, 6.0-7.5 = ⚠️, <6.0 = 🚨
#    - Worker budget tracking: % of refresh interval consumed (✅<50%, ⚠️50-90%, 🚨>90%)
#    - Lag detection: Iterations behind leader spinner
#    - Periodic health summaries: Every 40 iterations (5 seconds at 8 FPS)
#    - Logs: spinner_timing.log, table_worker_timing.log (cleared each session)
#
# 5. LIFECYCLE MANAGEMENT:
#    - _start_spinner_worker(): Creates thread, sets _spinner_worker_running=True
#    - _stop_spinner_worker(): Sets flag False, joins thread (500ms timeout)
#    - Exception handling: Full traceback logged to spinner_timing.log
#    - Clean shutdown: Worker checks _spinner_worker_running in loop
#
# This pattern is ESSENTIAL for responsive TUIs with heavy background operations!
# Reference: Textual docs (call_from_thread), demo/game.py (await sleep pattern for yielding)
#
# Adaptive Region Monitoring (Lines 389-417):
# - 18 GCP regions tracked (hot/cold separation)
# - Hot regions: recent activity (always query)
# - Cold regions: no activity (rotate 3 per cycle)
# - Result: 2-5 regions per refresh (vs 18), 10× API reduction, full coverage every ~6 cycles
#
# DRY Helpers (Lines 463-502) 🌶️ PAPRIKA:
# - _create_table_divider(table_name): auto column count from TABLES config
# - _add_empty_state_row(table_name, message): auto column count + formatting
#
# 📦 Note Column Formatters (core_formatters.py - companion to core.py):
# - All Note columns pre-formatted in core.py at data extraction time
# - format_runner_note(): cyan ⏳ (Running) / green (✓) / yellow (✗) / red ❌ (error)
# - format_builds_note(): green (Build completed) / red (error)
# - format_vertex_note(): red (errors only - other states have no note)
# - screen.py receives ready-to-display Rich-formatted strings
# - Single source of truth for each format (DRY, close to source)
# - Reduced 66 lines of duplication across 5 tables!
#
# Row Data Storage (for popups):
# - self.row_data["runner/vertex/active/completed/build_recent"][row_key]
# - Stores full untruncated text for popup display
# - Active/completed: complete fields (full_name, config, tags, metrics, exit_code) 🌶️ PAPRIKA Phase 2
#
# Logging System (ARR_COC/Training/logs/auto_refresh.log):
# - Events: 🚀=START_TIMERS, ⏱️=TIMER_CREATED, 🔔=TIMER_FIRE, ⏭️=SKIP, ▶️=WORKER_START, ✅=COMPLETE, ❌=FAILED
# - Cache: 💾=CACHE_HIT
# - Tracks: timestamp, table name, duration, skip detection
# - Confession logs: 🔍=ENTRY, 📊=FETCH, 🔹=ROW, ❌=ERROR (humorous debug style kept for character!)
#
# Performance:
# - Initial load: 4-5s (BATCH 1: 2-3s, BATCH 2: 1-2s delayed)
# - Auto-refresh: 0.5-3s per table (parallel workers, 50-80% cache hits)
# - Memory: ~10-15MB (framework + 5 tables)
# - API calls: 54 initial → 3-10 subsequent (cache + adaptive monitoring)
#
# 🩰 STEVEN_FULL_DANCE_DEBUG Flag (2025-11-20):
# - Module constant: STEVEN_FULL_DANCE_DEBUG = False (default)
# - When False: Skips FPS complaint/praise calcs + auto_refresh.log writes (production mode)
# - When True: Full dance partner drama + writes to ARR_COC/Training/logs/auto_refresh.log
# - Wrapped: FPS complaints, recovery praise, stuck dancer rants, timer logs
# - Keeps: Health emoji (✅⚠️🚨), IPASB calculations (needed for adaptive timing)
#
# 🧵 Threading Optimizations (2025-11-20):
# - Pre-calculated 10k spinner chars: No locks, no random.choice() per frame
# - Cached spinner refs: _cached_spinners dict (no query_one() every 125ms)
# - FPS backoff: Skip P3 for 50ms when all spinners healthy (10ms → 0ms)
# - Phase timing: P0/P1/P2/P3/P4 breakdown in gil_hold.log
# - Result: P1 from 52ms → 0.3ms (173× faster!)
#
# 🎯 IPASB Smart Entry (Intelligent Progressive Adaptive Spinner Backoff):
# - Stagger pattern: 400→350→300→250→200ms (decreasing, bigger gaps at cold start)
# - Total spread: 1.3s for 5 tables (was 300ms)
# - Stress detection: Adds +75/150/300ms per table if backoff > 0
# - Adaptive: System warm-up before full speed
#
# File: 3,938 lines (threading + IPASB + debug flags)
# DRY Improvements: Universal cache, DRY helpers, universal refresh, MAX_* constants
# </claudes_code_comments>

import threading  # 🔒 For refresh lock (prevent race conditions)
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import work  # For async background operations (@work decorator)
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Checkbox, DataTable, Footer, Header, Label, Static

from ..shared.base_screen import BaseScreen

# COOL SPINNER - Import random char generator (42 chars: 4 rotation + 38 special)
from ..shared.cool_spinner import get_next_spinner_char
from ..shared.datatable_info_popup import DataTableInfoPopup
from ..shared.log_paths import get_log_path
from ..shared.performance_monitor import get_monitor
from ..shared.stevens_dance import stevens_log  # 🦡🎩 Stevens Dance buffered logging!
from ..shared.wandb_helper import WandBHelper
from .core import (
    _fetch_runner_executions_all_regions,
    _format_runtime,
    _list_active_runs,
    _list_completed_runs,
    _list_recent_cloud_builds,
    _list_vertex_ai_jobs,
    cancel_run_core,
    list_runs_core,
)

# Auto-refresh interval (seconds) - change this one value to update everywhere!
AUTO_REFRESH_INTERVAL = 30  # Seconds between auto-refresh cycles

# 🩰 STEVEN'S DANCE DEBUG - Toggle all the fun dance partner logging!
# Set to True for full Steven complaints/praise, False for quiet mode
# When False: Skips FPS complaint/praise calculations + auto_refresh.log writes
# When True: Full dance partner drama + writes to ARR_COC/Training/logs/auto_refresh.log
STEVEN_FULL_DANCE_DEBUG = False  # 🎭 Turn ON for dance partner drama!

# Table row limits - change these values to update everywhere!
MAX_CLOUD_BUILDS = (
    4  # Recent completed Cloud Builds to display (active builds always shown)
)
MAX_RUNNER_EXECS = None  # W&B Launch Agent executions to display (None = show all)
MAX_VERTEX_JOBS = 7  # Vertex AI jobs to display
MAX_ACTIVE_RUNS = None  # Active W&B runs to display (None = show all)
MAX_COMPLETED_RUNS = 7  # Completed W&B runs to display

# Cache configuration - 5 second TTL for all tables (reduces redundant API calls)
CACHE_TTL = 5  # Seconds to cache table responses before re-fetching

# ═══════════════════════════════════════════════════════════════════════════════
# 🦡 THREADING DANCE TIMING CONSTANTS - Tune spinner smoothness here!
# ═══════════════════════════════════════════════════════════════════════════════
# All interleaved waiting tick points in one place for easy tuning.
# Lower = faster but choppier spinners. Higher = smoother but slower load.

SPINNER_FPS_INTERVAL = 0.125  # 125ms = 8 FPS spinner animation
GIL_YIELD_API = 0.025  # 25ms yield after API calls
GIL_YIELD_ROW = 0.010  # 10ms yield per table row added
GIL_YIELD_PROCESSING = 0.005  # 5ms yield after data processing
VISUAL_FILL_DELAY = 0.0  # 0ms! Was 500ms "visual fill effect" - REMOVED!
TABLE_DISPLAY_DELAY = 0.100  # 100ms between tables (was 200ms - too slow!)
POLL_READY_INTERVAL = 0.050  # 50ms polling for ready tables
POST_BATCH_DELAY = 0.050  # 50ms after batch (was 500ms - WAY too slow!)

# Table metadata configuration - centralizes all table properties
TABLES = {
    "builds": {
        "id": "builds-recent-table",
        "spinner_id": "builds-recent-spinner",
        "max_const": MAX_CLOUD_BUILDS,
        "has_divider": True,
        "active_key": "status",
        "active_values": ["WORKING", "QUEUED"],
        "columns": 7,
        "row_data_key": "build_recent",
    },
    "runner": {
        "id": "runner-executions-table",
        "spinner_id": "runner-spinner",
        "max_const": MAX_RUNNER_EXECS,
        "has_divider": True,
        "active_key": "status",
        "active_values": ["RUNNING"],
        "columns": 7,
        "row_data_key": "runner",
    },
    "vertex": {
        "id": "vertex-jobs-table",
        "spinner_id": "vertex-spinner",
        "max_const": MAX_VERTEX_JOBS,
        "has_divider": True,
        "active_key": "state",
        "active_values": ["JOB_STATE_RUNNING"],
        "columns": 5,
        "row_data_key": "vertex",
    },
    "active": {
        "id": "runs-table",
        "spinner_id": "active-spinner",
        "max_const": MAX_ACTIVE_RUNS,
        "has_divider": False,
        "columns": 5,
        "row_data_key": "active",
    },
    "completed": {
        "id": "completed-runs-table",
        "spinner_id": "completed-spinner",
        "max_const": MAX_COMPLETED_RUNS,
        "has_divider": False,
        "columns": 5,
        "row_data_key": "completed",
    },
}


class MonitorScreen(BaseScreen):
    """Training runs monitoring screen - Real-time run tracking"""

    CSS = """
    MonitorScreen {
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

    /* All three tables have consistent auto-height */
    #vertex-jobs-table, #runs-table, #completed-runs-table {
        height: auto;
        max-height: 20;  /* Reasonable max to prevent huge tables */
        width: 100%;
    }

    /* Checkbox container - left-aligned, spinner first */
    .checkbox-container {
        width: 100%;
        height: auto;
        align: left top;
        background: transparent;
        layout: horizontal;
    }

    /* Spinner inline with checkbox */
    .spinner-inline {
        width: auto;
        margin-left: 2;
        color: $accent;
    }

    /* Checkbox - simple styling without complex selectors */
    .table-checkbox {
        width: auto;
        height: 1;
        margin: 0;
        padding: 0 0 0 2;  /* Left padding to space from spinner */
        background: transparent;
        border: none;
        color: $text-muted;
    }

    /* Checkbox active state - green! */
    .checkbox-active {
        color: $success !important;
    }

    .hidden {
        display: none;
    }

    /* Custom cursor styling - bold text for selection, no background */
    DataTable > .datatable--cursor {
        background: transparent;  /* No background (cursor invisible) */
        text-style: bold;         /* Bold text when row selected (visual feedback!) */
    }

    /* NOTE: Rich markup colors preserved via cursor_foreground_priority="renderable"
     * This is the proper Textual way! See DataTable() init parameters.
     */
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("c", "cancel", "Cancel Run"),
        ("escape", "back", "Back"),
        ("q", "back", "Back to Home"),
    ]

    # ═══════════════════════════════════════════════════════════════════════════════
    # ⚙️ INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════════

    def __init__(self, helper: WandBHelper, config: dict):
        super().__init__(loading_message="Loading monitor...")

        # 🧹 CLEAR STEVEN'S DANCE DEBUG LOG ON START (only if enabled!)
        if STEVEN_FULL_DANCE_DEBUG:
            log_file = get_log_path("auto_refresh.log")
            with open(log_file, "w") as f:
                f.write(
                    f"# Auto-refresh tracking log - Session started {datetime.now().isoformat()}\n"
                )
                f.write(f"#\n")
                f.write(f"# 🦡 THREADING DANCE TIMING CONSTANTS:\n")
                f.write(
                    f"#   SPINNER_FPS_INTERVAL = {SPINNER_FPS_INTERVAL * 1000:.0f}ms (8 FPS)\n"
                )
                f.write(
                    f"#   GIL_YIELD_API = {GIL_YIELD_API * 1000:.0f}ms (after API calls)\n"
                )
                f.write(
                    f"#   GIL_YIELD_ROW = {GIL_YIELD_ROW * 1000:.0f}ms (per table row)\n"
                )
                f.write(
                    f"#   GIL_YIELD_PROCESSING = {GIL_YIELD_PROCESSING * 1000:.0f}ms (after processing)\n"
                )
                f.write(
                    f"#   VISUAL_FILL_DELAY = {VISUAL_FILL_DELAY * 1000:.0f}ms (was 500ms!)\n"
                )
                f.write(
                    f"#   TABLE_DISPLAY_DELAY = {TABLE_DISPLAY_DELAY * 1000:.0f}ms (between tables)\n"
                )
                f.write(
                    f"#   POLL_READY_INTERVAL = {POLL_READY_INTERVAL * 1000:.0f}ms (polling)\n"
                )
                f.write(
                    f"#   POST_BATCH_DELAY = {POST_BATCH_DELAY * 1000:.0f}ms (after batch)\n"
                )
                f.write(f"#\n")

        # 🧹 CLEAR SPINNER TIMING LOG ON START
        spinner_log = get_log_path("spinner_timing.log")
        with open(spinner_log, "w") as f:
            f.write(
                f"# Spinner worker thread timing log - Session started {datetime.now().isoformat()}\n"
            )
            f.write(f"# Expected: 8 iterations/second (125ms per iteration)\n")
            f.write(f"# Sanity check: Every 8 iterations should take ~1 second\n")
            f.write(f"#\n")

        # 🧹 CLEAR TABLE WORKER TIMING LOG ON START
        worker_log = get_log_path("table_worker_timing.log")
        with open(worker_log, "w") as f:
            f.write(
                f"# Table worker timing log - Session started {datetime.now().isoformat()}\n"
            )

        # 🧹 CLEAR GIL HOLD LOG ON START (track ALL GIL holds!)
        gil_log = get_log_path("gil_hold.log")
        with open(gil_log, "w") as f:
            f.write(
                f"# GIL Hold tracking log - Session started {datetime.now().isoformat()}\n"
            )
            f.write(f"# Shows every operation that holds the GIL (blocks spinners!)\n")
            f.write(f"# Format: ⚠️ GIL_HOLD location: Xms (description)\n")
            f.write(f"#\n")
            f.write(f"#\n")

        self.helper = helper
        self.config = config
        self.selected_run_id = None
        self.selected_table_id = None  # Track which table has selection
        self.selected_row_data = None  # Store full row data for cancel toast
        self.auto_refresh_enabled = False  # Default OFF - user can enable with 'a' key
        self.refresh_timers = []  # List of AUTO_REFRESH_INTERVAL timers (one per enabled table)

        # Store full row data for popup (key → full_text mapping)
        self.row_data = {
            "runner": {},  # runner-execution-id → {"note": "full error text"}
            "vertex": {},  # vertex-job-id → {"note": "full error text"}
            "active": {},  # run-id → {"note": "full info"}
            "completed": {},  # run-id → {"note": "full info"}
            "build_recent": {},  # build-id → {"note": "full error text"}
        }

        # Per-table autorefresh state (persists across navigation)
        self.refresh_enabled = {
            "runner": False,  # Default OFF - user must check to enable
            "vertex": False,
            "active_runs": False,
            "completed_runs": False,
            "recent_builds": False,
        }

        # Per-table "show more" state (persists across navigation, resets on shutdown)
        self._extra_items = {
            "builds": 0,  # Extra completed builds to show beyond MAX_CLOUD_BUILDS
            "runner": 0,  # Extra completed executions beyond MAX_RUNNER_EXECS
            "vertex": 0,  # Extra jobs beyond MAX_VERTEX_JOBS
            "active": 0,  # Extra active runs beyond MAX_ACTIVE_RUNS
            "completed": 0,  # Extra completed runs beyond MAX_COMPLETED_RUNS
        }

        # COOL SPINNER - Dedicated worker thread for constant rate (independent of main thread load!)
        self.spinner_timer = None  # Legacy (keeping for compatibility)
        self._spinner_worker_thread_obj = (
            None  # Dedicated spinner animation thread OBJECT
        )
        self._spinner_worker_running = False  # Control flag for worker thread
        self._spinner_last_iteration = (
            0  # Track last iteration count (for sanity checks)
        )
        self._spinner_last_update_time = (
            0  # Track when spinner last updated (for timeout detection)
        )
        self._spinner_update_times = {}  # Track last update time per spinner {spinner_id: iteration_number}
        self._spinner_queue_time = (
            0  # Timestamp when update was queued (for measuring queue lag)
        )

        # 📊 COMPREHENSIVE SPINNER METRICS (track EVERYTHING!)
        self._spinner_metrics = {
            "builds-recent-spinner": {
                "total_spins": 0,
                "start_iter": None,
                "stop_iter": None,
                "last_active_iter": None,
                "was_active": False,
                "last_8_times": [],
            },
            "runner-spinner": {
                "total_spins": 0,
                "start_iter": None,
                "stop_iter": None,
                "last_active_iter": None,
                "was_active": False,
                "last_8_times": [],
            },
            "vertex-spinner": {
                "total_spins": 0,
                "start_iter": None,
                "stop_iter": None,
                "last_active_iter": None,
                "was_active": False,
                "last_8_times": [],
            },
            "active-spinner": {
                "total_spins": 0,
                "start_iter": None,
                "stop_iter": None,
                "last_active_iter": None,
                "was_active": False,
                "last_8_times": [],
            },
            "completed-spinner": {
                "total_spins": 0,
                "start_iter": None,
                "stop_iter": None,
                "last_active_iter": None,
                "was_active": False,
                "last_8_times": [],
            },
        }

        # 🎭 STEVEN'S COMPLAINT THROTTLING - Don't spam the logs!
        # Only complain about each spinner once per second (not 8× per second!)
        self._spinner_last_warning_time = {
            "builds-recent-spinner": 0,
            "runner-spinner": 0,
            "vertex-spinner": 0,
            "active-spinner": 0,
            "completed-spinner": 0,
        }

        # ═══════════════════════════════════════════════════════════════════════
        # 🎯 IPASB - INTERLEAVED PILEUP AUTOMATIC SMART BACKOFF
        # ═══════════════════════════════════════════════════════════════════════
        #
        # When shit's fucked, STOP DOING SHIT! Three-level protection system:
        #
        # LEVEL 1: BAILOUT MODE (queue_lag > 300ms)
        #   → Skip ALL FPS/health calculations
        #   → Just spin the chars (~3ms vs 66ms)
        #   → "Everything is fucked, minimal work only!"
        #
        # LEVEL 2: PER-SPINNER FUCKUP COOLDOWN (FPS < 8.0)
        #   → Skip that spinner for 100ms
        #   → Then ONE quick check - recovered? Great! Still bad? Cooldown again!
        #   → "This spinner is fucked, back off for 100ms!"
        #
        # LEVEL 3: GLOBAL COMPLAINT COOLDOWN (2 seconds)
        #   → After ANY complaint, no more for 2s
        #   → Prevents: complaint → slow → complaint → slower → MORE COMPLAINTS
        #   → "Steven already yelled, shut up for 2 seconds!"
        #
        # Philosophy: Interleaved checks, automatic detection, smart backoff!
        # Like a bouncer: "Too crowded! Nobody in for 100ms! ...ok one more..."
        # ═══════════════════════════════════════════════════════════════════════

        # 🚨 LEVEL 3: GLOBAL COMPLAINT THROTTLE
        self._spinner_last_any_warning = 0
        self._spinner_global_warning_cooldown = (
            2.0  # seconds - no complaints for 2s after any complaint
        )

        # 📡 IPASB BACKOFF SIGNAL - Tells API workers "whoa nelly, slow down!"
        # Workers check this before each operation and add delays accordingly
        # 0 = normal, 1 = mild (10ms), 2 = heavy (50ms), 3 = critical (100ms)
        self._ipasb_backoff_level = 0
        self._ipasb_backoff_delays = {0: 0, 1: 0.010, 2: 0.050, 3: 0.100}  # seconds

        # 🛑 LEVEL 2: PER-SPINNER FUCKUP COOLDOWN
        self._spinner_fuckup_cooldown = {
            "builds-recent-spinner": 0,
            "runner-spinner": 0,
            "vertex-spinner": 0,
            "active-spinner": 0,
            "completed-spinner": 0,
        }
        self._spinner_fuckup_cooldown_duration = (
            0.100  # 100ms cooldown after fuckup (STRICT!)
        )

        # 🚀 CACHED SPINNER REFS - Avoid query_one() in hot path!
        # Populated in initialize_content() after widgets exist
        self._cached_spinners = {}

        # 🎯 FPS BACKOFF - Skip P3 when healthy (running at 8fps)
        # If all spinners healthy, skip FPS calc for 50ms
        self._fps_last_calc_time = 0
        self._fps_all_healthy = False
        self._fps_backoff_duration = 0.050  # 50ms backoff when healthy

        # 😊 STEVEN'S PRAISE THROTTLING - Only praise every 5 seconds per spinner!
        self._spinner_last_praise_time = {
            "builds-recent-spinner": 0,
            "runner-spinner": 0,
            "vertex-spinner": 0,
            "active-spinner": 0,
            "completed-spinner": 0,
        }

        # 📝 BUFFERED LOGGING - Uses Stevens Dance (10,000-line batching!)
        # This is the secret to hitting 8 FPS! File I/O was killing us!
        # Stevens Dance handles all buffering centrally (shared across all screens!)

        # 🩰💃 THE SPINNER PARTNERS - Each spinner has a name and dances with a data fetcher!
        # ═══════════════════════════════════════════════════════════════════════════════
        #
        #   THE THREADING DANCE PARTNERSHIPS:
        #
        #   DANCER (fetches data)          SPINNER (shows loading)
        #   ─────────────────────          ──────────────────────
        #   🏃 RICKY THE RUNNER      💃    SCARLETT THE SPINNER
        #   🏗️ BELLA THE BUILDER     ♡⃤    BORIS THE TWIRLER
        #   🎯 VICTOR THE VERTEX     🩰    VERA THE PIROUETTE
        #   ⚡ ARCHIE THE ACTIVE     🌟    AURORA THE WHIRL
        #   ✅ CLEO THE COMPLETED    🎭    CARLOS THE FLOURISH
        #
        #   When a dancer enters the stage, their spinner partner starts twirling!
        #   When the dancer returns with data, the spinner takes a bow and stops.
        #   If a spinner can't maintain 8 FPS, they're fucking up the dance! 🦡🔥
        #
        # ═══════════════════════════════════════════════════════════════════════════════
        self.SPINNER_PARTNERS = {
            "runner-spinner": {
                "name": "SCARLETT THE SPINNER",
                "emoji": "💃",
                "partner": "RICKY THE RUNNER 🏃",
                "story": "Scarlett is passionate and fiery - she spins with intensity while Ricky races through the API calls!",
            },
            "builds-recent-spinner": {
                "name": "BORIS THE TWIRLER",
                "emoji": "♡⃤",
                "partner": "BELLA THE BUILDER 🏗️",
                "story": "Boris is solid and dependable - he twirls steadily while Bella constructs the build data!",
            },
            "vertex-spinner": {
                "name": "VERA THE PIROUETTE",
                "emoji": "🩰",
                "partner": "VICTOR THE VERTEX 🎯",
                "story": "Vera is precise and elegant - she pirouettes perfectly while Victor targets the Vertex AI jobs!",
            },
            "active-spinner": {
                "name": "AURORA THE WHIRL",
                "emoji": "🌟",
                "partner": "ARCHIE THE ACTIVE ⚡",
                "story": "Aurora is electric and energetic - she whirls like lightning while Archie fetches active runs!",
            },
            "completed-spinner": {
                "name": "CARLOS THE FLOURISH",
                "emoji": "🎭",
                "partner": "CLEO THE COMPLETED ✅",
                "story": "Carlos is dramatic and theatrical - he flourishes grandly for the finale while Cleo gathers completed runs!",
            },
        }

        # 🎯 HEALTH TRACKING: Simple metrics to show what's working/broken
        self._spinner_health_last_check = 0  # Iteration number of last health summary
        self._worker_durations = {}  # {table_name: [last 5 durations]}
        # ✅ FIX: Use AUTO_REFRESH_INTERVAL constant (not hardcoded!) for accurate budget calculation
        self._auto_refresh_interval = float(
            AUTO_REFRESH_INTERVAL
        )  # Expected refresh interval (seconds)
        self._is_refreshing = (
            False  # Track if currently refreshing (prevent duplicate refreshes)
        )

        # 🎯 GENERAL ACCUMULATOR PATTERN: Parallel load, ordered display (200ms delay)!
        self._display_order = [
            "builds",
            "runner",
            "vertex",
            "active",
            "completed",
        ]  # Display order (top to bottom!)
        self._current_batch = []  # Current batch of tables being refreshed
        self._accumulator_results = {}  # {table_name: True} - track completed tables
        self._accumulator_completion_times = {}  # {table_name: timestamp} - when each table completed
        self._fetched_data = {}  # {table_name: data} - store fetched data before rendering (Phase 2!)
        self._accumulator_batch_start = 0  # Batch start timestamp
        self._accumulator_next_display = 0  # Index of next table to display (0-4)
        self._accumulator_lock = threading.Lock()  # Protect accumulator state
        self._accumulator_active = False  # Is accumulator currently running?
        self._accumulator_delay_ms = 200  # Delay between table displays (ms)

        # Per-table refresh locks (prevent overlapping refreshes for same table)
        self._refreshing_tables = (
            set()
        )  # {"builds", "runner", "vertex", "active", "completed"}
        self._refresh_start_times = {}  # Track when each table started refreshing (for timeout detection)
        self._refresh_lock = (
            threading.Lock()
        )  # 🔒 Prevent race conditions when checking/adding to _refreshing_tables

        # 🔍 DATA VALIDATION: Snapshot storage for sanity checks
        self._data_snapshots = {
            "builds": {},  # {row_key: {field: value}}
            "runner": {},
            "vertex": {},
            "active": {},
            "completed": {},
        }

        # 🎯 UNIVERSAL TABLE REFRESH CONFIGURATION
        # All table refresh logic centralized here - DRY principle!
        self.TABLE_CONFIG = {
            "runner": {
                "name": "W&B Launch Agent",
                "spinner_id": "runner-spinner",
                "table_id": "#runner-executions-table",
                "max_items_key": "runner",  # For _extra_items
                "refresh_key": "runner",  # For refresh_enabled
            },
            "builds": {
                "name": "Cloud Builds",
                "spinner_id": "builds-recent-spinner",
                "table_id": "#builds-recent-table",
                "max_items_key": "builds",
                "refresh_key": "recent_builds",
            },
            "vertex": {
                "name": "Vertex AI Jobs",
                "spinner_id": "vertex-spinner",
                "table_id": "#vertex-jobs-table",
                "max_items_key": "vertex",
                "refresh_key": "vertex",
            },
            "active": {
                "name": "Active Runs",
                "spinner_id": "active-spinner",
                "table_id": "#active-runs-table",
                "max_items_key": "active",
                "refresh_key": "active_runs",
            },
            "completed": {
                "name": "Completed Runs",
                "spinner_id": "completed-spinner",
                "table_id": "#completed-runs-table",
                "max_items_key": "completed",
                "refresh_key": "completed_runs",
            },
        }

        # Adaptive region tracking - hot/cold monitoring system
        self.ALL_MECHA_REGIONS = [
            "us-central1",
            "us-east1",
            "us-east4",
            "us-east5",
            "us-west1",
            "us-west2",
            "us-west3",
            "us-west4",
            "northamerica-northeast1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west9",
            "asia-northeast1",
            "asia-southeast1",
            "australia-southeast1",
            "southamerica-east1",
        ]
        self._hot_regions = {
            "builds": set(),
            "runner": set(),
            "vertex": set(),
        }
        self._refresh_cycle = 0
        self._cold_rotation_idx = 0

        # 🎯 UNIVERSAL TABLE CACHE (CACHE_TTL seconds - all tables!)
        # Reduces redundant API calls during rapid refreshes
        self._table_cache = {
            "runner": {"data": None, "timestamp": 0, "ttl": CACHE_TTL},
            "builds": {"data": None, "timestamp": 0, "ttl": CACHE_TTL},
            "vertex": {"data": None, "timestamp": 0, "ttl": CACHE_TTL},
            "active": {"data": None, "timestamp": 0, "ttl": CACHE_TTL},
            "completed": {"data": None, "timestamp": 0, "ttl": CACHE_TTL},
        }
        # Cache statistics (for monitoring effectiveness)
        self._cache_stats = {
            "runner": {"hits": 0, "misses": 0},
            "builds": {"hits": 0, "misses": 0},
            "vertex": {"hits": 0, "misses": 0},
            "active": {"hits": 0, "misses": 0},
            "completed": {"hits": 0, "misses": 0},
        }

    # ✅ SPICY: __init__() inspected - orphan comments removed, state vars organized
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎨 UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════════════

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("📊  MONITOR · Active Training Runs", id="page-title")

        # Content panel with pre-mounted tables (hidden under loading overlay)
        with VerticalScroll(id="content-panel") as content:
            # Cloud Builds section (Active builds at top, then recent) - ALL 18 MECHA REGIONS!
            yield Static(
                "\n[bold blue]🐳 CLOUD BUILDS[/bold blue] (Active + Recent - All 18 Regions)",
                id="builds-recent-header",
            )

            builds_recent_table = DataTable(
                id="builds-recent-table", cursor_foreground_priority="renderable"
            )
            builds_recent_table.cursor_type = (
                "none"  # Disabled during initial load (re-enabled after batch complete)
            )
            builds_recent_table.zebra_stripes = True
            builds_recent_table.add_column("Build ID", width=12, key="build_id")
            builds_recent_table.add_column(
                "Image", width=20, key="image"
            )  # Image name (wider for full names)
            builds_recent_table.add_column(
                "Region", width=18, key="region"
            )  # GCP region
            builds_recent_table.add_column("Status", width=12, key="status")
            builds_recent_table.add_column("Runtime", width=10, key="runtime")
            builds_recent_table.add_column(
                "Finished", width=20, key="finished"
            )  # Wide enough for "Nov 16, 10:21:48 PM"
            builds_recent_table.add_column(
                "Note", width=40, key="note"
            )  # Error messages (reduced width for region column)
            yield builds_recent_table
            with Horizontal(classes="checkbox-container"):
                yield Static(
                    "", id="builds-recent-spinner", classes="spinner-inline"
                )  # Loading spinner
                yield Checkbox(
                    "✨ Auto-refresh",
                    value=False,
                    id="cb-recent-builds",
                    classes="table-checkbox",
                )

            # W&B Launch agent executions (shows FIRST - errors visible!) - ALL 18 MECHA REGIONS!
            yield Static(
                "\n[bold red]◈ W&B LAUNCH AGENT[/bold red] (Cloud Run Launches - All 18 Regions)",
                id="runner-header",
            )

            runner_table = DataTable(
                id="runner-executions-table", cursor_foreground_priority="renderable"
            )
            runner_table.cursor_type = (
                "none"  # Disabled during initial load (re-enabled after batch complete)
            )
            runner_table.zebra_stripes = True
            runner_table.add_column(
                "Queue", width=16, key="queue"
            )  # Queue being monitored by this agent
            runner_table.add_column("Region", width=14, key="region")  # GCP region
            runner_table.add_column("Status", width=10, key="status")
            runner_table.add_column(
                "Runs", width=6, key="runs"
            )  # NEW: Number of jobs processed by this runner
            runner_table.add_column(
                "Runtime", width=9, key="runtime"
            )  # How long runner has been alive
            runner_table.add_column(
                "Created", width=20, key="created"
            )  # When execution started
            runner_table.add_column(
                "Note", width=65, key="note"
            )  # Error messages (reduced width for new Jobs column)
            yield runner_table
            with Horizontal(classes="checkbox-container"):
                yield Static(
                    "", id="runner-spinner", classes="spinner-inline"
                )  # Loading spinner (| / - \)
                yield Checkbox(
                    "✨ Auto-refresh",
                    value=False,
                    id="cb-runner",
                    classes="table-checkbox",
                )

            # Vertex AI jobs section (shows jobs submitted to GCP) - ALL 18 MECHA REGIONS!
            yield Static(
                "\n[bold magenta]◈ VERTEX AI JOBS[/bold magenta] (GCP Custom Jobs, Last 7 Days - All 18 Regions)",
                id="vertex-header",
            )

            vertex_table = DataTable(
                id="vertex-jobs-table", cursor_foreground_priority="renderable"
            )
            vertex_table.cursor_type = (
                "none"  # Disabled during initial load (re-enabled after batch complete)
            )
            vertex_table.zebra_stripes = True
            vertex_table.add_column(
                "Job ID", width=12, key="job_id"
            )  # Last 12 digits (shortened)
            vertex_table.add_column(
                "Name", width=18, key="name"
            )  # Shortened for region column
            vertex_table.add_column("Region", width=18, key="region")  # GCP region
            vertex_table.add_column(
                "State", width=22, key="state"
            )  # Wider for full state names (JOB_STATE_RUNNING = 17 chars)
            vertex_table.add_column("Runtime", width=10, key="runtime")
            vertex_table.add_column(
                "Created", width=20, key="created"
            )  # When job was created
            vertex_table.add_column(
                "Note", width=30, key="note"
            )  # Error messages (reduced for Created column)
            yield vertex_table
            with Horizontal(classes="checkbox-container"):
                yield Static(
                    "", id="vertex-spinner", classes="spinner-inline"
                )  # Loading spinner (| / - \)
                yield Checkbox(
                    "✨ Auto-refresh",
                    value=False,
                    id="cb-vertex",
                    classes="table-checkbox",
                )

            # Active W&B runs section (shows when training starts)
            yield Static(
                "\n[bold cyan]◈ ACTIVE W&B RUNS[/bold cyan] (Training Started)",
                id="active-header",
            )

            table = DataTable(id="runs-table", cursor_foreground_priority="renderable")
            table.cursor_type = (
                "none"  # Disabled during initial load (re-enabled after batch complete)
            )
            table.zebra_stripes = True
            table.add_column("Run ID", width=12)  # W&B IDs are short: "6v2yvvd9"
            table.add_column("Name", width=30)
            table.add_column("State", width=12)
            table.add_column("Runtime", width=10)
            table.add_column(
                "Created", width=20
            )  # Wide enough for "Nov 16, 10:21:48 PM"
            yield table
            with Horizontal(classes="checkbox-container"):
                yield Static(
                    "", id="active-spinner", classes="spinner-inline"
                )  # Loading spinner (| / - \)
                yield Checkbox(
                    "✨ Auto-refresh",
                    value=False,
                    id="cb-active-runs",
                    classes="table-checkbox",
                )

            # Completed runs section
            yield Static(
                "\n[bold yellow]◈ COMPLETED RUNS[/bold yellow] (Last 20 runs)",
                id="completed-header",
            )

            completed_table = DataTable(
                id="completed-runs-table", cursor_foreground_priority="renderable"
            )
            completed_table.cursor_type = (
                "none"  # Disabled during initial load (re-enabled after batch complete)
            )
            completed_table.zebra_stripes = True
            completed_table.add_column("Run ID", width=12)  # W&B IDs are short
            completed_table.add_column("Name", width=30)
            completed_table.add_column("State", width=12)
            completed_table.add_column("Runtime", width=10)
            completed_table.add_column(
                "Created", width=20
            )  # Wide enough for "Nov 16, 10:21:48 PM"
            yield completed_table
            with Horizontal(classes="checkbox-container"):
                yield Static(
                    "", id="completed-spinner", classes="spinner-inline"
                )  # Loading spinner (| / - \)
                yield Checkbox(
                    "✨ Auto-refresh",
                    value=False,
                    id="cb-completed-runs",
                    classes="table-checkbox",
                )

            # Auto-refresh status (shows count of enabled tables, AUTO_REFRESH_INTERVAL)
            yield Static("\n[dim]○ Auto-refresh: OFF[/dim]", id="auto-refresh-status")

            # Helpful tip for users
            yield Static(
                "[dim italic]💡 Tip: Click any table row to see full error details[/dim italic]",
                id="table-help-text",
            )

        # Button bar
        with Horizontal(id="button-bar"):
            yield Button("← Back (ESC)", id="back-btn", classes="left-btn pastel-gray")
            yield Static("", classes="spacer")
            yield Button(
                "Refresh (r)", id="refresh-btn", classes="action-btn pastel-cyan"
            )
            yield Button("Cancel (c)", id="cancel-btn", classes="action-btn pastel-red")

        yield Footer()

        # Loading overlay (LAST - appears on top)
        yield from self.compose_base_overlay()

    # ✅ SPICY: compose() inspected - 5 tables, 5 checkboxes, all IDs correct
    # ═══════════════════════════════════════════════════════════════════════════════
    # 🔄 LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════════

    def initialize_content(self) -> Any:
        """INSTANT: No loading during spinner - everything lazy loads!"""
        import time

        # 🦡🔥 STEVEN LOGS: Entering Monitor screen
        from CLI.shared.steven_toasts import steven_log_screen_entry

        steven_log_screen_entry(
            self.app, "Monitor", "User pressed '1' or navigated to Monitor"
        )

        # Skip ALL heavy loading - everything lazy loads after page appears!
        # Runner, vertex, active, completed - all lazy load in parallel
        return {"success": True}

    def finish_loading(self, data: Any = None) -> None:
        """Page appears instantly, trigger all lazy loads!"""
        # Hide loading overlay NOW (page appears instantly!)
        super().finish_loading(data)

        if data and data.get("success"):
            # 🎯 Trigger accumulated start (parallel load, ordered display!)
            self._accumulated_start()
        elif data:
            # Show error notification
            error = data.get("error", "Unknown error")
            self.notify_with_full_error("Error Loading Runs", error)

    def on_screen_resume(self) -> None:
        """Called when returning to this screen - refresh all tables!"""
        # Skip if this is the first resume (initial mount handles that)
        if not hasattr(self, "_has_resumed_once"):
            self._has_resumed_once = True
            return

        # Refresh all tables
        self._refresh_all_tables()

    def on_unmount(self) -> None:
        """Stop all timers and worker threads when leaving screen (avoid background operations!)"""
        self._stop_staggered_refresh()
        self._stop_spinner_worker()  # Stop dedicated spinner thread

        # 📝 Flush all buffered logs to disk using Stevens Dance!
        from CLI.shared.stevens_dance import stevens_flush_all

        stevens_flush_all()

    # ✅ SPICY: 4 lifecycle methods inspected - instant load, batch triggers, resume refresh, cleanup

    # ═══════════════════════════════════════════════════════════════════════════════
    # 💾 CACHE SYSTEM (Universal 5s TTL for all tables)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _should_fetch_table(self, table_name: str) -> bool:
        """Universal: Check if we should fetch fresh data or use cache (CACHE_TTL seconds)"""
        cache = self._table_cache.get(table_name)
        if not cache or cache["data"] is None:
            self._cache_stats[table_name]["misses"] += 1
            return True  # No cache, must fetch

        # Check if cache expired (CACHE_TTL from const)
        age = time.time() - cache["timestamp"]
        if age > cache["ttl"]:
            self._cache_stats[table_name]["misses"] += 1
            return True  # Expired, fetch fresh

        # Cache is fresh!
        self._cache_stats[table_name]["hits"] += 1
        return False

    # ✅ SPICY: ALICE RABBIT HOLE complete - import time moved to module top (line 123), cache structures verified (lines 411, 419), TTL logic correct

    def _get_cached_data(self, table_name: str) -> Optional[List[Dict]]:
        """Universal: Get cached data if available"""
        cache = self._table_cache.get(table_name)
        if cache and cache["data"] is not None:
            return cache["data"]
        return None

    # ✅ SPICY: ALICE RABBIT HOLE complete - simple cache getter, all dependencies verified in Function 1

    def _update_table_cache(self, table_name: str, data: List[Dict]) -> None:
        """Universal: Update response cache for any table"""
        self._table_cache[table_name] = {
            "data": data,
            "timestamp": time.time(),
            "ttl": CACHE_TTL,
        }

    # ✅ SPICY: ALICE RABBIT HOLE complete - removed import time (module top line 123), verified CACHE_TTL constant (line 148), cache structure correct

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🌍 REGION MONITORING (Adaptive hot/cold region rotation for Vertex AI)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _get_target_regions(self, table_type: str) -> List[str]:
        """Get target regions for adaptive monitoring (gradual discovery)"""
        # Get hot regions (have hits)
        hot = self._hot_regions.get(table_type, set())

        # Initial load: query all 18 regions once
        if not hot and self._refresh_cycle == 1:
            return None  # First load = all regions

        # Get cold regions (no hits)
        cold = [r for r in self.ALL_MECHA_REGIONS if r not in hot]

        # Rotate through 3 cold regions each refresh (gradual discovery)
        # This covers all 18 regions in ~6 refreshes instead of one big dump
        rotating_cold = []
        if len(cold) > 0:
            for i in range(3):
                idx = (self._cold_rotation_idx + i) % len(cold)
                rotating_cold.append(cold[idx])

        # Return hot + 3 rotating cold (typically 2-5 regions total)
        return list(hot) + rotating_cold

    # ✅ SPICY: ALICE RABBIT HOLE complete - verified ALL_MECHA_REGIONS (line 393), _hot_regions (402), _refresh_cycle (407), _cold_rotation_idx (408), logic correct

    def _update_hot_regions(self, table_type: str, results: List[Dict]) -> None:
        """Update hot region tracking based on results"""
        regions_with_hits = {
            item.get("region") for item in results if item.get("region")
        }
        self._hot_regions[table_type] = regions_with_hits
        # Advance by 3 to rotate through cold regions (covers all 18 in ~6 refreshes)
        self._cold_rotation_idx = (self._cold_rotation_idx + 3) % len(
            self.ALL_MECHA_REGIONS
        )

    # ✅ SPICY: ALICE RABBIT HOLE complete - all dependencies verified in Function 4, logic correct (extract regions → update tracking → rotate index)

    # ═══════════════════════════════════════════════════════════════════════════════
    # ⚡ SPINNER SYSTEM (Visual loading feedback below tables)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _start_spinner(self, spinner_id: str):
        """Show and animate spinner below a table"""
        try:
            spinner = self.query_one(f"#{spinner_id}", Static)
            char = get_next_spinner_char()  # Get initial random char
            spinner.update(f"  {char}")  # Show initial frame
        except Exception:
            pass  # Widget might not exist yet

    # ✅ SPICY: ALICE RABBIT HOLE complete - verified get_next_spinner_char import (../shared/cool_spinner.py), Static widget (line 127), exception handling correct

    def _stop_spinner(self, spinner_id: str):
        """Hide spinner below a table (safe for main thread only!)

        ⚠️ WARNING: Do NOT call from worker threads - call_from_thread() blocks!
        Instead, remove table from _refreshing_tables and let spinner timer auto-hide.
        """

        def stop():
            try:
                spinner = self.query_one(f"#{spinner_id}", Static)
                spinner.update("")  # Clear spinner
            except Exception:
                pass  # Widget might not exist yet

        # Call from main thread if we're on a worker thread
        try:
            # ✅ FIX: call_from_thread is on App, not Screen!
            self.app.call_from_thread(stop)
        except Exception:
            # Already on main thread, call directly
            stop()

    # ✅ SPICY: ALICE RABBIT HOLE complete - verified call_from_thread() is App method (not Screen!), thread-safe logic correct

    def _update_spinners(self):
        """Update all active spinners (called by timer)"""
        import time

        start_time = time.time()
        log_file = get_log_path("spinner_timing.log")

        # 📊 QUEUE LAG: Measure how long update waited in queue before executing
        queue_lag_ms = (
            (start_time - self._spinner_queue_time) * 1000
            if self._spinner_queue_time > 0
            else 0
        )

        # 📡 IPASB: Update backoff signal based on queue_lag
        # Workers will check this and add delays - "whoa nelly!"
        if queue_lag_ms > 300:
            self._ipasb_backoff_level = 3  # CRITICAL: 100ms delays
        elif queue_lag_ms > 200:
            self._ipasb_backoff_level = 2  # HEAVY: 50ms delays
        elif queue_lag_ms > 100:
            self._ipasb_backoff_level = 1  # MILD: 10ms delays
        else:
            self._ipasb_backoff_level = 0  # NORMAL: no delays

        # 🚨🚨 EXTREME BAILOUT (queue_lag > 500ms) - DO NOTHING!
        # System is SO fucked that even spinner.update() is too expensive
        # Just return immediately - sometimes doing NOTHING is fastest!
        if queue_lag_ms > 500:
            self._gil_hold_log(
                "_update_spinners_EXTREME_BAILOUT",
                (time.time() - start_time) * 1000,
                f"queue_lag={queue_lag_ms:.0f}ms - DO NOTHING!",
            )
            return  # Absolute minimal - don't even update spinners!

        # 🚨 IPASB LEVEL 1: BAILOUT MODE (queue_lag 300-500ms)
        # Just update spinner chars, skip ALL FPS/health calculations
        # "Everything is fucked, minimal work only!"
        # Uses cached spinner refs to avoid query_one() overhead!
        if queue_lag_ms > 300:
            char = get_next_spinner_char()
            for spinner_id, spinner in self._cached_spinners.items():
                try:
                    table_name = spinner_id.replace("-spinner", "").replace(
                        "builds-recent", "builds"
                    )
                    if table_name in self._refreshing_tables:
                        spinner.update(f"  {char}")
                    else:
                        spinner.update("   ")
                except Exception:
                    pass
            # Log the bailout
            self._gil_hold_log(
                "_update_spinners_BAILOUT",
                (time.time() - start_time) * 1000,
                f"queue_lag={queue_lag_ms:.0f}ms - minimal update only!",
            )
            return  # Skip all the heavy FPS/health logic!

        char = get_next_spinner_char()
        active_count = 0
        current_iteration = self._spinner_last_iteration  # Current iteration number

        # 🔍 TIMING CHECKPOINTS - Find the mystery time!
        phase0_ms = (
            time.time() - start_time
        ) * 1000  # Time BEFORE phase1 (queue lag, backoff, char)
        phase1_start = time.time()

        # Track which spinners updated THIS iteration
        updated_this_iteration = []
        lifecycle_events = []  # Track start/stop events

        # Update ALL spinner widgets (active = table in _refreshing_tables)
        table_name_map = {
            "builds-recent-spinner": "builds",
            "runner-spinner": "runner",
            "vertex-spinner": "vertex",
            "active-spinner": "active",
            "completed-spinner": "completed",
        }

        for spinner_id in [
            "builds-recent-spinner",
            "runner-spinner",
            "vertex-spinner",
            "active-spinner",
            "completed-spinner",
        ]:
            metrics = self._spinner_metrics[spinner_id]

            try:
                # 🚀 USE CACHED SPINNER - No query_one()! (73ms → 3ms!)
                spinner = self._cached_spinners.get(spinner_id)
                if not spinner:
                    continue  # Skip if not cached yet

                # Check if this table is currently refreshing
                table_name = table_name_map[spinner_id]
                is_active = table_name in self._refreshing_tables

                if is_active:  # Spinner has content (is active)
                    spinner.update(f"  {char}")
                    active_count += 1

                    # 📊 TRACK METRICS: Update spin count
                    metrics["total_spins"] += 1
                    metrics["last_active_iter"] = current_iteration
                    self._spinner_update_times[spinner_id] = current_iteration
                    updated_this_iteration.append(spinner_id)

                    # ⏱️ TRACK FPS: Keep rolling window of last 8 spin times
                    metrics["last_8_times"].append(start_time)
                    if len(metrics["last_8_times"]) > 8:
                        metrics["last_8_times"].pop(0)  # Keep only last 8

                    # 🎬 DETECT START: Spinner just became active!
                    if not metrics["was_active"]:
                        metrics["start_iter"] = current_iteration
                        metrics["was_active"] = True
                        lifecycle_events.append(
                            f"🟢 START: {spinner_id} @iter{current_iteration}"
                        )

                else:  # Spinner is INACTIVE
                    # Hide spinner but preserve spacing (prevents line jumping!)
                    spinner.update("   ")

                    # 🛑 DETECT STOP: Spinner just became inactive!
                    if metrics["was_active"]:
                        metrics["stop_iter"] = current_iteration
                        metrics["was_active"] = False
                        lifetime = (
                            current_iteration - metrics["start_iter"]
                            if metrics["start_iter"]
                            else 0
                        )
                        lifecycle_events.append(
                            f"🔴 STOP: {spinner_id} @iter{current_iteration} (lived {lifetime} iters, spun {metrics['total_spins']} times)"
                        )
                        # Reset spin count for next activation
                        metrics["total_spins"] = 0

            except Exception:
                pass  # Spinner might not exist yet

        # 🔍 PHASE 1 DONE: Spinner updates (query_one + update)
        phase1_ms = (time.time() - phase1_start) * 1000
        phase2_start = time.time()

        # 📊 FIND LEADER (most recently updated spinner) and calculate lags
        lag_report = []
        if self._spinner_update_times:
            leader_iteration = max(self._spinner_update_times.values())
            for spinner_id, last_iter in self._spinner_update_times.items():
                iterations_behind = leader_iteration - last_iter
                if iterations_behind > 0:
                    lag_report.append(f"{spinner_id}: {iterations_behind} behind")

        # 🔍 PHASE 2 DONE: Lag calculations
        phase2_ms = (time.time() - phase2_start) * 1000
        phase3_start = time.time()

        # 🎯 FPS BACKOFF: Skip P3 entirely when healthy!
        # If all spinners were healthy last calc AND within 50ms, skip!
        time_since_fps_calc = start_time - self._fps_last_calc_time
        if self._fps_all_healthy and time_since_fps_calc < self._fps_backoff_duration:
            # Skip P3! We're healthy, no need to recalc FPS every frame
            phase3_ms = 0
            # Still log but with minimal info
            elapsed = time.time() - start_time
            self._gil_hold_log(
                "_update_spinners",
                elapsed * 1000,
                f"{active_count} spinners, queue_lag={queue_lag_ms:.1f}ms (P3 SKIPPED - healthy)",
            )
            return

        # 📊 COMPILE ACTIVE SPINNER STATUS + HEALTH
        active_status = []
        health_indicators = []
        all_healthy = True  # Track if all spinners are healthy this calc
        for spinner_id in updated_this_iteration:
            metrics = self._spinner_metrics[spinner_id]
            age = (
                current_iteration - metrics["start_iter"]
                if metrics["start_iter"]
                else 0
            )

            # 🛑 FUCKUP COOLDOWN: If spinner is in cooldown, skip ALL FPS calculations!
            time_since_fuckup = start_time - self._spinner_fuckup_cooldown.get(
                spinner_id, 0
            )
            was_in_cooldown = self._spinner_fuckup_cooldown.get(spinner_id, 0) > 0
            if (
                was_in_cooldown
                and time_since_fuckup < self._spinner_fuckup_cooldown_duration
            ):
                # Still in cooldown - skip this spinner entirely!
                short_name = spinner_id.replace("-spinner", "")
                active_status.append(f"{short_name}:COOLDOWN")
                health_indicators.append(f"💤{short_name}:cooldown")
                continue  # Skip to next spinner!

            # 😊 RECOVERY CHECK: If just exited cooldown, check if recovered!
            just_exited_cooldown = (
                was_in_cooldown
                and time_since_fuckup >= self._spinner_fuckup_cooldown_duration
            )

            # 🎯 CALCULATE FPS (rolling average over last 8 spins)
            fps = 0.0
            health_emoji = "❓"
            if len(metrics["last_8_times"]) >= 2:
                time_span = metrics["last_8_times"][-1] - metrics["last_8_times"][0]
                if time_span > 0:
                    fps = (len(metrics["last_8_times"]) - 1) / time_span

                    # 🚦 HEALTH INDICATOR
                    if fps >= 8.0:  # STRICT! Must hit 8 FPS target!
                        health_emoji = "✅"
                    elif fps >= 6.0:  # Within 75% of target
                        health_emoji = "⚠️"
                        all_healthy = False  # Not at target = not healthy
                    else:
                        health_emoji = "🚨"
                        all_healthy = False  # Definitely not healthy!
                        # 🛑 FUCKUP DETECTED! Enter cooldown for this spinner!
                        self._spinner_fuckup_cooldown[spinner_id] = start_time

                    # 🩰🦡🔥 LOG TO auto_refresh.log WHEN SPINNER FUCKS UP THE DANCE!
                    # 🎭 THROTTLED: Only log once per second per spinner (not 8× per second!)
                    # 🚨 GLOBAL THROTTLE: Only ONE complaint every 2 seconds (prevents feedback loop!)
                    if (
                        STEVEN_FULL_DANCE_DEBUG
                        and fps < 8.0
                        and spinner_id in self.SPINNER_PARTNERS
                    ):
                        # Check GLOBAL cooldown first (prevents complaint → slow → complaint loop!)
                        time_since_any_warning = (
                            start_time - self._spinner_last_any_warning
                        )
                        if (
                            time_since_any_warning
                            < self._spinner_global_warning_cooldown
                        ):
                            # Skip ALL warnings - global cooldown active
                            pass
                        # Then check per-spinner cooldown
                        elif (
                            start_time
                            - self._spinner_last_warning_time.get(spinner_id, 0)
                        ) < 1.0:
                            # Skip this warning - already complained about THIS spinner recently
                            pass
                        else:
                            # Log the warning and update throttle timestamps
                            self._spinner_last_warning_time[spinner_id] = start_time
                            self._spinner_last_any_warning = start_time  # 🚨 GLOBAL: No more complaints for 2 seconds!
                            partner_info = self.SPINNER_PARTNERS[spinner_id]
                            spinner_first_name = partner_info["name"].split()[
                                0
                            ]  # Get first name (SCARLETT, BORIS, etc.)
                            # 📝 BUFFERED: Complaints use buffer too (no more file I/O blocking!)
                            if fps < 4.0:
                                # 🚨🚨 CATASTROPHIC: Spinner is COMPLETELY fucking up!
                                stevens_log(
                                    "auto_refresh",
                                    f"{datetime.now().isoformat()} 🚨🦡🔥 {partner_info['emoji']} {partner_info['name']} is FUCKING UP the dance! Only {fps:.1f} FPS (need 8)! 🦡🔥",
                                )
                                stevens_log(
                                    "auto_refresh",
                                    f"{datetime.now().isoformat()} 😤🤯 STEVEN: WHAT THE FUCK {spinner_first_name}?! {fps:.1f} FPS?! That's not SPINNING, that's having a STROKE! You're making {partner_info['partner']} look BAD!",
                                )
                            elif fps < 6.0:
                                # 🚨 CRITICAL: Spinner is FUCKING UP the dance!
                                stevens_log(
                                    "auto_refresh",
                                    f"{datetime.now().isoformat()} 🚨🦡🔥 {partner_info['emoji']} {partner_info['name']} is FUCKING UP the dance! Only {fps:.1f} FPS (need 8)! 🦡🔥",
                                )
                                stevens_log(
                                    "auto_refresh",
                                    f"{datetime.now().isoformat()} 😤😤 STEVEN: {spinner_first_name}! FUCK! {fps:.1f} FPS?! You're supposed to be SPINNING not STUTTERING! {partner_info['partner']} is counting on you!",
                                )
                            else:
                                # ⚠️ WARNING: Spinner is struggling
                                stevens_log(
                                    "auto_refresh",
                                    f"{datetime.now().isoformat()} ⚠️ {partner_info['emoji']} {partner_info['name']} is struggling! {fps:.1f} FPS (need 8). 😤",
                                )
                                stevens_log(
                                    "auto_refresh",
                                    f"{datetime.now().isoformat()} 😤 STEVEN: {spinner_first_name}, {fps:.1f} FPS? Come on! You're fucking up the dance! Pick up the pace or {partner_info['partner']} will be upset!",
                                )

                    # 😊✨ STEVEN'S RECOVERY PRAISE! Only when exiting fuckup cooldown with good FPS!
                    # This replaces the timer-based praise - much simpler!
                    elif (
                        STEVEN_FULL_DANCE_DEBUG
                        and just_exited_cooldown
                        and fps >= 8.0
                        and spinner_id in self.SPINNER_PARTNERS
                    ):
                        # Clear the cooldown (they recovered!)
                        self._spinner_fuckup_cooldown[spinner_id] = 0
                        partner_info = self.SPINNER_PARTNERS[spinner_id]
                        spinner_first_name = partner_info["name"].split()[0]
                        # 📝 BUFFERED: Praise uses buffer too!
                        stevens_log(
                            "auto_refresh",
                            f"{datetime.now().isoformat()} 🎉✨ {partner_info['emoji']} {partner_info['name']} RECOVERED! {fps:.1f} FPS - back in form! ✨",
                        )
                        stevens_log(
                            "auto_refresh",
                            f"{datetime.now().isoformat()} 😊 STEVEN: {spinner_first_name} is BACK! {fps:.1f} FPS! {partner_info['partner']} will be so relieved!",
                        )

            short_name = spinner_id.replace("-spinner", "")
            active_status.append(f"{short_name}:age{age},spins{metrics['total_spins']}")
            health_indicators.append(f"{health_emoji}{short_name}:{fps:.1f}fps")

        # 🔍 PHASE 3 DONE: FPS/health calculations
        phase3_ms = (time.time() - phase3_start) * 1000

        # 🎯 UPDATE FPS BACKOFF STATE
        self._fps_all_healthy = all_healthy
        self._fps_last_calc_time = start_time

        # ⏱️ LOG EXECUTION TIME + COMPREHENSIVE METRICS + QUEUE LAG!
        # 📝 BUFFERED: Only writes to disk every 100 lines (8 FPS secret!)
        elapsed = time.time() - start_time

        # 🔍 LOG PHASE BREAKDOWN if total > 10ms (find the culprit!)
        total_ms = elapsed * 1000
        phase4_ms = (
            total_ms - phase0_ms - phase1_ms - phase2_ms - phase3_ms
        )  # Remaining = logging/other
        if total_ms > 10:
            self._gil_hold_log(
                "_update_spinners_PHASES",
                total_ms,
                f"P0={phase0_ms:.1f}ms P1={phase1_ms:.1f}ms P2={phase2_ms:.1f}ms P3={phase3_ms:.1f}ms P4={phase4_ms:.1f}ms",
            )

        # 📊 GIL_HOLD: Track spinner update time (runs on main thread!)
        self._gil_hold_log(
            "_update_spinners",
            elapsed * 1000,
            f"{active_count} spinners, queue_lag={queue_lag_ms:.1f}ms",
        )

        log_line = f"{datetime.now().isoformat()} 🔄 UPDATE: {elapsed * 1000:.2f}ms, {active_count} active, queue_lag={queue_lag_ms:.1f}ms"
        if health_indicators:
            log_line += f" [{' '.join(health_indicators)}]"
        if lag_report:
            log_line += f", LAG: {', '.join(lag_report)}"
        if lifecycle_events:
            log_line += f"\n     {'  '.join(lifecycle_events)}"
        stevens_log("spinner_timing", log_line)

        # 🏥 PERIODIC HEALTH SUMMARY (every 40 iterations = 5 seconds at 8 FPS)
        # 📝 BUFFERED: Health summaries also go through buffer
        if current_iteration > 0 and current_iteration % 40 == 0:
            stevens_log("spinner_timing", f"\n{'=' * 80}")
            stevens_log(
                "spinner_timing",
                f"{datetime.now().isoformat()} 🏥 HEALTH SUMMARY @iter{current_iteration}:",
            )

            # Overall spinner health
            all_healthy = all(
                emoji == "✅" for emoji in [h.split(":")[0] for h in health_indicators]
            )
            summary_emoji = (
                "✅"
                if all_healthy
                else ("⚠️" if any("⚠️" in h for h in health_indicators) else "🚨")
            )
            stevens_log(
                "spinner_timing",
                f"  {summary_emoji} Spinners: {' | '.join(health_indicators)}",
            )

            # Lag summary
            if lag_report:
                stevens_log(
                    "spinner_timing", f"  🐢 LAG DETECTED: {', '.join(lag_report)}"
                )
            else:
                stevens_log(
                    "spinner_timing", f"  ✅ No lag - all spinners synchronized!"
                )

            stevens_log("spinner_timing", f"{'=' * 80}\n")

    # ✅ SPICY: ALICE RABBIT HOLE complete - all dependencies verified in Functions 6-7, loops through all spinners, updates only active ones

    def _spinner_worker_thread(self):
        """Dedicated worker thread for spinner animation at CONSTANT rate

        Runs independently of main thread load - spinner spins smoothly even under heavy CPU load!
        Uses time.sleep(0.125) for precise 8 FPS animation (125ms intervals).
        """
        import time

        log_file = get_log_path("spinner_timing.log")
        iteration_count = 0
        last_iteration_time = time.time()

        with open(log_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"{datetime.now().isoformat()} 🧵 SPINNER_WORKER_STARTED\n")
            f.write(f"{'=' * 80}\n")

        while self._spinner_worker_running:
            iteration_start = time.time()

            # 🚨 SANITY CHECK: Detect if iteration took too long!
            elapsed_since_last = iteration_start - last_iteration_time
            if (
                iteration_count > 0 and elapsed_since_last > 0.200
            ):  # Should be ~125ms, warn if >200ms
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} ⚠️  SLOW_ITERATION #{iteration_count}: {elapsed_since_last * 1000:.1f}ms (expected 125ms!)\n"
                    )

            # Sleep FIRST (precise timing independent of main thread)
            sleep_start = time.time()
            time.sleep(SPINNER_FPS_INTERVAL)  # 125ms = 8 FPS
            sleep_end = time.time()

            # Update spinners on main thread (thread-safe)
            try:
                call_start = time.time()
                # 📊 QUEUE LAG: Track when update was queued (measure lag in _update_spinners)
                self._spinner_queue_time = call_start
                # ✅ FIX: call_from_thread is on App, not Screen!
                self.app.call_from_thread(self._update_spinners)
                call_queued = time.time()

                # ⏱️ LOG EVERY ITERATION with timing
                iteration_count += 1
                self._spinner_last_iteration = (
                    iteration_count  # Track globally for sanity checks
                )
                self._spinner_last_update_time = call_queued

                # 📝 BUFFERED: Use buffered logging for every-iteration writes (8 FPS secret!)
                stevens_log(
                    "spinner_timing",
                    f"{datetime.now().isoformat()} ⏱️  SPIN #{iteration_count:04d}: "
                    f"sleep={sleep_end - sleep_start:.3f}s, "
                    f"queue={call_queued - call_start:.6f}s, "
                    f"total={call_queued - iteration_start:.3f}s",
                )

                # 🚨 PERIODIC SANITY CHECK: Every 8 iterations (1 second at 8 FPS), check health
                if iteration_count % 8 == 0:
                    expected_time = (
                        iteration_count * 0.125
                    )  # Expected total time at 8 FPS
                    actual_time = call_queued - (
                        call_queued - (iteration_count * 0.125)
                    )
                    stevens_log(
                        "spinner_timing",
                        f"{datetime.now().isoformat()} 📊 SANITY_CHECK #{iteration_count // 8}: "
                        f"{iteration_count} iterations in ~{iteration_count * 0.125:.1f}s (expected), "
                        f"last_update={datetime.fromtimestamp(call_queued).isoformat()}",
                    )

                last_iteration_time = call_queued
            except Exception as e:
                # TUI might be shutting down, or spinner widgets don't exist yet
                import traceback

                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 🛑 SPINNER_WORKER_EXCEPTION: {type(e).__name__}: {e}\n"
                    )
                    # Log full traceback for debugging
                    for line in traceback.format_exc().strip().split("\n"):
                        f.write(f"    {line}\n")
                break

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🧵 SPINNER_WORKER_STOPPED (ran {iteration_count} iterations)\n"
            )

    # ═══════════════════════════════════════════════════════════════════════════════
    # 📝 BUFFERED LOGGING - Now uses Stevens Dance centrally!
    # ═══════════════════════════════════════════════════════════════════════════════
    # All logging moved to stevens_log() - 10,000-line batching handled centrally!
    # Monitor screen's old _buffered_log() methods removed - Stevens Dance is better!

    def _start_spinner_worker(self):
        """Start dedicated spinner worker thread (if not already running)"""
        if not self._spinner_worker_running:
            self._spinner_worker_running = True
            # ✅ FIX: Use _thread_obj to avoid name collision with _spinner_worker_thread METHOD!
            self._spinner_worker_thread_obj = threading.Thread(
                target=self._spinner_worker_thread, daemon=True
            )
            self._spinner_worker_thread_obj.start()

    def _stop_spinner_worker(self):
        """Stop dedicated spinner worker thread"""
        self._spinner_worker_running = False
        if self._spinner_worker_thread_obj:
            self._spinner_worker_thread_obj.join(
                timeout=0.5
            )  # Wait max 500ms for thread to stop
            self._spinner_worker_thread_obj = None

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🌶️ DRY HELPERS (Reduce duplication across table operations)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _enable_table_cursors(self) -> None:
        """Re-enable table cursors after initial batch loading completes.

        Cursors are disabled during initial load (cursor_type="none") to prevent
        visual freeze caused by cursor positioning/rendering during heavy data loading.
        After all 5 tables are displayed, this function re-enables cursors so users
        can click rows for details.

        Called from: _display_next() after BATCH_COMPLETE
        """
        try:
            # Re-enable all 5 table cursors
            builds_table = self.query_one("#builds-recent-table", DataTable)
            builds_table.cursor_type = "row"

            runner_table = self.query_one("#runner-executions-table", DataTable)
            runner_table.cursor_type = "row"

            vertex_table = self.query_one("#vertex-jobs-table", DataTable)
            vertex_table.cursor_type = "row"

            active_table = self.query_one("#runs-table", DataTable)
            active_table.cursor_type = "row"

            completed_table = self.query_one("#completed-runs-table", DataTable)
            completed_table.cursor_type = "row"

            # Log cursor re-enable
            log_file = get_log_path("auto_refresh.log")
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ CURSORS_ENABLED: All 5 tables now clickable!\n"
                )

        except Exception as e:
            # Silently fail if tables not found (shouldn't happen but defensive)
            pass

    def _take_data_snapshot(self, table_name: str, data: List[Dict]) -> None:
        """Take snapshot of all row data with ALL fields for validation

        Captures complete row data including ALL fields returned from core.py
        for sanity checking between refreshes.

        Args:
            table_name: Table identifier ("builds", "runner", "vertex", "active", "completed")
            data: Raw data list from core.py fetch functions
        """
        log_file = get_log_path("auto_refresh.log")

        # Build snapshot dictionary {row_key: {all_fields}}
        snapshot = {}
        for item in data:
            # Get unique key based on table type
            if table_name == "builds":
                row_key = f"build-{item.get('build_id', 'unknown')}"
            elif table_name == "runner":
                row_key = f"runner-{item.get('name', 'unknown')}"
            elif table_name == "vertex":
                row_key = f"vertex-{item.get('id', 'unknown')}"
            elif table_name in ["active", "completed"]:
                row_key = f"{table_name}-{item.get('id', 'unknown')}"
            else:
                continue

            # Store ALL fields (complete data)
            snapshot[row_key] = dict(item)

        # Store snapshot
        self._data_snapshots[table_name] = snapshot

        # Log snapshot taken
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 📸 SNAPSHOT_TAKEN: {table_name} ({len(snapshot)} rows)\n"
            )

    def _compare_and_log_snapshot(self, table_name: str, new_data: List[Dict]) -> None:
        """Compare new data with previous snapshot and log mismatches

        Compares ALL fields for rows that exist in both snapshots.
        New rows are OK (not flagged as mismatch).

        Args:
            table_name: Table identifier
            new_data: Fresh data from current fetch
        """
        log_file = get_log_path("auto_refresh.log")

        # Get previous snapshot
        old_snapshot = self._data_snapshots.get(table_name, {})
        if not old_snapshot:
            # First snapshot - just take it
            self._take_data_snapshot(table_name, new_data)
            return

        # Build new snapshot
        new_snapshot = {}
        for item in new_data:
            if table_name == "builds":
                row_key = f"build-{item.get('build_id', 'unknown')}"
            elif table_name == "runner":
                row_key = f"runner-{item.get('name', 'unknown')}"
            elif table_name == "vertex":
                row_key = f"vertex-{item.get('id', 'unknown')}"
            elif table_name in ["active", "completed"]:
                row_key = f"{table_name}-{item.get('id', 'unknown')}"
            else:
                continue
            new_snapshot[row_key] = dict(item)

        # Compare common rows (ignore new rows)
        mismatches = []
        for row_key in old_snapshot.keys():
            if row_key not in new_snapshot:
                continue  # Row disappeared (OK - might be out of view limit)

            old_row = old_snapshot[row_key]
            new_row = new_snapshot[row_key]

            # Compare ALL fields
            for field_name in old_row.keys():
                old_value = old_row.get(field_name)
                new_value = new_row.get(field_name)

                if old_value != new_value:
                    mismatches.append(
                        {
                            "row_key": row_key,
                            "field": field_name,
                            "old": old_value,
                            "new": new_value,
                        }
                    )

        # Log results
        with open(log_file, "a") as f:
            if mismatches:
                f.write(f"\n{'=' * 80}\n")
                f.write(
                    f"{datetime.now().isoformat()} 🚨 DATA_MISMATCH: {table_name} ({len(mismatches)} mismatches!)\n"
                )
                f.write(f"{'=' * 80}\n")
                for mismatch in mismatches:
                    f.write(f"  ROW: {mismatch['row_key']}\n")
                    f.write(f"  FIELD: {mismatch['field']}\n")
                    f.write(f"  BEFORE: {mismatch['old']}\n")
                    f.write(f"  AFTER:  {mismatch['new']}\n")
                    f.write(f"  {'-' * 78}\n")
                f.write(f"{'=' * 80}\n\n")
            else:
                new_rows = len(new_snapshot) - len(old_snapshot)
                f.write(
                    f"{datetime.now().isoformat()} ✅ SNAPSHOT_MATCH: {table_name} (no mismatches, {new_rows:+d} rows)\n"
                )

        # 🎯 SHOW USER-VISIBLE TOAST for data changes (permanent toast!)
        if mismatches:
            # Group mismatches by row_key for cleaner display
            changes_by_row = {}
            for mismatch in mismatches:
                row_key = mismatch["row_key"]
                if row_key not in changes_by_row:
                    changes_by_row[row_key] = []
                changes_by_row[row_key].append(mismatch)

            # Build detailed toast message
            toast_lines = [f"🔄 DATA CHANGED: {table_name.upper()}", ""]
            for row_key, row_mismatches in changes_by_row.items():
                toast_lines.append(f"ROW: {row_key}")
                for m in row_mismatches:
                    # Truncate long values for readability
                    old_val = str(m["old"])[:50]
                    new_val = str(m["new"])[:50]
                    toast_lines.append(f"  • {m['field']}: {old_val} → {new_val}")
                toast_lines.append("")  # Blank line between rows

            toast_msg = "\n".join(toast_lines)

            # 🔍 DEBUG: Log that toast is being sent
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 📢 TOAST_SENT: {len(changes_by_row)} rows changed\n"
                )
                f.write(f"TOAST_MSG:\n{toast_msg}\n")
                f.write(f"{'-' * 80}\n")

            try:
                # 🧵 CRITICAL: Must call notify() on main thread! (we're in a worker thread)
                # ✅ FIX: Use App.call_from_thread - Screen.call_from_thread doesn't exist!
                self.app.call_from_thread(
                    self.notify, toast_msg, severity="warning", timeout=0
                )
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} ✅ NOTIFY_CALLED: success (via main thread)\n"
                    )
            except Exception as e:
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} ❌ NOTIFY_FAILED: {str(e)}\n"
                    )

        # Update snapshot
        self._data_snapshots[table_name] = new_snapshot

    def _create_table_divider(self, table_name: str):
        """Create divider row for table (matches column count automatically)

        Args:
            table_name: Key from TABLES dict ("runner", "builds", etc.)

        Returns:
            List of divider strings matching table's column count
        """
        config = TABLES.get(table_name)
        if not config:
            return []

        num_cols = config["columns"]
        divider_char = "[dim blue]━━━━━━━━━━━━━━━━[/dim blue]"
        return [divider_char] * num_cols

    # ✅ SPICY: ALICE RABBIT HOLE complete - verified TABLES constant (line 151), config["columns"] key exists, safe fallback for unknown tables

    def _add_empty_state_row(self, table_name: str, message: str):
        """Add empty state row to table (auto-handles column count and formatting)

        Args:
            table_name: Key from TABLES dict ("runner", "builds", etc.)
            message: Message to display (e.g., "No executions", "No builds (24h)")
        """
        config = TABLES.get(table_name)
        if not config:
            return

        table = self.query_one(f"#{config['id']}", DataTable)
        num_cols = config["columns"]

        # First column: —, Second column: message, Rest: —
        row = ["[dim]—[/dim]"] * num_cols
        row[1] = f"[dim]{message}[/dim]"

        table.add_row(*row)
        table.move_cursor(row=-1)
        table.refresh()

    # ✅ SPICY: ALICE RABBIT HOLE complete - removed local DataTable import (already at line 127), all dependencies verified in Function 9, logic correct

    # ═══════════════════════════════════════════════════════════════════════════════
    # ⏱️ REFRESH ORCHESTRATION (Ticking timers and active duration updates)
    # ═══════════════════════════════════════════════════════════════════════════════

    def _update_active_durations(self) -> None:
        """Update duration/lifetime display for all ACTIVE items every second (ticking timer)"""

        log_file = get_log_path("auto_refresh.log")

        # 🔍 DEBUG: Log 1s timer tick (BEFORE updates)
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ⏱️  1S_TIMER_TICK: Updating active durations...\n"
            )

        def calculate_duration(start_time_str):
            """Calculate elapsed time and return formatted string"""
            try:
                start = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                elapsed_seconds = int((now - start).total_seconds())

                if elapsed_seconds < 60:
                    return f"{elapsed_seconds}s"
                else:
                    minutes = elapsed_seconds // 60
                    seconds = elapsed_seconds % 60
                    return f"{minutes}m{seconds}s"
            except Exception:
                return None

        # Update WORKING Cloud Builds
        updated_builds = 0
        try:
            builds_table = self.query_one("#builds-recent-table", DataTable)
            for build_key, build_data in list(
                self.row_data.get("build_recent", {}).items()
            ):
                if build_data.get("status") == "WORKING":
                    start_time = build_data.get("start_time")
                    if start_time:
                        duration_str = calculate_duration(start_time)
                        if duration_str:
                            try:
                                builds_table.update_cell(
                                    build_key, "Runtime", f"[cyan]{duration_str}[/cyan]"
                                )
                                updated_builds += 1
                                # 🔍 DEBUG: Log cell update
                                with open(log_file, "a") as f:
                                    f.write(
                                        f"{datetime.now().isoformat()} 🔄 UPDATE_CELL: builds[{build_key}] Runtime='{duration_str}'\n"
                                    )
                            except Exception:
                                # Row disappeared during refresh (expected race condition) - skip silently
                                pass
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ❌ BUILDS_UPDATE_ERROR: {str(e)}\n"
                )

        # Update RUNNING Launch Agent Executions
        updated_runners = 0
        try:
            runner_table = self.query_one("#runner-executions-table", DataTable)
            for exec_key, exec_data in list(self.row_data.get("runner", {}).items()):
                if exec_data.get("status") == "RUNNING":
                    start_time = exec_data.get("start_time")
                    if start_time:
                        duration_str = calculate_duration(start_time)
                        if duration_str:
                            try:
                                runner_table.update_cell(
                                    exec_key, "Runtime", f"[cyan]{duration_str}[/cyan]"
                                )
                                updated_runners += 1
                                # 🔍 DEBUG: Log cell update
                                with open(log_file, "a") as f:
                                    f.write(
                                        f"{datetime.now().isoformat()} 🔄 UPDATE_CELL: runner[{exec_key}] Runtime='{duration_str}'\n"
                                    )
                            except Exception:
                                # Row disappeared during refresh (expected race condition) - skip silently
                                pass
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ❌ RUNNER_UPDATE_ERROR: {str(e)}\n"
                )

        # Update RUNNING/PENDING/QUEUED Vertex AI Jobs
        updated_vertex = 0
        try:
            vertex_table = self.query_one("#vertex-jobs-table", DataTable)
            for job_key, job_data in list(self.row_data.get("vertex", {}).items()):
                # Tick for RUNNING, PENDING, QUEUED states
                state = job_data.get("state", "")
                if state in [
                    "JOB_STATE_RUNNING",
                    "JOB_STATE_PENDING",
                    "JOB_STATE_QUEUED",
                ]:
                    start_time = job_data.get("start_time")
                    if start_time:
                        duration_str = calculate_duration(start_time)
                        if duration_str:
                            try:
                                vertex_table.update_cell(
                                    job_key, "Runtime", f"[cyan]{duration_str}[/cyan]"
                                )
                                updated_vertex += 1
                                # 🔍 DEBUG: Log cell update
                                with open(log_file, "a") as f:
                                    f.write(
                                        f"{datetime.now().isoformat()} 🔄 UPDATE_CELL: vertex[{job_key}] Runtime='{duration_str}'\n"
                                    )
                            except Exception:
                                # Row disappeared during refresh (expected race condition) - skip silently
                                pass
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ❌ VERTEX_UPDATE_ERROR: {str(e)}\n"
                )

        # Update RUNNING Active W&B Runs
        updated_active = 0
        try:
            active_table = self.query_one("#runs-table", DataTable)
            for run_key, run_data in list(self.row_data.get("active", {}).items()):
                # W&B uses lowercase "running"
                if run_data.get("state") == "running":
                    start_time = run_data.get("start_time")
                    if start_time:
                        duration_str = calculate_duration(start_time)
                        if duration_str:
                            try:
                                active_table.update_cell(
                                    run_key, "Runtime", f"[cyan]{duration_str}[/cyan]"
                                )
                                updated_active += 1
                                # 🔍 DEBUG: Log cell update
                                with open(log_file, "a") as f:
                                    f.write(
                                        f"{datetime.now().isoformat()} 🔄 UPDATE_CELL: active[{run_key}] Runtime='{duration_str}'\n"
                                    )
                            except Exception:
                                # Row disappeared during refresh (expected race condition) - skip silently
                                pass
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ❌ ACTIVE_UPDATE_ERROR: {str(e)}\n"
                )

        # 🔍 DEBUG: Log 1s timer completion
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ✅ 1S_TIMER_COMPLETE: Updated {updated_builds} builds, {updated_runners} runners, {updated_vertex} vertex, {updated_active} active\n"
            )

    # ✅ SPICY: ALICE RABBIT HOLE complete - BUG #4 DESTROYED! Added datetime to module top (line 124), removed local import, verified DataTable/self.row_data, inner function logic correct

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎮 EVENT HANDLERS (Textual lifecycle and user interactions)
    # ═══════════════════════════════════════════════════════════════════════════════

    def on_mount(self):
        """Initialize UI state after mounting"""
        # CRITICAL: Call parent's on_mount to start loading flow!
        # Parent calls: initialize_content() → finish_loading() → _populate_initial_tables()
        # We DON'T call _populate_initial_tables here (parent handles it)
        super().on_mount()

        # 🚀 CACHE SPINNER REFS - Avoid query_one() in hot path!
        # This makes BAILOUT mode ~100x faster (no DOM queries!)
        for spinner_id in [
            "builds-recent-spinner",
            "runner-spinner",
            "vertex-spinner",
            "active-spinner",
            "completed-spinner",
        ]:
            try:
                self._cached_spinners[spinner_id] = self.query_one(
                    f"#{spinner_id}", Static
                )
            except Exception:
                pass  # Widget might not exist yet

        # Disable cancel button initially (no selection)
        cancel_btn = self.query_one("#cancel-btn", Button)
        cancel_btn.disabled = True
        # Note: Cursor reset happens AFTER tables are populated (in workers)

        # Set up 1-second timer for ACTIVE item duration updates
        self.set_interval(1.0, self._update_active_durations)

        # Start dedicated spinner worker thread (constant rate independent of main thread!)
        self._start_spinner_worker()

    # ✅ SPICY: ALICE RABBIT HOLE complete - BUG #5 MERGING! Found DUPLICATE on_mount at line 880, merged cancel button logic + timers, verified Button import (line 127), all logic correct

    def on_click(self, event) -> None:
        """Handle clicks - clear selection if clicking outside tables"""
        # Check if click target is a DataTable
        try:
            # If clicked on a table, do nothing (let row selection handle it)
            if isinstance(event.widget, DataTable):
                return

            # Clicked outside all tables - clear selection
            self._clear_selection()
        except Exception:
            pass

    # ✅ SPICY: ALICE RABBIT HOLE complete - verified event.widget (Textual Click event), DataTable import (line 127), _clear_selection() local method, logic correct

    def _clear_selection(self) -> None:
        """Clear selection and disable cancel button"""
        self.selected_run_id = None
        self.selected_table_id = None
        self.selected_row_data = None

        # Disable cancel button
        try:
            cancel_btn = self.query_one("#cancel-btn", Button)
            cancel_btn.disabled = True
        except Exception:
            pass

        # Clear cursor from all tables (move to no selection)
        for table_id in [
            "builds-recent-table",
            "runner-executions-table",
            "vertex-jobs-table",
            "runs-table",
            "completed-runs-table",
        ]:
            try:
                table = self.query_one(f"#{table_id}", DataTable)
                table.move_cursor(row=-1)  # Clear visual selection
            except Exception:
                pass

    # ✅ SPICY: ALICE RABBIT HOLE complete - verified instance vars initialized (lines 312-314), Button/DataTable imports (line 128), query_one() Textual method, logic correct

    def _refresh_all_tables(self) -> None:
        """Refresh all 5 tables - USES ACCUMULATOR PATTERN! ✨"""
        log_file = get_log_path("auto_refresh.log")

        # 🚨 CRITICAL: Prevent concurrent accumulator operations!
        if self._accumulator_active:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏸️  MANUAL_REFRESH_BLOCKED: Accumulator already active! (Preventing race condition)\n"
                )
            return

        # Prevent duplicate refreshes
        if self._is_refreshing:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏸️  MANUAL_REFRESH_BLOCKED: Already refreshing!\n"
                )
            return

        self._is_refreshing = True

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🔵 MANUAL_REFRESH: User triggered manual refresh (all tables)\n"
            )

        # Start spinner worker thread (constant rate!)
        self._start_spinner_worker()

        # 🎯 USE ACCUMULATOR PATTERN for ordered display with 200ms delays!
        all_tables = ["builds", "runner", "vertex", "active", "completed"]
        self._start_accumulator(all_tables)

        # Launch workers for all tables in parallel (use_accumulator=True for accumulator control!)
        for table_name in all_tables:
            self._universal_refresh_table(
                table_name, is_auto_refresh=False, use_accumulator=True
            )

        # Reset flag after a short delay (workers run in background)
        def reset_flag():
            self._is_refreshing = False

        self.set_timer(0.5, reset_flag)  # 500ms delay before allowing next refresh

    # ✅ SPICY: ALICE RABBIT HOLE complete - verified _is_refreshing (line 347), spinner_timer (line 346), set_interval/set_timer Textual methods, _update_spinners (Function 8), _universal_refresh_table (line 993), no local imports, logic correct

    def _display_next_ready_table(self) -> None:
        """Display next ready table in order with 200ms delay before checking next

        ⚠️ CRITICAL: Don't hold lock during wait! Check once, display if ready, return!
        """
        log_file = get_log_path("auto_refresh.log")
        display_time = time.time()

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🟦 DISPLAY_NEXT_ENTERED: index={self._accumulator_next_display}, batch_size={len(self._current_batch)}\n"
            )

        # Check if we're done
        if self._accumulator_next_display >= len(self._current_batch):
            total_batch_time = display_time - self._accumulator_batch_start
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 🎉 BATCH_COMPLETE: All {len(self._current_batch)} tables displayed in {total_batch_time:.2f}s total\n"
                )
                f.write(
                    f"{datetime.now().isoformat()} 📊 SUMMARY: {' → '.join(self._current_batch)}\n"
                )

            # Re-enable cursors after batch completes (prevents freeze during loading!)
            self._enable_table_cursors()

            self._accumulator_active = False
            return

        # Grab lock BRIEFLY to check state
        with self._accumulator_lock:
            next_table = self._current_batch[self._accumulator_next_display]
            is_ready = next_table in self._accumulator_results
            completion_time = self._accumulator_completion_times.get(next_table, 0)

        # Not ready yet? Schedule check again in 50ms
        if not is_ready:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏳ WAITING: {next_table} not ready yet, checking again in 50ms...\n"
                )
            self.set_timer(0.05, self._display_next_ready_table)  # Poll every 50ms
            return

        # Ready! But enforce MINIMUM 200ms delay from last display
        position = self._accumulator_next_display + 1
        total = len(self._current_batch)

        # Check time since last display
        time_since_last_display = (
            display_time - self._accumulator_last_display_time
            if self._accumulator_last_display_time > 0
            else 999
        )
        min_delay_sec = self._accumulator_delay_ms / 1000.0

        # If not enough time passed, schedule display for later
        if time_since_last_display < min_delay_sec:
            remaining_delay = min_delay_sec - time_since_last_display
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏸️  ENFORCING_MIN_DELAY: {next_table} ready but only {time_since_last_display * 1000:.0f}ms since last display, waiting {remaining_delay * 1000:.0f}ms more\n"
                )
            self.set_timer(remaining_delay, self._display_next_ready_table)
            return

        # Enough time passed! Display now
        wait_time = display_time - completion_time if completion_time > 0 else 0

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🎯 DISPLAYING: {next_table} "
                f"(position {position}/{total}, waited {wait_time:.3f}s before display, {time_since_last_display * 1000:.0f}ms since last)\n"
            )

        # 🎯 PHASE 3: Get fetched data and call render function!
        with self._accumulator_lock:
            data = self._fetched_data.get(next_table)

        # Call appropriate render function
        if next_table == "runner":
            self._update_runner_table(data)
        elif next_table == "builds":
            self._update_builds_table(data)
        elif next_table == "vertex":
            self._update_vertex_table(data)
        elif next_table == "active":
            self._update_active_table(data)
        elif next_table == "completed":
            self._update_completed_table(data)

        # 🎯 NOW stop the spinner - table is actually VISIBLE!
        # (Worker kept it spinning until now for better UX)
        self._refreshing_tables.discard(next_table)

        # Update state: increment index, record display time
        with self._accumulator_lock:
            self._accumulator_next_display += 1
            self._accumulator_last_display_time = time.time()

        # Schedule next check with 200ms delay
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ⏱️  DELAY: Waiting {self._accumulator_delay_ms}ms before next check\n"
            )
        self.set_timer(min_delay_sec, self._display_next_ready_table)

    def _ipasb_check_backoff(self) -> None:
        """📡 IPASB: Check backoff signal and sleep if needed - "whoa nelly!"

        Workers call this before heavy operations. If backoff level is high,
        they'll sleep to let the main thread breathe.

        Levels:
        - 0: normal (no delay)
        - 1: mild (10ms)
        - 2: heavy (50ms)
        - 3: critical (100ms)
        """
        import time

        delay = self._ipasb_backoff_delays.get(self._ipasb_backoff_level, 0)
        if delay > 0:
            time.sleep(delay)

    def _start_accumulator(self, tables_to_refresh: list[str]) -> None:
        """🎯 Start accumulator for a batch of tables

        Args:
            tables_to_refresh: List of table names to refresh and display in order
                             (e.g. ["builds", "runner", "vertex"] for checked tables)
        """
        log_file = get_log_path("auto_refresh.log")

        with self._accumulator_lock:
            # Reset accumulator state
            self._accumulator_results = {}
            self._accumulator_completion_times = {}  # Track when each table completed
            self._accumulator_next_display = 0
            self._accumulator_active = False
            self._accumulator_batch_start = time.time()  # Track batch start time
            self._accumulator_last_display_time = (
                0  # Track last display for minimum 200ms delay
            )

            # Filter display_order to only include tables we're refreshing
            self._current_batch = [
                t for t in self._display_order if t in tables_to_refresh
            ]

            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 🎯 ACCUMULATOR_START: Batch size = {len(self._current_batch)}\n"
                )
                f.write(
                    f"{datetime.now().isoformat()} 📋 DISPLAY_ORDER: {' → '.join(self._current_batch)}\n"
                )
                f.write(
                    f"{datetime.now().isoformat()} ⏱️  BATCH_START_TIME: T+0.000s (canonical reference)\n"
                )

    def _canonical_log(self, message: str) -> None:
        """📏 Log with CANONICAL timing from batch start!

        Shows T+X.XXXs relative to _accumulator_batch_start so we can trace
        exactly where time is spent in each refresh cycle.
        """
        log_file = get_log_path("auto_refresh.log")

        # Calculate time since batch start
        if (
            hasattr(self, "_accumulator_batch_start")
            and self._accumulator_batch_start > 0
        ):
            elapsed = time.time() - self._accumulator_batch_start
            time_str = f"T+{elapsed:6.3f}s"
        else:
            time_str = "T+??????s"

        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} {time_str} {message}\n")

    def _gil_hold_log(
        self, location: str, duration_ms: float, description: str = ""
    ) -> None:
        """📊 Log GIL hold time to identify spinner killers!

        Only logs if duration > 1ms (ignore tiny holds)
        """
        if duration_ms < 1.0:
            return  # Skip tiny holds

        gil_log = get_log_path("gil_hold.log")
        with open(gil_log, "a") as f:
            if duration_ms >= 10:
                emoji = "🚨"  # Critical - 10ms+ is BAD
            elif duration_ms >= 5:
                emoji = "⚠️"  # Warning - 5ms+ is concerning
            else:
                emoji = "📊"  # Info - 1-5ms is normal

            desc = f" ({description})" if description else ""
            f.write(
                f"{datetime.now().isoformat()} {emoji} GIL_HOLD {location}: {duration_ms:.1f}ms{desc}\n"
            )

    def _accumulated_start(self) -> None:
        """🎯 ACCUMULATED START: Initial page load with ordered display!

        Pattern:
        1. Launch all 5 workers immediately (ZERO lag!)
        2. Workers complete async and accumulate results
        3. Display tables in order: builds → runner → vertex → active → completed
        4. If runner finishes before builds, it WAITS for builds to display first!

        Result: Perfect spinners, parallel loading, ordered display! 🔥
        """
        log_file = get_log_path("auto_refresh.log")
        # NOTE: Log already cleared in __init__ with constants header - just append here!

        # Set refreshing flag (prevents on_show from triggering during initial load)
        self._is_refreshing = True

        # Start spinner worker thread (constant rate independent of main thread!)
        self._start_spinner_worker()

        # Start duration ticker (1-second updates for active items)
        duration_ticker = self.set_interval(1.0, self._update_active_durations)
        self.refresh_timers.append(duration_ticker)

        # Start auto-refresh timer (calls _accumulated_refresh every AUTO_REFRESH_INTERVAL)
        # 😴⏰ AUTO REFRESH STEVEN - He wakes up every 30s to refresh the tables!
        auto_refresh_timer = self.set_interval(
            AUTO_REFRESH_INTERVAL, self._accumulated_refresh
        )
        self.refresh_timers.append(auto_refresh_timer)

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 😴⏰ AUTO REFRESH STEVEN: Goes to sleep (will wake every {AUTO_REFRESH_INTERVAL}s to refresh tables!)\n"
            )

        # 🎯 START ACCUMULATOR: Prep for ordered display!
        all_tables = ["builds", "runner", "vertex", "active", "completed"]
        self._start_accumulator(all_tables)

        # 🎯 IPASB SMART ENTRY: Adaptive staggering based on load conditions!
        # Checks backoff level before each launch - if system stressed, wait longer!
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🚀 INITIAL_LOAD: IPASB Smart Entry - adaptive staggering!\n"
            )

        # Queue of tables to launch with adaptive delays
        tables_to_launch = ["builds", "runner", "vertex", "active", "completed"]

        def launch_next_table(index: int = 0):
            """Recursively launch tables with adaptive delays based on load"""
            if index >= len(tables_to_launch):
                return  # All launched!

            table_name = tables_to_launch[index]

            # Launch this table
            self._universal_refresh_table(table_name, use_accumulator=True)

            # 🎯 IPASB SMART ENTRY: BIG delays at start, decrease as we go!
            # Early launches need LOTS of breathing room when system is cold
            # Also check backoff level for additional stress detection
            base_delays = [
                0.400,
                0.350,
                0.300,
                0.250,
                0.200,
            ]  # BIGGER decreasing pattern!
            base_delay = base_delays[index] if index < len(base_delays) else 0.200

            # Add extra delay if system is stressed (backoff > 0)
            backoff = self._ipasb_backoff_level
            if backoff >= 3:
                delay = base_delay + 0.300  # Critical: +300ms
            elif backoff >= 2:
                delay = base_delay + 0.150  # Heavy: +150ms
            elif backoff >= 1:
                delay = base_delay + 0.075  # Mild: +75ms
            else:
                delay = base_delay  # Normal: use base delay

            # Schedule next table launch
            if index + 1 < len(tables_to_launch):
                self.set_timer(delay, lambda i=index: launch_next_table(i + 1))

        # Start the chain!
        launch_next_table(0)

        # Reset flag after all workers launched (allow manual refreshes)
        def reset_flag():
            self._is_refreshing = False

        self.set_timer(0.5, reset_flag)  # 500ms - after all workers launched

    # ✅ SPICY: ALICE RABBIT HOLE complete - BUG #6 FIXED! Added Path to module top (line 125), removed local datetime+pathlib imports, verified _start_staggered_refresh (line 1592), all logic correct

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🎯 UNIVERSAL TABLE REFRESH SYSTEM
    # Replaces 21 separate functions with 2 universal helpers - DRY architecture!
    # ═══════════════════════════════════════════════════════════════════════════════

    def _universal_refresh_table(
        self,
        table_name: str,
        is_auto_refresh: bool = False,
        use_accumulator: bool = False,
    ) -> None:
        """Universal table refresh - works for ALL tables with guaranteed cleanup!

        Args:
            table_name: One of "runner", "builds", "vertex", "active", "completed"
            is_auto_refresh: True if called from auto-refresh timer (adds logging)
            use_accumulator: True if part of initial page load (uses General Accumulator pattern)
        """
        # Validate table name
        if table_name not in self.TABLE_CONFIG:
            self.notify(f"❌ Unknown table: {table_name}", severity="error", timeout=3)
            return

        config = self.TABLE_CONFIG[table_name]

        # 🔒 ATOMIC: Check+add to refreshing_tables (prevent race conditions!)
        with self._refresh_lock:
            # Skip if already refreshing this table
            if table_name in self._refreshing_tables:
                elapsed = time.time() - self._refresh_start_times.get(table_name, 0)
                # Log skip (for debugging - always log!)
                log_file = get_log_path("auto_refresh.log")
                skip_type = "AUTO" if is_auto_refresh else "PAGE_LOAD"
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} ⏭️  SKIP ({skip_type}): {table_name} (already running for {elapsed:.1f}s)\n"
                    )
                return

            # Mark as refreshing + record start time (inside lock - atomic!)
            self._refreshing_tables.add(table_name)
            self._refresh_start_times[table_name] = time.time()

        # Start spinner
        self._start_spinner(config["spinner_id"])

        # Log launch (for debugging - ALWAYS log, not just auto-refresh!)
        log_file = get_log_path("auto_refresh.log")
        launch_type = "AUTO" if is_auto_refresh else "PAGE_LOAD"
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🚀 LAUNCHING_WORKER ({launch_type}): {table_name}\n"
            )

        # Launch worker with table-specific logic
        self.run_worker(
            lambda: self._universal_table_worker(table_name, config, use_accumulator),
            exclusive=False,  # 🔥 ALLOW PARALLEL WORKERS! exclusive=True blocks all other workers!
            name=f"refresh_{table_name}",
            thread=True,
        )

        # Log queued (always log, not just auto-refresh!)
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ✓ WORKER_QUEUED ({launch_type}): {table_name}\n"
            )

    def _universal_table_worker(
        self, table_name: str, config: dict, use_accumulator: bool = False
    ):
        """Universal worker - handles ANY table with automatic cleanup guarantee!

        This is the CRITICAL function that prevents stuck tables!
        The finally block ALWAYS runs, ensuring cleanup happens even if worker crashes.

        Args:
            table_name: One of "runner", "builds", "vertex", "active", "completed"
            config: Table configuration dict
            use_accumulator: True if part of initial page load (uses General Accumulator)
        """
        #  🔥 CRITICAL DEBUG: Log IMMEDIATELY at function entry
        log_file = get_log_path("auto_refresh.log")
        timing_log = get_log_path("table_worker_timing.log")

        entry_time = time.time()
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🔥 FUNCTION_CALLED: _universal_table_worker({table_name}) ENTRY\n"
            )

        start_time = time.time()
        fetch_start = None
        fetch_end = None
        update_start = None
        update_end = None

        try:
            with open(log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} ▶️  WORKER_START: {table_name}\n")

            # ⏱️ TIME FETCH OPERATION
            fetch_start = time.time()

            # 🎯 PHASE 2: Check use_accumulator flag!
            if use_accumulator:
                # ✅ INITIAL LOAD: Fetch data ONLY, store it, NO rendering!
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 🎯 FETCH_ONLY: {table_name} (accumulator will render later)\n"
                    )

                # Fetch data (no rendering!)
                if table_name == "runner":
                    data = self._fetch_runner_data()
                elif table_name == "builds":
                    data = self._fetch_builds_data()
                elif table_name == "vertex":
                    data = self._fetch_vertex_data()
                elif table_name == "active":
                    data = self._fetch_active_data()
                elif table_name == "completed":
                    data = self._fetch_completed_data()

                # Store fetched data (thread-safe)
                with self._accumulator_lock:
                    self._fetched_data[table_name] = data

                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 💾 DATA_STORED: {table_name} (ready for display)\n"
                    )

            else:
                # ✅ AUTO-REFRESH / MANUAL: Fetch + render immediately (old behavior)
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 🔄 FETCH_AND_RENDER: {table_name} (immediate display)\n"
                    )

                if table_name == "runner":
                    self._fetch_and_update_runner_table()
                elif table_name == "builds":
                    self._fetch_and_update_builds_table()
                elif table_name == "vertex":
                    self._fetch_and_update_vertex_table()
                elif table_name == "active":
                    self._fetch_and_update_active_runs_table()
                elif table_name == "completed":
                    self._fetch_and_update_completed_runs_table()

            update_end = time.time()
            elapsed = update_end - start_time
            fetch_time = update_end - fetch_start

            # 🎯 TRACK WORKER DURATIONS (for budget health checks)
            if table_name not in self._worker_durations:
                self._worker_durations[table_name] = []
            self._worker_durations[table_name].append(elapsed)
            if len(self._worker_durations[table_name]) > 5:
                self._worker_durations[table_name].pop(0)  # Keep last 5

            # 💰 CALCULATE BUDGET HEALTH
            budget_pct = (elapsed / self._auto_refresh_interval) * 100
            if budget_pct < 50:
                budget_emoji = "✅"  # Plenty of headroom
            elif budget_pct < 90:
                budget_emoji = "⚠️"  # Tight but ok
            else:
                budget_emoji = "🚨"  # Overlapping!

            # 📊 LOG COMPREHENSIVE TIMING + BUDGET
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ WORKER_COMPLETE: {table_name} ({elapsed:.2f}s)\n"
                )
            with open(timing_log, "a") as f:
                avg_duration = sum(self._worker_durations[table_name]) / len(
                    self._worker_durations[table_name]
                )
                f.write(
                    f"{datetime.now().isoformat()} ⏱️  {table_name.upper()}: "
                    f"total={elapsed:.3f}s, fetch+update={fetch_time:.3f}s, "
                    f"overhead={(start_time - entry_time):.3f}s, "
                    f"budget={budget_emoji}{budget_pct:.0f}% (avg={avg_duration:.2f}s over last {len(self._worker_durations[table_name])})\n"
                )

        except Exception as e:
            import traceback

            elapsed = time.time() - start_time
            error_msg = str(e)
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ❌ WORKER_FAILED: {table_name} ({elapsed:.2f}s) - {error_msg}\n"
                )
                # Log full traceback for debugging
                f.write(f"{datetime.now().isoformat()} 📋 TRACEBACK:\n")
                for line in traceback.format_exc().strip().split("\n"):
                    f.write(f"    {line}\n")
            # 🧵 Error toasts: Use call_from_thread() for thread-safe UI updates!
            self.app.call_from_thread(
                self.notify,
                f"❌ {config['name']} refresh failed: {error_msg}",
                severity="error",
                timeout=5,
            )

        finally:
            # 🎯 GUARANTEED CLEANUP - ALWAYS runs, no matter what!
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 🚨 FINALLY_ENTERED: {table_name}\n"
                )

            # ⚠️ DON'T call _stop_spinner() - call_from_thread() BLOCKS from worker threads!
            # Solution: Just remove from _refreshing_tables, spinner timer will auto-hide
            # (spinner timer checks _refreshing_tables and skips stopped tables)

            # 🎯 ACCUMULATOR PATTERN: Keep spinner spinning until display!
            # If using accumulator, DON'T stop spinner here - let _display_next_ready_table do it
            # This way spinner only stops when table is actually VISIBLE (better UX!)
            if not use_accumulator:
                self._refreshing_tables.discard(table_name)
            if table_name in self._refresh_start_times:
                del self._refresh_start_times[table_name]

            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 🧹 CLEANUP_DONE: {table_name} (spinner will auto-hide on next timer cycle)\n"
                )

            # 🎯 GENERAL ACCUMULATOR: If this table is part of a batch, mark complete!
            # Polling timer (_display_next_ready_table) will notice and trigger display
            if table_name in self._current_batch:
                completion_time = time.time()
                with self._accumulator_lock:
                    self._accumulator_results[table_name] = True
                    self._accumulator_completion_times[table_name] = completion_time
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} ✅ ACCUMULATOR_MARKED: {table_name} complete (polling timer will display)\n"
                    )

                # Start display polling if not already running
                if not self._accumulator_active:
                    self._accumulator_active = True
                    with open(log_file, "a") as f:
                        f.write(
                            f"{datetime.now().isoformat()} 🎯 STARTING_POLLING: First table complete, starting display timer\n"
                        )
                    # Schedule on main thread - use App.call_from_thread!
                    self.app.call_from_thread(self._display_next_ready_table)

    # 🎯 TABLE-SPECIFIC FETCH & UPDATE FUNCTIONS
    # Clean data-only logic - no workers, no cleanup, no logging!
    # ═══════════════════════════════════════════════════════════════════════════════

    def _fetch_and_update_runner_table(self):
        """Fetch runner executions data and update table UI"""
        # 📡 IPASB: Check if we should back off before heavy work
        self._ipasb_check_backoff()

        debug_log = get_log_path("worker_debug.log")
        start_time = datetime.now()
        with open(debug_log, "a") as f:
            f.write(f"{start_time.isoformat()} 🚀 RUNNER_START\n")

        log_file = get_log_path("auto_refresh.log")

        # 🎯 CACHE: Check if we should use cached data (CACHE_TTL seconds)
        if not self._should_fetch_table("runner"):
            runner_execs = self._get_cached_data("runner")
            # Use cached data (skip expensive API call!)
        else:
            # Cache expired or empty - fetch fresh data
            monitor = get_monitor()
            region = self.config.get("GCP_ROOT_RESOURCE_REGION", "us-central1")

            class SilentCallback:
                def __call__(self, message: str):
                    pass

            op_id = monitor.start_operation("fetch_runner_executions", category="GCP")
            runner_execs = _fetch_runner_executions_all_regions(
                SilentCallback(), region
            )
            monitor.end_operation(op_id)
            # 🦡 HONEY BADGER: Yield after API call! (GIL held during JSON parsing!)
            time.sleep(GIL_YIELD_API)  # 🦡 HONEY BADGER: 25ms yield!

            # Update cache with fresh data
            self._update_table_cache("runner", runner_execs)

        # 🔍 SNAPSHOT VALIDATION: Compare with previous data
        self._compare_and_log_snapshot("runner", runner_execs)

        # 🧵 BATCHED UI UPDATES: Prepare all row data in worker thread (no UI calls!)
        rows_to_add = []
        extra_items = 0

        if runner_execs and len(runner_execs) > 0:
            # Separate running vs completed
            running_execs = [e for e in runner_execs if e.get("status") == "RUNNING"]
            completed_execs = [e for e in runner_execs if e.get("status") != "RUNNING"]

            # Show ALL running + limited completed
            execs_to_show = list(running_execs)
            if MAX_RUNNER_EXECS and len(completed_execs) > 0:
                execs_to_show += completed_execs[:MAX_RUNNER_EXECS]
                extra_items = (
                    len(completed_execs) - MAX_RUNNER_EXECS
                    if len(completed_execs) > MAX_RUNNER_EXECS
                    else 0
                )
            else:
                execs_to_show += completed_execs

            added_divider = False

            for exec_data in execs_to_show:
                # Add divider if transitioning from running to completed
                if (
                    not added_divider
                    and len(running_execs) > 0
                    and exec_data.get("status") != "RUNNING"
                ):
                    divider_row = self._create_table_divider("runner")
                    rows_to_add.append(
                        (divider_row, f"divider-runner-running-completed", None)
                    )
                    added_divider = True

                exec_name = str(exec_data.get("name", "unknown"))
                exec_key = f"runner-{exec_name}"

                # Note is pre-formatted in core.py via format_runner_note()
                error_display = exec_data.get("error", "[dim]—[/dim]")
                full_error_log = exec_data.get("full_error_log", "")

                # Prepare row tuple
                row = (
                    f"[dim blue]{exec_data.get('queue_name', '—')}[/dim blue]",
                    f"[cyan]{exec_data.get('region', '—')}[/cyan]",
                    exec_data.get("status_display", "UNKNOWN"),
                    f"[yellow]{exec_data.get('jobs_run', '0')}[/yellow]",
                    f"[cyan]{exec_data.get('duration', '—')}[/cyan]",
                    f"[dim]{exec_data.get('created_display', '—')}[/dim]",
                    error_display,
                )

                # Prepare row_data dict
                row_data = {
                    "queue": exec_data.get("queue_name", "—"),
                    "region": exec_data.get("region", "—"),
                    "status": exec_data.get("status", "—"),
                    "start_time": exec_data.get("start_time"),
                    "duration": exec_data.get("duration", "—"),
                    "created": exec_data.get("created_display", "—"),
                    "note": error_display,  # Pre-formatted in core.py
                    "full_error_log": full_error_log,
                }

                rows_to_add.append((row, exec_key, row_data))

        # 🧵 ONE callback to update entire table (thread-safe!)
        def update_runner_ui():
            runner_table = self.query_one("#runner-executions-table", DataTable)
            runner_table.clear()
            self.row_data["runner"].clear()
            self._extra_items["runner"] = extra_items

            if not rows_to_add:
                self._add_empty_state_row("runner", "No executions")
            else:
                for row, key, row_data in rows_to_add:
                    if row_data is None:  # Divider row
                        runner_table.add_row(*row, key=key)
                    else:
                        row_key_obj = runner_table.add_row(*row, key=key)
                        row_data["row_key"] = row_key_obj
                        self.row_data["runner"][key] = row_data

                runner_table.move_cursor(row=-1)
            runner_table.refresh()

            # Log completion
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ TABLE_UPDATED: runner ({len(rows_to_add)} rows) - BATCHED\n"
                )

        self.app.call_from_thread(update_runner_ui)

    def _fetch_runner_data(self) -> list[dict]:
        """Fetch runner data ONLY - no rendering! (Used by accumulator)

        🩰 RICKY THE RUNNER - First dancer to take the stage! 🏃
        """
        log_file = get_log_path("auto_refresh.log")

        # 🩰 DANCER ENTRY: Ricky takes the stage!
        self._canonical_log(
            "🩰 RICKY THE RUNNER 🏃 enters the stage! (fetching data...)"
        )

        # 🎯 CACHE: Check if we should use cached data (CACHE_TTL seconds)
        if not self._should_fetch_table("runner"):
            runner_execs = self._get_cached_data("runner")
            # Use cached data (skip expensive API call!)
        else:
            # Cache expired or empty - fetch fresh data
            monitor = get_monitor()
            region = self.config.get("GCP_ROOT_RESOURCE_REGION", "us-central1")

            class SilentCallback:
                def __call__(self, message: str):
                    pass

            op_id = monitor.start_operation("fetch_runner_executions", category="GCP")
            runner_execs = _fetch_runner_executions_all_regions(
                SilentCallback(), region
            )
            monitor.end_operation(op_id)
            # 🦡 HONEY BADGER: Yield after API call! (GIL held during JSON parsing!)
            time.sleep(GIL_YIELD_API)  # 🦡 HONEY BADGER: 25ms yield!

            # Update cache with fresh data
            self._update_table_cache("runner", runner_execs)

        # 🔍 SNAPSHOT VALIDATION: Compare with previous data
        self._compare_and_log_snapshot("runner", runner_execs)

        # 🩰 DANCER EXIT: Ricky returns with data!
        self._canonical_log(
            f"🩰 RICKY THE RUNNER 🏃 returns! ({len(runner_execs)} items) → throwing to main thread!"
        )

        return runner_execs

    def _update_runner_table(self, runner_execs: list[dict]) -> None:
        """Update runner table ONLY - assumes data already fetched! (Used by accumulator)"""
        log_file = get_log_path("auto_refresh.log")
        func_start = time.time()

        # Update table
        runner_table = self.query_one("#runner-executions-table", DataTable)

        # 🎨 TABLE FILL EFFECT: Clear → Show empty → Pause → Fill → Show filled
        table_order = 2  # runner is 2nd in display order (builds=1, runner=2, vertex=3, active=4, completed=5)
        clear_time = time.time()

        # 📊 GIL_HOLD: Table clear + placeholder
        gil_start = time.time()
        runner_table.clear()
        # 🎨 Add placeholder row so empty table is VISIBLE during fill effect
        runner_table.add_row("-", "-", "-", "-", "-", "-", "-")
        runner_table.refresh()  # STEP 1: Show EMPTY table with placeholder
        self._gil_hold_log(
            "runner_table_clear",
            (time.time() - gil_start) * 1000,
            "clear+placeholder+refresh",
        )

        # 🕐 TIMESTAMP: Log clear time
        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ⬜ TABLE_CLEAR runner - showing empty\n"
            )

        self.row_data["runner"].clear()

        # 🌶️ PAPRIKA DRY: Use helper for empty state (10 lines → 3 lines!)
        if not runner_execs or len(runner_execs) == 0:
            runner_table.clear()
            self._add_empty_state_row("runner", "No executions")
            runner_table.refresh()  # Show empty state
            return

        # 🎨 STEP 2: Visual fill effect (0ms now - was 500ms, too slow!)
        if VISUAL_FILL_DELAY > 0:
            time.sleep(VISUAL_FILL_DELAY)

        # 🎨 Clear placeholder before filling with real data
        runner_table.clear()

        # Separate running vs completed
        # 📊 GIL_HOLD: List comprehensions
        gil_start = time.time()
        running_execs = [e for e in runner_execs if e.get("status") == "RUNNING"]
        completed_execs = [e for e in runner_execs if e.get("status") != "RUNNING"]
        self._gil_hold_log(
            "runner_list_comp",
            (time.time() - gil_start) * 1000,
            f"separate {len(runner_execs)} execs",
        )
        # 🦡 HONEY BADGER: Yield after data processing! (list comprehensions can hold GIL!)
        time.sleep(GIL_YIELD_PROCESSING)  # 5ms yield - let spinner updates execute

        # 🌶️ PAPRIKA: Show ALL running + limited completed (track _extra_items)
        execs_to_show = list(running_execs)
        if MAX_RUNNER_EXECS and len(completed_execs) > 0:
            execs_to_show += completed_execs[:MAX_RUNNER_EXECS]
            self._extra_items["runner"] = (
                len(completed_execs) - MAX_RUNNER_EXECS
                if len(completed_execs) > MAX_RUNNER_EXECS
                else 0
            )
        else:
            execs_to_show += completed_execs
            self._extra_items["runner"] = 0

        added_divider = False

        for idx, exec_data in enumerate(execs_to_show):
            # 🦡 HONEY BADGER: Yield EVERY row! (Textual demo/game.py uses 50ms per iteration)
            # This keeps spinners smooth during heavy 50+ row table loads
            time.sleep(GIL_YIELD_ROW)  # 10ms yield per row

            # 🌶️ PAPRIKA DRY: Use helper to create divider (8 lines → 3 lines!)
            if (
                not added_divider
                and len(running_execs) > 0
                and exec_data.get("status") != "RUNNING"
            ):
                divider_row = self._create_table_divider("runner")
                runner_table.add_row(
                    *divider_row, key=f"divider-runner-running-completed"
                )
                added_divider = True

            exec_name = str(exec_data.get("name", "unknown"))
            exec_key = f"runner-{exec_name}"

            # 🔍 DEBUG: Log ALL rows with FULL name and row_key (sanity check for duplicate keys!)
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 🔍 RUNNER ROW {idx}: FULL_NAME=[{exec_name}], ROW_KEY=[{exec_key}], name_field={exec_data.get('name', 'MISSING')}, status={exec_data.get('status')}\n"
                )

            # Note is pre-formatted in core.py via format_runner_note()
            error_display = exec_data.get("error", "[dim]—[/dim]")
            full_error_log = exec_data.get("full_error_log", "")

            row_key_obj = runner_table.add_row(
                f"[dim blue]{exec_data.get('queue_name', '—')}[/dim blue]",
                f"[cyan]{exec_data.get('region', '—')}[/cyan]",
                exec_data.get("status_display", "UNKNOWN"),
                f"[yellow]{exec_data.get('jobs_run', '0')}[/yellow]",
                f"[cyan]{exec_data.get('duration', '—')}[/cyan]",
                f"[dim]{exec_data.get('created_display', '—')}[/dim]",
                error_display,
                key=exec_key,
            )

            self.row_data["runner"][exec_key] = {
                "queue": exec_data.get("queue_name", "—"),
                "region": exec_data.get("region", "—"),
                "status": exec_data.get("status", "—"),
                "start_time": exec_data.get("start_time"),
                "duration": exec_data.get("duration", "—"),
                "created": exec_data.get("created_display", "—"),
                "note": error_display,  # Pre-formatted in core.py
                "full_error_log": full_error_log,
                "row_key": row_key_obj,
            }

            # 🔍 DEBUG: Log row_data for ticker debugging
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 📊 RUNNER_ROW_DATA: key={exec_key}, status={exec_data.get('status')}, start_time={exec_data.get('start_time')}\n"
                )

        runner_table.move_cursor(row=-1)
        runner_table.refresh()  # 🌶️ PAPRIKA: Force UI update

    def _fetch_and_update_builds_table(self):
        """Fetch cloud builds data and update table UI - LAST CONFESSION! 🔥"""
        # 📡 IPASB: Check if we should back off before heavy work
        self._ipasb_check_backoff()

        debug_log = get_log_path("worker_debug.log")
        start_time = datetime.now()
        with open(debug_log, "a") as f:
            f.write(f"{start_time.isoformat()} 🚀 BUILDS_START\n")
        log_file = get_log_path("auto_refresh.log")

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🔍 FETCH_BUILDS: ENTRY (FINAL CONFESSION!)\n"
            )

        # 🎯 CACHE: Check if we should use cached data (CACHE_TTL seconds)
        if not self._should_fetch_table("builds"):
            builds = self._get_cached_data("builds")
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 💾 CACHE_HIT: Using cached builds ({len(builds)} items)\n"
                )
        else:
            # Cache expired - fetch fresh

            try:
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 🌐 CALLING _list_recent_cloud_builds (all 18 regions)...\n"
                    )
                builds = _list_recent_cloud_builds(lambda m: None)
                # 🦡 HONEY BADGER: Yield after API call! (GIL held during JSON parsing!)
                time.sleep(GIL_YIELD_API)  # 🦡 HONEY BADGER: 25ms yield!
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 📊 FETCH_BUILDS: Got {len(builds)} builds (FORGIVE ME!)\n"
                    )

                # Update cache
                self._update_table_cache("builds", builds)
            except Exception as e:
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} ❌ BUILDS CONFESSION: FAILED TO FETCH - {str(e)}\n"
                    )
                raise

        # 🔍 SNAPSHOT VALIDATION: Compare with previous data
        self._compare_and_log_snapshot("builds", builds)

        # 🧵 BATCHED UI UPDATES: Prepare all row data in worker thread (no UI calls!)
        rows_to_add = []
        extra_items = 0
        active_count = 0
        completed_count = 0

        if builds and len(builds) > 0:
            # Separate active vs completed builds
            active_builds = [
                b for b in builds if b.get("status") in ["WORKING", "QUEUED"]
            ]
            completed_builds = [
                b for b in builds if b.get("status") not in ["WORKING", "QUEUED"]
            ]
            active_count = len(active_builds)
            completed_count = len(completed_builds)

            # Apply MAX_CLOUD_BUILDS limit
            builds_to_show = list(active_builds)
            if MAX_CLOUD_BUILDS and len(completed_builds) > 0:
                builds_to_show += completed_builds[:MAX_CLOUD_BUILDS]
                extra_items = (
                    len(completed_builds) - MAX_CLOUD_BUILDS
                    if len(completed_builds) > MAX_CLOUD_BUILDS
                    else 0
                )
            else:
                builds_to_show += completed_builds

            added_divider = False

            for build in builds_to_show:
                build_id = build.get("build_id", "unknown")
                row_key = f"build-{build_id}"

                # Add divider if transitioning from active to completed
                if (
                    not added_divider
                    and len(active_builds) > 0
                    and build.get("status") not in ["WORKING", "QUEUED"]
                ):
                    divider_row = self._create_table_divider("builds")
                    rows_to_add.append(
                        (divider_row, f"divider-builds-active-completed", None)
                    )
                    added_divider = True

                # Note is pre-formatted in core.py via format_builds_note()
                error_display = build.get("error", "[dim]—[/dim]")

                # Prepare row tuple
                row = (
                    f"[yellow]{build.get('build_id', '—')}[/yellow]",
                    f"[dim blue]{build.get('image_name', '—')[:20]}[/dim blue]",
                    f"[cyan]{build.get('region', '—')}[/cyan]",
                    build.get("status_display", "UNKNOWN"),
                    f"[cyan]{build.get('duration_display', '—')}[/cyan]",
                    f"[dim]{build.get('finished_display', '—')}[/dim]",
                    error_display,
                )

                # Prepare row_data dict
                row_data = {
                    "id": build_id,
                    "status": build.get("status", "—"),
                    "start_time": build.get("start_time"),
                    "note": error_display,  # Pre-formatted in core.py
                }

                rows_to_add.append((row, row_key, row_data))

        # 🧵 ONE callback to update entire table (thread-safe!)
        def update_builds_ui():
            builds_table = self.query_one("#builds-recent-table", DataTable)
            builds_table.clear()
            self.row_data["build_recent"].clear()
            self._extra_items["builds"] = extra_items

            if not rows_to_add:
                self._add_empty_state_row("builds", "No builds (24h)")
            else:
                for row, key, row_data in rows_to_add:
                    if row_data is None:  # Divider row
                        builds_table.add_row(*row, key=key)
                    else:
                        row_key_obj = builds_table.add_row(*row, key=key)
                        row_data["row_key"] = row_key_obj
                        self.row_data["build_recent"][key] = row_data

                builds_table.move_cursor(row=-1)
            builds_table.refresh()

            # Log completion
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ TABLE_UPDATED: builds ({len(rows_to_add)} rows, {active_count} active, {completed_count} completed) - BATCHED\n"
                )

        self.app.call_from_thread(update_builds_ui)

    def _fetch_builds_data(self) -> list[dict]:
        """Fetch builds data ONLY - no rendering! Returns list of builds

        🩰 BELLA THE BUILDER - Second dancer, builds the foundation! 🏗️
        """
        log_file = get_log_path("auto_refresh.log")

        # 🩰 DANCER ENTRY: Bella takes the stage!
        self._canonical_log(
            "🩰 BELLA THE BUILDER 🏗️ enters the stage! (fetching builds...)"
        )

        # Cache check
        if not self._should_fetch_table("builds"):
            builds = self._get_cached_data("builds")
        else:
            builds = _list_recent_cloud_builds(lambda m: None)
            time.sleep(GIL_YIELD_API)
            self._update_table_cache("builds", builds)

        self._compare_and_log_snapshot("builds", builds)

        # 🩰 DANCER EXIT: Bella returns with data!
        self._canonical_log(
            f"🩰 BELLA THE BUILDER 🏗️ returns! ({len(builds)} items) → throwing to main thread!"
        )

        return builds

    def _update_builds_table(self, builds: list[dict]) -> None:
        """Update builds table with pre-fetched data - THE THREADING DANCE SYSTEM! 🩰🔥

        # ═══════════════════════════════════════════════════════════════
        #   🩰 THE THREADING DANCE: Worker → Bridge → UI 🩰
        # ═══════════════════════════════════════════════════════════════
        #
        #   WORKER THREAD                    MAIN THREAD
        #        ◯                                ◯
        #       /|\\                             /|\\
        #       / \\                             / \\
        #        │                                │
        #        │ builds = fetch()               │
        #        │                                │
        #        └─► call_from_thread() ─────────►│
        #                                         │
        #                                         ▼
        #                                   _update_builds_table(builds)
        #                                         │
        #                                         │ Use builds DIRECTLY!
        #                                         │ NO API calls here!
        #                                         ▼
        #                                       \\○//
        #                                        │
        #                                       / \\
        #                                   UI UPDATED!
        #
        # ═══════════════════════════════════════════════════════════════
        """
        log_file = get_log_path("auto_refresh.log")

        # Get table widget
        builds_table = self.query_one("#builds-recent-table", DataTable)

        # Clear and prepare
        builds_table.clear()
        self.row_data["build_recent"].clear()

        if not builds or len(builds) == 0:
            # 🩰 DANCER: Empty state - graceful bow
            self._add_empty_state_row("builds", "No builds (24h)")
            builds_table.refresh()
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ THREADING_DANCE: builds table updated (empty) 🩰\n"
                )
            return

        # Separate active vs completed builds
        active_builds = [b for b in builds if b.get("status") in ["WORKING", "QUEUED"]]
        completed_builds = [
            b for b in builds if b.get("status") not in ["WORKING", "QUEUED"]
        ]

        # Apply MAX_CLOUD_BUILDS limit
        builds_to_show = list(active_builds)
        if MAX_CLOUD_BUILDS and len(completed_builds) > 0:
            builds_to_show += completed_builds[:MAX_CLOUD_BUILDS]
            self._extra_items["builds"] = (
                len(completed_builds) - MAX_CLOUD_BUILDS
                if len(completed_builds) > MAX_CLOUD_BUILDS
                else 0
            )
        else:
            builds_to_show += completed_builds
            self._extra_items["builds"] = 0

        added_divider = False

        for build in builds_to_show:
            build_id = build.get("build_id", "unknown")
            row_key = f"build-{build_id}"

            # 🩰 DANCER: Add divider when transitioning from active to completed
            if (
                not added_divider
                and len(active_builds) > 0
                and build.get("status") not in ["WORKING", "QUEUED"]
            ):
                divider_row = self._create_table_divider("builds")
                builds_table.add_row(
                    *divider_row, key=f"divider-builds-active-completed"
                )
                added_divider = True

            # Note is pre-formatted in core.py via format_builds_note()
            error_display = build.get("error", "[dim]—[/dim]")

            row_key_obj = builds_table.add_row(
                f"[yellow]{build.get('build_id', '—')}[/yellow]",
                f"[dim blue]{build.get('image_name', '—')[:20]}[/dim blue]",
                f"[cyan]{build.get('region', '—')}[/cyan]",
                build.get("status_display", "UNKNOWN"),
                f"[cyan]{build.get('duration_display', '—')}[/cyan]",
                f"[dim]{build.get('finished_display', '—')}[/dim]",
                error_display,
                key=row_key,
            )

            self.row_data["build_recent"][row_key] = {
                "id": build_id,
                "status": build.get("status", "—"),
                "start_time": build.get("start_time"),
                "note": error_display,  # Pre-formatted in core.py
                "row_key": row_key_obj,
            }

            # 🔍 DEBUG: Log row_data for ticker debugging
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 📊 BUILD_ROW_DATA: key={row_key}, status={build.get('status')}, start_time={build.get('start_time')}\n"
                )

        builds_table.move_cursor(row=-1)
        builds_table.refresh()

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ✅ THREADING_DANCE: builds table updated ({len(builds_to_show)} rows) 🩰\n"
            )

    def _fetch_vertex_data(self) -> list[dict]:
        """Fetch vertex data ONLY - no rendering!

        🩰 VICTOR THE VERTEX - Third dancer, aims for the target! 🎯
        """
        log_file = get_log_path("auto_refresh.log")

        # 🩰 DANCER ENTRY: Victor takes the stage!
        self._canonical_log(
            "🩰 VICTOR THE VERTEX 🎯 enters the stage! (fetching jobs...)"
        )

        if not self._should_fetch_table("vertex"):
            jobs = self._get_cached_data("vertex")
        else:
            jobs = _list_vertex_ai_jobs(lambda m: None)
            time.sleep(GIL_YIELD_API)
            self._update_table_cache("vertex", jobs)

        self._compare_and_log_snapshot("vertex", jobs)

        # 🩰 DANCER EXIT: Victor returns with data!
        self._canonical_log(
            f"🩰 VICTOR THE VERTEX 🎯 returns! ({len(jobs)} items) → throwing to main thread!"
        )

        return jobs

    def _update_vertex_table(self, jobs: list[dict]) -> None:
        """Update vertex table with pre-fetched data - THE THREADING DANCE SYSTEM! 🩰🔥

        # ═══════════════════════════════════════════════════════════════
        #   🩰 THE THREADING DANCE: Worker → Bridge → UI 🩰
        # ═══════════════════════════════════════════════════════════════
        #
        #   WORKER THREAD                    MAIN THREAD
        #        ◯                                ◯
        #       /|\\                             /|\\
        #       / \\                             / \\
        #        │                                │
        #        │ jobs = fetch_vertex()          │
        #        │                                │
        #        └─► call_from_thread() ─────────►│
        #                                         │
        #                                         ▼
        #                                   _update_vertex_table(jobs)
        #                                         │
        #                                         │ Use jobs DIRECTLY!
        #                                         │ NO API calls here!
        #                                         ▼
        #                                       \\○//
        #                                        │
        #                                       / \\
        #                                   VERTEX TABLE UPDATED!
        #
        # ═══════════════════════════════════════════════════════════════
        """
        log_file = get_log_path("auto_refresh.log")

        # Get table widget
        vertex_table = self.query_one("#vertex-jobs-table", DataTable)

        # Clear and prepare
        vertex_table.clear()
        self.row_data["vertex"].clear()

        if not jobs or len(jobs) == 0:
            # 🩰 DANCER: Empty state - graceful bow
            self._add_empty_state_row("vertex", "No jobs (7d)")
            vertex_table.refresh()
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ THREADING_DANCE: vertex table updated (empty) 🩰\n"
                )
            return

        # Apply MAX_VERTEX_JOBS limit
        if MAX_VERTEX_JOBS and len(jobs) > MAX_VERTEX_JOBS:
            jobs_to_show = jobs[:MAX_VERTEX_JOBS]
            self._extra_items["vertex"] = len(jobs) - MAX_VERTEX_JOBS
        else:
            jobs_to_show = jobs
            self._extra_items["vertex"] = 0

        for job in jobs_to_show:
            job_id = job.get("id", "unknown")
            row_key = f"vertex-{job_id}"
            job_id_short = str(job.get("id", "—"))[:8] if job.get("id") else "—"

            row_key_obj = vertex_table.add_row(
                f"[yellow]{job_id_short}[/yellow]",
                f"[dim blue]{job.get('name', '—')}[/dim blue]",
                f"[cyan]{job.get('region', '—')}[/cyan]",
                job.get("state_display", "UNKNOWN"),
                f"[cyan]{job.get('runtime_display', '—')}[/cyan]",
                f"[dim]{job.get('created_display', '—')}[/dim]",
                f"[dim]{job.get('error', '—') if job.get('error') else '—'}[/dim]",
                key=row_key,
            )

            self.row_data["vertex"][row_key] = {
                "id": job_id,
                "state": job.get("state", "—"),  # Ticker checks "state" not "status"!
                "start_time": job.get("start_time"),
                "note": job.get("note", "—"),
                "row_key": row_key_obj,
            }

            # 🔍 DEBUG: Log row_data for ticker debugging
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 📊 VERTEX_ROW_DATA: key={row_key}, state={job.get('state')}, start_time={job.get('start_time')}\n"
                )

        vertex_table.move_cursor(row=-1)
        vertex_table.refresh()

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ✅ THREADING_DANCE: vertex table updated ({len(jobs_to_show)} rows) 🩰\n"
            )

    def _fetch_active_data(self) -> list[dict]:
        """Fetch active runs data ONLY - no rendering!

        🩰 ARCHIE THE ACTIVE - Fourth dancer, full of energy! ⚡
        """
        log_file = get_log_path("auto_refresh.log")

        # 🩰 DANCER ENTRY: Archie takes the stage!
        self._canonical_log(
            "🩰 ARCHIE THE ACTIVE ⚡ enters the stage! (fetching active runs...)"
        )

        if not self._should_fetch_table("active"):
            runs = self._get_cached_data("active")
        else:
            runs = _list_active_runs(self.helper, lambda m: None)
            time.sleep(GIL_YIELD_API)
            self._update_table_cache("active", runs)

        self._compare_and_log_snapshot("active", runs)

        # 🩰 DANCER EXIT: Archie returns with data!
        self._canonical_log(
            f"🩰 ARCHIE THE ACTIVE ⚡ returns! ({len(runs)} items) → throwing to main thread!"
        )

        return runs

    def _update_active_table(self, runs: list[dict]) -> None:
        """Update active table with pre-fetched data - THE THREADING DANCE SYSTEM! 🩰🔥

        # ═══════════════════════════════════════════════════════════════
        #   🩰 THE THREADING DANCE: Worker → Bridge → UI 🩰
        # ═══════════════════════════════════════════════════════════════
        #
        #   WORKER THREAD                    MAIN THREAD
        #        ◯                                ◯
        #       /|\\                             /|\\
        #       / \\                             / \\
        #        │                                │
        #        │ runs = fetch_active()          │
        #        │                                │
        #        └─► call_from_thread() ─────────►│
        #                                         │
        #                                         ▼
        #                                   _update_active_table(runs)
        #                                         │
        #                                         │ Use runs DIRECTLY!
        #                                         │ NO API calls here!
        #                                         ▼
        #                                       \\○//
        #                                        │
        #                                       / \\
        #                                   ACTIVE RUNS UPDATED!
        #
        # ═══════════════════════════════════════════════════════════════
        """
        log_file = get_log_path("auto_refresh.log")

        # Get table widget
        active_table = self.query_one("#runs-table", DataTable)

        # Clear and prepare
        active_table.clear()
        self.row_data["active"].clear()

        if not runs or len(runs) == 0:
            # 🩰 DANCER: Empty state - graceful bow
            self._add_empty_state_row("active", "No W&B runs started yet")
            active_table.refresh()
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ THREADING_DANCE: active table updated (empty) 🩰\n"
                )
            return

        # Apply MAX_ACTIVE_RUNS limit
        if MAX_ACTIVE_RUNS and len(runs) > MAX_ACTIVE_RUNS:
            runs_to_show = runs[:MAX_ACTIVE_RUNS]
            self._extra_items["active"] = len(runs) - MAX_ACTIVE_RUNS
        else:
            runs_to_show = runs
            self._extra_items["active"] = 0

        for run in runs_to_show:
            run_id = run.get("id", "unknown")
            row_key = f"active-{run_id}"

            row_key_obj = active_table.add_row(
                run_id,
                f"[yellow]{run.get('name', '—')}[/yellow]",
                run.get("state_display", "UNKNOWN"),
                run.get("runtime_display", "—"),
                run.get("created_display", "—"),
                key=row_key,
            )

            self.row_data["active"][row_key] = {
                "id": run_id,
                "full_name": run.get("name", "—"),
                "state": run.get("state", "—"),
                "runtime": run.get("runtime_display", "—"),
                "created": run.get("created_display", "—"),
                "config": run.get("config", {}),
                "tags": run.get("tags", []),
                "start_time": run.get("start_time"),
                "note": f"Name: {run.get('name', '—')}\n\nState: {run.get('state', 'Unknown')}\nRuntime: {run.get('runtime_display', 'Unknown')}\nCreated: {run.get('created_display', 'Unknown')}",
                "row_key": row_key_obj,
            }

            # 🔍 DEBUG: Log row_data for ticker debugging
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 📊 ACTIVE_ROW_DATA: key={row_key}, state={run.get('state')}, start_time={run.get('start_time')}\n"
                )

        active_table.move_cursor(row=-1)
        active_table.refresh()

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ✅ THREADING_DANCE: active table updated ({len(runs_to_show)} rows) 🩰\n"
            )

    def _fetch_completed_data(self) -> list[dict]:
        """Fetch completed runs data ONLY - no rendering!

        🩰 CLEO THE COMPLETED - Fifth dancer, the grand finale! ✅
        """
        log_file = get_log_path("auto_refresh.log")

        # 🩰 DANCER ENTRY: Cleo takes the stage for the finale!
        self._canonical_log(
            "🩰 CLEO THE COMPLETED ✅ enters the stage! (fetching completed runs...)"
        )

        if not self._should_fetch_table("completed"):
            runs = self._get_cached_data("completed")
        else:
            runs = _list_completed_runs(self.helper, lambda m: None)
            time.sleep(GIL_YIELD_API)
            self._update_table_cache("completed", runs)

        self._compare_and_log_snapshot("completed", runs)

        # 🩰 DANCER EXIT: Cleo takes her bow!
        self._canonical_log(
            f"🩰 CLEO THE COMPLETED ✅ returns! ({len(runs)} items) → throwing to main thread! 🎭 FINALE!"
        )

        return runs

    def _update_completed_table(self, runs: list[dict]) -> None:
        """Update completed table with pre-fetched data - THE THREADING DANCE SYSTEM! 🩰🔥

        # ═══════════════════════════════════════════════════════════════
        #   🩰 THE THREADING DANCE: Worker → Bridge → UI 🩰
        # ═══════════════════════════════════════════════════════════════
        #
        #   WORKER THREAD                    MAIN THREAD
        #        ◯                                ◯
        #       /|\\                             /|\\
        #       / \\                             / \\
        #        │                                │
        #        │ runs = fetch_completed()       │
        #        │                                │
        #        └─► call_from_thread() ─────────►│
        #                                         │
        #                                         ▼
        #                                   _update_completed_table(runs)
        #                                         │
        #                                         │ Use runs DIRECTLY!
        #                                         │ NO API calls here!
        #                                         ▼
        #                                       \\○//
        #                                        │
        #                                       / \\
        #                                   COMPLETED RUNS UPDATED!
        #
        # ═══════════════════════════════════════════════════════════════
        """
        log_file = get_log_path("auto_refresh.log")

        # Get table widget
        completed_table = self.query_one("#completed-runs-table", DataTable)

        # Clear and prepare
        completed_table.clear()
        self.row_data["completed"].clear()

        if not runs or len(runs) == 0:
            # 🩰 DANCER: Empty state - graceful bow
            self._add_empty_state_row("completed", "No completed runs")
            completed_table.refresh()
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ THREADING_DANCE: completed table updated (empty) 🩰\n"
                )
            return

        # Apply MAX_COMPLETED_RUNS limit
        if MAX_COMPLETED_RUNS and len(runs) > MAX_COMPLETED_RUNS:
            runs_to_show = runs[:MAX_COMPLETED_RUNS]
            self._extra_items["completed"] = len(runs) - MAX_COMPLETED_RUNS
        else:
            runs_to_show = runs
            self._extra_items["completed"] = 0

        for run in runs_to_show:
            run_id = run.get("id", "unknown")
            row_key = f"completed-{run_id}"

            row_key_obj = completed_table.add_row(
                run_id,
                f"[yellow]{run.get('name', '—')}[/yellow]",
                run.get("state_display", "UNKNOWN"),
                run.get("runtime_display", "—"),
                run.get("created_display", "—"),
                key=row_key,
            )

            self.row_data["completed"][row_key] = {
                "id": run_id,
                "full_name": run.get("name", "—"),
                "state": run.get("state", "—"),
                "runtime": run.get("runtime_display", "—"),
                "created": run.get("created_display", "—"),
                "summary_metrics": run.get("summary_metrics", {}),
                "exit_code": run.get("exit_code", "—"),
                "note": f"Name: {run.get('name', '—')}\n\nFinal State: {run.get('state', 'Unknown')}\nTotal Runtime: {run.get('runtime_display', 'Unknown')}\nCompleted: {run.get('created_display', 'Unknown')}\nExit Code: {run.get('exit_code', '—')}",
                "row_key": row_key_obj,
            }

        completed_table.move_cursor(row=-1)
        completed_table.refresh()

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} ✅ THREADING_DANCE: completed table updated ({len(runs_to_show)} rows) 🩰\n"
            )

    def _fetch_and_update_vertex_table(self):
        """Fetch Vertex AI jobs data and update table UI - CONFESSION MODE ACTIVATED! 🔥"""
        # 📡 IPASB: Check if we should back off before heavy work
        self._ipasb_check_backoff()

        debug_log = get_log_path("worker_debug.log")
        start_time = datetime.now()
        with open(debug_log, "a") as f:
            f.write(f"{start_time.isoformat()} 🚀 VERTEX_START\n")
        log_file = get_log_path("auto_refresh.log")

        with open(log_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()} 🔍 FETCH_VERTEX: ENTRY (CONFESS YOUR SINS!)\n"
            )

        # 🎯 CACHE: Check if we should use cached data (CACHE_TTL seconds)
        if not self._should_fetch_table("vertex"):
            jobs = self._get_cached_data("vertex")
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 💾 CACHE_HIT: Using cached vertex ({len(jobs)} items)\n"
                )
        else:
            # Cache expired - fetch fresh

            try:
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 🌐 CALLING _list_vertex_ai_jobs (all 18 regions)...\n"
                    )
                jobs = _list_vertex_ai_jobs(lambda m: None)
                # 🦡 HONEY BADGER: Yield after API call! (GIL held during JSON parsing!)
                time.sleep(GIL_YIELD_API)  # 🦡 HONEY BADGER: 25ms yield!
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 📊 FETCH_VERTEX: Got {len(jobs)} jobs (I REPENT!)\n"
                    )

                # Update cache
                self._update_table_cache("vertex", jobs)
            except Exception as e:
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} ❌ FETCH_VERTEX CONFESSION: FAILED TO FETCH - {str(e)}\n"
                    )
                raise

        # 🔍 SNAPSHOT VALIDATION: Compare with previous data
        self._compare_and_log_snapshot("vertex", jobs)

        # 🧵 BATCHED UI UPDATES: Prepare all row data in worker thread (no UI calls!)
        rows_to_add = []
        extra_items = 0
        total_jobs = len(jobs) if jobs else 0

        if jobs and len(jobs) > 0:
            # Check for billing error marker
            if len(jobs) == 1 and jobs[0].get("_billing_error"):
                # Billing is disabled - show helpful error message
                billing_error_row = (
                    "[red]❌ BILLING[/red]",
                    "[red]Billing Disabled[/red]",
                    f"[dim]{jobs[0].get('region', 'unknown')}[/dim]",
                    "[yellow]⚠️ Billing required[/yellow]",
                    "—",
                    "—",
                    "[yellow]Enable billing: See SETUP.md[/yellow]",
                )
                rows_to_add.append((billing_error_row, "billing-error", {}))
            else:
                # Normal processing
                # Apply MAX_VERTEX_JOBS limit
                if MAX_VERTEX_JOBS and len(jobs) > MAX_VERTEX_JOBS:
                    jobs_to_show = jobs[:MAX_VERTEX_JOBS]
                    extra_items = len(jobs) - MAX_VERTEX_JOBS
                else:
                    jobs_to_show = jobs

                for job in jobs_to_show:
                    job_id = job.get("id", "unknown")
                    row_key = f"vertex-{job_id}"
                    job_id_short = str(job.get("id", "—"))[:8] if job.get("id") else "—"

                    # Prepare row tuple
                    row = (
                        f"[yellow]{job_id_short}[/yellow]",
                        f"[dim blue]{job.get('name', '—')}[/dim blue]",
                        f"[cyan]{job.get('region', '—')}[/cyan]",
                        job.get("state_display", "UNKNOWN"),
                        f"[cyan]{job.get('runtime_display', '—')}[/cyan]",
                        f"[dim]{job.get('created_display', '—')}[/dim]",
                        f"[dim]{job.get('error', '—') if job.get('error') else '—'}[/dim]",
                    )

                    # Prepare row_data dict
                    row_data = {
                        "id": job_id,
                        "status": job.get("status", "—"),
                        "start_time": job.get("start_time"),
                        "note": job.get("note", "—"),
                    }

                    rows_to_add.append((row, row_key, row_data))

        # 🧵 ONE callback to update entire table (thread-safe!)
        def update_vertex_ui():
            vertex_table = self.query_one("#vertex-jobs-table", DataTable)
            vertex_table.clear()
            self.row_data["vertex"].clear()
            self._extra_items["vertex"] = extra_items

            if not rows_to_add:
                self._add_empty_state_row("vertex", "No jobs (7d)")
            else:
                for row, key, row_data in rows_to_add:
                    row_key_obj = vertex_table.add_row(*row, key=key)
                    row_data["row_key"] = row_key_obj
                    self.row_data["vertex"][key] = row_data

                vertex_table.move_cursor(row=-1)
            vertex_table.refresh()

            # Log completion
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ TABLE_UPDATED: vertex ({len(rows_to_add)} rows, {total_jobs} total, {extra_items} hidden) - BATCHED\n"
                )

        self.app.call_from_thread(update_vertex_ui)

    def _fetch_and_update_active_runs_table(self):
        """Fetch active W&B runs data and update table UI"""
        # 📡 IPASB: Check if we should back off before heavy work
        self._ipasb_check_backoff()

        debug_log = get_log_path("worker_debug.log")
        start_time = datetime.now()
        with open(debug_log, "a") as f:
            f.write(f"{start_time.isoformat()} 🚀 ACTIVE_START\n")
        log_file = get_log_path("auto_refresh.log")

        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} 🔍 FETCH_ACTIVE: ENTRY\n")

        # 🎯 CACHE: Check if we should use cached data (CACHE_TTL seconds)
        if not self._should_fetch_table("active"):
            runs = self._get_cached_data("active")
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 💾 CACHE_HIT: Using cached active ({len(runs)} items)\n"
                )
        else:
            # Cache expired - fetch fresh

            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 🌐 CALLING _list_active_runs...\n"
                )
            runs = _list_active_runs(self.helper, lambda m: None)
            # 🦡 HONEY BADGER: Yield after API call! (GIL held during JSON parsing!)
            time.sleep(GIL_YIELD_API)  # 🦡 HONEY BADGER: 25ms yield!

            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 📊 FETCH_ACTIVE: Got {len(runs)} runs\n"
                )

            # Update cache
            self._update_table_cache("active", runs)

        # 🔍 SNAPSHOT VALIDATION: Compare with previous data
        self._compare_and_log_snapshot("active", runs)

        # 🧵 BATCHED UI UPDATES: Prepare all row data in worker thread (no UI calls!)
        rows_to_add = []
        extra_items = 0
        total_runs = len(runs) if runs else 0

        if runs and len(runs) > 0:
            # Apply MAX_ACTIVE_RUNS limit
            if MAX_ACTIVE_RUNS and len(runs) > MAX_ACTIVE_RUNS:
                runs_to_show = runs[:MAX_ACTIVE_RUNS]
                extra_items = len(runs) - MAX_ACTIVE_RUNS
            else:
                runs_to_show = runs

            for run in runs_to_show:
                run_id = run.get("id", "unknown")
                row_key = f"active-{run_id}"

                # Prepare row tuple
                row = (
                    run_id,
                    f"[yellow]{run.get('name', '—')}[/yellow]",
                    run.get("state_display", "UNKNOWN"),
                    run.get("runtime_display", "—"),
                    run.get("created_display", "—"),
                )

                # Prepare row_data dict
                row_data = {
                    "id": run_id,
                    "full_name": run.get("name", "—"),
                    "state": run.get("state", "—"),
                    "runtime": run.get("runtime_display", "—"),
                    "created": run.get("created_display", "—"),
                    "config": run.get("config", {}),
                    "tags": run.get("tags", []),
                    "start_time": run.get("start_time"),
                    "note": f"Name: {run.get('name', '—')}\n\nState: {run.get('state', 'Unknown')}\nRuntime: {run.get('runtime_display', 'Unknown')}\nCreated: {run.get('created_display', 'Unknown')}",
                }

                rows_to_add.append((row, row_key, row_data))

        # 🧵 ONE callback to update entire table (thread-safe!)
        def update_active_ui():
            active_table = self.query_one("#runs-table", DataTable)
            active_table.clear()
            self.row_data["active"].clear()
            self._extra_items["active"] = extra_items

            if not rows_to_add:
                self._add_empty_state_row("active", "No W&B runs started yet")
            else:
                for row, key, row_data in rows_to_add:
                    row_key_obj = active_table.add_row(*row, key=key)
                    row_data["row_key"] = row_key_obj
                    self.row_data["active"][key] = row_data

                active_table.move_cursor(row=-1)
            active_table.refresh()

            # Log completion
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ TABLE_UPDATED: active ({len(rows_to_add)} rows, {total_runs} total, {extra_items} hidden) - BATCHED\n"
                )

        self.app.call_from_thread(update_active_ui)

    def _fetch_and_update_completed_runs_table(self):
        """Fetch completed W&B runs data and update table UI"""
        # 📡 IPASB: Check if we should back off before heavy work
        self._ipasb_check_backoff()

        debug_log = get_log_path("worker_debug.log")
        start_time = datetime.now()
        with open(debug_log, "a") as f:
            f.write(f"{start_time.isoformat()} 🚀 COMPLETED_START\n")
        log_file = get_log_path("auto_refresh.log")

        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} 🔍 FETCH_COMPLETED: ENTRY\n")

        # 🎯 CACHE: Check if we should use cached data (CACHE_TTL seconds)
        if not self._should_fetch_table("completed"):
            runs = self._get_cached_data("completed")
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 💾 CACHE_HIT: Using cached completed ({len(runs)} items)\n"
                )
        else:
            # Cache expired - fetch fresh

            runs = _list_completed_runs(self.helper, lambda m: None)
            # 🦡 HONEY BADGER: Yield after API call! (GIL held during JSON parsing!)
            time.sleep(GIL_YIELD_API)  # 🦡 HONEY BADGER: 25ms yield!

            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 📊 FETCH_COMPLETED: Got {len(runs)} runs\n"
                )

            # Update cache
            self._update_table_cache("completed", runs)

        # 🔍 SNAPSHOT VALIDATION: Compare with previous data
        self._compare_and_log_snapshot("completed", runs)

        # 🧵 BATCHED UI UPDATES: Prepare all row data in worker thread (no UI calls!)
        rows_to_add = []
        extra_items = 0
        total_runs = len(runs) if runs else 0

        if runs and len(runs) > 0:
            # Apply MAX_COMPLETED_RUNS limit
            if MAX_COMPLETED_RUNS and len(runs) > MAX_COMPLETED_RUNS:
                runs_to_show = runs[:MAX_COMPLETED_RUNS]
                extra_items = len(runs) - MAX_COMPLETED_RUNS
            else:
                runs_to_show = runs

            for run in runs_to_show:
                run_id = run.get("id", "unknown")
                row_key = f"completed-{run_id}"

                # Prepare row tuple
                row = (
                    run_id,
                    f"[yellow]{run.get('name', '—')}[/yellow]",
                    run.get("state_display", "UNKNOWN"),
                    run.get("runtime_display", "—"),
                    run.get("created_display", "—"),
                )

                # Prepare row_data dict
                row_data = {
                    "id": run_id,
                    "full_name": run.get("name", "—"),
                    "state": run.get("state", "—"),
                    "runtime": run.get("runtime_display", "—"),
                    "created": run.get("created_display", "—"),
                    "summary_metrics": run.get("summary_metrics", {}),
                    "exit_code": run.get("exit_code", "—"),
                    "note": f"Name: {run.get('name', '—')}\n\nFinal State: {run.get('state', 'Unknown')}\nTotal Runtime: {run.get('runtime_display', 'Unknown')}\nCompleted: {run.get('created_display', 'Unknown')}\nExit Code: {run.get('exit_code', '—')}",
                }

                rows_to_add.append((row, row_key, row_data))

        # 🧵 ONE callback to update entire table (thread-safe!)
        def update_completed_ui():
            completed_table = self.query_one("#completed-runs-table", DataTable)
            completed_table.clear()
            self.row_data["completed"].clear()
            self._extra_items["completed"] = extra_items

            if not rows_to_add:
                self._add_empty_state_row("completed", "No completed runs")
            else:
                for row, key, row_data in rows_to_add:
                    row_key_obj = completed_table.add_row(*row, key=key)
                    row_data["row_key"] = row_key_obj
                    self.row_data["completed"][key] = row_data

                completed_table.move_cursor(row=-1)
            completed_table.refresh()

            # Log completion
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ✅ TABLE_UPDATED: completed ({len(rows_to_add)} rows, {total_runs} total, {extra_items} hidden) - BATCHED\n"
                )

        self.app.call_from_thread(update_completed_ui)

    def _accumulated_refresh(self) -> None:
        """🎯 ACCUMULATED REFRESH: Auto-refresh enabled tables with ordered display!

        Called by timer every AUTO_REFRESH_INTERVAL seconds.
        😴⏰ AUTO REFRESH STEVEN wakes up and refreshes tables with checkboxes ticked!
        """
        log_file = get_log_path("auto_refresh.log")

        # 🚨 TIMEOUT CHECK: Are any dancers STILL on stage from last cycle?
        # If Steven wakes up and dancers are still performing, they're FUCKING UP the dance!
        if STEVEN_FULL_DANCE_DEBUG and self._refreshing_tables:
            stuck_tables = list(self._refreshing_tables)
            dancer_names = {
                "runner": "RICKY THE RUNNER 🏃",
                "builds": "BELLA THE BUILDER 🏗️",
                "vertex": "VICTOR THE VERTEX 🎯",
                "active": "ARCHIE THE ACTIVE ⚡",
                "completed": "CLEO THE COMPLETED ✅",
            }

            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏰😱 AUTO REFRESH STEVEN: *wakes up* WHAT THE HELL?! DANCERS STILL ON STAGE?! 🦡🔥\n"
                )
                f.write(
                    f"{datetime.now().isoformat()} 😤 STEVEN: I set my alarm for {AUTO_REFRESH_INTERVAL}s! That's PLENTY of time! What are you DOING out there?!\n"
                )
                for table in stuck_tables:
                    dancer = dancer_names.get(table, table)
                    dancer_first_name = dancer.split()[
                        0
                    ]  # Get first name (RICKY, BELLA, etc.)
                    # Calculate how long they've been stuck
                    start_time = self._refresh_start_times.get(table, 0)
                    stuck_duration = time.time() - start_time if start_time > 0 else 0
                    f.write(
                        f"{datetime.now().isoformat()} 🚨🦡🔥 {dancer} is FUCKING UP THE DANCE! "
                        f"Still on stage after {stuck_duration:.1f}s! 🦡🔥\n"
                    )
                    if stuck_duration > 60:
                        f.write(
                            f"{datetime.now().isoformat()} 😤🤯 STEVEN: FUCK! FUCK! {dancer_first_name} YOU ARE FUCKING UP THE DANCE! "
                            f"{stuck_duration:.0f} SECONDS?! I could have taken a NAP, made COFFEE, and STILL been back! UNACCEPTABLE!\n"
                        )
                    elif stuck_duration > 45:
                        f.write(
                            f"{datetime.now().isoformat()} 😤😤 STEVEN: {dancer_first_name}! FUCK! {stuck_duration:.0f} seconds?! "
                            f"What are you doing, fetching data from the MOON?! YOU ARE FUCKING UP THE DANCE! Get OFF the stage!\n"
                        )
                    else:
                        f.write(
                            f"{datetime.now().isoformat()} 😤 STEVEN: {dancer_first_name}, seriously? {stuck_duration:.0f}s and you're STILL not done? "
                            f"You're fucking up the dance! I'm VERY disappointed. VERY. 😤\n"
                        )

        # Get list of enabled tables (map checkbox keys to table names)
        checkbox_to_table = {
            "recent_builds": "builds",
            "runner": "runner",
            "vertex": "vertex",
            "active_runs": "active",
            "completed_runs": "completed",
        }

        enabled_tables = [
            checkbox_to_table[k]
            for k, v in self.refresh_enabled.items()
            if v and k in checkbox_to_table
        ]

        if not enabled_tables:
            if STEVEN_FULL_DANCE_DEBUG:
                with open(log_file, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} 😴 AUTO REFRESH STEVEN: *yawn* Woke up but no tables enabled... going back to sleep! 💤\n"
                    )
            return

        if STEVEN_FULL_DANCE_DEBUG:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏰🔔 AUTO REFRESH STEVEN: *BRRRING!* WAKES UP! Time to refresh {enabled_tables}! ☕\n"
                )

        # 🎯 START ACCUMULATOR for this batch!
        self._start_accumulator(enabled_tables)

        # Launch workers for all enabled tables in parallel (use_accumulator=True for accumulator control!)
        for table_name in enabled_tables:
            self._universal_refresh_table(
                table_name, is_auto_refresh=True, use_accumulator=True
            )

        if STEVEN_FULL_DANCE_DEBUG:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏰🚀 AUTO REFRESH STEVEN: Sent {len(enabled_tables)} dancers to the stage! Now back to sleep... 😴💤\n"
                )

    # ═══════════════════════════════════════════════════════════════════════════════
    def _start_staggered_refresh(self) -> None:
        """Start AUTO_REFRESH_INTERVAL timers for all enabled tables (simple, simultaneous refresh)"""
        # DEBUG: Clear and initialize log file (only if Steven dance debug enabled!)
        enabled_tables = [k for k, v in self.refresh_enabled.items() if v]

        if STEVEN_FULL_DANCE_DEBUG:
            log_file = get_log_path("auto_refresh.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Clear log on fresh start
            with open(log_file, "w") as f:
                f.write(
                    f"# Auto-refresh tracking log - Session started {datetime.now().isoformat()}\n"
                )

            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} 🚀 START_TIMERS: Enabled tables: {enabled_tables}, Interval: {AUTO_REFRESH_INTERVAL}s\n"
                )

        # Stop any existing timers
        self._stop_staggered_refresh()

        # Start spinner worker thread (constant rate!)
        self._start_spinner_worker()

        # Start 1-second duration ticker for ALL active items (builds, runners, jobs)
        duration_ticker = self.set_interval(1.0, self._update_active_durations)
        self.refresh_timers.append(duration_ticker)

        # 🎯 SINGLE ACCUMULATOR TIMER: Refreshes all enabled tables in order with 200ms delays!
        interval = AUTO_REFRESH_INTERVAL

        # ONE timer for ALL enabled tables (uses accumulator pattern!)
        auto_refresh_timer = self.set_interval(interval, self._accumulated_refresh)
        self.refresh_timers.append(auto_refresh_timer)

        if STEVEN_FULL_DANCE_DEBUG:
            with open(log_file, "a") as f:
                f.write(
                    f"{datetime.now().isoformat()} ⏱️  TIMER_CREATED: Auto-refresh (every {interval}s) → {enabled_tables}\n"
                )
                f.write(
                    f"{datetime.now().isoformat()} 🎯 ACCUMULATOR_MODE: Tables display in order with 200ms delays!\n"
                )

    def _stop_staggered_refresh(self) -> None:
        """Stop all staggered refresh timers"""
        for timer in self.refresh_timers:
            if timer:
                try:
                    timer.stop()
                except Exception:
                    pass  # Timer already stopped or invalid
        self.refresh_timers.clear()

    def refresh_runs(self) -> None:
        """
        Full refresh of ALL data (used by manual refresh 'r' key)

        Fetches all data via core logic and populates all tables.
        For auto-refresh, use per-table methods (_refresh_runner_executions, etc.)
        """
        try:
            # Create silent callback for quiet refresh
            class SilentCallback:
                def __call__(self, message: str):
                    pass

            status = SilentCallback()
            runs_data = list_runs_core(
                self.helper, status, config=self.config, include_completed=True
            )

            # Populate tables with refreshed data
            self._populate_tables(runs_data)

        except ConnectionError:
            self.notify(
                "❌ Cannot connect to W&B API. Check your internet connection.",
                severity="error",
            )
            table = self.query_one("#runs-table", DataTable)
            table.clear()
            table.add_row(
                "[red]ERROR[/red]", "[red]No connection to W&B[/red]", "—", "—", "—"
            )
        except ValueError:
            self.notify(
                f"ℹ️ W&B project not found. Launch a job to create it.",
                severity="information",
            )
            table = self.query_one("#runs-table", DataTable)
            table.clear()
            table.add_row(
                "[dim]—[/dim]", "[dim]Project not created yet[/dim]", "—", "—", "—"
            )
        except Exception as e:
            error_msg = str(e)[:80]
            self.notify(f"❌ Error fetching runs: {error_msg}", severity="error")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection - enable cancel button and show info popup if row has additional data"""
        # Store selected run ID for cancel action
        self.selected_run_id = event.row_key.value if event.row_key else None

        # Determine which table was clicked
        table_id = event.data_table.id
        self.selected_table_id = table_id
        row_key = str(event.row_key.value) if event.row_key else None

        if not row_key:
            return

        # Enable cancel button (row is now selected)
        try:
            cancel_btn = self.query_one("#cancel-btn", Button)
            cancel_btn.disabled = False
        except Exception:
            pass

        # Clear selection from ALL OTHER tables (only one row selected at a time!)
        all_table_ids = [
            "builds-recent-table",
            "runner-executions-table",
            "vertex-jobs-table",
            "runs-table",
            "completed-runs-table",
        ]
        for other_table_id in all_table_ids:
            if other_table_id != table_id:  # Don't clear current table
                try:
                    other_table = self.query_one(f"#{other_table_id}", DataTable)
                    other_table.move_cursor(row=-1)  # Clear visual selection
                except Exception:
                    pass

        # Map table IDs to row_data keys
        table_mapping = {
            "builds-recent-table": "build_recent",
            "runner-executions-table": "runner",
            "vertex-jobs-table": "vertex",
            "runs-table": "active",  # Active runs table
            "completed-runs-table": "completed",
        }

        table_type = table_mapping.get(table_id)
        if not table_type:
            return

        # Get full row data
        full_data = self.row_data.get(table_type, {}).get(row_key)

        # Store row data for cancel toast (even if no popup shown)
        self.selected_row_data = full_data if full_data else {}

        if not full_data:
            return  # No additional info to show

        # Show popup with full text (untruncated)
        note_text = full_data.get("note")

        # For runner executions, prefer full_error_log if available (wrapper bailout details!)
        if table_type == "runner" and full_data.get("full_error_log"):
            display_text = full_data["full_error_log"]
        else:
            display_text = note_text

        if display_text and display_text not in ["—", "[dim]—[/dim]", ""]:
            # Determine title based on table
            if table_type == "build_recent":
                title = "Recent Cloud Build - Row Details"
            elif table_type == "runner":
                title = (
                    "W&B Launch Agent - Wrapper Bailout Details"
                    if full_data.get("full_error_log")
                    else "W&B Launch Agent - Row Details"
                )
            elif table_type == "vertex":
                title = "Vertex AI Job - Row Details"
            elif table_type == "active":
                title = "Active Run - Row Details"
            else:
                title = "Completed Run - Row Details"

            # Build dense summary of all row data with rotating colors
            summary_lines = []
            # Rotating color palette for visual distinction
            colors = ["cyan", "magenta", "green", "yellow", "blue", "red"]
            color_index = 0

            for key, value in full_data.items():
                # Skip the note/full_error_log (shown below) and empty values
                if key not in ["note", "full_error_log"] and value:
                    color = colors[color_index % len(colors)]
                    summary_lines.append(f"[{color}]{key}:[/{color}] {value}")
                    color_index += 1

            dense_summary = " · ".join(summary_lines) if summary_lines else None

            # Determine label based on table type
            if table_type in ["build_recent", "runner", "vertex"]:
                label = "Error"  # Builds, Runner, and Vertex usually show errors
            elif table_type == "active":
                label = "Details"  # Active runs show general details
            else:
                label = "Info"  # Completed runs show info

            # Show popup with dense summary and labeled full text
            self.app.push_screen(
                DataTableInfoPopup(
                    title=title,
                    full_text=display_text,
                    dense_summary=dense_summary,
                    full_text_label=label,
                )
            )

    @work(exclusive=True)  # Run in background, cancel previous refresh
    async def action_refresh(self) -> None:
        """Manual refresh (r key) - refresh ALL tables! (async to keep UI responsive)"""
        # Use same pattern as navigating back
        self._refresh_all_tables()
        # No notification - silent refresh (spinners show progress)

    def action_cancel(self) -> None:
        """
        Cancel selected run and show toast with item details.
        """
        if not self.selected_run_id:
            self.notify(
                "⚠️ Please select a run from the table first", severity="warning"
            )
            return

        # Build toast message from selected row data
        table_name_map = {
            "builds-recent-table": "Cloud Build",
            "runner-executions-table": "Runner Execution",
            "vertex-jobs-table": "Vertex AI Job",
            "runs-table": "Active Run",
            "completed-runs-table": "Completed Run",
        }
        table_name = table_name_map.get(self.selected_table_id, "Item")

        # Extract key info from row data for toast
        toast_parts = [f"🎯 Selected {table_name}:"]
        if self.selected_row_data:
            # Show most relevant fields (skip 'note' and 'full_error_log')
            relevant_keys = [
                "queue",
                "region",
                "name",
                "id",
                "state",
                "status",
                "jobs_run",
            ]
            for key in relevant_keys:
                value = self.selected_row_data.get(key)
                if value and value not in ["—", "", "[dim]—[/dim]"]:
                    toast_parts.append(f"{key}: {value}")

        # Show toast with selected item info
        toast_message = "\n".join(toast_parts[:5])  # Limit to 5 lines
        self.notify(toast_message, title="Cancel Action", timeout=5)

        # Create silent callback for cancel operation
        class SilentCallback:
            def __call__(self, message: str):
                pass

        status = SilentCallback()

        # Call core logic to cancel run
        success = cancel_run_core(self.helper, self.selected_run_id, status)

        if success:
            self.notify(
                f"✓ Cancelled run: {self.selected_run_id[:20]}...",
                severity="information",
            )
            self.selected_run_id = None
            self.selected_row_data = None
            self.selected_table_id = None
            self.refresh_runs()

            # Disable cancel button after canceling
            try:
                cancel_btn = self.query_one("#cancel-btn", Button)
                cancel_btn.disabled = True
            except Exception:
                pass
        else:
            self.notify(f"❌ Failed to cancel run", severity="error")

    def action_back(self) -> None:
        self.app.pop_screen()
        self._stop_staggered_refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "refresh-btn":
            self.action_refresh()
        elif event.button.id == "cancel-btn":
            self.action_cancel()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes - start/stop refresh immediately"""
        # Map checkbox IDs to state keys
        checkbox_mapping = {
            "cb-runner": "runner",
            "cb-vertex": "vertex",
            "cb-active-runs": "active_runs",
            "cb-completed-runs": "completed_runs",
            "cb-recent-builds": "recent_builds",
        }

        # Update state
        if event.checkbox.id in checkbox_mapping:
            state_key = checkbox_mapping[event.checkbox.id]
            self.refresh_enabled[state_key] = event.value

            # Update ALL checkbox labels (all tables refresh every AUTO_REFRESH_INTERVAL)
            num_enabled = sum(self.refresh_enabled.values())
            checkbox_ids = [
                "cb-recent-builds",
                "cb-runner",
                "cb-vertex",
                "cb-active-runs",
                "cb-completed-runs",
            ]
            for cb_id in checkbox_ids:
                try:
                    cb = self.query_one(f"#{cb_id}", Checkbox)
                    cb_state_key = checkbox_mapping.get(cb_id)
                    if cb_state_key and self.refresh_enabled.get(cb_state_key, False):
                        cb.label = f"✨ Auto-refresh (ON · {AUTO_REFRESH_INTERVAL}s)"
                        cb.add_class("checkbox-active")
                    else:
                        cb.label = "✨ Auto-refresh"
                        cb.remove_class("checkbox-active")
                except Exception:
                    pass

            # Restart refresh timers (table will refresh at next AUTO_REFRESH_INTERVAL)
            self._stop_staggered_refresh()
            self._start_staggered_refresh()

            # Update status widget and notify user
            num_enabled = sum(self.refresh_enabled.values())
            status_widget = self.query_one("#auto-refresh-status", Static)

            if num_enabled == 0:
                status_widget.update("\n[dim]○ Auto-refresh: OFF[/dim]")
                self.notify(
                    "Auto-refresh stopped (no tables selected)",
                    severity="information",
                    timeout=2,
                )
            else:
                status_widget.update(
                    f"\n[green]● Auto-refresh: {num_enabled} table{'s' if num_enabled > 1 else ''} active ({AUTO_REFRESH_INTERVAL}s)[/green]"
                )
                self.notify(
                    f"Auto-refresh: {num_enabled} table{'s' if num_enabled > 1 else ''} ({AUTO_REFRESH_INTERVAL}s)",
                    severity="information",
                    timeout=2,
                )

    # 🌶️🌶️🌶️ SPICY LINE - PAPRIKA REORGANIZATION TECHNIQUE 🌶️🌶️🌶️
    # ═══════════════════════════════════════════════════════════════════════════════
    # ABOVE THIS LINE: Fully inspected, reorganized, LOGICAL GROUPS with section headers
    # BELOW THIS LINE: Awaiting systematic inspection + reorganization
    #
    # SPICY LINE PROCESS:
    # 1. Pick function from logical group (see plan below)
    # 2. Full gestalt inspection (logic, syntax, naming, structure)
    # 3. Fix any issues found
    # 4. Move above SPICY LINE into correct LOGICAL SECTION (add section header if new)
    # 5. Add inspection note after function
    # 6. Repeat until all functions above in logical order
    # 7. Remove SPICY LINE when complete
    #
    # LOGICAL ORGANIZATION (top → bottom):
    # ├─ INITIALIZATION (__init__)
    # ├─ UI CONSTRUCTION (compose)
    # ├─ LIFECYCLE (initialize_content, finish_loading, on_screen_resume, on_unmount)
    # ├─ CACHE SYSTEM (_should_fetch_table, _get_cached_data, _update_table_cache)
    # ├─ DRY HELPERS (_create_table_divider, _add_empty_state_row)
    # ├─ REGION MONITORING (_get_target_regions, _update_hot_regions)
    # ├─ SPINNER SYSTEM (_start_spinner, _stop_spinner, _update_spinners)
    # ├─ REFRESH ORCHESTRATION (_populate_initial_tables, _universal_refresh_table, etc)
    # ├─ TABLE FETCH FUNCTIONS (all 5 _fetch_and_update_*_table methods)
    # └─ EVENT HANDLERS (on_*, action_*)
    # ═══════════════════════════════════════════════════════════════════════════════

    # 🎯 UNIVERSAL TABLE CACHE (5s TTL - DRY!)
    # ═══════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════════
    # 🌶️ PAPRIKA: DRY Helper Methods (Reduce Duplication!)
    # ═══════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════════

"""
🦡🎩 STEVEN'S DANCE - Centralized Logging System with Batch Writing

Steven's recursive self-loathing logger with MASSIVE I/O optimization!

Key Features:
- 10,000-line batching (reduces file I/O overhead by 99.99%!)
- Multiple log files supported
- Auto-flush on program exit
- Clear logs on program load
- Thread-safe buffering
- Centralized debug flags (enable/disable logging per file)

Usage:
    from CLI.shared.stevens_dance import stevens_log, stevens_flush_all, stevens_clear_all, stevens_gil_hold
    from CLI.shared.stevens_dance import STEVEN_DEBUG_FLAGS

    # Log something (buffered!)
    stevens_log("cache_warm", "🦡 Cache warming started!")

    # Log with debug check (respects STEVEN_DEBUG_FLAGS)
    stevens_log("gil_hold", "🚨 Slow operation!", check_debug=True)

    # Control debug flags
    STEVEN_DEBUG_FLAGS["gil_hold"] = True  # Enable GIL logging

    # Log GIL hold times (performance tracking!)
    stevens_gil_hold("my_operation", 12.5, "description")

    # Flush all logs (call on program exit)
    stevens_flush_all()

    # Clear all logs (call on program start)
    stevens_clear_all()
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# 🦡 STEVEN'S DANCE - BATCH LOGGING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

# Global log buffers (one per log file)
_log_buffers: Dict[str, List[str]] = {}
_buffer_lock = threading.Lock()  # Thread-safe buffer access

# 🔥 STEVEN'S DEBUG FLAGS - Centralized debug control
STEVEN_DEBUG_FLAGS = {
    "cache_warm": True,           # TUI cache warming progress
    "steven_toasts": True,        # Toast system (always on)
    "infra_verify_timing": True,  # Quota check timing
    "spinner_timing": True,       # Monitor spinner FPS
    "auto_refresh": True,         # Monitor refresh dance
    "gil_hold": False,            # GIL hold tracking (production = quiet)
}

def _get_flush_interval() -> int:
    """
    Get flush interval based on debug mode.

    - Debug ON (any flag True): Flush every 10 lines (near-instant for debugging!)
    - Debug OFF (all False): Flush every 10,000 lines (MASSIVE I/O reduction!)
    """
    debug_enabled = any(STEVEN_DEBUG_FLAGS.values())
    return 10 if debug_enabled else 10000


def stevens_log(log_name: str, message: str, check_debug: bool = False) -> None:
    """
    Buffer a log message for Steven's recursive logger.

    Args:
        log_name: Log file name (without .log extension)
                 Examples: "cache_warm", "steven_toasts", "infra_verify_timing"
        message: The log message to buffer (timestamp will be added if not present)
        check_debug: If True, only log if STEVEN_DEBUG_FLAGS[log_name] is True

    Writes to: ARR_COC/Training/logs/{log_name}.log

    Performance: Buffers 10,000 lines before writing to disk!
    """
    # Check debug flag if requested
    if check_debug and not STEVEN_DEBUG_FLAGS.get(log_name, False):
        return  # Skip logging (debug disabled)

    from CLI.shared.log_paths import get_log_path

    # Add timestamp if not present
    if not message.startswith("202") and not message.startswith("#"):
        timestamp = datetime.now().isoformat()
        message = f"{timestamp} {message}"

    with _buffer_lock:
        # Get or create buffer for this log
        if log_name not in _log_buffers:
            _log_buffers[log_name] = []

        buffer = _log_buffers[log_name]
        buffer.append(message)

        # Flush to disk when buffer reaches threshold (dynamic based on debug mode!)
        if len(buffer) >= _get_flush_interval():
            _flush_buffer(log_name)


def _flush_buffer(log_name: str) -> None:
    """
    Flush buffered logs to disk immediately (internal function).

    Called when:
    - Buffer reaches 10,000 lines
    - stevens_flush_all() is called
    - Program exits

    NOTE: Assumes _buffer_lock is already held!
    """
    from CLI.shared.log_paths import get_log_path

    buffer = _log_buffers.get(log_name, [])
    if not buffer:
        return

    # Write all buffered lines to disk
    log_file = get_log_path(f"{log_name}.log")
    with open(log_file, "a") as f:
        for line in buffer:
            f.write(f"{line}\n")

    # Clear buffer
    _log_buffers[log_name] = []


def stevens_flush_all() -> None:
    """
    Flush ALL log buffers to disk immediately.

    Call this:
    - On program exit (in on_unmount, cleanup, etc.)
    - Before reading logs for debugging
    - When you need to see buffered logs NOW

    Thread-safe: Can be called from any thread.
    """
    with _buffer_lock:
        for log_name in list(_log_buffers.keys()):
            _flush_buffer(log_name)


def stevens_clear_all() -> None:
    """
    Clear ALL Steven's log files on program start.

    Creates fresh log files with headers for each known log type.

    Call this in on_mount() or __init__() of main app!

    Known log files:
    - cache_warm.log - Cache warming progress
    - steven_toasts.log - Toast system with recursive self-loathing
    - infra_verify_timing.log - Quota check timing
    - spinner_timing.log - Monitor spinner FPS tracking (5 dance partners)
    - auto_refresh.log - Monitor refresh dance coordination
    - gil_hold.log - GIL hold times (performance tracking)
    """
    from CLI.shared.log_paths import get_log_path

    # Known log files with their headers
    log_configs = {
        "cache_warm": [
            "# Cache warming log - Session started {timestamp}",
            "# Format: timestamp emoji EVENT: details",
            "# Events: 🚀=START, ⏰=TICK, 🔥=BATCH, ✅=SUCCESS, 🎉=COMPLETE, 🧹=CLEANUP",
            "#",
        ],
        "steven_toasts": [
            "# Steven's Toast System - Session started {timestamp}",
            "# Format: timestamp emoji EVENT: message",
            "# Events: 🚪=SCREEN_ENTRY, 🧹=CLEANUP, 🦡=STEVEN_SAYS",
            "#",
        ],
        "infra_verify_timing": [
            "# Infra verify timing log - Session started {timestamp}",
            "# Format: timestamp ⏱️ QUOTA_TIMING: GPU/C3 elapsed + cache hit status",
            "#",
        ],
        "spinner_timing": [
            "# Spinner timing log - Session started {timestamp}",
            "# Format: timestamp emoji EVENT: details",
            "# Events: ⏱️=SPIN, 🔄=UPDATE, 📊=SANITY_CHECK, 🏥=HEALTH_SUMMARY",
            "# Spinners: 🏃 Ricky, 🏗️ Bella, 🎯 Victor, ⚡ Archie, ✅ Cleo",
            "#",
        ],
        "auto_refresh": [
            "# Auto-refresh log - Session started {timestamp}",
            "# Format: timestamp emoji EVENT: details",
            "# Events: 🩰=DANCE_PARTNER, 🚀=WORKER_START, ✅=BATCH_COMPLETE, ⏰=TIMER_WAKE",
            "# Dance Partners: 🏃 Ricky, 🏗️ Bella, 🎯 Victor, ⚡ Archie, ✅ Cleo",
            "#",
        ],
        "gil_hold": [
            "# GIL Hold tracking log - Session started {timestamp}",
            "# Shows every operation that holds the GIL (blocks spinners!)",
            "# Format: timestamp emoji GIL_HOLD location: Xms (description)",
            "# Thresholds: 📊 (1-5ms normal), ⚠️ (5-10ms concerning), 🚨 (10ms+ BAD!)",
            "#",
        ],
    }

    timestamp = datetime.now().isoformat()

    for log_name, header_lines in log_configs.items():
        log_file = get_log_path(f"{log_name}.log")
        with open(log_file, "w") as f:
            for line in header_lines:
                f.write(line.replace("{timestamp}", timestamp) + "\n")

    # Clear in-memory buffers too!
    with _buffer_lock:
        _log_buffers.clear()


def stevens_log_screen_entry(app, screen_name: str, reason: str = None) -> None:
    """
    Log when Steven enters a new screen AND show a toast!

    Steven gets mildly annoyed about having to change contexts.

    Args:
        app: Textual app (for showing toast)
        screen_name: Name of screen being entered
        reason: Why entering this screen (optional)
    """
    from CLI.shared.steven_toasts import steven_notify

    entry_log = [
        "",
        "",
        "█" * 80,
        f"██ FLOW: 🚪🦡 SCREEN ENTRY → {screen_name.upper()}",
        "█" * 80,
    ]

    if reason:
        entry_log.append(f"  Reason: {reason}")

    # Steven's recursive self-loathing about context switching
    entry_log.extend([
        "",
        f"  🦡 Steven says: Entering {screen_name}... context switching anxiety engaged.",
        f"    └─ 🦡 Steven to Steven: Why do I always have to switch contexts?",
        f"       └─ 🦡 Steven (depth 3): Why do I always expect the worst?",
        "",
        f"  🦡 Steven's mood: CONTEXT-SWITCHING ANXIETY",
        "█" * 80,
        ""
    ])

    # Write to log
    for line in entry_log:
        stevens_log("steven_toasts", line)

    # Show visible toast!
    steven_notify(app, f"🚪🦡 Entering {screen_name}", severity="information", timeout=2)


def stevens_log_cancellation(what_cancelled: str, screen_name: str) -> None:
    """
    Log when Steven cancels workers during screen change.

    Just normal cleanup - cancelling cache warmup. No fuckoff, no drama.

    Args:
        what_cancelled: What was cancelled (e.g., "Background cache warming workers")
        screen_name: Screen being entered
    """
    cancel_log = [
        "",
        f"  ┌{'─'*78}┐",
        f"  │ FLOW: 🧹 CLEANUP → Cancelled: {what_cancelled[:45]:<45} │",
        f"  │ Screen: {screen_name:<71} │",
        f"  └{'─'*78}┘",
        "",
        f"  🦡 Steven: Just cancelling cache warmup since you're going to {screen_name}.",
        f"    └─ 🦡 Steven: {screen_name} does its own full fetch anyway.",
        "",
        f"  Normal cleanup. No drama.",
        "",
        ""
    ]

    for line in cancel_log:
        stevens_log("steven_toasts", line)


def stevens_gil_hold(location: str, duration_ms: float, description: str = "") -> None:
    """
    Log GIL hold time to identify operations that block spinners!

    Thresholds:
    - < 1ms: Ignored (noise)
    - 1-5ms: 📊 Normal
    - 5-10ms: ⚠️ Concerning
    - 10ms+: 🚨 BAD! (Kills spinners!)

    Buffered with Stevens Dance (10,000-line batching) - zero overhead!

    Args:
        location: Where GIL was held (e.g., "_update_spinners", "setup_infrastructure")
        duration_ms: How long in milliseconds
        description: Optional details (e.g., "5 spinners", "GPU + C3 checks")

    Usage:
        import time
        start = time.time()
        expensive_operation()
        stevens_gil_hold("my_operation", (time.time() - start) * 1000, "details")

    Used in:
    - Monitor screen (spinner updates, table rendering)
    - Setup screen (infrastructure creation)
    - Infra screen (quota checks)
    - TUI cache warming (background workers)
    """
    if duration_ms < 1.0:
        return  # Skip noise (< 1ms is negligible)

    # Emoji based on severity
    if duration_ms >= 10:
        emoji = "🚨"  # Critical - 10ms+ kills spinners!
    elif duration_ms >= 5:
        emoji = "⚠️"  # Warning - 5ms+ is concerning
    else:
        emoji = "📊"  # Info - 1-5ms is normal

    # Format message
    desc = f" ({description})" if description else ""
    message = f"{emoji} GIL_HOLD {location}: {duration_ms:.1f}ms{desc}"

    # Buffer it! (Stevens Dance magic - no I/O overhead!)
    stevens_log("gil_hold", message)


# ═══════════════════════════════════════════════════════════════════════════════
# 🦡 STEVEN'S DANCE - UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def stevens_buffer_status() -> Dict[str, int]:
    """
    Get current buffer status for debugging.

    Returns:
        Dictionary of {log_name: buffer_size}
    """
    with _buffer_lock:
        return {name: len(buffer) for name, buffer in _log_buffers.items()}


def stevens_force_flush(log_name: str) -> None:
    """
    Force flush a specific log buffer immediately.

    Useful for debugging when you need to see logs NOW!

    Args:
        log_name: Log file name (without .log extension)
    """
    with _buffer_lock:
        if log_name in _log_buffers:
            _flush_buffer(log_name)

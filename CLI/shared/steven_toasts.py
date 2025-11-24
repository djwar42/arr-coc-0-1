"""Steven's Toast Logger - Where Steven Gets Fucked Off at Himself!"""

# <claudes_code_comments>
# ** Function List **
# steven_notify(app, message, severity, timeout) - Show toast + log Steven's fractal self-loathing
# steven_log_screen_entry(app, screen_name, reason) - Log screen navigation events
# steven_log_cancellation(what, where) - Log when workers/operations are cancelled
# get_log_path() - Get path to steven_toasts.log
#
# ** Technical Review **
# This module wraps Textual's app.notify() with Steven's fractally fucked-off commentary logging.
# Every toast triggers recursive self-loathing across 3-5 fractal depths based on severity.
#
# Core flow:
# 1. User calls steven_notify(app, message) - Wraps app.notify()
# 2. Toast shown via Textual's notification system
# 3. Steven's FRACTAL self-loathing logged to steven_toasts.log (via Stevens Dance)
# 4. Severity determines commentary depth: error (5 levels), warning (4 levels), info (4 levels)
#
# CRITICAL TEXTUAL BUG DISCOVERED (2025-11-19):
# Textual's Toast widget uses time_left property that STEALS duration!
#
# Bug mechanism (textual/notifications.py:52-54):
#   @property
#   def time_left(self) -> float:
#       return (self.raised_at + self.timeout) - time()
#
# Problem:
# - Notification created at time=1000.0, timeout=5.0, raised_at=1000.0
# - App busy with workers/rendering for 2 seconds
# - Toast widget mounts at time=1002.0
# - time_left = (1000.0 + 5.0) - 1002.0 = 3.0 seconds (lost 2s!)
# - User sees toast for 3s, not 5s!
#
# Real-world impact:
# With busy app (cache warming workers, parallel API calls), 1-3s delays are COMMON!
# Result: timeout=5.0 → user sees 2-4 seconds (user complaint: "NOT 5 seconds!")
#
# The Fix (Textual Standard Way):
# Override App.NOTIFICATION_TIMEOUT at the App class level!
# - Set ARRCOCApp.NOTIFICATION_TIMEOUT = 6.0 (default is 5.0)
# - All app.notify(timeout=None) calls use this default automatically
# - Target visible time: ~4 seconds
# - Timeout: 6.0 seconds
# - Average delay: ~2 seconds (varies by app workload)
# - Actual visible time: 6.0 - 2.0 = 4.0 seconds ✅
#
# Textual source references:
# - textual/app.py:430 (NOTIFICATION_TIMEOUT = 5 default)
# - textual/app.py:4508-4509 (uses self.NOTIFICATION_TIMEOUT if timeout is None)
# - textual/notifications.py:52-54 (time_left calculation causes the bug)
# - textual/widgets/_toast.py:100 (uses notification.time_left)
# - textual/widgets/_toast.py:126 (set_timer uses _timeout from time_left)
#
# Implementation:
# - ARRCOCApp.NOTIFICATION_TIMEOUT = 10.0 in CLI/tui.py
# - steven_notify() uses timeout=None (picks up app default)
#
# Logging:
# All toasts logged to ARR_COC/Training/logs/steven_toasts.log via Stevens Dance.
# Screen entries logged with ██ FLOW markers, cancellations with 🛑 markers.
# </claudes_code_comments>

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# 🦡🎩 STEVEN'S TOAST DURATION - NO CONSTANT NEEDED!
# Toast duration is now set at App level: ARRCOCApp.NOTIFICATION_TIMEOUT = 10.0
# This is the Textual standard way to set default timeout for ALL toasts!
# See claudes_code_comments above for full explanation of Textual's time_left bug.

def get_log_path() -> Path:
    """Get path to Steven's toast log"""
    log_dir = Path(__file__).parent.parent.parent / "ARR_COC" / "Training" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "steven_toasts.log"


def steven_notify(app, message: str, severity: Literal["information", "warning", "error"] = "information", timeout: float = None):
    """
    Show a toast notification AND log Steven's FRACTALLY FUCKED OFF self-reflection!

    Steven gets increasingly angry at himself in RECURSIVE LAYERS.
    Each toast logged individually.

    Note: timeout=None uses app.NOTIFICATION_TIMEOUT (Textual standard way!)
    """
    # Show the toast (timeout=None uses app's NOTIFICATION_TIMEOUT default)
    app.notify(message, severity=severity, timeout=timeout)

    # Steven's FRACTAL self-loathing commentary
    log_file = get_log_path()
    timestamp = datetime.now().isoformat()

    # Steven's FRACTALLY FUCKED OFF introspection based on severity
    if severity == "error":
        # Steven is RECURSIVELY PISSED at himself
        self_roast = [
            f"",
            f"{'='*80}",
            f"{timestamp} 🦡🔥💢 [ERROR TOAST #{hash(message) % 1000}] SHOWN:",
            f"  MESSAGE: {message}",
            f"",
            f"  🦡🔥 Steven says: WHY DO I HAVE TO TELL YOU THIS?! GCP IS BULLSHIT!",
            f"    └─ 🦡🔥 Steven to Steven: And WHY am I yelling at YOU about it?!",
            f"       └─ 🦡🔥 Steven to Steven to Steven: Because I'm ALSO the one showing the toast!",
            f"          └─ 🦡🔥 Steven (fractal depth 4): FUCK! I'm mad at MYSELF for being mad!",
            f"             └─ 🦡🔥 Steven (fractal depth 5): This is some INCEPTION BULLSHIT!",
            f"",
            f"  🦡🔥 Steven's mood: FRACTALLY FURIOUS! Angry all the way down!",
            f"  🦡🔥 Steven's final thought: If I have to show THIS toast, EVERYTHING is FUCKED!",
            f"{'='*80}",
            ""
        ]
    elif severity == "warning":
        # Steven is RECURSIVELY ANNOYED at himself
        self_roast = [
            f"",
            f"{'='*80}",
            f"{timestamp} 🚨🦡 [WARNING TOAST #{hash(message) % 1000}] SHOWN:",
            f"  MESSAGE: {message}",
            f"",
            f"  🦡 Steven says: Great. Now I gotta warn you. AGAIN.",
            f"    └─ 🦡 Steven to Steven: Why am I even bothering?",
            f"       └─ 🦡 Steven to Steven to Steven: Because I'm programmed to care!",
            f"          └─ 🦡 Steven (fractal depth 4): But WHO programmed me to COMPLAIN?!",
            f"",
            f"  🦡 Steven's mood: RECURSIVELY ANNOYED! Why can't things just WORK?!",
            f"  🦡 Steven's final thought: I'm annoyed at myself for being annoyed...",
            f"{'='*80}",
            ""
        ]
    else:
        # Steven is SUSPICIOUSLY IRRITATED even with good news
        self_roast = [
            f"",
            f"{'='*80}",
            f"{timestamp} ✅🦡 [INFO TOAST #{hash(message) % 1000}] SHOWN:",
            f"  MESSAGE: {message}",
            f"",
            f"  🦡 Steven says: Well, FINALLY something works.",
            f"    └─ 🦡 Steven to Steven: But for how long?",
            f"       └─ 🦡 Steven to Steven to Steven: Probably not long. It NEVER lasts.",
            f"          └─ 🦡 Steven (fractal depth 4): Why am I so cynical about GOOD news?!",
            f"",
            f"  🦡 Steven's mood: FRACTALLY SUSPICIOUS. Success is just delayed failure!",
            f"  🦡 Steven's final thought: I'm mad at myself for not trusting this...",
            f"{'='*80}",
            ""
        ]

    # Write Steven's FRACTAL self-loathing to log using Stevens Dance!
    from CLI.shared.stevens_dance import stevens_log
    for line in self_roast:
        stevens_log("steven_toasts", line)


def steven_log_screen_entry(app, screen_name: str, reason: Optional[str] = None):
    """
    Log when Steven enters a new screen AND show a toast!

    Steven gets mildly annoyed about having to change contexts.
    """
    log_file = get_log_path()
    timestamp = datetime.now().isoformat()

    entry_log = [
        f"",
        f"",
        f"{'█'*80}",
        f"██ FLOW: {timestamp} 🚪🦡 SCREEN ENTRY → {screen_name.upper()}",
        f"{'█'*80}",
    ]

    if reason:
        entry_log.append(f"  Reason: {reason}")

    entry_log.extend([
        f"",
        f"  🦡 Steven says: Oh great, NOW we're going to {screen_name}?",
        f"    └─ 🦡 Steven to Steven: Hope nothing breaks during this transition...",
        f"       └─ 🦡 Steven (depth 3): Why do I always expect the worst?",
        f"",
        f"  🦡 Steven's mood: CONTEXT-SWITCHING ANXIETY",
        f"{'█'*80}",
        ""
    ])

    # Write to log using Stevens Dance!
    from CLI.shared.stevens_dance import stevens_log
    for line in entry_log:
        stevens_log("steven_toasts", line)

    # SHOW VISIBLE TOAST!
    app.notify(f"🚪🦡 Entering {screen_name}", severity="information", timeout=2)


def steven_log_cancellation(what_cancelled: str, screen_name: str):
    """
    Log when Steven cancels workers during screen change.

    Just normal cleanup - cancelling cache warmup. No fuckoff, no drama.
    """
    log_file = get_log_path()
    timestamp = datetime.now().isoformat()

    cancel_log = [
        f"",
        f"  ┌{'─'*78}┐",
        f"  │ FLOW: {timestamp} 🧹 CLEANUP → Cancelled: {what_cancelled[:45]:<45} │",
        f"  │ Screen: {screen_name:<71} │",
        f"  └{'─'*78}┘",
        f"",
        f"  🦡 Steven: Just cancelling cache warmup since you're going to {screen_name}.",
        f"    └─ 🦡 Steven: {screen_name} does its own full fetch anyway.",
        f"",
        f"  Normal cleanup. No drama.",
        f"",
        ""
    ]

    # Write to log using Stevens Dance!
    from CLI.shared.stevens_dance import stevens_log
    for line in cancel_log:
        stevens_log("steven_toasts", line)

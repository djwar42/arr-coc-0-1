#!/usr/bin/env python3
"""
Automatic Performance Coupling Analyzer

Runs Vervaekean performance monitoring automatically and prints a simple,
focused exit summary when TUI closes. Captures what user experienced.

Usage:
    # Just run TUI normally - monitoring happens automatically!
    python CLI/tui.py

    # On exit, simple summary prints showing what happened
"""

import json
import sys
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import atexit


class AutoPerformanceReporter:
    """Automatic performance monitoring - prints simple exit summary"""

    def __init__(self):
        self.session_start = datetime.now()
        self.monitor = None
        self.report_generated = False  # Prevent duplicate reports

        # Session metrics (tracks what happened)
        from CLI.shared.session_metrics import SessionMetrics
        self.metrics = SessionMetrics()

    def setup(self):
        """Initialize monitoring (call at TUI startup)"""
        from CLI.shared.performance_monitor import get_monitor, reset_monitor

        # Reset monitor (fresh session)
        reset_monitor()
        self.monitor = get_monitor()

        # Register cleanup handlers
        atexit.register(self.generate_report)  # Normal exit

        # Handle Ctrl+C (SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Handle termination (SIGTERM)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"📊 Performance monitoring enabled")
        print(f"   Exit summary will print on TUI close")

    def _signal_handler(self, signum, frame):
        """Handle signals (Ctrl+C, SIGTERM) - generate report before exit"""
        signal_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else f"Signal {signum}"
        print(f"\n⚠️  Received {signal_name} - generating performance report...")

        # Generate report
        self.generate_report()

        # Re-raise to allow normal signal handling
        sys.exit(0)

    def generate_report(self):
        """Generate simple exit summary showing what user experienced"""
        # Prevent duplicate reports (if both signal handler and atexit trigger)
        if self.report_generated:
            return

        if not self.monitor:
            return

        self.report_generated = True

        # Get stats from session metrics
        stats = self.metrics.get_stats()

        # Print simple focused summary
        from CLI.shared.vervaekean_exit_summary import generate_exit_summary
        summary_text = generate_exit_summary(stats)
        print(summary_text)

    def get_metrics(self):
        """Get session metrics instance for recording events"""
        return self.metrics



# Global reporter instance
_global_reporter: AutoPerformanceReporter = None


def enable_auto_reporting():
    """Enable automatic performance reporting (call at TUI startup)"""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = AutoPerformanceReporter()
        _global_reporter.setup()


def get_reporter() -> AutoPerformanceReporter:
    """Get global reporter instance"""
    return _global_reporter


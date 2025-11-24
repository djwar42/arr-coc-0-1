#!/usr/bin/env python3
"""
Simple Session Metrics Tracker

Tracks essential timing and stats during TUI sessions.
"""

import time
from typing import Dict, Any


class SessionMetrics:
    """Track essential session metrics"""

    def __init__(self):
        self.start_time = time.time()

        # Build tracking
        self.builds_completed = 0
        self.builds_skipped = 0

        # Hash detection
        self.hash_detection_working = False

        # Errors
        self.errors = 0

        # Build timing
        self.build_times = []  # [(image_name, duration_seconds)]

    def record_build(self, image_name: str, duration: float, skipped: bool = False):
        """Record build completion or skip"""
        if skipped:
            self.builds_skipped += 1
            self.hash_detection_working = True
        else:
            self.builds_completed += 1
            self.build_times.append((image_name, duration))

    def record_error(self):
        """Record an error"""
        self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        session_duration = time.time() - self.start_time
        total_builds = self.builds_completed + self.builds_skipped

        # Calculate average build time
        avg_build_time = (
            sum(t for _, t in self.build_times) / len(self.build_times)
            if self.build_times else 0.0
        )

        return {
            "session_duration": session_duration,
            "builds_completed": self.builds_completed,
            "builds_skipped": self.builds_skipped,
            "total_builds": total_builds,
            "hash_detection_working": self.hash_detection_working,
            "errors": self.errors,
            "build_times": self.build_times,
            "avg_build_time": avg_build_time,
        }

#!/usr/bin/env python3
"""
Simple Timing Helper

Context manager for timing major operations and recording to session metrics.

Usage:
    from CLI.shared.timing import timed_operation

    # Automatic timing
    with timed_operation("build_base_image"):
        # Do work...
        pass

    # Or manual control
    with timed_operation("setup") as timer:
        # Do work...
        if should_skip:
            timer.skip()  # Mark as skipped (won't record as build)
"""

import time
from contextlib import contextmanager
from typing import Optional


class Timer:
    """Simple timer for tracking operation duration"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.skipped = False
        self.is_build = name.startswith("build_")

    def skip(self):
        """Mark this operation as skipped"""
        self.skipped = True

    def duration(self) -> float:
        """Get current duration in seconds"""
        return time.time() - self.start_time


@contextmanager
def timed_operation(name: str, record_as_build: bool = None):
    """
    Time an operation and optionally record to session metrics.

    Args:
        name: Operation name (e.g., "build_base_image", "setup", "teardown")
        record_as_build: Override build detection (default: auto-detect from name)

    Yields:
        Timer instance with .skip() and .duration() methods

    Example:
        # Auto-record builds
        with timed_operation("build_base_image"):
            subprocess.run(["gcloud", "builds", "submit", ...])

        # Manual skip
        with timed_operation("build_training_image") as timer:
            if hash_exists:
                timer.skip()
                return
            # Build...

        # Non-build timing (just logged, not recorded)
        with timed_operation("teardown"):
            cleanup_resources()
    """
    timer = Timer(name)

    # Override build detection if specified
    if record_as_build is not None:
        timer.is_build = record_as_build

    try:
        yield timer
    finally:
        duration = timer.duration()

        # Try to record to session metrics (if reporter exists)
        try:
            from CLI.shared.auto_performance_reporter import get_reporter
            reporter = get_reporter()

            if reporter and timer.is_build:
                # Extract image name from operation name
                # e.g., "build_base_image" → "base"
                image_name = name.replace("build_", "").replace("_image", "")

                metrics = reporter.get_metrics()
                metrics.record_build(
                    image_name=image_name,
                    duration=duration,
                    skipped=timer.skipped
                )

                if timer.skipped:
                    print(f"✓ Skipped {image_name} ({duration:.1f}s to check)")
                else:
                    print(f"✓ Built {image_name} ({duration:.1f}s = {duration/60:.1f} min)")

        except Exception:
            # Metrics recording failed - just continue (non-critical)
            pass


# Example usage
if __name__ == "__main__":
    from CLI.shared.auto_performance_reporter import enable_auto_reporting

    # Enable tracking
    enable_auto_reporting()

    # Simulate operations
    print("Simulating session with timing...")

    with timed_operation("build_base_image") as timer:
        print("Building base image...")
        time.sleep(0.1)  # Simulate work

    with timed_operation("build_training_image") as timer:
        print("Checking training image hash...")
        time.sleep(0.05)
        print("Hash matches - skipping!")
        timer.skip()

    with timed_operation("build_runner_image") as timer:
        print("Checking runner image hash...")
        time.sleep(0.05)
        print("Hash matches - skipping!")
        timer.skip()

    # Print results
    from CLI.shared.auto_performance_reporter import get_reporter
    reporter = get_reporter()
    if reporter:
        stats = reporter.get_metrics().get_stats()
        print("\nSession Stats:")
        print(f"  Duration: {stats['session_duration']:.2f}s")
        print(f"  Builds completed: {stats['builds_completed']}")
        print(f"  Builds skipped: {stats['builds_skipped']}")
        print(f"  Build times: {stats['build_times']}")

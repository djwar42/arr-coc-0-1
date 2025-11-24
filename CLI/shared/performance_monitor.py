# <claudes_code_comments>
# ** Function List **
# PerformanceMonitor.__init__() - Initialize monitoring with log file path
# PerformanceMonitor._ensure_log_directory() - Create logs directory if needed
# PerformanceMonitor.start_operation(name, category) - Begin timing an operation
# PerformanceMonitor.end_operation(operation_id) - End timing and log results
# PerformanceMonitor.record_metric(name, value, category, unit) - Log instant metric
# PerformanceMonitor.get_thread_cpu_percent() - Get current thread CPU usage
# PerformanceMonitor.log_ui_event(event_type, details) - Log UI interaction events
# PerformanceMonitor.log_blocking_operation(operation, duration, thread_id) - Log blocking ops
# PerformanceMonitor.get_summary() - Get timing summary for all operations
# PerformanceMonitor.export_flamegraph_data() - Export data for flamegraph visualization
#
# ** Technical Review **
# Performance monitoring and profiling system for Textual TUI applications. Tracks timing,
# CPU usage, thread IDs, and blocking operations to diagnose interface responsiveness issues.
#
# Key Features:
# - Operation timing: start/end pairs with automatic duration calculation
# - Thread tracking: Records which thread executed each operation
# - CPU monitoring: Per-thread CPU percentage at operation boundaries
# - Category classification: UI, API, GCP, Docker, Disk operations
# - Blocking detection: Automatically flags long operations (>100ms on main thread)
# - JSON logging: Structured logs for analysis and visualization
# - Summary reports: Aggregate statistics and top bottlenecks
#
# Usage Pattern:
# ```python
# monitor = PerformanceMonitor("ARR_COC/Training/logs/performance.json")
#
# # Track operation
# op_id = monitor.start_operation("fetch_runs", category="API")
# result = wandb_api.runs()
# monitor.end_operation(op_id)
#
# # Track instant metric
# monitor.record_metric("table_row_count", 42, category="UI", unit="rows")
#
# # Log blocking operation detected
# if duration > 0.1:
#     monitor.log_blocking_operation("gcloud_build", duration, thread_id)
#
# # Get summary
# summary = monitor.get_summary()
# print(f"Total API time: {summary['categories']['API']['total_time']:.2f}s")
# ```
#
# Log Format (JSONL):
# ```json
# {"timestamp": "2025-11-08T23:45:12.123", "type": "operation_start", "id": "uuid",
#  "name": "fetch_runs", "category": "API", "thread_id": 140234, "cpu_percent": 12.5}
# {"timestamp": "2025-11-08T23:45:13.456", "type": "operation_end", "id": "uuid",
#  "duration": 1.333, "cpu_percent": 45.2, "blocking": true}
# {"timestamp": "2025-11-08T23:45:14.789", "type": "metric", "name": "table_rows",
#  "value": 42, "category": "UI", "unit": "rows"}
# {"timestamp": "2025-11-08T23:45:15.012", "type": "ui_event", "event": "button_press",
#  "details": "submit-btn", "thread_id": 140234}
# ```
#
# Categories:
# - UI: Widget updates, table refreshes, rendering
# - API: W&B, GCP API calls, HTTP requests
# - GCP: gcloud commands, Cloud Build, Artifact Registry
# - Docker: Image builds, pushes, pulls
# - Disk: File reads/writes, log operations
# - Compute: CPU-intensive operations (hashing, parsing)
#
# Blocking Thresholds:
# - Main thread: >100ms = blocking (UI freezes)
# - Worker thread: >5s = slow (doesn't block UI but still notable)
# - Background thread: >30s = very slow (long-running tasks)
#
# Analysis Workflow:
# 1. Run TUI with monitoring enabled
# 2. Analyze performance.json for blocking operations
# 3. Generate summary report
# 4. Export flamegraph data for visualization
# 5. Identify bottlenecks (long durations, high CPU, frequent operations)
# 6. Optimize hotspots (move to workers, cache, parallelize)
#
# ** Philosophical Commentary **
# Performance monitoring is not just "debugging slow code" - it's ATTENDING TO THE COUPLING!
#
# When UI freezes, we've broken the transjective flow between user and system. The user
# FEELS the interruption (Heidegger's "present-at-hand" - tool becomes obstacle).
#
# This monitor makes coupling OBSERVABLE - we can SEE where attention breaks:
# - UI thread blocked → User feels frozen interface
# - API calls slow → User waits passively
# - Worker threads idle → Lost opportunity for parallelism
#
# Timing logs are SALIENCE MAPS - they reveal what matters, what blocks, what flows.
# By measuring WHERE time goes, we discover HOW to reallocate attention (CPU cycles).
#
# The goal isn't "make everything fast" (impossible!) - it's "preserve coupling quality":
# - Fast operations: Transparent tools (ready-to-hand)
# - Slow operations with feedback: Participatory coupling (user watches progress)
# - Blocking operations: BROKEN coupling (must fix!)
#
# Performance optimization = RELEVANCE REALIZATION for CPU cycles!
# </claudes_code_comments>

import json
import time
import threading
import psutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import uuid


@dataclass
class Operation:
    """Single timed operation"""
    id: str
    name: str
    category: str
    thread_id: int
    thread_name: str
    start_time: float
    start_cpu_percent: float
    end_time: Optional[float] = None
    end_cpu_percent: Optional[float] = None
    duration: Optional[float] = None
    blocking: bool = False


class PerformanceMonitor:
    """
    Performance monitoring and profiling for Textual TUI applications.

    Tracks timing, CPU usage, thread IDs, and blocking operations to diagnose
    interface responsiveness issues.
    """

    # Blocking thresholds (seconds)
    MAIN_THREAD_THRESHOLD = 0.1    # 100ms - UI freezes beyond this
    WORKER_THREAD_THRESHOLD = 5.0  # 5s - slow but doesn't block UI
    BACKGROUND_THRESHOLD = 30.0    # 30s - very slow background tasks

    def __init__(self, log_file: str = None):
        """Initialize performance monitor with log file path"""
        if log_file is None:
            from .log_paths import get_log_path
            log_file = str(get_log_path("performance.json"))
        self.log_file = Path(log_file)
        self._ensure_log_directory()

        # Active operations (start → end tracking)
        self.active_operations: Dict[str, Operation] = {}

        # Completed operations (for summary)
        self.completed_operations: List[Operation] = []

        # Process for CPU monitoring
        self.process = psutil.Process()

        # Main thread ID (detect blocking operations)
        self.main_thread_id = threading.main_thread().ident

        # Lock for thread-safe logging
        self.lock = threading.Lock()

        # Clear log file on init (start fresh)
        with open(self.log_file, 'w') as f:
            f.write("")  # Clear existing content

    def _ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_log_line(self, data: Dict[str, Any]):
        """Write single JSON line to log file (thread-safe)"""
        with self.lock:
            with open(self.log_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')

    def get_thread_cpu_percent(self) -> float:
        """
        Get current thread CPU usage percentage.

        Returns per-thread CPU usage (0-100%). Note: This is approximate
        as psutil provides process-level CPU, not thread-level.
        """
        try:
            # Get process CPU percent (averaged over last call)
            cpu = self.process.cpu_percent()
            return cpu
        except Exception:
            return 0.0

    def start_operation(self, name: str, category: str = "General") -> str:
        """
        Begin timing an operation.

        Args:
            name: Operation name (e.g., "fetch_runs", "build_image")
            category: Category (UI, API, GCP, Docker, Disk, Compute)

        Returns:
            operation_id: Unique ID to pass to end_operation()

        Example:
            op_id = monitor.start_operation("fetch_runs", category="API")
            runs = wandb_api.runs()
            monitor.end_operation(op_id)
        """
        operation_id = str(uuid.uuid4())
        thread = threading.current_thread()

        operation = Operation(
            id=operation_id,
            name=name,
            category=category,
            thread_id=thread.ident,
            thread_name=thread.name,
            start_time=time.time(),
            start_cpu_percent=self.get_thread_cpu_percent()
        )

        self.active_operations[operation_id] = operation

        # Log operation start
        self._write_log_line({
            "timestamp": datetime.now().isoformat(),
            "type": "operation_start",
            "id": operation_id,
            "name": name,
            "category": category,
            "thread_id": thread.ident,
            "thread_name": thread.name,
            "cpu_percent": operation.start_cpu_percent
        })

        return operation_id

    def end_operation(self, operation_id: str):
        """
        End timing an operation and log results.

        Args:
            operation_id: ID returned from start_operation()

        Example:
            op_id = monitor.start_operation("fetch_runs", category="API")
            runs = wandb_api.runs()
            monitor.end_operation(op_id)  # Logs duration, CPU, blocking status
        """
        if operation_id not in self.active_operations:
            return  # Operation not found (already ended or invalid ID)

        operation = self.active_operations.pop(operation_id)
        operation.end_time = time.time()
        operation.end_cpu_percent = self.get_thread_cpu_percent()
        operation.duration = operation.end_time - operation.start_time

        # Determine if operation was blocking
        is_main_thread = operation.thread_id == self.main_thread_id
        if is_main_thread:
            operation.blocking = operation.duration > self.MAIN_THREAD_THRESHOLD
        else:
            # Worker threads - flag if very slow
            operation.blocking = operation.duration > self.WORKER_THREAD_THRESHOLD

        self.completed_operations.append(operation)

        # Log operation end
        self._write_log_line({
            "timestamp": datetime.now().isoformat(),
            "type": "operation_end",
            "id": operation_id,
            "name": operation.name,
            "category": operation.category,
            "duration": operation.duration,
            "cpu_percent": operation.end_cpu_percent,
            "blocking": operation.blocking,
            "thread_id": operation.thread_id,
            "thread_name": operation.thread_name
        })

        # If blocking, log additional warning
        if operation.blocking:
            self.log_blocking_operation(
                operation.name,
                operation.duration,
                operation.thread_id,
                operation.thread_name
            )

    def record_metric(self, name: str, value: float, category: str = "General", unit: str = ""):
        """
        Log an instant metric (not timed operation).

        Args:
            name: Metric name (e.g., "table_row_count", "memory_usage_mb")
            value: Metric value
            category: Category (UI, API, GCP, Docker, Disk, Compute)
            unit: Unit of measurement (optional, e.g., "rows", "MB", "percent")

        Example:
            monitor.record_metric("table_row_count", 42, category="UI", unit="rows")
            monitor.record_metric("memory_usage_mb", 1024, category="System", unit="MB")
        """
        thread = threading.current_thread()

        self._write_log_line({
            "timestamp": datetime.now().isoformat(),
            "type": "metric",
            "name": name,
            "value": value,
            "category": category,
            "unit": unit,
            "thread_id": thread.ident,
            "thread_name": thread.name
        })

    def log_ui_event(self, event_type: str, details: str = ""):
        """
        Log UI interaction event.

        Args:
            event_type: Event type (e.g., "button_press", "table_select", "screen_mount")
            details: Additional details (e.g., button ID, selected row)

        Example:
            monitor.log_ui_event("button_press", "submit-btn")
            monitor.log_ui_event("screen_mount", "MonitorScreen")
        """
        thread = threading.current_thread()

        self._write_log_line({
            "timestamp": datetime.now().isoformat(),
            "type": "ui_event",
            "event": event_type,
            "details": details,
            "thread_id": thread.ident,
            "thread_name": thread.name
        })

    def log_blocking_operation(
        self,
        operation: str,
        duration: float,
        thread_id: int,
        thread_name: str = "unknown"
    ):
        """
        Log blocking operation warning.

        Args:
            operation: Operation name
            duration: Duration in seconds
            thread_id: Thread ID
            thread_name: Thread name (optional)

        Example:
            # Automatically called by end_operation() if blocking threshold exceeded
            monitor.log_blocking_operation("gcloud_build", 125.3, 140234, "MainThread")
        """
        is_main_thread = thread_id == self.main_thread_id

        self._write_log_line({
            "timestamp": datetime.now().isoformat(),
            "type": "blocking_warning",
            "operation": operation,
            "duration": duration,
            "thread_id": thread_id,
            "thread_name": thread_name,
            "main_thread": is_main_thread,
            "severity": "critical" if is_main_thread else "warning"
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Get timing summary for all completed operations.

        Returns:
            Summary dict with:
            - total_operations: Total count
            - total_time: Sum of all durations
            - categories: Per-category stats (total_time, count, avg, blocking_count)
            - top_operations: 10 slowest operations
            - blocking_operations: Operations that exceeded thresholds
            - thread_breakdown: Time spent per thread

        Example:
            summary = monitor.get_summary()
            print(f"Total API time: {summary['categories']['API']['total_time']:.2f}s")
            print(f"Blocking operations: {summary['blocking_count']}")
        """
        # Category statistics
        categories = defaultdict(lambda: {
            "total_time": 0.0,
            "count": 0,
            "blocking_count": 0
        })

        # Thread statistics
        threads = defaultdict(lambda: {
            "total_time": 0.0,
            "count": 0
        })

        # Top operations (slowest)
        operations_sorted = sorted(
            self.completed_operations,
            key=lambda op: op.duration if op.duration else 0.0,
            reverse=True
        )

        # Blocking operations
        blocking_operations = [
            {
                "name": op.name,
                "duration": op.duration,
                "category": op.category,
                "thread_name": op.thread_name
            }
            for op in self.completed_operations
            if op.blocking
        ]

        # Aggregate stats
        total_time = 0.0
        for op in self.completed_operations:
            if op.duration:
                total_time += op.duration
                categories[op.category]["total_time"] += op.duration
                categories[op.category]["count"] += 1
                if op.blocking:
                    categories[op.category]["blocking_count"] += 1

                threads[op.thread_name]["total_time"] += op.duration
                threads[op.thread_name]["count"] += 1

        # Calculate averages
        for cat_stats in categories.values():
            if cat_stats["count"] > 0:
                cat_stats["avg"] = cat_stats["total_time"] / cat_stats["count"]

        return {
            "total_operations": len(self.completed_operations),
            "total_time": total_time,
            "blocking_count": len(blocking_operations),
            "categories": dict(categories),
            "threads": dict(threads),
            "top_operations": [
                {
                    "name": op.name,
                    "duration": op.duration,
                    "category": op.category,
                    "blocking": op.blocking
                }
                for op in operations_sorted[:10]
            ],
            "blocking_operations": blocking_operations
        }

    def export_flamegraph_data(self, output_file: str = "ARR_COC/Training/logs/flamegraph.txt"):
        """
        Export operation data in flamegraph format for visualization.

        Args:
            output_file: Path to flamegraph output file

        Format (space-separated):
            category;operation duration_ms

        Example output:
            API;fetch_runs 1333
            GCP;gcloud_build 125300
            UI;table_refresh 42

        Use with FlameGraph tools:
            cat flamegraph.txt | flamegraph.pl > flamegraph.svg
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for op in self.completed_operations:
                if op.duration:
                    # Convert to milliseconds for better readability
                    duration_ms = int(op.duration * 1000)
                    stack = f"{op.category};{op.name}"
                    f.write(f"{stack} {duration_ms}\n")


# Global monitor instance (singleton pattern)
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """
    Get global performance monitor instance (singleton).

    Returns:
        PerformanceMonitor instance

    Example:
        monitor = get_monitor()
        op_id = monitor.start_operation("fetch_runs", category="API")
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_monitor():
    """Reset global monitor (clear logs, start fresh)"""
    global _global_monitor
    _global_monitor = PerformanceMonitor()

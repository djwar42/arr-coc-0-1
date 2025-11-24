"""
Runner Cleanup - Automatic cleanup of old Cloud Run executions

Cleans up idle runners older than 60 minutes before each launch.

This is THE STANDARD PATTERN for distributed queue systems:
- W&B Launch queue is shared by all runners (FIFO load balancing)
- Multiple runners can pick up jobs from the same queue
- Old idle runners should be cleaned up automatically
- Agents with --max-jobs=1 exit after one job (we already have this!)

Why 60 minutes:
- Cloud Run job timeout is 60 minutes (configured max)
- Any runner running > 60min has hit timeout (stuck/failed)
- Recent launches (< 60min) stay alive (might still be working)
- Simple, predictable cleanup policy aligned with timeout
"""

import json
import subprocess
from datetime import datetime, timedelta
from typing import List, Tuple

from ..shared.callbacks import StatusCallback


def cleanup_old_runners(
    status: StatusCallback,
    older_than_hours: int = 1,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Cancel Cloud Run executions older than N hours (default 1h = 60min)

    Args:
        status: Callback for status messages
        older_than_hours: Cancel executions older than this (default 1h = 60min)
        dry_run: If True, show what would be canceled without canceling

    Returns:
        Tuple of (canceled_count, total_running_count)

    Example:
        >>> canceled, total = cleanup_old_runners(status, older_than_hours=1)
        >>> # Canceled 3 old runners (>60min), 2 recent ones kept running
    """
    from .mecha.mecha_regions import ALL_MECHA_REGIONS

    status(f"🧹 Cleaning old runners (>{older_than_hours}h)...")

    # Calculate cutoff time
    cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

    canceled_count = 0
    total_running = 0

    # Check all MECHA regions for running executions
    for region in ALL_MECHA_REGIONS:
        try:
            # List running executions in this region
            result = subprocess.run(
                [
                    "gcloud", "run", "jobs", "executions", "list",
                    "--job=vertex-ai-launcher",
                    f"--region={region}",
                    "--format=json",
                    "--filter=status.conditions[0].type=Completed AND status.conditions[0].status=Unknown"  # RUNNING
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                # Region might not have the job, skip silently
                continue

            executions = json.loads(result.stdout) if result.stdout.strip() else []

            for execution in executions:
                total_running += 1

                # Get execution start time
                start_time_str = execution.get("status", {}).get("startTime")
                if not start_time_str:
                    continue

                # Parse start time (format: 2025-11-15T07:32:45.123456Z)
                start_time = datetime.strptime(start_time_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")

                # Check if older than cutoff
                if start_time < cutoff_time:
                    execution_name = execution.get("metadata", {}).get("name", "unknown")
                    age_hours = (datetime.utcnow() - start_time).total_seconds() / 3600

                    if dry_run:
                        status(f"  [dim]Would cancel: {execution_name} ({region}, {age_hours:.1f}h old)[/dim]")
                    else:
                        status(f"  ✗ Canceling: {execution_name} ({region}, {age_hours:.1f}h old)")

                        # Cancel the execution
                        cancel_result = subprocess.run(
                            [
                                "gcloud", "run", "jobs", "executions", "cancel",
                                execution_name,
                                f"--region={region}",
                                "--quiet"
                            ],
                            capture_output=True,
                            timeout=30
                        )

                        if cancel_result.returncode == 0:
                            canceled_count += 1

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            # Skip regions that error out
            continue

    # Report results
    if dry_run:
        status(f"  [dim]Dry run: Would cancel {canceled_count}/{total_running} runners[/dim]")
    elif canceled_count > 0:
        status(f"  ✓ Canceled {canceled_count} old runners ({total_running - canceled_count} recent kept)")
    else:
        status(f"  ✓ No old runners found ({total_running} recent kept)")

    return (canceled_count, total_running)

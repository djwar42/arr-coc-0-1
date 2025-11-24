"""
W&B Launch Runner Diagnostics

Complex Cloud Logging parsing and error detection for W&B Launch runners.
Used by both `launch` (failure diagnostics) and `monitor` (real-time streaming).

This module contains the sophisticated log parsing logic that was originally in
_stream_execution_logs(). It's been extracted here for reusability and clarity.
"""

import json
import subprocess
import time
from typing import Tuple, List, Optional
from rich.console import Console

console = Console()


def parse_runner_logs_for_errors(
    execution_name: str,
    project_id: str,
    region: str,
    timeout_seconds: int = 900,
    status_callback=None
) -> Tuple[bool, bool, str]:
    """
    Parse Cloud Logging for W&B Launch runner errors.

    This is the SLOW PATH - comprehensive error detection with detailed diagnostics.
    Use this when you need to understand WHY a runner failed, not just IF it failed.

    Args:
        execution_name: Cloud Run job execution name
        project_id: GCP project ID
        region: GCP region
        timeout_seconds: Maximum time to wait for logs
        status_callback: Function to call for status updates (default: console.print)

    Returns:
        Tuple of (has_error, success, all_logs)
        - has_error: True if error keywords found in logs
        - success: True if Cloud Run execution succeeded (exit code 0)
        - all_logs: Full log text for detailed analysis
    """
    if status_callback is None:
        status_callback = console.print

    status = status_callback
    start_time = time.time()
    has_error = False
    success = False
    execution_done = False
    all_logs = []

    # Error patterns to detect fatal issues
    error_patterns = [
        "wandb: ERROR",
        "Machine type",
        "not supported",
        "PermissionDenied",
        "QuotaExceeded",
        "Invalid argument",
        "RESOURCE_EXHAUSTED",
        "failed to create",
        "Error:",
        "Exception:",
        "Traceback"
    ]

    status("   [dim]Starting Cloud Logging analysis...[/dim]")

    while not execution_done and (time.time() - start_time) < timeout_seconds:
        try:
            # Check execution status
            describe_cmd = [
                "gcloud",
                "run",
                "jobs",
                "executions",
                "describe",
                execution_name,
                "--region",
                region,
                "--project",
                project_id,
                "--format=json"
            ]

            result = subprocess.run(
                describe_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                try:
                    exec_info = json.loads(result.stdout)
                    conditions = exec_info.get("status", {}).get("conditions", [])

                    for condition in conditions:
                        if condition.get("type") in ["Completed", "Ready"]:
                            execution_done = True
                            success = condition.get("status") == "True"
                            elapsed = int(time.time() - start_time)
                            status(f"   [yellow]Cloud Run execution completed ({elapsed}s)[/yellow]")
                            status(f"   [magenta]🔍 DEBUG: execution_done=True, success={success}[/magenta]")
                            break
                except json.JSONDecodeError:
                    pass

            # Fetch logs
            logs_cmd = [
                "gcloud",
                "logging",
                "read",
                f'resource.type="cloud_run_job" AND resource.labels.job_name="vertex-ai-launcher" AND resource.labels.location="{region}" AND labels."run.googleapis.com/execution_name"="{execution_name}"',
                "--format=json",
                "--limit=500",
                "--project",
                project_id
            ]

            logs_result = subprocess.run(
                logs_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if logs_result.returncode == 0 and logs_result.stdout:
                try:
                    log_entries = json.loads(logs_result.stdout)
                    for entry in reversed(log_entries):  # Chronological order
                        log_text = entry.get("textPayload", "")
                        if log_text and log_text not in all_logs:
                            all_logs.append(log_text)

                            # Check for error patterns
                            if not has_error:
                                for pattern in error_patterns:
                                    if pattern.lower() in log_text.lower():
                                        has_error = True
                                        status(f"   [red]⚠️  Error pattern detected: {pattern}[/red]")
                                        break
                except json.JSONDecodeError:
                    pass

        except (subprocess.TimeoutExpired, Exception) as e:
            status(f"   [dim]Log fetch error: {str(e)[:50]}[/dim]")

        if not execution_done:
            time.sleep(1)

    return has_error, success, "\n".join(all_logs)


def fetch_detailed_error_context(
    execution_name: str,
    project_id: str,
    region: str,
    context_lines: int = 500,  # Increased from 100 to get full stack traces
    status_callback=None
) -> List[str]:
    """
    Fetch detailed error context from Cloud Logging.

    Use this AFTER detecting a failure to get comprehensive diagnostics.
    Uses JSON format to avoid Cloud Logging's text truncation (...<N lines>...).

    Args:
        execution_name: Cloud Run job execution name
        project_id: GCP project ID
        region: GCP region
        context_lines: Number of log lines to fetch (default 500)
        status_callback: Function to call for status updates

    Returns:
        List of log lines with full context (untruncated)
    """
    if status_callback is None:
        status_callback = console.print

    status = status_callback

    status("   [dim]Fetching detailed error context...[/dim]")

    logs_cmd = [
        "gcloud",
        "logging",
        "read",
        f'resource.type="cloud_run_job" AND resource.labels.job_name="vertex-ai-launcher" AND resource.labels.location="{region}" AND labels."run.googleapis.com/execution_name"="{execution_name}"',
        "--format=json",  # Use JSON to avoid truncation!
        f"--limit={context_lines}",
        "--project",
        project_id
    ]

    try:
        result = subprocess.run(
            logs_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout.strip():
            try:
                log_entries = json.loads(result.stdout)
                lines = []
                # Extract textPayload from each entry (reversed for chronological order)
                for entry in reversed(log_entries):
                    text = entry.get("textPayload", "")
                    if text.strip():
                        lines.append(text)
                return lines
            except json.JSONDecodeError as e:
                status(f"   [red]Failed to parse JSON logs: {str(e)}[/red]")
                return []
    except (subprocess.TimeoutExpired, Exception) as e:
        status(f"   [red]Failed to fetch error context: {str(e)}[/red]")

    return []

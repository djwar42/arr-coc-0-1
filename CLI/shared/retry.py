"""
Shared Retry Logic - Fuck It Retry Pattern

Used by both setup and teardown for resilient cloud operations.

Philosophy:
- Setup: "Already exists" errors → IT'S FINE, move along
- Teardown: "Not found" errors → VERIFY it's really gone, then move along
- All operations: Retry with 1s, 4s, 8s backoff on failures
"""

import time


def retry_with_backoff(operation_func, max_attempts=4, operation_name="operation"):
    """
    Retry an operation with fixed backoff delays: 1s, 4s, 8s (fuck it, restart strategy).

    Args:
        operation_func: Function to retry (should return tuple: (success: bool, error_msg: str))
        max_attempts: Maximum number of attempts (default: 4)
        operation_name: Name for logging (default: "operation")

    Returns:
        tuple: (success: bool, error_msg: str or None)

    Backoff schedule (cloud-optimized):
        Attempt 1: immediate
        Attempt 2: 1s delay
        Attempt 3: 4s delay
        Attempt 4: 8s delay

    Total max wait: 13 seconds (1s + 4s + 8s)

    Example:
        def try_create_bucket():
            result = subprocess.run([...])
            if result.returncode == 0:
                return (True, None)
            elif "already exists" in result.stderr.lower():
                return (True, None)  # Idempotent success!
            else:
                return (False, result.stderr[:200])

        success, error = retry_with_backoff(try_create_bucket, operation_name="GCS bucket creation")
        if not success:
            print(f"Failed: {error}")
    """
    RETRY_DELAYS = [0, 1, 4, 8]  # Delays before each attempt (0 = immediate first try)

    for attempt in range(1, max_attempts + 1):
        success, error_msg = operation_func()

        if success:
            return (True, None)

        # Failed - check if we should retry
        if attempt < max_attempts:
            delay = RETRY_DELAYS[attempt]  # Next delay (1s, 4s, or 8s)
            time.sleep(delay)
        else:
            # Final attempt failed
            return (False, error_msg)

    return (False, "Max retries exceeded")


def format_retry_error_report(
    operation_name: str,
    attempt_errors: list,
    retry_delays: str = "1s, 4s, 8s",
    next_steps: list = None,
) -> str:
    """
    Format a detailed error report after all retries fail.

    Args:
        operation_name: Name of the operation that failed
        attempt_errors: List of error messages from each attempt
        retry_delays: String describing retry delays used
        next_steps: List of suggested next steps (optional)

    Returns:
        Formatted error report string

    Example:
        errors = ["Attempt 1: 404 error", "Attempt 2: timeout"]
        report = format_retry_error_report(
            "Cloud Function deployment",
            errors,
            next_steps=[
                "Check GCP console",
                "Review logs",
                "Run teardown and retry"
            ]
        )
    """
    lines = [
        "",
        f"   ❌ {operation_name} failed after {len(attempt_errors)} attempts!",
        "",
        "   Attempts summary:",
    ]

    for i, err in enumerate(attempt_errors, 1):
        lines.append(f"      {i}. {err}")

    lines.extend([
        "",
        f"   Retry delays used: {retry_delays} (total 13s)",
        "   This is STEP-level retry (not entire setup/teardown)",
        "",
    ])

    if next_steps:
        lines.append("   💡 Next steps:")
        for i, step in enumerate(next_steps, 1):
            lines.append(f"      {i}. {step}")
        lines.append("")

    return "\n".join(lines)


def is_already_exists_error(error_message: str) -> bool:
    """
    Check if error message indicates resource already exists (idempotent success for setup).

    Args:
        error_message: Error message from GCP command

    Returns:
        True if error is "already exists" type (409, ALREADY_EXISTS, etc.)

    Examples:
        >>> is_already_exists_error("ERROR: ALREADY_EXISTS: the repository already exists")
        True
        >>> is_already_exists_error("ServiceException: 409 bucket already exists")
        True
        >>> is_already_exists_error("ERROR: PERMISSION_DENIED")
        False
    """
    error_lower = error_message.lower()
    return (
        "already_exists" in error_lower
        or "already exists" in error_lower
        or "409" in error_message  # HTTP 409 Conflict
    )


def is_not_found_error(error_message: str) -> bool:
    """
    Check if error message indicates resource not found (idempotent success for teardown).

    Args:
        error_message: Error message from GCP command

    Returns:
        True if error is "not found" type (404, NOT_FOUND, etc.)

    Examples:
        >>> is_not_found_error("ERROR: NOT_FOUND: Function not found")
        True
        >>> is_not_found_error("BucketNotFoundException: 404 bucket does not exist")
        True
        >>> is_not_found_error("ERROR: PERMISSION_DENIED")
        False
    """
    error_lower = error_message.lower()
    return (
        "not_found" in error_lower
        or "not found" in error_lower
        or "does not exist" in error_lower
        or "404" in error_message  # HTTP 404 Not Found
    )


# Retry delays constant (shared across all operations)
RETRY_DELAYS = [0, 1, 4, 8]  # 0s, 1s, 4s, 8s (total 13s max)
MAX_ATTEMPTS = 4

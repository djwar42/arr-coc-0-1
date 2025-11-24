"""
Pricing Infrastructure Teardown

Deletes:
1. Cloud Scheduler: arr-coc-pricing-scheduler
2. Cloud Function: arr-coc-pricing-runner

Preserves:
- Artifact Registry: arr-coc-pricing repository (historical pricing data)
"""

# <claudes_code_comments>
# ** Function List **
# teardown_pricing_infrastructure(status_callback) - Main teardown entry point
# delete_scheduler(status) - Delete arr-coc-pricing-scheduler job
# delete_cloud_function(status) - Delete arr-coc-pricing-runner function
# revoke_actAs_permission(status) - Revoke OIDC permissions (silent)
# disable_cloudbilling_api(status) - Disable Cloud Billing API (silent)
#
# ** Technical Review **
# Deletes pricing infrastructure while PRESERVING historical data. Phase 1: Deletes
# Cloud Scheduler job (arr-coc-pricing-scheduler) via gcloud scheduler jobs delete.
# Phase 2: Deletes Cloud Function (arr-coc-pricing-runner, gen2) via gcloud functions
# delete. Phase 3: PRESERVES Artifact Registry repository (arr-coc-pricing) containing
# historical pricing data. All operations ignore NOT_FOUND errors (resource already
# deleted), but warn on other errors. Constants: PROJECT_ID, REGION (us-central1),
# FUNCTION_NAME (arr-coc-pricing-runner), SCHEDULER_JOB (arr-coc-pricing-scheduler).
# Graceful error handling allows teardown to continue if resources missing.
# Flow: scheduler → function → revoke OIDC → preserve repository → complete.
# </claudes_code_comments>

import subprocess
import time

# Shared retry logic (fuck it retry pattern with 1s, 4s, 8s backoff)
from ..shared.retry import (
    retry_with_backoff,
    format_retry_error_report,
    is_not_found_error,
    RETRY_DELAYS,
    MAX_ATTEMPTS,
)

# Pricing infrastructure config (single source of truth for all pricing constants)
from ..shared.pricing.pricing_config import (
    PROJECT_ID,
    REGION,
    FUNCTION_NAME,
    SCHEDULER_JOB,
    SCHEDULER_INTERVAL_MINUTES,
    REPOSITORY,
    PACKAGE,
)


# OLD retry_with_backoff function removed - now using shared version from ..shared.retry
# (Keeping this comment for context)
def _OLD_retry_with_backoff(operation_func, max_attempts=4, operation_name="operation"):
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


def teardown_pricing_infrastructure(status_callback):
    """Teardown pricing infrastructure (header shown as step 8/8 in main teardown)"""
    status = status_callback

    # Header is shown as step (8/8) in main teardown logs
    # Just show the substeps here

    # 1. Delete Cloud Scheduler
    status("   ⊗  Deleting Cloud Scheduler...")
    delete_scheduler(status)

    # 2. Delete Cloud Function
    status("   ⊗  Deleting Cloud Function...")
    delete_cloud_function(status)

    # 2.5 Revoke OIDC permissions
    status("   ⊗  Revoking OIDC permissions...")
    revoke_actAs_permission(status)

    # 2.6 Disable Cloud Billing API
    status("   ⊗  Disabling Cloud Billing API...")
    disable_cloudbilling_api(status)

    # 3. Artifact Registry pricing repository is PRESERVED (contains historical data)
    status("   [cyan]ℹ Pricing repository preserved (historical data intact)[/cyan]")

    status("   [cyan]⚡Pricing teardown complete - Roger![/cyan]")


def delete_scheduler(status):
    """Delete Cloud Scheduler job with retry logic (fuck it, restart)"""

    def try_delete():
        result = subprocess.run(
            [
                "gcloud", "scheduler", "jobs", "delete", SCHEDULER_JOB,
                f"--location={REGION}",
                f"--project={PROJECT_ID}",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return (True, None)

        # Ignore "not found" errors (already deleted - idempotent success!)
        stderr_lower = result.stderr.lower()
        if ("not found" in stderr_lower
            or "not_found" in stderr_lower
            or "has not been used" in stderr_lower):
            return (True, None)

        # Real error - return for retry
        error_msg = f"Exit code: {result.returncode}, STDERR: {result.stderr[:100]}"
        return (False, error_msg)

    # Retry with backoff (4 attempts: 0s, 1s, 4s, 8s delays)
    success, error_msg = retry_with_backoff(
        try_delete,
        max_attempts=4,
        operation_name="scheduler deletion"
    )

    if not success:
        status(f"    ⚠️  Scheduler deletion failed after 4 attempts")
        if error_msg:
            status(f"    {error_msg}")


def delete_cloud_function(status):
    """Delete Cloud Function with retry logic (fuck it, restart)"""

    def try_delete():
        result = subprocess.run(
            [
                "gcloud", "functions", "delete", FUNCTION_NAME,
                "--gen2",
                f"--region={REGION}",
                f"--project={PROJECT_ID}",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return (True, None)

        # Ignore "not found" errors (already deleted - idempotent success!)
        stderr_lower = result.stderr.lower()
        if ("not found" in stderr_lower
            or "not_found" in stderr_lower
            or "has not been used" in stderr_lower):
            return (True, None)

        # Real error - return for retry
        error_msg = f"Exit code: {result.returncode}, STDERR: {result.stderr[:100]}"
        return (False, error_msg)

    # Retry with backoff (4 attempts: 0s, 1s, 4s, 8s delays)
    success, error_msg = retry_with_backoff(
        try_delete,
        max_attempts=4,
        operation_name="cloud function deletion"
    )

    if not success:
        status(f"    ⚠️  Function deletion failed after 4 attempts")
        if error_msg:
            status(f"    {error_msg}")


def revoke_actAs_permission(status):
    """Revoke Service Account User role from current user"""
    # Get current user email
    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        status("    ⚠️  Could not get current user email")
        return

    user_email = result.stdout.strip()
    service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"

    # Check if service account exists first (idempotency check)
    check_sa = subprocess.run(
        [
            "gcloud", "iam", "service-accounts", "describe",
            service_account,
            f"--project={PROJECT_ID}",
        ],
        capture_output=True,
        text=True,
    )

    if check_sa.returncode != 0:
        # SA doesn't exist - nothing to revoke (idempotent success!)
        status("    ℹ  App Engine SA not found - nothing to revoke")
        return

    # Remove Service Account User role with condition
    result = subprocess.run(
        [
            "gcloud", "iam", "service-accounts", "remove-iam-policy-binding",
            service_account,
            f"--member=user:{user_email}",
            "--role=roles/iam.serviceAccountUser",
            '--condition=expression=resource.service=="cloudscheduler.googleapis.com",title=OIDCSchedulerOnly',
            f"--project={PROJECT_ID}",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Ignore "not found" or "no binding" errors (already removed)
        stderr_lower = result.stderr.lower()
        if "not found" not in stderr_lower and "no role binding" not in stderr_lower:
            status(f"    ⚠️  OIDC permission revocation failed (exit code: {result.returncode})")
            status(f"    STDERR: {result.stderr if result.stderr else '(empty)'}")
            status(f"    STDOUT: {result.stdout if result.stdout else '(empty)'}")


def disable_cloudbilling_api(status):
    """Disable Cloud Billing API"""
    result = subprocess.run(
        [
            "gcloud", "services", "disable", "cloudbilling.googleapis.com",
            f"--project={PROJECT_ID}",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # API disable can fail if still in use - that's ok
        status(f"    ⚠️  Cloud Billing API disable warning (may still be in use)")
    # Success case: no output (silent success)


def cleanup_artifact_registry(status):
    """Delete pricing repository from Artifact Registry"""
    # Delete entire repository (contains only pricing package)
    result = subprocess.run(
        [
            "gcloud", "artifacts", "repositories", "delete", "arr-coc-pricing",
            f"--location={REGION}",
            f"--project={PROJECT_ID}",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Ignore "not found" errors (already deleted)
        stderr_lower = result.stderr.lower()
        if "not found" not in stderr_lower and "not_found" not in stderr_lower:
            status(f"    ⚠️  Artifact Registry cleanup failed (exit code: {result.returncode})")
            status(f"    STDERR: {result.stderr if result.stderr else '(empty)'}")
            status(f"    STDOUT: {result.stdout if result.stdout else '(empty)'}")

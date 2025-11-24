"""
Setup Steps - Infrastructure Setup Functions

Each function implements one step of the 8-step setup process.
All functions follow the same pattern:
- Receive: project_id, region (if needed), config (if needed), status callback
- Return: bool (True = success, False = failure)
- Call status() immediately for user feedback (no deferred logs)
"""

# <claudes_code_comments>
# ** Function List **
# _enable_apis(project_id, status) - Enable required GCP APIs (8 APIs at once using parallel execution)
# _setup_registry(project_id, region, status) - Create deletable Artifact Registry (arr-coc-registry)
# _setup_persistent_registry(project_id, region, status) - Create persistent registry (arr-coc-registry-persistent, PyTorch only)
# _setup_secrets(project_id, wandb_api_key, status) - Create/update W&B API key secret in Secret Manager
# _create_service_account(project_id, status) - Create service account for Vertex AI + grant IAM roles
# _setup_build_pool(project_id, region, status) - Create Cloud Build worker pool for C3 instances
# _grant_cloudbuild_permissions(project_id, status) - Grant IAM roles to Cloud Build SA for C3 instances
# _setup_vpc_peering(project_id, status) - Set up VPC peering for Service Networking API
#
# ** Technical Review **
# Setup steps create GCP infrastructure for ARR-COC training.
#
# TWO ARTIFACT REGISTRIES:
# 1. arr-coc-registry (deletable) - Fast-building images: ml-stack, trainer, launcher
# 2. arr-coc-registry-persistent (NEVER deleted) - PyTorch base only (~15GB, 2-4 hours)
# Separation prevents expensive PyTorch rebuilds across teardown/setup cycles.
#
# PARALLELIZATION PATTERNS:
# - API enablement: 8 APIs enabled at once (run_gcloud_batch_parallel) - SAFE to parallelize
# - IAM role grants: MUST be sequential! IAM operations are read-modify-write - concurrent
#   modifications cause "concurrent policy changes" errors. Each grant must complete before next.
# - Docker image deletion: Safe to parallelize (independent operations)
#
# Flow: APIs → Registry → Persistent Registry → Secrets → Service Account → Build Pool → Cloud Build IAM → VPC Peering
# Each step is idempotent (already-exists = success).
# Steps use run_gcloud_with_retry for resilience (3 retries, exponential backoff).
# </claudes_code_comments>

# ═══════════════════════════════════════════════════════════════════════
# TWO-SPACE STANDARD SYSTEM (Output Formatting)
# IMPORTANT: Don't remove - consistent formatting across setup/teardown
# ═══════════════════════════════════════════════════════════════════════
# SPACING AFTER ICONS (spaces between icon and text):
#   ℹ shows like: "   ℹ APIs for Storage"       - 1 space  (info/context)
#   ⊕ shows like: "   ⊕  Creating registry..."   - 2 spaces (operation/action)
#   ⚡ shows like: "   ⚡GCP APIs passed - Roger!"  - 0 spaces (step complete - Lightning Pow Pow!)
#   ✓ shows like: "   ✓    Registry created"    - 4 spaces (success/done)
#   ✗ shows like: "   ✗    Creation failed"     - 4 spaces (failure/error)
#   🪙 shows like: "   🪙  Using pricing (4.1KB)" - 2 spaces (pricing data)
#   ☁️ shows like: "   ☁️  Function deployed"    - 2 spaces (cloud resource)
#   📄 shows like: "   📄  Checked 5000 SKUs"    - 2 spaces (progress update)
#   🔄 shows like: "   🔄  Fetching pricing"     - 2 spaces (fetching/loading)
#   🚀 shows like: "   🚀  Triggering function"  - 2 spaces (triggering/launching)
#   ⏳ shows like: "   ⏳  Deploying..."         - 2 spaces (waiting/in-progress)
#
# OTHER RULES:
#   - All icons have 3 spaces before them (prefix)
#   - Headers: "Creating Service Account... (4/9)" (no prefix)
#   - Indentation levels: L0=0sp | L1=3sp | L2=9sp | L3=15sp
#   - LIGHTNING POW POW FINALE: Every step ends "   ⚡[Step] {text} - Roger!" (0 spaces after ⚡)
#   - Follow when possible, break for clarity when needed
#
# LIGHTNING POW POW RULES:
#   - ✅ Success → ⚡Pow Pow!
#   - ❌ Halting failure (returns False) → NO Pow Pow
#   - ⚠️ Non-halting failure (continues) → ⚡Pow Pow!
# ═══════════════════════════════════════════════════════════════════════

import subprocess
import time
from typing import Dict
from pathlib import Path

# Import from relative path
from ..shared.types import StatusCallback
from ..shared.api_helpers import run_gcloud_with_retry, run_gcloud_batch_parallel


def _enable_apis(project_id: str, status: StatusCallback) -> bool:
    """
    Step 1/9: Enable required GCP APIs

    APIs enabled:
    - Vertex AI (aiplatform)
    - Artifact Registry (artifactregistry)
    - Cloud Storage (storage)
    - Compute Engine (compute)
    - Cloud Run (run)
    - Secret Manager (secretmanager)
    - Cloud Build (cloudbuild)
    - Service Networking (servicenetworking - required for VPC peering)
    """

    apis = [
        "servicenetworking.googleapis.com",  # Required for VPC peering - 34 chars
        "artifactregistry.googleapis.com",  # 33 chars
        "secretmanager.googleapis.com",  # 30 chars
        "cloudbuild.googleapis.com",  # 27 chars
        "aiplatform.googleapis.com",  # 25 chars
        "storage.googleapis.com",  # 24 chars
        "compute.googleapis.com",  # 24 chars
        "run.googleapis.com",  # 20 chars
    ]

    status(f"   ⊕  Enabling {len(apis)} APIs...")

    # Build all commands for parallel execution
    commands = [
        {
            "cmd": ["gcloud", "services", "enable", api, f"--project={project_id}"],
            "max_retries": 3,
            "timeout": 30,
            "operation_name": f"enable {api} API",
        }
        for api in apis
    ]

    # Execute all API enables in parallel (max 8 at once)
    results = run_gcloud_batch_parallel(commands, max_workers=8)

    # Process results (results are tuples: (index, result_or_none, error_or_none))
    for api, (idx, result, error) in zip(apis, results):
        if error:
            # Non-fatal - API might already be enabled
            status(f"   [yellow]⚠️  Warning: {api} enable issue[/yellow]")
            status(f"      {error[:4000]}")
        elif result and result.returncode == 0:
            status(f"   [green]✓    Enabled {api}[/green]")
        else:
            # Fallback (shouldn't happen)
            status(f"   [yellow]⚠️  Warning: {api} unexpected result[/yellow]")

    status("   [cyan]⚡GCP APIs passed - Roger![/cyan]")
    return True


def _setup_persistent_registry(
    project_id: str,
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Step 2a/9: Setup Persistent Artifact Registry (NEVER DELETED)

    Registry name: arr-coc-registry-persistent (PyTorch base image only)

    This registry stores the arr-pytorch-base image which takes 2-4 hours to build.
    NEVER deleted during teardown - user must manually remove if needed (~15GB).
    """

    registry_name = "arr-coc-registry-persistent"

    # Check if registry exists
    check_registry = run_gcloud_with_retry(
        [
            "gcloud", "artifacts", "repositories", "describe",
            registry_name,
            "--location", region,
            f"--project={project_id}",
            "--format=json",
        ],
        max_retries=3,
        timeout=30,
        operation_name="check persistent registry exists",
    )

    if check_registry.returncode == 0:
        # Registry exists - verify health
        try:
            import json

            stdout = check_registry.stdout
            json_start = stdout.find('{')
            if json_start >= 0:
                registry_info = json.loads(stdout[json_start:])
            else:
                raise json.JSONDecodeError("No JSON found", stdout, 0)

            actual_format = registry_info.get("format", "").upper()
            name = registry_info.get("name", "")
            actual_location = ""
            if "/locations/" in name:
                parts = name.split("/locations/")
                if len(parts) > 1:
                    actual_location = parts[1].split("/")[0]

            format_ok = actual_format == "DOCKER"
            location_ok = actual_location == region

            if format_ok and location_ok:
                status(f"   [dim]ℹ[/dim] Format: {actual_format}, Location: {actual_location}")
                status(f"   [green]✓    Persistent registry exists: {registry_name}[/green]")
                status(f"   [cyan]⚡Persistent Artifacts passed - Roger![/cyan]")
            else:
                status(f"   [yellow]⚠️  Persistent registry misconfigured:[/yellow]")
                if not format_ok:
                    status(f"      Format mismatch: {actual_format} (expected: DOCKER)")
                if not location_ok:
                    status(f"      Location mismatch: {actual_location} (expected: {region})")

        except (json.JSONDecodeError, KeyError) as e:
            status(f"   [yellow]⚠️  Persistent registry exists but could not verify: {e}[/yellow]")

        return True

    # Registry doesn't exist - create it
    status(f"   ⊕  Creating persistent registry: {registry_name}...")

    create_registry = run_gcloud_with_retry(
        [
            "gcloud", "artifacts", "repositories", "create",
            registry_name,
            "--repository-format=docker",
            f"--location={region}",
            f"--project={project_id}",
            "--description=Persistent artifacts (PyTorch base image)",
        ],
        max_retries=3,
        timeout=60,
        operation_name="create persistent registry",
    )

    if create_registry.returncode != 0:
        stderr_lower = create_registry.stderr.lower()
        if "already_exists" in stderr_lower or "already exists" in stderr_lower:
            status(f"   [green]✓    Persistent registry exists: {registry_name}[/green] [dim](created by parallel process)[/dim]")
            status(f"   [cyan]⚡Persistent Artifacts passed - Roger![/cyan]")
            return True
        else:
            status(f"   [red]✗    Failed to create persistent registry: {create_registry.stderr[:4000]}[/red]")
            return False

    status(f"   [green]✓    Persistent registry created: {registry_name}[/green]")
    status(f"   [cyan]⚡Persistent Artifacts passed - Roger![/cyan]")
    return True


def _setup_registry(
    project_id: str,
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Step 2b/9: Setup Artifact Registry for Docker images

    Registry name: arr-coc-registry (fast-building images only)

    This registry stores fast-building Docker images: ml-stack, trainer, launcher.
    Can be safely deleted during teardown.
    """

    # SHARED INFRASTRUCTURE (all ARR-COC prototypes share this)
    registry_name = "arr-coc-registry"

    # Check if registry exists
    check_registry = run_gcloud_with_retry(
        [
            "gcloud", "artifacts", "repositories", "describe",
            registry_name,
            "--location", region,
            f"--project={project_id}",
            "--format=json",  # CRITICAL: Need JSON output for health check parsing!
        ],
        max_retries=3,
        timeout=30,
        operation_name="check artifact registry exists",
    )

    if check_registry.returncode == 0:
        # Registry exists - verify health
        # Health check: Verify format and location (from OLD GOOD CODE)
        try:
            import json

            # Parse JSON (skip any human-readable header text before the JSON)
            stdout = check_registry.stdout
            json_start = stdout.find('{')
            if json_start >= 0:
                registry_info = json.loads(stdout[json_start:])
            else:
                raise json.JSONDecodeError("No JSON found", stdout, 0)

            actual_format = registry_info.get("format", "").upper()

            # Extract location from "name" field (format: "projects/X/locations/REGION/repositories/Y")
            name = registry_info.get("name", "")
            actual_location = ""
            if "/locations/" in name:
                parts = name.split("/locations/")
                if len(parts) > 1:
                    actual_location = parts[1].split("/")[0]

            format_ok = actual_format == "DOCKER"
            location_ok = actual_location == region

            if format_ok and location_ok:
                # Info first, then success, then Roger! last
                status(f"   [dim]ℹ[/dim] Format: {actual_format}, Location: {actual_location}")
                status(f"   [green]✓    Registry exists: {registry_name}[/green]")
                status(f"   [cyan]⚡Artifact Registry passed - Roger![/cyan]")
            else:
                # Misconfigured registry - warn user
                status(f"   [yellow]⚠️  Registry exists but misconfigured:[/yellow]")
                if not format_ok:
                    status(f"      Format mismatch: {actual_format} (expected: DOCKER)")
                    status(f"      ⚠️  MANUAL FIX REQUIRED: Delete registry and re-run setup")
                if not location_ok:
                    status(f"      Location mismatch: {actual_location} (expected: {region})")
                    status(f"      ⚠️  MANUAL FIX REQUIRED: Cannot migrate registry locations")

                # Don't fail - registry can still work, just suboptimal
                if not format_ok:
                    status(f"      ⚠️  Builds will fail until registry format is fixed!")

        except (json.JSONDecodeError, KeyError) as e:
            status(f"   [yellow]⚠️  Registry exists but could not verify config: {e}[/yellow]")
            # Continue anyway - registry might still work

        return True

    # Registry doesn't exist - create it
    status(f"   ⊕  Creating registry: {registry_name}...")

    create_registry = run_gcloud_with_retry(
        [
            "gcloud", "artifacts", "repositories", "create",
            registry_name,
            "--repository-format=docker",
            f"--location={region}",
            f"--project={project_id}",
            "--description=ARR-COC training images",
        ],
        max_retries=3,
        timeout=60,
        operation_name="create artifact registry",
    )

    if create_registry.returncode != 0:
        # Check if registry was just created by another parallel process (IDEMPOTENT)
        stderr_lower = create_registry.stderr.lower()
        if "already_exists" in stderr_lower or "already exists" in stderr_lower:
            # Another parallel process just created it - that's OK! (idempotent success)
            status(f"   [green]✓    Registry exists: {registry_name}[/green] [dim](created by parallel process)[/dim]")
            status(f"   [cyan]⚡Artifact Registry passed - Roger![/cyan]")
            return True
        else:
            # Real error - fail
            status(f"   [red]✗    Failed to create registry: {create_registry.stderr[:4000]}[/red]")
            return False

    status(f"   [green]✓    Registry created: {registry_name}[/green]")
    status(f"   [cyan]⚡Artifact Registry passed - Roger![/cyan]")
    return True


def _setup_buckets(
    config: Dict[str, str],
    project_id: str,
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Step 3/9: GCS buckets (ON-DEMAND)

    Buckets are created ON-DEMAND during launch:
    - Regional staging buckets: created when ZEUS picks a GPU region
    - Regional checkpoints buckets: created alongside staging buckets
    - Pattern: gs://{project_id}-{project_name}-{region}-staging
    - Pattern: gs://{project_id}-{project_name}-{region}-checkpoints

    All regional, all on-demand!
    """
    status("   [green]✓[/green]  GCS buckets: Created on-demand during launch")
    status("   [cyan]⚡GCS buckets passed - Roger![/cyan]")
    return True


def _create_service_account(
    config: Dict[str, str],
    project_id: str,
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Step 4/9: Create Vertex AI service account

    Reuses launch function - service account creation logic shared with launch code.
    """

    # Import the shared function (reuse launch function)
    from ..launch.core import _create_service_account as create_sa

    # Call shared function
    success = create_sa(config, region, status)
    if success:
        status("   [cyan]⚡Service Account passed - Roger![/cyan]")
    return success


def _create_queue(
    config: Dict[str, str],
    entity: str,
    status: StatusCallback,
) -> bool:
    """
    Step 6/9: Verify W&B Launch Queue exists

    IMPORTANT: Queue must be created manually via W&B web UI!
    This function only VERIFIES the queue exists.

    DO NOT use api.create_run_queue() - causes ghost queues!
    """

    import wandb

    queue_name = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")

    status(f"   ⊕  Verifying W&B queue: {queue_name}...")

    try:
        # Verify queue exists (DO NOT CREATE - must be done manually!)
        api = wandb.Api()

        # Try to access the queue
        queue = api.run_queue(entity, queue_name)

        # CRITICAL: Access queue.id to verify it's REAL (not a ghost queue)
        queue_id = queue.id

        status(f"   [green]✓    Queue verified: {queue_name}[/green]")
        status(f"   [green]✓    Queue ID: {queue_id}[/green]")
        status(f"   [cyan]⚡W&B Queue passed - Roger![/cyan]")
        return True

    except Exception as e:
        # Queue doesn't exist or is a ghost queue
        status(f"   [red]✗    Queue '{queue_name}' not found[/red]")
        status(f"   ⚠️  Queue must be created manually via W&B web UI")
        status(f"   ⚠️  Entity: {entity}")
        status(f"   ⚠️  Error: {str(e)[:200]}")
        return False


def _setup_cloudbuild_iam(project_id: str, status: StatusCallback) -> bool:
    """
    Step 7/8: Grant Cloud Build service account permissions

    Cloud Build default SA needs compute.admin + compute.networkUser to create C3 instances.
    These permissions are required for worker pools to function.

    Roles granted:
    - roles/compute.networkUser: Use VPC peering
    - roles/compute.admin: Create/manage compute instances
    """

    # Get project number (needed for Cloud Build SA email)
    project_number = run_gcloud_with_retry(
        ["gcloud", "projects", "describe", project_id, "--format=value(projectNumber)"],
        max_retries=3,
        timeout=10,
        operation_name="get project number",
    )

    if project_number.returncode != 0:
        status("   [yellow]⚠️  Could not determine project number[/yellow]")
        return True  # Non-fatal

    # Construct Cloud Build service account email
    project_num = project_number.stdout.strip()
    cloudbuild_sa = f"{project_num}@cloudbuild.gserviceaccount.com"

    # Roles needed for Cloud Build to create C3 instances in worker pools
    cloudbuild_roles = [
        "roles/compute.networkUser",  # Use VPC peering
        "roles/compute.admin",        # Create/manage compute instances
    ]

    # Grant each role SEQUENTIALLY (IAM operations are read-modify-write, NOT safe to parallelize!)
    for role in cloudbuild_roles:
        grant_cb = run_gcloud_with_retry(
            [
                "gcloud", "projects", "add-iam-policy-binding", project_id,
                f"--member=serviceAccount:{cloudbuild_sa}",
                f"--role={role}",
                "--condition=None",
            ],
            max_retries=3,
            timeout=15,
            operation_name=f"grant IAM role {role}",
        )

        # Idempotent check
        if grant_cb.returncode == 0 or "already has role" in grant_cb.stderr.lower():
            status(f"   [green]✓    Granted {role}[/green]")
        else:
            status(f"   [yellow]⚠️  Warning: Failed to grant {role}[/yellow]")
            status(f"      {grant_cb.stderr[:4000]}")

    status("   [cyan]⚡Cloud Build passed - Roger![/cyan]")
    return True


def _setup_vpc_peering(project_id: str, status: StatusCallback) -> bool:
    """
    Step 8/8: Set up VPC peering for Service Networking API

    Required for Cloud Build worker pools to access private networks.
    Worker pools won't function without VPC peering.

    Creates:
    - IP address range for VPC peering (google-managed-services-default)
    - VPC peering connection to servicenetworking.googleapis.com
    """

    # Standard GCP naming for VPC peering range
    peering_range_name = "google-managed-services-default"

    # Step A: Create IP address range (with retry logic)
    create_range = run_gcloud_with_retry(
        [
            "gcloud", "compute", "addresses", "create",
            peering_range_name,
            "--global",
            "--purpose=VPC_PEERING",
            "--prefix-length=16",
            "--network=default",
            f"--project={project_id}",
        ],
        max_retries=3,
        timeout=60,
        operation_name="create VPC peering address range",
    )

    if create_range.returncode == 0:
        status(f"   [green]✓    Created VPC peering address range[/green]")
    elif "already exists" in create_range.stderr.lower():
        status(f"   [green]✓    VPC peering address range already exists[/green]")
    else:
        status(f"   [red]✗    Failed to create peering range: {create_range.stderr[:4000]}[/red]")
        return False

    # Step B: Connect VPC peering (with retry logic)
    connect_peering = run_gcloud_with_retry(
        [
            "gcloud", "services", "vpc-peerings", "connect",
            "--service=servicenetworking.googleapis.com",
            f"--ranges={peering_range_name}",
            "--network=default",
            f"--project={project_id}",
        ],
        max_retries=3,
        timeout=120,  # VPC peering is slow
        operation_name="connect VPC peering",
    )

    if connect_peering.returncode == 0:
        status(f"   [green]✓    Connected VPC peering[/green]")
    elif "already peered" in connect_peering.stderr.lower():
        status(f"   [green]✓    VPC already peered[/green]")
    else:
        status(f"   [red]✗    Failed to connect peering: {connect_peering.stderr[:4000]}[/red]")
        return False

    status("   [cyan]⚡VPC peering passed - Roger![/cyan]")
    return True

"""
Pricing Infrastructure Setup

Creates:
1. Repository: arr-coc-pricing (GENERIC format)
2. Cloud Function: arr-coc-pricing-runner (fetches pricing)
3. Cloud Scheduler: arr-coc-pricing-scheduler (every 20 min)
"""

# ═══════════════════════════════════════════════════════════════════════
# TWO-SPACE STANDARD SYSTEM (Output Formatting)
# IMPORTANT: Don't remove - consistent formatting across setup/teardown
# ═══════════════════════════════════════════════════════════════════════
# SPACING AFTER ICONS (spaces between icon and text):
#   ℹ shows like: "   ℹ APIs for Storage"       - 1 space  (info/context)
#   ⊕ shows like: "   ⊕  Creating registry..."   - 2 spaces (operation/action)
#   ⚡ shows like: "   ⚡GCP APIs passed - Roger!"  - 0 spaces  (step complete - Lightning Pow Pow!)
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

# <claudes_code_comments>
# ** Function List **
# setup_pricing_infrastructure(status_callback) - main setup entry point for pricing
# create_pricing_repository(status) - creates arr-coc-pricing GENERIC repository
# bootstrap_pricing(status) - fetches initial pricing or validates existing schema
# deploy_cloud_function(status) - deploys arr-coc-pricing-runner Cloud Function
# create_scheduler(status) - creates arr-coc-pricing-scheduler Cloud Scheduler job
# _fetch_pricing_inline(status) - inline GCP Billing API fetch with complete SKU data
#
# ** Technical Review **
# Autonomous pricing system with Cloud Function + Scheduler + schema validation.
# Phase 0: Creates arr-coc-pricing GENERIC repository (JSON storage). Phase 1: Bootstrap
# with schema validation - fetches existing pricing from Artifact Registry, validates
# against PRICING_SCHEMA from pricing/pricing_config.py (checks c3_machines, e2_machines,
# gpus_spot, gpus_ondemand all populated). If schema mismatch detected (missing fields
# or empty data), forces fresh fetch from GCP Billing API (~30k SKUs scanned). Stores
# COMPLETE SKU data (price + metadata) for ALL pricing tiers: spot, on-demand, commitment
# (1yr/3yr). Data structure: Lists of SKU objects sorted by price (cheapest first).
# Each SKU includes: {price, description, sku_id, usage_type}. Consumer code accesses
# cheapest via [0]["price"]. Shows progress every 5000 SKUs, outputs detailed GPU/machine
# counts on completion. Phase 1.5: Grants OIDC permission to App Engine SA. Phase 2:
# Deploys Cloud Function (Python 3.12, 512MB, 540s timeout) with same pricing logic.
# Phase 3: Creates scheduler (*/20 * * * * = every 20 min). Pricing data includes C3 spot,
# E2 on-demand, GPU spot/on-demand/commitment for all types (T4, L4, V100, P4, P100, A100,
# H100, H200). Idempotent - safe to run multiple times. Schema validation ensures code
# updates trigger refetch.
# </claudes_code_comments>

import json
import subprocess
import time
from pathlib import Path
import requests

from ..shared.api_helpers import run_requests_with_retry

# Pricing infrastructure config (single source of truth for all pricing constants)
from ..shared.pricing.pricing_config import (
    FUNCTION_NAME,
    PACKAGE,
    PROJECT_ID,
    REGION,
    REPOSITORY,
    SCHEDULER_INTERVAL_MINUTES,
    SCHEDULER_JOB,
    get_required_fields,
)

# Shared retry logic (fuck it retry pattern with 1s, 4s, 8s backoff)
from ..shared.retry import (
    MAX_ATTEMPTS,
    RETRY_DELAYS,
    format_retry_error_report,
    is_already_exists_error,
    retry_with_backoff,
)


def setup_pricing_infrastructure(status_callback):
    """Setup pricing infrastructure"""
    status = status_callback

    # 1. Create generic repository first (needed for pricing upload!)
    status("   ⊕  Creating pricing repository...")
    create_pricing_repository(status)

    # 2. Deploy Cloud Function FIRST (creates App Engine SA if needed)
    status("   ⏳  Deploying cloud function...")
    deploy_cloud_function(status)

    # 3. Grant OIDC permissions AFTER deploy (App Engine SA now exists)
    grant_actAs_permission(status)

    # 4. Bootstrap pricing (check existing or fetch fresh - needs repo to upload!)
    status("   ⊕  Fetching initial pricing...")
    bootstrap_pricing(status)

    # 5. Create Cloud Scheduler
    status("   ⊕  Creating scheduler (every 20 min)...")
    create_scheduler(status)

    status("   [green]✓    Pricing infrastructure deployed[/green]")
    status("   [cyan]⚡Pricing infrastructure passed - Roger![/cyan]")

    return True


def create_pricing_repository(status):
    """Create generic repository for pricing storage if it doesn't exist"""
    # Check if repository exists
    result = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            "arr-coc-pricing",
            f"--location={REGION}",
            f"--project={PROJECT_ID}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        # Repository already exists
        return

    # Create generic repository
    result = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "create",
            "arr-coc-pricing",
            f"--repository-format=generic",
            f"--location={REGION}",
            f"--project={PROJECT_ID}",
            "--description=ARR-COC pricing data storage",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Repository creation failed: {result.stderr}")


def bootstrap_pricing(status):
    """
    Check Artifact Registry for pricing, fetch fresh if needed.

    Flow:
    1. Try to download from Artifact Registry
    2. If exists → verify (5 prices + recency)
    3. If valid → skip 1000 SKU fetch
    4. If not → fetch fresh from GCP Billing API (~1 minute)
    5. Upload to Artifact Registry
    6. Never saves to local file (Artifact Registry is single source of truth)
    """
    from CLI.shared.pricing import (
        fetch_pricing_no_save,
        get_pricing_age_minutes,
        upload_pricing_to_artifact_registry,
    )

    # Try to fetch existing pricing from Artifact Registry
    try:
        existing_data, version, file_size_kb = fetch_pricing_no_save()

        # Schema validation: Auto-detect from pricing/pricing_config.py
        # This imports the canonical schema definition shared by Cloud Function
        expected_fields = get_required_fields()

        # Check for schema mismatch (missing fields or empty required fields)
        schema_valid = True
        missing_fields = []

        for field, should_have_data in expected_fields.items():
            if field not in existing_data:
                schema_valid = False
                missing_fields.append(f"{field} (missing)")
            elif should_have_data:
                # For dict fields, check if populated (not empty)
                if isinstance(existing_data[field], dict) and len(existing_data[field]) == 0:
                    schema_valid = False
                    missing_fields.append(f"{field} (empty)")

        if not schema_valid:
            status(
                f"   ⚠️  Pricing schema mismatch: {', '.join(missing_fields)} - fetching fresh..."
            )
            raise ValueError(f"Schema validation failed: {', '.join(missing_fields)}")

        # Calculate age
        age_minutes = get_pricing_age_minutes(existing_data)

        # Format age message
        if age_minutes < 1:
            age_str = "just now"
        elif age_minutes < 60:
            age_str = f"{int(age_minutes)} minutes ago"
        else:
            hours = int(age_minutes / 60)
            age_str = f"{hours} hour{'s' if hours > 1 else ''} ago"

        # CRITICAL: Only use existing pricing if fresh enough (< rotation interval)
        # Rotation interval = 20 min (matches scheduler: */20 * * * *)
        if age_minutes < SCHEDULER_INTERVAL_MINUTES:
            status(
                f"   🪙  Using existing pricing ({file_size_kb:.1f} KB, version {version}, {age_str})"
            )
            # Trigger Cloud Function and confirm it started
            status("   🚀  Triggering Cloud Function (first run)...")
            _trigger_and_verify_function(status)
            return
        else:
            status(
                f"   ⚠️  Existing pricing is {age_str} (> {SCHEDULER_INTERVAL_MINUTES} min) - fetching fresh..."
            )
    except FileNotFoundError:
        status("   ⊕  No existing pricing data - fetching fresh....")
    except Exception as e:
        status("   ✗    Could not retrieve pricing - fetching fresh...")

    # No valid existing data - fetch fresh
    status("      🔄  Fetching fresh GCP pricing (~1 minute)...")

    # Fetch pricing from GCP Billing API (inline, no external module)
    pricing_data = _fetch_pricing_inline(status)

    # Sort all pricing lists by price (cheapest first)
    for region_data in pricing_data.get('c3_machines', {}).values():
        if 'cpu_per_core_spot' in region_data:
            region_data['cpu_per_core_spot'].sort(key=lambda x: x['price'])
        if 'ram_per_gb_spot' in region_data:
            region_data['ram_per_gb_spot'].sort(key=lambda x: x['price'])

    for region_data in pricing_data.get('e2_machines', {}).values():
        if 'cpu_per_core_ondemand' in region_data:
            region_data['cpu_per_core_ondemand'].sort(key=lambda x: x['price'])
        if 'ram_per_gb_ondemand' in region_data:
            region_data['ram_per_gb_ondemand'].sort(key=lambda x: x['price'])

    for region_data in pricing_data.get('gpus_spot', {}).values():
        region_data.sort(key=lambda x: x['price'])

    for region_data in pricing_data.get('gpus_ondemand', {}).values():
        region_data.sort(key=lambda x: x['price'])

    # Count regions and GPU types
    c3_regions = len(pricing_data.get('c3_machines', {}))
    e2_regions = len(pricing_data.get('e2_machines', {}))
    gpu_spot_regions = len(pricing_data.get('gpus_spot', {}))
    gpu_ondemand_regions = len(pricing_data.get('gpus_ondemand', {}))

    # Count GPU SKUs by type (now working with lists of SKU objects)
    gpu_types_spot = {}
    gpu_types_ondemand = {}
    for region_data in pricing_data.get('gpus_spot', {}).values():
        for sku in region_data:
            desc = sku['description']
            for gpu_type in ['T4', 'L4', 'V100', 'P4', 'P100', 'A100', 'H100', 'H200']:
                if gpu_type in desc:
                    gpu_types_spot[gpu_type] = gpu_types_spot.get(gpu_type, 0) + 1
    for region_data in pricing_data.get('gpus_ondemand', {}).values():
        for sku in region_data:
            desc = sku['description']
            for gpu_type in ['T4', 'L4', 'V100', 'P4', 'P100', 'A100', 'H100', 'H200']:
                if gpu_type in desc:
                    gpu_types_ondemand[gpu_type] = gpu_types_ondemand.get(gpu_type, 0) + 1

    status(f"           ✓    Pricing fetched")
    status(f"                • C3 machines (spot): {c3_regions} regions")
    status(f"                • E2 machines (on-demand): {e2_regions} regions")

    # GPU spot breakdown
    if gpu_types_spot:
        gpu_spot_summary = ", ".join([f"{gpu}={count}" for gpu, count in sorted(gpu_types_spot.items())])
        status(f"                • GPUs (spot): {gpu_spot_regions} regions - {gpu_spot_summary}")
    else:
        status(f"                • GPUs (spot): {gpu_spot_regions} regions")

    # GPU on-demand breakdown
    if gpu_types_ondemand:
        gpu_ondemand_summary = ", ".join([f"{gpu}={count}" for gpu, count in sorted(gpu_types_ondemand.items())])
        status(f"                • GPUs (on-demand): {gpu_ondemand_regions} regions - {gpu_ondemand_summary}")
    else:
        status(f"                • GPUs (on-demand): {gpu_ondemand_regions} regions")

    # Upload to Artifact Registry (no local save)
    upload_pricing_to_artifact_registry(pricing_data)

    # Trigger Cloud Function and confirm it started
    status("      🚀  Triggering Cloud Function (first run)...")
    _trigger_and_verify_function(status)


def _fetch_pricing_inline(status):
    """
    Fetch pricing from GCP Billing API (bootstrap only).

    NOTE: Cloud Billing API = GCP's public price catalog API.
    Provides spot prices for compute, GPUs, storage across all regions.
    Does NOT access billing accounts or payment info - just public pricing.
    """
    import urllib.parse
    from datetime import datetime

    # Get access token
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to get access token. Run: gcloud auth login")

    access_token = result.stdout.strip()

    # Fetch all pricing from public catalog
    # Structure: Store ALL SKUs as lists (sorted by price after collection)
    pricing_data = {
        "updated": datetime.utcnow().isoformat() + "Z",
        "c3_machines": {},  # {region: {cpu_per_core_spot: [skus], ram_per_gb_spot: [skus]}}
        "e2_machines": {},  # {region: {cpu_per_core_ondemand: [skus], ram_per_gb_ondemand: [skus]}}
        "gpus_spot": {},    # {region: [skus]}
        "gpus_ondemand": {},  # {region: [skus]} (includes commitment pricing!)
    }

    page_token = None
    skus_checked = 0

    while True:
        # Query API
        url = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"
        params = {"pageSize": 500}
        if page_token:
            params["pageToken"] = page_token

        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        # Use retry wrapper (secure - token not visible in ps aux!)
        response = run_requests_with_retry(
            "GET",
            full_url,
            headers={"Authorization": f"Bearer {access_token}"},
            max_retries=3,
            timeout=30,
            operation_name="fetch Cloud Billing API pricing",
        )

        if response.status_code != 200:
            # Sanitize error (don't leak auth token)
            raise RuntimeError(f"API request failed: HTTP {response.status_code}")

        data = response.json()

        if not data or "skus" not in data:
            break

        for sku in data["skus"]:
            skus_checked += 1

            # Progress every 5000 SKUs
            if skus_checked % 5000 == 0:
                status(f"           📄  Checked {skus_checked} SKUs...")

            description = sku.get("description", "").lower()
            category = sku.get("category", {})
            usage_type = category.get("usageType", "")
            regions = sku.get("serviceRegions", [])

            # Extract price with field validation
            try:
                pricing = sku.get("pricingInfo", [{}])[0]
                expr = pricing.get("pricingExpression", {})
                rates = expr.get("tieredRates", [])
                if rates:
                    price_data = rates[0].get("unitPrice", {})
                    nanos_raw = price_data.get("nanos", 0)
                    units_raw = price_data.get("units", 0) or 0

                    # Try converting to float (GCP sometimes returns strings)
                    # Keep as floats for precision (prices like $0.00513/hr)
                    try:
                        nanos = float(nanos_raw)  # Handles strings like "123" or "123.45"
                    except (ValueError, TypeError):
                        status(f"   ⚠️  Cannot convert 'nanos' to number: {nanos_raw} (type: {type(nanos_raw).__name__})")
                        status(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                        nanos = 0.0

                    try:
                        units = float(units_raw)  # Handles strings like "123" or "123.45"
                    except (ValueError, TypeError):
                        status(f"   ⚠️  Cannot convert 'units' to number: {units_raw} (type: {type(units_raw).__name__})")
                        status(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                        units = 0.0
                    price = units + (nanos / 1_000_000_000.0)

                    # Validate price is reasonable
                    if price < 0:
                        status(f"   ⚠️  Negative price: ${price:.6f}")
                        status(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                        price = 0.0

                    if price > 10000:  # $10k/hour is suspicious
                        status(f"   ⚠️  Suspiciously high price: ${price:.2f}/hour")
                        status(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                else:
                    continue
            except Exception as e:
                # Silent skip for unparseable SKUs (keep it clean)
                continue

            # Filter for C3 spot pricing
            if (
                "c3" in description
                and "spot" in description
                and usage_type == "Preemptible"
            ):
                for region in regions:
                    if region == "global":
                        continue
                    if region not in pricing_data["c3_machines"]:
                        pricing_data["c3_machines"][region] = {
                            "cpu_per_core_spot": [],
                            "ram_per_gb_spot": []
                        }

                    # Store full SKU data (all SKUs, sorted by price later)
                    sku_data = {
                        "price": price,
                        "description": sku.get("description", ""),
                        "sku_id": sku.get("skuId", ""),
                        "usage_type": usage_type
                    }

                    if "cpu" in description and "core" in description:
                        pricing_data["c3_machines"][region]["cpu_per_core_spot"].append(sku_data)
                    elif "ram" in description:
                        pricing_data["c3_machines"][region]["ram_per_gb_spot"].append(sku_data)

            # Filter for E2 on-demand pricing
            if (
                "e2" in description
                and "instance" in description
                and "preemptible" not in description
                and "spot" not in description
            ):
                for region in regions:
                    if region == "global":
                        continue
                    if region not in pricing_data["e2_machines"]:
                        pricing_data["e2_machines"][region] = {
                            "cpu_per_core_ondemand": [],
                            "ram_per_gb_ondemand": []
                        }

                    # Store full SKU data (all SKUs, sorted by price later)
                    sku_data = {
                        "price": price,
                        "description": sku.get("description", ""),
                        "sku_id": sku.get("skuId", ""),
                        "usage_type": usage_type
                    }

                    if "core" in description:
                        pricing_data["e2_machines"][region]["cpu_per_core_ondemand"].append(sku_data)
                    elif "ram" in description:
                        pricing_data["e2_machines"][region]["ram_per_gb_ondemand"].append(sku_data)

            # Filter for GPU pricing (ALL types - spot, on-demand, commitment)
            if "gpu" in description or "tpu" in description:
                desc_full = sku.get("description", "")

                # Store full SKU data (including commitment tiers!)
                sku_data = {
                    "price": price,
                    "description": desc_full,
                    "sku_id": sku.get("skuId", ""),
                    "usage_type": usage_type
                }

                if "Spot" in desc_full or "Preemptible" in desc_full:
                    # Spot/Preemptible pricing
                    for region in regions:
                        if region == "global":
                            continue
                        if region not in pricing_data["gpus_spot"]:
                            pricing_data["gpus_spot"][region] = []
                        pricing_data["gpus_spot"][region].append(sku_data)
                else:
                    # On-demand + Commitment pricing (store ALL in gpus_ondemand)
                    # Consumer can filter by usage_type later
                    for region in regions:
                        if region == "global":
                            continue
                        if region not in pricing_data["gpus_ondemand"]:
                            pricing_data["gpus_ondemand"][region] = []
                        pricing_data["gpus_ondemand"][region].append(sku_data)

        # Next page
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return pricing_data


def _trigger_and_verify_function(status):
    """
    Trigger Cloud Function and watch logs for startup signal.

    Times out after 30 seconds. Success = function started running.
    """
    import time

    # Trigger function using gcloud (handles authentication automatically)
    # Run in background - don't wait for completion
    subprocess.Popen(
        [
            "gcloud",
            "functions",
            "call",
            FUNCTION_NAME,
            "--gen2",
            f"--region={REGION}",
            f"--project={PROJECT_ID}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Watch logs for startup signal (90s timeout for cold start)
    status("   ⊕  Watching logs for startup signal...")

    start_time = time.time()
    timeout = 90  # Increased for cold start
    check_count = 0

    while time.time() - start_time < timeout:
        check_count += 1
        elapsed = int(time.time() - start_time)

        # Show progress every 10 seconds
        if check_count % 5 == 0:  # Every 10 seconds (5 checks × 2 sec)
            status(f"      [dim]... checking ({elapsed}s elapsed)[/dim]")

        # Query Cloud Logging for recent logs (Gen2 functions use cloud_run_revision)
        result = subprocess.run(
            [
                "gcloud",
                "logging",
                "read",
                f'resource.type="cloud_run_revision" AND resource.labels.service_name="{FUNCTION_NAME}" AND textPayload:"PRICING_RUNNER_STARTED"',
                "--limit=1",
                "--format=value(textPayload)",
                f"--project={PROJECT_ID}",
                "--freshness=2m",  # Increased freshness window
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if "PRICING_RUNNER_STARTED" in result.stdout:
            status("")
            status(
                f"   [green]✓    Cloud Function verified (started in {elapsed}s)[/green]"
            )
            return

        time.sleep(2)  # Poll every 2 seconds

    # Timeout - function might still be cold starting
    status(
        "   [yellow]Log verification timeout (90s) - function may still be starting[/yellow]"
    )
    status("      Scheduler will trigger again in 20 minutes")


def grant_actAs_permission(status):
    """
    Grant Service Account User role to current user (for OIDC) with idempotent retry logic.

    Idempotency: Checks if permission already exists before trying to grant.
    Retry: 3 attempts with exponential backoff (2s, 4s) for race conditions.
    """
    import subprocess

    # Get current user email
    result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        status("   Could not get current user email")
        raise RuntimeError("Failed to get gcloud account")

    user_email = result.stdout.strip()
    service_account = f"{PROJECT_ID}@appspot.gserviceaccount.com"

    # Define the operation to retry
    def try_grant_permission():
        # IDEMPOTENCY: Check if permission already exists BEFORE trying to grant
        check_result = subprocess.run(
            [
                "gcloud",
                "iam",
                "service-accounts",
                "get-iam-policy",
                service_account,
                f"--project={PROJECT_ID}",
                "--format=json",
            ],
            capture_output=True,
            text=True,
        )

        if check_result.returncode == 0:
            # Parse policy to check if binding already exists
            try:
                import json

                policy = json.loads(check_result.stdout)
                bindings = policy.get("bindings", [])

                for binding in bindings:
                    if binding.get("role") == "roles/iam.serviceAccountUser":
                        members = binding.get("members", [])
                        if f"user:{user_email}" in members:
                            # Permission already exists - idempotent success!
                            return (True, None)
            except (json.JSONDecodeError, KeyError):
                pass  # If we can't parse, just try to grant

        # Permission doesn't exist - grant it
        result = subprocess.run(
            [
                "gcloud",
                "iam",
                "service-accounts",
                "add-iam-policy-binding",
                service_account,
                f"--member=user:{user_email}",
                "--role=roles/iam.serviceAccountUser",
                '--condition=expression=resource.service=="cloudscheduler.googleapis.com",title=OIDCSchedulerOnly',
                f"--project={PROJECT_ID}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return (True, None)
        else:
            # Check if error says it already exists (race condition - another process just added it)
            if (
                "already has" in result.stderr.lower()
                or "already exists" in result.stderr.lower()
            ):
                return (True, None)  # Success - permission exists now
            else:
                error_msg = f"Error: {result.stderr[:200]}"
                return (False, error_msg)

    # Retry with backoff (4 attempts: 0s, 1s, 4s, 8s delays)
    success, error_msg = retry_with_backoff(
        try_grant_permission, max_attempts=4, operation_name="OIDC permission grant"
    )

    if not success:
        status("")
        status("   Failed to grant OIDC permissions (after 4 attempts)")
        status(f"      User: {user_email}")
        status(f"      Service Account: {service_account}")
        if error_msg:
            status(f"      Error: {error_msg[:100]}")
        status("")
        raise RuntimeError("OIDC permission grant failed - setup cannot continue")
    # Silent success - only show errors


def deploy_cloud_function(status):
    """
    Deploy Cloud Function to GCP with timeout and retry.

    Timeout: 10 minutes per attempt
    Retry: 2 attempts total
    Progress: Shows early status + streaming output
    """
    import os
    import subprocess
    from pathlib import Path

    DEPLOY_TIMEOUT_SECONDS = 600  # 10 minutes per attempt

    # Enable APIs (show progress early)
    for api in [
        "cloudfunctions.googleapis.com",
        "cloudbuild.googleapis.com",
        "cloudbilling.googleapis.com",
    ]:
        result = subprocess.run(
            ["gcloud", "services", "enable", api, f"--project={PROJECT_ID}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            status(f"            API enable warning ({api}):")
            status(f"            {result.stderr}")

    function_dir = (
        Path(__file__).parent.parent / "shared" / "pricing" / "cloud_function"
    )

    env = os.environ.copy()
    env["CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK"] = "1"

    # STEP-LEVEL RETRY: Try deployment up to 4 times (not entire setup, just this step!)
    # Handles race conditions and transient GCP errors (404, 409, etc.)
    # Retry delays: 1s, 4s, 8s (cloud-optimized backoff)
    RETRY_DELAYS = [0, 1, 4, 8]
    attempt_errors = []  # Track errors from each attempt for final reporting
    for attempt in [1, 2, 3, 4]:
        if attempt > 1:
            delay = RETRY_DELAYS[attempt - 1]
            status(f"               Retry #{attempt - 1} - waiting {delay}s...")
            time.sleep(delay)  # Use cloud-optimized backoff: 1s, 4s, or 8s

        # Stream output with indentation (Popen for line-by-line processing with timeout)
        import queue
        import sys
        import threading

        status(
            f"               Starting deployment (timeout: {DEPLOY_TIMEOUT_SECONDS // 60} min)..."
        )

        process = subprocess.Popen(
            [
                "gcloud",
                "functions",
                "deploy",
                FUNCTION_NAME,
                "--gen2",
                f"--region={REGION}",
                "--runtime=python312",
                "--entry-point=fetch_pricing",
                f"--source={function_dir}",
                "--trigger-http",
                "--allow-unauthenticated",
                "--timeout=540s",
                "--memory=512MB",
                "--max-instances=1",
                f"--project={PROJECT_ID}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env,
        )

        # Stream output with timeout monitoring
        start_time = time.time()
        returncode = None

        try:
            for line in process.stdout:
                sys.stdout.write(f"               {line}")
                sys.stdout.flush()

                # Check timeout while streaming
                elapsed = time.time() - start_time
                if elapsed > DEPLOY_TIMEOUT_SECONDS:
                    process.kill()
                    status(
                        f"               ⚠️  Deployment timeout ({DEPLOY_TIMEOUT_SECONDS // 60} min) - killing process"
                    )
                    returncode = -1
                    break

            # Wait for process to finish (if not killed)
            if returncode is None:
                try:
                    returncode = process.wait(timeout=30)  # 30s grace period
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = -1
                    status("               ⚠️  Process cleanup timeout")

        except Exception as e:
            process.kill()
            status(f"               ❌ Deployment error: {str(e)[:100]}")
            returncode = -1

        if returncode == 0:
            # Success
            status("               ☁️  Cloud Function created successfully")
            status("")
            return

        # Failed - check if function actually deployed (gcloud crash bug OR race condition)
        # Wait a bit if this might be a race (another process deploying)
        status("               Checking deployment status...")
        time.sleep(5)  # Let other process finish if racing

        check = subprocess.run(
            [
                "gcloud",
                "functions",
                "describe",
                FUNCTION_NAME,
                "--gen2",
                f"--region={REGION}",
                f"--project={PROJECT_ID}",
                "--format=value(state)",
            ],
            capture_output=True,
            text=True,
        )

        if check.returncode == 0 and check.stdout.strip() == "ACTIVE":
            # Function deployed successfully (either gcloud crash OR another parallel process deployed it)
            status(
                "               ☁️  Cloud Function verified ACTIVE (deployed by this or parallel process)"
            )
            status("")
            return

        # Actual failure - capture error and retry if not last attempt
        error_msg = f"Attempt {attempt}: Deploy failed (returncode={returncode}), function not ACTIVE"
        attempt_errors.append(error_msg)

        if attempt < 4:
            status(
                "               Deployment failed - fuck it, restarting THIS STEP..."
            )
            continue  # Retry loop will handle backoff (1s, 4s, or 8s)

    # All 4 attempts failed - show detailed error report
    status("")
    status("   ❌ Cloud Function deployment failed after 4 attempts!")
    status("")
    status("   Attempts summary:")
    for i, err in enumerate(attempt_errors, 1):
        status(f"      {i}. {err}")
    status("")
    status("   Retry delays used: 1s, 4s, 8s (total 13s)")
    status("   This is STEP-level retry (not entire setup)")
    status("")
    status("   💡 Next steps:")
    status("      1. Check GCP console for function status")
    status("      2. Review Cloud Build logs if build failed")
    status("      3. Verify APIs enabled: cloudfunctions, cloudbuild, run")
    status("      4. Run 'python CLI/cli.py teardown' then retry setup")
    status("")
    raise RuntimeError(
        f"Cloud Function deployment failed after {len(attempt_errors)} attempts"
    )


def create_scheduler(status):
    """Create Cloud Scheduler job"""
    # Enable Cloud Scheduler API first (wait for completion - no async)
    # Silent - only show errors
    result = subprocess.run(
        [
            "gcloud",
            "services",
            "enable",
            "cloudscheduler.googleapis.com",
            f"--project={PROJECT_ID}",
            # No --async flag means it waits for completion
        ],
        capture_output=True,
        text=True,
    )

    # Silent success for API enable
    if result.returncode != 0:
        status("")
        status("            Cloud Scheduler API enable warning:")
        status(f"            {result.stderr}")
        status("")

    # Get function URL (needed for scheduler)
    result = subprocess.run(
        [
            "gcloud",
            "functions",
            "describe",
            FUNCTION_NAME,
            "--gen2",
            f"--region={REGION}",
            "--format=value(serviceConfig.uri)",
            f"--project={PROJECT_ID}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to get function URL: {result.stderr}")

    function_url = result.stdout.strip()
    expected_schedule = f"*/{SCHEDULER_INTERVAL_MINUTES} * * * *"  # Every N minutes
    expected_sa = f"{PROJECT_ID}@appspot.gserviceaccount.com"

    # Check if scheduler job exists AND is correctly configured (silent)

    check_result = subprocess.run(
        [
            "gcloud",
            "scheduler",
            "jobs",
            "describe",
            SCHEDULER_JOB,
            f"--location={REGION}",
            f"--project={PROJECT_ID}",
            "--format=json",
        ],
        capture_output=True,
        text=True,
    )

    scheduler_exists = False
    scheduler_healthy = False
    needs_recreation = False

    if check_result.returncode == 0:
        # Scheduler exists - check configuration
        import json

        try:
            job_info = json.loads(check_result.stdout)

            scheduler_exists = True

            # Check critical configuration
            actual_schedule = job_info.get("schedule", "")
            actual_uri = job_info.get("httpTarget", {}).get("uri", "")
            actual_sa = (
                job_info.get("httpTarget", {})
                .get("oidcToken", {})
                .get("serviceAccountEmail", "")
            )
            state = job_info.get("state", "UNKNOWN")

            config_ok = (
                actual_schedule == expected_schedule
                and actual_uri == function_url
                and actual_sa == expected_sa
                and state == "ENABLED"
            )

            if config_ok:
                scheduler_healthy = True
            else:
                needs_recreation = True

        except (json.JSONDecodeError, KeyError) as e:
            needs_recreation = True

    # Decide action
    skip_creation = scheduler_exists and scheduler_healthy

    if not skip_creation:
        # Need to create/recreate scheduler
        import os

        env = os.environ.copy()
        env["CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK"] = "1"

        if scheduler_exists and needs_recreation:
            # Delete broken scheduler first (silent)
            delete_result = subprocess.run(
                [
                    "gcloud",
                    "scheduler",
                    "jobs",
                    "delete",
                    SCHEDULER_JOB,
                    f"--location={REGION}",
                    f"--project={PROJECT_ID}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                env=env,
            )

            if delete_result.returncode == 0:
                scheduler_exists = False
            else:
                # Show error if delete fails
                status(
                    f"               Could not delete scheduler: {delete_result.stderr}"
                )
                status("               Attempting creation anyway...")

        # Create scheduler job (silent)
        result = subprocess.run(
            [
                "gcloud",
                "scheduler",
                "jobs",
                "create",
                "http",
                SCHEDULER_JOB,
                f"--location={REGION}",
                f"--schedule={expected_schedule}",
                f"--uri={function_url}",
                "--http-method=GET",
                # SECURITY: Use OIDC token for authentication
                f"--oidc-service-account-email={expected_sa}",
                f"--oidc-token-audience={function_url}",
                f"--project={PROJECT_ID}",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            status("")
            status("            Cloud Scheduler creation failed!")
            status(f"            {result.stderr}")
            raise RuntimeError("Scheduler creation failed (see error above)")

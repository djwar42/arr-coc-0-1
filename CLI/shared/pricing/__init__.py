"""
GCP Pricing Operations

Fetches C3/E2/GPU pricing from Artifact Registry and provides helper functions
for extracting pricing tiers and calculating machine costs.
"""

# <claudes_code_comments>
# ** Function List **
# _get_access_token() - Gets OAuth2 token from gcloud (secure - never logged)
# _get_latest_version() - Gets latest version via Artifact Registry API (HTTP)
# fetch_pricing_no_save() - Fetches pricing via gcloud CLI (system temp, auto-cleaned)
# upload_pricing_to_artifact_registry(pricing_data) - Uploads via gcloud CLI (system temp, auto-cleaned)
# get_pricing_age_minutes(pricing_data) - Calculates pricing age in minutes
# format_pricing_age(age_minutes) - Formats age for display
# get_spot_price(sku_list) - Returns cheapest spot/preemptible price from SKU list
# get_standard_price(sku_list) - Returns cheapest on-demand (standard) price from SKU list
# get_commitment_1yr_price(sku_list) - Returns cheapest 1-year commitment price from SKU list
# get_commitment_3yr_price(sku_list) - Returns cheapest 3-year commitment price from SKU list
# all_prices(sku_list) - Returns all pricing tiers with names, prices, metadata
# get_machine_hourly_cost(machine_type, region) - Calculates total hourly cost for C3/E2 machines
#
# ** Technical Review **
# Manages C3/E2/GPU pricing data with Artifact Registry as single source of truth. NO local
# caching! Both fetch and upload use gcloud CLI (only reliable method for generic repos).
# Temp files created in SYSTEM temp (/var/folders/.../T/ on Mac) - NEVER in codebase!
# Auto-cleaned by context manager. Token from gcloud auth (OAuth2, short-lived, secure).
# Versions are timestamp-based (1.0.YYYYMMDD-HHMMSS). Age calculation parses ISO
# timestamps from pricing_data['updated']. Formatting: <1min="just now", <60min="X min
# ago", <24h="X hours ago", >=24h="X days ago". Helper functions extract specific pricing
# tiers from SKU lists (which are pre-sorted by price, cheapest first). Filters by
# usage_type (Spot/Preemptible/OnDemand/COMMIT) or description keywords (1 Year/3 Year).
# all_prices() returns complete list for user comparison. get_machine_hourly_cost() parses
# machine type strings (c3-standard-176, E2_HIGHCPU_8), calculates RAM from vCPUs, fetches
# component pricing (CPU + RAM), and returns total hourly cost. Used by MECHA campaign stats.
# Constants: PROJECT_ID, REPOSITORY (arr-coc-pricing), LOCATION (us-central1), PACKAGE
# (gcp-pricing). Security: tokens never logged, errors sanitized (no temp paths leaked).
# RETRY LOGIC: fetch_pricing_no_save() uses run_gcloud_with_retry (3 retries, exponential backoff)
# and _get_latest_version() uses run_requests_with_retry for robust pricing fetch. Critical for
# ZEUS pricing system - prevents "No thunder-ready regions" errors from transient API failures.
# </claudes_code_comments>

import subprocess
import json
from datetime import datetime
import requests

from ..api_helpers import run_requests_with_retry


# PROJECT_ID loaded from config
def _get_project_id():
    from ...config.constants import load_training_config
    return load_training_config().get("GCP_PROJECT_ID", "")

REPOSITORY = "arr-coc-pricing"
LOCATION = "us-central1"
PACKAGE = "gcp-pricing"


def _get_access_token():
    """Get access token from gcloud (secure - never logged)"""
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _get_latest_version():
    """Get the latest version of the pricing package (HTTP - no gcloud CLI)"""
    token = _get_access_token()

    # Artifact Registry API: List versions (newest first)
    url = (
        f"https://artifactregistry.googleapis.com/v1/"
        f"projects/{_get_project_id()}/locations/{LOCATION}/repositories/{REPOSITORY}/"
        f"packages/{PACKAGE}/versions"
        f"?pageSize=1&orderBy=createTime%20desc"
    )

    response = run_requests_with_retry(
        "GET",
        url,
        headers={"Authorization": f"Bearer {token}"},
        max_retries=3,
        timeout=10,
        operation_name="get latest Artifact Registry version",
    )

    if response.status_code == 404:
        raise FileNotFoundError("No pricing versions found")

    if response.status_code != 200:
        # Try to extract error reason from response (e.g., "BILLING_DISABLED")
        error_reason = ""
        try:
            error_data = response.json()
            error_reason = error_data.get("error", {}).get("details", [{}])[0].get("reason", "")
        except Exception:
            pass  # Response might not be JSON or structured differently

        # Include reason if available (helps detect billing issues!)
        if error_reason:
            raise RuntimeError(f"Failed to get latest version: HTTP {response.status_code} ({error_reason})")
        else:
            raise RuntimeError(f"Failed to get latest version: HTTP {response.status_code}")

    data = response.json()
    versions = data.get("versions", [])

    if not versions:
        raise FileNotFoundError("No pricing versions found")

    # Extract version from full path: "projects/.../versions/1.0.20250111-142530" → "1.0.20250111-142530"
    version_path = versions[0]["name"]
    return version_path.split("/")[-1]


def fetch_pricing_no_save():
    """
    Fetch latest pricing from Artifact Registry (gcloud CLI - temp files auto-cleaned).

    Downloads using gcloud CLI (only method that works for generic repos).
    Temp files are automatically cleaned up by context manager.

    Returns:
        tuple: (pricing_data, version, file_size_kb)
            - pricing_data: dict with C3 machines + GPUs spot/on-demand
            - version: str like "1.0.20251112-165852"
            - file_size_kb: float of file size in KB

    Raises:
        FileNotFoundError: No pricing found in Artifact Registry
        RuntimeError: Download failed
    """
    import tempfile
    from pathlib import Path
    from ..api_helpers import run_gcloud_with_retry

    version = _get_latest_version()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download using gcloud with retry logic (handles transient failures!)
        result = run_gcloud_with_retry(
            cmd=[
                "gcloud", "artifacts", "generic", "download",
                f"--project={_get_project_id()}",
                f"--repository={REPOSITORY}",
                f"--location={LOCATION}",
                f"--package={PACKAGE}",
                f"--version={version}",
                f"--destination={tmpdir}",
            ],
            max_retries=3,
            timeout=30,
            operation_name="fetch pricing data",
        )

        if result.returncode != 0:
            if "NOT_FOUND" in result.stderr or "not found" in result.stderr.lower():
                raise FileNotFoundError("Pricing not found in Artifact Registry")
            else:
                # Sanitize error (don't leak temp paths)
                error_msg = result.stderr.replace(str(tmpdir), "[temp]")
                raise RuntimeError(f"Download failed: {error_msg[:200]}")

        # Look for gcp-live-pricing.json (uploaded by Cloud Function)
        pricing_files = list(Path(tmpdir).rglob("gcp-live-pricing.json"))
        if not pricing_files:
            raise FileNotFoundError("gcp-live-pricing.json not found after download")

        # Get file size
        file_path = pricing_files[0]
        file_size_bytes = file_path.stat().st_size
        file_size_kb = file_size_bytes / 1024

        # Load and return with metadata
        with open(file_path) as f:
            pricing_data = json.load(f)
            return (pricing_data, version, file_size_kb)




def upload_pricing_to_artifact_registry(pricing_data):
    """Upload pricing JSON to Artifact Registry (gcloud CLI - handles large files)"""
    import tempfile
    from pathlib import Path

    # Generate timestamp-based version (e.g., "1.0.20250111-142530")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version = f"1.0.{timestamp}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save to temp file (gcloud CLI requires file path for upload)
        pricing_file = Path(tmpdir) / "gcp-live-pricing.json"
        with open(pricing_file, "w") as f:
            json.dump(pricing_data, f, indent=2)

        # Get file size for progress message
        file_size_kb = pricing_file.stat().st_size / 1024
        print(f"      Uploading to Artifact Registry ({file_size_kb:.1f} KB, version {version})...")

        # Upload to Artifact Registry using gcloud (handles large files + chunked upload)
        result = subprocess.run(
            [
                "gcloud", "artifacts", "generic", "upload",
                f"--project={_get_project_id()}",
                f"--repository={REPOSITORY}",
                f"--location={LOCATION}",
                f"--package={PACKAGE}",
                f"--version={version}",
                f"--source={pricing_file}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Sanitize error (don't leak temp paths)
            raise RuntimeError(f"Upload failed: {result.stderr[:200]}")


def get_pricing_age_minutes(pricing_data):
    """Calculate pricing age in minutes from 'updated' timestamp"""
    updated_str = pricing_data.get("updated")
    if not updated_str:
        return float('inf')  # Unknown age

    # Parse ISO timestamp
    updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
    age = datetime.now(updated.tzinfo) - updated
    return age.total_seconds() / 60


def format_pricing_age(age_minutes):
    """Format age for display"""
    if age_minutes < 1:
        return "just now"
    elif age_minutes < 60:
        return f"{int(age_minutes)} minute{'s' if age_minutes > 1 else ''} ago"
    elif age_minutes < 1440:  # < 24 hours
        hours = int(age_minutes / 60)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = int(age_minutes / 1440)
        return f"{days} day{'s' if days > 1 else ''} ago"


# ============================================================================
# Pricing Helper Functions (for extracting specific tiers from SKU lists)
# ============================================================================

def get_spot_price(sku_list):
    """
    Get cheapest spot/preemptible price from SKU list.

    SKU lists are pre-sorted by price (cheapest first), so this filters for
    spot/preemptible SKUs and returns the first match.

    Args:
        sku_list: List of SKU dicts with {price, description, sku_id, usage_type}

    Returns:
        float: Cheapest spot price, or None if no spot SKUs found

    Example:
        >>> cpu_skus = pricing_data["c3_machines"]["us-central1"]["cpu_per_core_spot"]
        >>> get_spot_price(cpu_skus)
        0.012
    """
    spot_skus = [s for s in sku_list if s["usage_type"] in ["Preemptible", "Spot"]]
    return spot_skus[0]["price"] if spot_skus else None


def get_standard_price(sku_list):
    """
    Get cheapest on-demand (standard) price from SKU list.

    Args:
        sku_list: List of SKU dicts with {price, description, sku_id, usage_type}

    Returns:
        float: Cheapest on-demand price, or None if no on-demand SKUs found

    Example:
        >>> cpu_skus = pricing_data["e2_machines"]["us-central1"]["cpu_per_core_ondemand"]
        >>> get_standard_price(cpu_skus)
        0.0218
    """
    ondemand_skus = [s for s in sku_list if s["usage_type"] == "OnDemand"]
    return ondemand_skus[0]["price"] if ondemand_skus else None


def get_commitment_1yr_price(sku_list):
    """
    Get cheapest 1-year commitment price from SKU list.

    Searches for "1 Year" or "1yr" in SKU descriptions.

    Args:
        sku_list: List of SKU dicts with {price, description, sku_id, usage_type}

    Returns:
        float: Cheapest 1-year commitment price, or None if not found

    Example:
        >>> gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]
        >>> get_commitment_1yr_price(gpu_skus)
        0.245
    """
    commit_skus = [
        s for s in sku_list
        if "1 Year" in s["description"] or "1yr" in s["description"].lower()
    ]
    return commit_skus[0]["price"] if commit_skus else None


def get_commitment_3yr_price(sku_list):
    """
    Get cheapest 3-year commitment price from SKU list.

    Searches for "3 Year" or "3yr" in SKU descriptions.

    Args:
        sku_list: List of SKU dicts with {price, description, sku_id, usage_type}

    Returns:
        float: Cheapest 3-year commitment price, or None if not found

    Example:
        >>> gpu_skus = pricing_data["gpus_ondemand"]["us-central1"]
        >>> get_commitment_3yr_price(gpu_skus)
        0.175
    """
    commit_skus = [
        s for s in sku_list
        if "3 Year" in s["description"] or "3yr" in s["description"].lower()
    ]
    return commit_skus[0]["price"] if commit_skus else None


def all_prices(sku_list):
    """
    Get all available pricing tiers with human-readable names and metadata.

    Returns a list of all pricing options (spot, on-demand, commitments) with
    friendly names, prices, and complete SKU metadata. Useful for displaying
    pricing comparisons to users or in TUI/CLI.

    Args:
        sku_list: List of SKU dicts with {price, description, sku_id, usage_type}

    Returns:
        list: List of dicts with {name, price, description, sku_id, usage_type}

    Example:
        >>> gpu_skus = pricing_data["gpus_spot"]["us-central1"]
        >>> for option in all_prices(gpu_skus):
        ...     print(f"{option['name']}: ${option['price']}/hr")
        Spot (Preemptible): $0.14/hr
        On-Demand (Standard): $0.35/hr
        1-Year Commitment: $0.245/hr
        3-Year Commitment: $0.175/hr
    """
    options = []

    # Spot/Preemptible
    spot = get_spot_price(sku_list)
    if spot is not None:
        spot_sku = next(s for s in sku_list if s["usage_type"] in ["Preemptible", "Spot"])
        options.append({
            "name": "Spot (Preemptible)",
            "price": spot,
            "description": spot_sku["description"],
            "sku_id": spot_sku["sku_id"],
            "usage_type": spot_sku["usage_type"]
        })

    # On-Demand (Standard)
    standard = get_standard_price(sku_list)
    if standard is not None:
        standard_sku = next(s for s in sku_list if s["usage_type"] == "OnDemand")
        options.append({
            "name": "On-Demand (Standard)",
            "price": standard,
            "description": standard_sku["description"],
            "sku_id": standard_sku["sku_id"],
            "usage_type": standard_sku["usage_type"]
        })

    # 1-Year Commitment
    commit_1yr = get_commitment_1yr_price(sku_list)
    if commit_1yr is not None:
        commit_1yr_sku = next(
            s for s in sku_list
            if "1 Year" in s["description"] or "1yr" in s["description"].lower()
        )
        options.append({
            "name": "1-Year Commitment",
            "price": commit_1yr,
            "description": commit_1yr_sku["description"],
            "sku_id": commit_1yr_sku["sku_id"],
            "usage_type": commit_1yr_sku["usage_type"]
        })

    # 3-Year Commitment
    commit_3yr = get_commitment_3yr_price(sku_list)
    if commit_3yr is not None:
        commit_3yr_sku = next(
            s for s in sku_list
            if "3 Year" in s["description"] or "3yr" in s["description"].lower()
        )
        options.append({
            "name": "3-Year Commitment",
            "price": commit_3yr,
            "description": commit_3yr_sku["description"],
            "sku_id": commit_3yr_sku["sku_id"],
            "usage_type": commit_3yr_sku["usage_type"]
        })

    return options


def get_machine_hourly_cost(machine_type: str, region: str) -> float:
    """
    Get live hourly cost for a machine type in a region.

    Calculates total machine cost from component pricing (CPU + RAM).

    Args:
        machine_type: e.g. "c3-standard-176" or "E2_HIGHCPU_8"
        region: e.g. "us-west2"

    Returns:
        Hourly cost in USD, or 0.0 if pricing unavailable

    Example:
        >>> get_machine_hourly_cost("c3-standard-176", "us-west2")
        2.45  # $2.45/hour for 176 vCPU + 704 GB RAM (spot)
    """
    try:
        pricing_data, _, _ = fetch_pricing_no_save()

        # C3 machines (spot pricing - for MECHA worker pool)
        if machine_type.startswith("c3-standard-"):
            vcpus = int(machine_type.split("-")[-1])
            ram_gb = vcpus * 4  # C3: 4 GB RAM per vCPU

            c3_region_data = pricing_data.get("c3_machines", {}).get(region, {})
            cpu_skus = c3_region_data.get("cpu_per_core_spot", [])
            ram_skus = c3_region_data.get("ram_per_gb_spot", [])

            cpu_price = get_spot_price(cpu_skus) or 0.0
            ram_price = get_spot_price(ram_skus) or 0.0

            return (vcpus * cpu_price) + (ram_gb * ram_price)

        # E2 machines (on-demand - for CloudBuild default)
        elif machine_type == "E2_HIGHCPU_8":
            vcpus = 8
            ram_gb = 8  # E2_HIGHCPU: 1 GB RAM per vCPU

            e2_region_data = pricing_data.get("e2_machines", {}).get(region, {})
            cpu_skus = e2_region_data.get("cpu_per_core_ondemand", [])
            ram_skus = e2_region_data.get("ram_per_gb_ondemand", [])

            cpu_price = get_standard_price(cpu_skus) or 0.0
            ram_price = get_standard_price(ram_skus) or 0.0

            return (vcpus * cpu_price) + (ram_gb * ram_price)

        else:
            return 0.0

    except Exception:
        # Non-blocking - return 0 if pricing fetch fails
        return 0.0

"""
Cloud Function: GCP Pricing Fetcher

Triggered by Cloud Scheduler every 20 minutes.
Fetches live GCP C3 pricing and stores in Artifact Registry.
"""

# <claudes_code_comments>
# ** Function List **
# get_access_token() - retrieves GCP auth token from metadata server
# fetch_gcp_pricing() - queries Cloud Billing API for complete SKU data (all pricing tiers)
# upload_to_artifact_registry(data) - stores pricing JSON in Artifact Registry
# main(request) - Cloud Function entrypoint triggered by scheduler
#
# ** Technical Review **
# Cloud Function runs every 20 minutes via Cloud Scheduler. Fetches COMPLETE SKU data
# (price + metadata) for C3/E2 machines and all GPU types from GCP Cloud Billing API
# (~30k SKUs). Stores ALL pricing tiers: spot, on-demand, AND commitment (1yr/3yr).
# Data structure: Lists of SKU objects sorted by price (cheapest first). Each SKU includes:
# {price, description, sku_id, usage_type}. Consumer code accesses cheapest via [0]["price"].
# Filters by description keywords (c3/e2/gpu) and preemptible status. Shows progress
# every 5000 SKUs, outputs detailed GPU/machine counts on completion. Uploads versioned
# JSON to Artifact Registry (package: gcp-pricing). Returns pricing data with GPU counts.
# Uses metadata server auth (no keys needed). Schema matches PRICING_SCHEMA in
# pricing/pricing_config.py for validation compatibility.
# </claudes_code_comments>

import functions_framework
import json
import os
import requests
import time
from datetime import datetime
from pathlib import Path


def run_requests_with_retry(
    method: str,
    url: str,
    max_retries: int = 3,
    timeout: int = 30,
    operation_name: str = "HTTP request",
    **kwargs
):
    """Run HTTP request with retry logic and exponential backoff"""
    retry_delay = 2

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                timeout=timeout,
                **kwargs
            )
            return response

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"Failed to execute {operation_name} after {max_retries} attempts: Timeout"
                )

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(
                    f"Failed to execute {operation_name} after {max_retries} attempts: {str(e)}"
                )

    raise RuntimeError(f"Unexpected error in {operation_name}")


def get_access_token():
    """Get access token from metadata server"""
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    headers = {"Metadata-Flavor": "Google"}

    response = run_requests_with_retry(
        "GET",
        metadata_url,
        headers=headers,
        max_retries=3,
        timeout=10,
        operation_name="get GCP metadata token",
    )
    if response.status_code != 200:
        raise RuntimeError(f"Failed to get access token: {response.text}")

    return response.json()["access_token"]


def fetch_gcp_pricing():
    """
    Fetch machine and GPU pricing from GCP Cloud Billing API.

    Fetches: C3 machines (spot), E2 machines (on-demand), GPUs (spot + on-demand + commitment)

    Stores ALL SKUs as lists (sorted by price, cheapest first) with full metadata:
    - price, description, sku_id, usage_type

    Consumer code can access cheapest price: pricing_data[...][region][resource][0]["price"]

    NOTE: Cloud Billing API = GCP's public price catalog API.
    Provides pricing for compute, GPUs, storage across all regions.
    Does NOT access billing accounts or payment info - just public pricing.
    """
    print("🔍 Fetching live GCP pricing...")

    # Get access token from metadata server
    access_token = get_access_token()

    # Query billing API
    pricing_data = {
        "updated": datetime.utcnow().isoformat() + "Z",
        "c3_machines": {},
        "e2_machines": {},  # E2 for CloudBuild default (arr-ml-stack, arr-trainer, arr-vertex-launcher)
        "gpus_spot": {},
        "gpus_ondemand": {},
    }

    # Query API (paginated)
    page_token = None
    skus_checked = 0

    while True:
        # Build URL
        url = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"
        if page_token:
            url += f"?pageToken={page_token}&pageSize=500"
        else:
            url += "?pageSize=500"

        # Fetch page using retry wrapper
        headers = {"Authorization": f"Bearer {access_token}"}
        response = run_requests_with_retry(
            "GET",
            url,
            headers=headers,
            max_retries=3,
            timeout=30,
            operation_name="fetch Cloud Billing API pricing",
        )

        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} {response.text}")

        data = response.json()

        if "skus" not in data:
            break

        # Parse SKUs
        for sku in data["skus"]:
            skus_checked += 1

            description = sku.get("description", "").lower()
            category = sku.get("category", {})
            usage_type = category.get("usageType", "")
            regions = sku.get("serviceRegions", [])

            # Extract price with field validation
            try:
                pricing_info = sku.get("pricingInfo", [{}])[0]
                expr = pricing_info.get("pricingExpression", {})
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
                        print(f"   ⚠️  Cannot convert 'nanos' to number: {nanos_raw} (type: {type(nanos_raw).__name__})")
                        print(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                        nanos = 0.0

                    try:
                        units = float(units_raw)  # Handles strings like "123" or "123.45"
                    except (ValueError, TypeError):
                        print(f"   ⚠️  Cannot convert 'units' to number: {units_raw} (type: {type(units_raw).__name__})")
                        print(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                        units = 0.0
                    price = units + (nanos / 1_000_000_000.0)

                    # Validate price is reasonable
                    if price < 0:
                        print(f"   ⚠️  Negative price: ${price:.6f}")
                        print(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                        price = 0.0

                    if price > 10000:  # $10k/hour is suspicious
                        print(f"   ⚠️  Suspiciously high price: ${price:.2f}/hour")
                        print(f"      SKU: {sku.get('skuId', 'unknown')}, Description: {sku.get('description', '')[:60]}...")
                else:
                    continue
            except Exception as e:
                # Silent skip for unparseable SKUs (keep it clean)
                continue

            # C3 Machines (Spot only - for arr-pytorch-base MECHA builds)
            if "c3" in description and ("spot" in description or "preemptible" in description):
                if "c3d" in description or "c3a" in description:
                    continue

                for region in regions:
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

                    if "core" in description:
                        pricing_data["c3_machines"][region]["cpu_per_core_spot"].append(sku_data)
                    elif "ram" in description:
                        pricing_data["c3_machines"][region]["ram_per_gb_spot"].append(sku_data)

            # E2 Machines (On-demand - for CloudBuild default: arr-ml-stack, arr-trainer, arr-vertex-launcher)
            if "e2 instance" in description and "preemptible" not in description and "spot" not in description:
                for region in regions:
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

            # GPU Pricing (ALL types - for Vertex AI training)
            # Store: spot, on-demand, AND commitment pricing (1yr, 3yr)
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

        # Check for next page
        page_token = data.get("nextPageToken")
        if not page_token:
            break

        if skus_checked % 5000 == 0:
            print(f"📄 Checked {skus_checked} SKUs...")

            # Quick validation checks on partial data (lightweight, inline)
            if skus_checked >= 15000:  # Wait until we have substantial data
                # Count what we have so far
                c3_count = len(pricing_data['c3_machines'])
                e2_count = len(pricing_data['e2_machines'])
                gpu_spot_count = len(pricing_data['gpus_spot'])
                gpu_ondemand_count = len(pricing_data['gpus_ondemand'])

                # Check for missing data (verbose warnings)
                if c3_count == 0:
                    print("   ⚠️  WARNING: No C3 machine pricing found yet (expected ~43 regions)")
                if e2_count == 0:
                    print("   ⚠️  WARNING: No E2 machine pricing found yet (expected ~43 regions)")
                if gpu_spot_count == 0:
                    print("   ⚠️  WARNING: No GPU spot pricing found yet (expected ~43 regions)")
                if gpu_ondemand_count == 0:
                    print("   ⚠️  WARNING: No GPU on-demand pricing found yet (expected ~47 regions)")

                # Check for suspiciously low counts (at 15k+ SKUs we should have most data)
                if c3_count > 0 and c3_count < 30:
                    print(f"   ⚠️  WARNING: Only {c3_count} C3 regions found so far (expected ~43)")
                if e2_count > 0 and e2_count < 30:
                    print(f"   ⚠️  WARNING: Only {e2_count} E2 regions found so far (expected ~43)")
                if gpu_spot_count > 0 and gpu_spot_count < 30:
                    print(f"   ⚠️  WARNING: Only {gpu_spot_count} GPU spot regions found so far (expected ~43)")
                if gpu_ondemand_count > 0 and gpu_ondemand_count < 35:
                    print(f"   ⚠️  WARNING: Only {gpu_ondemand_count} GPU on-demand regions found so far (expected ~47)")

    # Sort all pricing lists by price (cheapest first)
    for region_data in pricing_data['c3_machines'].values():
        if 'cpu_per_core_spot' in region_data:
            region_data['cpu_per_core_spot'].sort(key=lambda x: x['price'])
        if 'ram_per_gb_spot' in region_data:
            region_data['ram_per_gb_spot'].sort(key=lambda x: x['price'])

    for region_data in pricing_data['e2_machines'].values():
        if 'cpu_per_core_ondemand' in region_data:
            region_data['cpu_per_core_ondemand'].sort(key=lambda x: x['price'])
        if 'ram_per_gb_ondemand' in region_data:
            region_data['ram_per_gb_ondemand'].sort(key=lambda x: x['price'])

    for region_data in pricing_data['gpus_spot'].values():
        region_data.sort(key=lambda x: x['price'])

    for region_data in pricing_data['gpus_ondemand'].values():
        region_data.sort(key=lambda x: x['price'])

    c3_regions = len(pricing_data['c3_machines'])
    e2_regions = len(pricing_data['e2_machines'])
    gpu_spot_regions = len(pricing_data['gpus_spot'])
    gpu_ondemand_regions = len(pricing_data['gpus_ondemand'])

    # Count GPU SKUs by type (across all regions)
    gpu_types_spot = {}
    gpu_types_ondemand = {}
    for region_data in pricing_data['gpus_spot'].values():
        for sku in region_data:
            desc = sku['description']
            for gpu_type in ['T4', 'L4', 'V100', 'P4', 'P100', 'A100', 'H100', 'H200']:
                if gpu_type in desc:
                    gpu_types_spot[gpu_type] = gpu_types_spot.get(gpu_type, 0) + 1
    for region_data in pricing_data['gpus_ondemand'].values():
        for sku in region_data:
            desc = sku['description']
            for gpu_type in ['T4', 'L4', 'V100', 'P4', 'P100', 'A100', 'H100', 'H200']:
                if gpu_type in desc:
                    gpu_types_ondemand[gpu_type] = gpu_types_ondemand.get(gpu_type, 0) + 1

    print(f"✅ Pricing fetched: {skus_checked} SKUs checked")
    print(f"   • C3 machines (spot): {c3_regions} regions - for Cloud Build PyTorch compilation")
    print(f"   • E2 machines (on-demand): {e2_regions} regions - for Cloud Build default images")

    # GPU spot breakdown
    gpu_spot_summary = ", ".join([f"{gpu}={count}" for gpu, count in sorted(gpu_types_spot.items())])
    print(f"   • GPUs (spot): {gpu_spot_regions} regions - {gpu_spot_summary}")

    # GPU on-demand breakdown
    gpu_ondemand_summary = ", ".join([f"{gpu}={count}" for gpu, count in sorted(gpu_types_ondemand.items())])
    print(f"   • GPUs (on-demand): {gpu_ondemand_regions} regions - {gpu_ondemand_summary}")

    # ========================================================================
    # VALIDATION CHECKS - Detect weird/missing pricing data
    # ========================================================================
    print("🔍 Running validation checks...")
    validation_warnings = []

    # Check 1: Regional coverage (expected ~43 C3/E2, ~47 GPU regions)
    if c3_regions < 38:  # 10% tolerance
        validation_warnings.append(f"⚠️  C3 regions low: {c3_regions} (expected ~43)")
    if e2_regions < 38:
        validation_warnings.append(f"⚠️  E2 regions low: {e2_regions} (expected ~43)")
    if gpu_spot_regions < 38:
        validation_warnings.append(f"⚠️  GPU spot regions low: {gpu_spot_regions} (expected ~43)")
    if gpu_ondemand_regions < 42:
        validation_warnings.append(f"⚠️  GPU on-demand regions low: {gpu_ondemand_regions} (expected ~47)")

    # Check 2: GPU type coverage (expected counts from Excessive Poking Study)
    gpu_expected = {"T4": 39, "L4": 41, "A100": 80, "V100": 28, "H100": 53, "P100": 29, "P4": 31}
    for gpu_type, expected in gpu_expected.items():
        actual = gpu_types_spot.get(gpu_type, 0)
        if actual < expected * 0.7:  # 30% tolerance
            validation_warnings.append(f"⚠️  {gpu_type} spot regions low: {actual} (expected ~{expected})")

        actual_ondemand = gpu_types_ondemand.get(gpu_type, 0)
        if actual_ondemand < expected * 0.7:
            validation_warnings.append(f"⚠️  {gpu_type} on-demand SKUs low: {actual_ondemand} (expected ~{expected})")

    # Check 3: Commitment pricing coverage (should be in ~85% of regions)
    regions_with_commitment = 0
    total_ondemand_regions = len(pricing_data['gpus_ondemand'])
    for region_data in pricing_data['gpus_ondemand'].values():
        has_commitment = any("Year" in sku['description'] for sku in region_data)
        if has_commitment:
            regions_with_commitment += 1

    commitment_coverage = regions_with_commitment / total_ondemand_regions if total_ondemand_regions > 0 else 0
    if commitment_coverage < 0.75:  # Expected 85%, allow 10% tolerance
        validation_warnings.append(
            f"⚠️  Commitment pricing coverage low: {regions_with_commitment}/{total_ondemand_regions} regions "
            f"({commitment_coverage*100:.0f}%, expected >85%)"
        )

    # Check 4: $0.00 SKUs (normal: ~100, suspicious: >150)
    zero_price_count = 0
    zero_price_commitment = 0
    for region_data in pricing_data['gpus_ondemand'].values():
        for sku in region_data:
            if sku['price'] == 0.0:
                zero_price_count += 1
                if "Year" in sku['description']:
                    zero_price_commitment += 1

    if zero_price_count > 150:
        validation_warnings.append(
            f"⚠️  Excessive $0.00 SKUs: {zero_price_count} (expected <150, likely Reserved/Calendar Mode)"
        )
    if zero_price_commitment > 0:
        validation_warnings.append(
            f"⚠️  Commitment SKUs with $0.00 price: {zero_price_commitment} (data quality issue)"
        )

    # Check 5: Missing pricing tiers for T4 (most common GPU)
    t4_has_spot = any(
        "T4" in sku['description'] and "TPU" not in sku['description']
        for region_data in pricing_data.get('gpus_spot', {}).values()
        for sku in region_data
    )
    t4_has_ondemand = False
    t4_has_1yr = False
    t4_has_3yr = False
    for region_data in pricing_data.get('gpus_ondemand', {}).values():
        for sku in region_data:
            if "T4" in sku['description'] and "TPU" not in sku['description']:
                if sku['usage_type'] == "OnDemand":
                    t4_has_ondemand = True
                if "1 Year" in sku['description']:
                    t4_has_1yr = True
                if "3 Year" in sku['description']:
                    t4_has_3yr = True

    if not t4_has_spot:
        validation_warnings.append("⚠️  T4 GPU missing spot pricing (critical pricing tier)")
    if not t4_has_ondemand:
        validation_warnings.append("⚠️  T4 GPU missing on-demand pricing (critical pricing tier)")
    if not t4_has_1yr:
        validation_warnings.append("⚠️  T4 GPU missing 1-year commitment pricing")
    if not t4_has_3yr:
        validation_warnings.append("⚠️  T4 GPU missing 3-year commitment pricing")

    # Output results
    if validation_warnings:
        print(f"⚠️  {len(validation_warnings)} validation warning(s) found:")
        for warning in validation_warnings:
            print(f"   {warning}")
    else:
        print("✅ All validation checks passed - pricing data looks perfect!")

    return pricing_data


def upload_to_artifact_registry(pricing_data, project_id):
    """Upload pricing JSON to Artifact Registry using REST API"""
    print("📦 Uploading to Artifact Registry...")

    # Get access token
    access_token = get_access_token()

    # Generate timestamp-based version (e.g., "1.0.20250111-142530")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version = f"1.0.{timestamp}"

    # Artifact Registry upload URL
    # Format: https://artifactregistry.googleapis.com/upload/v1/projects/{project}/locations/{location}/repositories/{repository}/genericArtifacts:create
    location = "us-central1"
    repository = "arr-coc-pricing"
    package_name = "gcp-pricing"

    upload_url = (
        f"https://artifactregistry.googleapis.com/upload/v1/"
        f"projects/{project_id}/locations/{location}/repositories/{repository}/"
        f"genericArtifacts:create"
        f"?package_id={package_name}&version_id={version}&filename=gcp-live-pricing.json"
    )

    # Prepare file content
    file_content = json.dumps(pricing_data, indent=2)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Upload file
    response = requests.post(upload_url, headers=headers, data=file_content)

    if response.status_code not in [200, 201]:
        raise RuntimeError(f"Upload failed: {response.status_code} {response.text}")

    print(f"✅ Uploaded version {version} to Artifact Registry")


@functions_framework.http
def fetch_pricing(request):
    """
    Cloud Function entry point

    Triggered by Cloud Scheduler every 20 minutes
    """
    project_id = os.environ.get("GCP_PROJECT_ID", "")

    # Startup signal for log monitoring
    print("🚀 PRICING_RUNNER_STARTED")

    try:
        # Fetch pricing
        pricing_data = fetch_gcp_pricing()

        # Upload to Artifact Registry
        upload_to_artifact_registry(pricing_data, project_id)

        # Success signal for log parsing
        print("🎯 PRICING_RUNNER_SUCCESS")

        return {
            "status": "success",
            "updated": pricing_data["updated"],
            "c3_regions": len(pricing_data["c3_machines"]),
            "e2_regions": len(pricing_data["e2_machines"]),
            "gpu_spot_regions": len(pricing_data["gpus_spot"]),
            "gpu_ondemand_regions": len(pricing_data["gpus_ondemand"]),
        }, 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }, 500

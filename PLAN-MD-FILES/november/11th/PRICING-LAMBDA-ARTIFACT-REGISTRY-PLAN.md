# Pricing System with Cloud Functions + Artifact Registry - IMPLEMENTATION PLAN

**Goal:** Auto-updating pricing system using Cloud Functions (Lambda) + Artifact Registry storage

**Date:** 2025-11-12
**Status:** ✅ IMPLEMENTED (2025-11-12), REFINED (2025-11-16)

**Estimated Time:** 2-3 hours

---

## ⚠️ UPDATE 2025-11-16: Simplified Implementation

**This plan was implemented as described, then simplified:**

✅ **Kept** (THE GOOD PRICING WAY):
- Cloud Function auto-fetch (every 20 min via Cloud Scheduler)
- Artifact Registry storage
- `fetch_pricing_no_save()` - Clean fetch without local caching
- Helper functions (`get_spot_price()`, `get_standard_price()`, etc.)

❌ **Removed** (2025-11-16):
- `check_and_update_pricing()` - Manual trigger + 24-hour staleness checks (85 lines deleted)
- Local pricing file saves in launch flow
- Manual Cloud Function triggering logic

**Result**: Simpler, cleaner code - Cloud Scheduler handles all refresh, CLI just fetches latest.

See: `THE_GOOD_PRICING_WAY.md` for current implementation

---

---

## 🚀 QUICK START GUIDE (New Session)

**Starting fresh? Follow these steps in order:**

### 1. Navigate to Project Directory
```bash
cd /Users/alfrednorth/Desktop/Code/arr-coc-ovis/RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1
```

### 2. Create All Files (Phase 1 + 2)
- [ ] Create `training/cli/shared/pricing/cloud_function/main.py` (see Phase 1, File 1)
- [ ] Create `training/cli/shared/pricing/cloud_function/requirements.txt` (see Phase 1, File 2)
- [ ] Create `training/cli/shared/artifact_pricing.py` (see Phase 2, File 3)

### 3. Modify Launch Integration (Phase 3)
- [ ] Edit `training/cli/launch/mecha/mecha_battle_epic.py` (see Phase 3, File 4)

### 4. Create Setup/Teardown (Phase 4 + 5)
- [ ] Create `training/cli/setup/pricing_setup.py` (see Phase 4, File 5)
- [ ] Edit `training/cli/setup/core.py` to integrate (see Phase 4, File 6)
- [ ] Create `training/cli/teardown/pricing_teardown.py` (see Phase 5, File 7)
- [ ] Edit `training/cli/teardown/core.py` to integrate (see Phase 5, File 8)

### 5. Test Everything (Phase 6)
```bash
# Run setup
python training/cli.py setup

# Try launch
python training/cli.py launch

# Teardown when done
python training/cli.py teardown
```

### 6. Commit (Phase 7)
```bash
git add training/cli/shared/pricing/cloud_function/
git add training/cli/shared/artifact_pricing.py
git add training/cli/setup/pricing_setup.py
git add training/cli/teardown/pricing_teardown.py
# ... (see Phase 7 for full commit command)
```

**Total implementation time:** ~2-3 hours

**Files to create:** 5 new files
**Files to modify:** 3 existing files

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Cloud Function Pricing Fetcher
- [x] Create `training/cli/shared/pricing/cloud_function/` directory
- [x] Create `main.py` (Cloud Function entry point)
- [x] Create `requirements.txt` (dependencies)
- [ ] Test function locally
- [ ] Deploy Cloud Function to GCP

### Phase 2: Artifact Registry Integration
- [x] Add pricing helper: `artifact_pricing.py`
- [ ] Test upload to Artifact Registry
- [ ] Test download from Artifact Registry
- [ ] Verify JSON format and freshness metadata

### Phase 3: Update Launch to Read from Artifact Registry
- [x] Modify `training/cli/launch/mecha/mecha_battle_epic.py`
- [x] Add fallback: Manual trigger if pricing missing
- [x] Add freshness display: "Using live prices from X minutes ago"
- [ ] Test launch with Artifact Registry pricing

### Phase 4: Setup Infrastructure
- [x] Add setup code: `training/cli/setup/pricing_setup.py`
- [x] Integrate with main setup flow
- [x] Bootstrap: Upload current JSON to Artifact Registry
- [x] Deploy Cloud Function
- [x] Create Cloud Scheduler (every 20 min)

### Phase 5: Teardown Infrastructure
- [x] Add teardown code: `training/cli/teardown/pricing_teardown.py`
- [x] Delete Cloud Scheduler job
- [x] Delete Cloud Function
- [x] Clean up Artifact Registry pricing artifacts
- [x] Integrate with main teardown flow

### Phase 6: Testing
- [ ] Test setup from scratch
- [ ] Test launch with existing pricing
- [ ] Test launch with missing pricing (fallback)
- [ ] Test scheduled updates (wait 20 min)
- [ ] Test teardown cleanup

### Phase 7: Git Commit
- [ ] Commit all changes with descriptive message

---

## 🔧 PHASE 1: Cloud Function Pricing Fetcher

### File 1: `training/cli/shared/pricing/cloud_function/main.py` (NEW)

**Location:** Create new directory and file

**Full Code:**

```python
"""
Cloud Function: GCP Pricing Fetcher

Triggered by Cloud Scheduler every 20 minutes.
Fetches live GCP C3 pricing and stores in Artifact Registry.
"""

import functions_framework
import subprocess
import json
from datetime import datetime
from pathlib import Path


def fetch_gcp_pricing():
    """Fetch C3 pricing from GCP Cloud Billing API"""
    print("🔍 Fetching live GCP pricing...")

    # Get access token
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to get access token")

    access_token = result.stdout.strip()

    # Query billing API
    pricing_data = {
        "updated": datetime.utcnow().isoformat() + "Z",
        "c3_machines": {},
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

        # Fetch page
        curl_result = subprocess.run(
            [
                "curl",
                "-s",
                "-H",
                f"Authorization: Bearer {access_token}",
                url,
            ],
            capture_output=True,
            text=True,
        )

        if curl_result.returncode != 0:
            raise RuntimeError(f"API request failed: {curl_result.stderr}")

        data = json.loads(curl_result.stdout)

        if "skus" not in data:
            break

        # Parse SKUs
        for sku in data["skus"]:
            skus_checked += 1

            description = sku.get("description", "").lower()
            category = sku.get("category", {})
            usage_type = category.get("usageType", "")
            regions = sku.get("serviceRegions", [])

            # Extract price
            try:
                pricing_info = sku.get("pricingInfo", [{}])[0]
                expr = pricing_info.get("pricingExpression", {})
                rates = expr.get("tieredRates", [])
                if rates:
                    price_data = rates[0].get("unitPrice", {})
                    nanos = int(price_data.get("nanos", 0))
                    units = int(price_data.get("units", 0) or 0)
                    price = units + (nanos / 1_000_000_000)
                else:
                    continue
            except:
                continue

            # C3 Machines (Spot only)
            if "c3" in description and ("spot" in description or "preemptible" in description):
                if "c3d" in description or "c3a" in description:
                    continue

                for region in regions:
                    if region not in pricing_data["c3_machines"]:
                        pricing_data["c3_machines"][region] = {}

                    if "core" in description:
                        existing_cpu = pricing_data["c3_machines"][region].get("cpu_per_core_spot", float('inf'))
                        if price < existing_cpu:
                            pricing_data["c3_machines"][region]["cpu_per_core_spot"] = price
                    elif "ram" in description:
                        existing_ram = pricing_data["c3_machines"][region].get("ram_per_gb_spot", float('inf'))
                        if price < existing_ram:
                            pricing_data["c3_machines"][region]["ram_per_gb_spot"] = price

        # Check for next page
        page_token = data.get("nextPageToken")
        if not page_token:
            break

        if skus_checked % 5000 == 0:
            print(f"📄 Checked {skus_checked} SKUs...")

    c3_regions = len(pricing_data['c3_machines'])
    print(f"✅ Pricing fetched: {c3_regions} regions, {skus_checked} SKUs checked")

    return pricing_data


def upload_to_artifact_registry(pricing_data, project_id):
    """Upload pricing JSON to Artifact Registry"""
    print("📦 Uploading to Artifact Registry...")

    # Save to temp file
    temp_file = Path("/tmp/gcp-live-pricing.json")
    with open(temp_file, "w") as f:
        json.dump(pricing_data, f, indent=2)

    # Upload to Artifact Registry (generic repository)
    # We'll use the existing arr-coc-registry but add a generic package
    result = subprocess.run(
        [
            "gcloud", "artifacts", "generic", "upload",
            "--project", project_id,
            "--repository=arr-coc-registry",
            "--location=us-central1",
            "--package=gcp-pricing",
            "--version=latest",
            f"--source={temp_file}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr}")

    print(f"✅ Uploaded to Artifact Registry")


@functions_framework.http
def fetch_pricing(request):
    """
    Cloud Function entry point

    Triggered by Cloud Scheduler every 20 minutes
    """
    project_id = "weight-and-biases-476906"

    try:
        # Fetch pricing
        pricing_data = fetch_gcp_pricing()

        # Upload to Artifact Registry
        upload_to_artifact_registry(pricing_data, project_id)

        return {
            "status": "success",
            "updated": pricing_data["updated"],
            "regions": len(pricing_data["c3_machines"]),
        }, 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }, 500
```

**Checkpoint:**
- [ ] File created
- [ ] No syntax errors

---

### File 2: `training/cli/shared/pricing/cloud_function/requirements.txt` (NEW)

**Full Content:**

```
functions-framework==3.*
```

**Checkpoint:**
- [ ] File created

---

## 🔧 PHASE 2: Artifact Registry Integration

### File 3: `training/cli/shared/artifact_pricing.py` (NEW)

**Location:** `training/cli/shared/artifact_pricing.py`

**Full Code:**

```python
"""
Artifact Registry Pricing Helper

Read/write pricing JSON from/to Artifact Registry.
"""

import subprocess
import json
import tempfile
from pathlib import Path
from datetime import datetime


PROJECT_ID = "weight-and-biases-476906"
REPOSITORY = "arr-coc-registry"
LOCATION = "us-central1"
PACKAGE = "gcp-pricing"
VERSION = "latest"


def download_pricing_from_artifact_registry():
    """
    Download latest pricing JSON from Artifact Registry

    Returns:
        dict: Pricing data with 'updated' timestamp

    Raises:
        FileNotFoundError: If pricing doesn't exist in registry
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download from Artifact Registry
        result = subprocess.run(
            [
                "gcloud", "artifacts", "generic", "download",
                f"--project={PROJECT_ID}",
                f"--repository={REPOSITORY}",
                f"--location={LOCATION}",
                f"--package={PACKAGE}",
                f"--version={VERSION}",
                f"--destination={tmpdir}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if "NOT_FOUND" in result.stderr:
                raise FileNotFoundError("Pricing not found in Artifact Registry")
            else:
                raise RuntimeError(f"Download failed: {result.stderr}")

        # Load JSON
        pricing_file = Path(tmpdir) / "gcp-live-pricing.json"
        with open(pricing_file) as f:
            return json.load(f)


def upload_pricing_to_artifact_registry(pricing_data):
    """Upload pricing JSON to Artifact Registry"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save to temp file
        pricing_file = Path(tmpdir) / "gcp-live-pricing.json"
        with open(pricing_file, "w") as f:
            json.dump(pricing_data, f, indent=2)

        # Upload to Artifact Registry
        result = subprocess.run(
            [
                "gcloud", "artifacts", "generic", "upload",
                f"--project={PROJECT_ID}",
                f"--repository={REPOSITORY}",
                f"--location={LOCATION}",
                f"--package={PACKAGE}",
                f"--version={VERSION}",
                f"--source={pricing_file}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Upload failed: {result.stderr}")


def get_pricing_age_minutes(pricing_data):
    """
    Calculate age of pricing data in minutes

    Args:
        pricing_data: Dict with 'updated' ISO timestamp

    Returns:
        float: Age in minutes
    """
    updated_str = pricing_data.get("updated")
    if not updated_str:
        return float('inf')

    updated_time = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
    now = datetime.utcnow().replace(tzinfo=updated_time.tzinfo)

    age = now - updated_time
    return age.total_seconds() / 60


def format_pricing_age(age_minutes):
    """
    Format pricing age for display

    Args:
        age_minutes: Age in minutes

    Returns:
        str: Human-readable age string
    """
    if age_minutes < 1:
        return "just now"
    elif age_minutes < 60:
        return f"{int(age_minutes)} minute{'s' if int(age_minutes) != 1 else ''} ago"
    elif age_minutes < 1440:  # Less than 24 hours
        hours = int(age_minutes / 60)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(age_minutes / 1440)
        return f"{days} day{'s' if days != 1 else ''} ago"
```

**Checkpoint:**
- [ ] File created
- [ ] No syntax errors

---

## 🔧 PHASE 3: Update Launch to Read from Artifact Registry

**⚠️ NOTE 2025-11-16: This entire section was DELETED in later cleanup**
- The `check_and_update_pricing()` function described below was removed
- Replaced with simple `fetch_pricing_no_save()` call (THE GOOD PRICING WAY)
- See commits: 5351fda, b35f5de, adbef3e

---

### File 4: Modify `training/cli/launch/mecha/mecha_battle_epic.py` [HISTORICAL]

**Location:** Line 356 (in `check_and_update_pricing()` function)

**Find this code:**
```python
def check_and_update_pricing() -> bool:
    """
    Check pricing freshness and update if needed.

    Returns:
        True if pricing was refreshed, False if using cached pricing
    """
    max_age_minutes = 20

    # Auto-update if stale or missing
    needs_refresh = False
    if not PRICING_JSON.exists():
        needs_refresh = True
    else:
        file_age = datetime.now() - datetime.fromtimestamp(PRICING_JSON.stat().st_mtime)
        age_minutes = file_age.total_seconds() / 60
        if age_minutes > max_age_minutes:
            needs_refresh = True

    if needs_refresh:
        # Will be shown by core.py as "⏳ Refreshing live prices..."
        _update_pricing_now()
        return True

    return False
```

**Replace with:**
```python
def check_and_update_pricing() -> bool:
    """
    Load pricing from Artifact Registry with fallback to manual trigger.

    Manual trigger conditions:
    - Pricing not found (FileNotFoundError)
    - Pricing older than 24 hours

    Returns:
        True if pricing was refreshed, False if using cached pricing
    """
    from training.cli.shared.artifact_pricing import (
        download_pricing_from_artifact_registry,
        get_pricing_age_minutes,
        format_pricing_age,
    )

    MAX_AGE_HOURS = 24
    MAX_AGE_MINUTES = MAX_AGE_HOURS * 60

    pricing_missing = False
    pricing_too_old = False

    try:
        # Try to download from Artifact Registry
        pricing_data = download_pricing_from_artifact_registry()

        # Calculate age
        age_minutes = get_pricing_age_minutes(pricing_data)
        age_str = format_pricing_age(age_minutes)

        # Check if too old (>24 hours)
        if age_minutes > MAX_AGE_MINUTES:
            pricing_too_old = True
            print(f"[yellow]⚠️  Pricing is stale ({age_str}, max {MAX_AGE_HOURS}h)[/yellow]")
        else:
            # Fresh enough - use it!
            print(f"[dim]✓ Using live prices from {age_str}[/dim]")

            # Save to local file for get_region_price() to use
            with open(PRICING_JSON, "w") as f:
                json.dump(pricing_data, f, indent=2)

            return False  # Using cached (from Artifact Registry)

    except FileNotFoundError:
        pricing_missing = True
        print("[yellow]⚠️  Pricing not found in Artifact Registry[/yellow]")

    # Manual trigger if missing OR too old
    if pricing_missing or pricing_too_old:
        print("[yellow]   Triggering manual pricing fetch...[/yellow]")

        # Trigger Cloud Function manually
        result = subprocess.run(
            [
                "gcloud", "functions", "call", "pricing-fetcher",
                "--gen2",
                "--region=us-central1",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )

        if result.returncode != 0:
            raise RuntimeError(f"Manual pricing fetch failed: {result.stderr}")

        print("[green]✓ Pricing fetched and stored[/green]")

        # Now download it
        pricing_data = download_pricing_from_artifact_registry()

        # Save to local file
        with open(PRICING_JSON, "w") as f:
            json.dump(pricing_data, f, indent=2)

        return True  # Just refreshed
```

**Checkpoint:**
- [ ] Code replaced
- [ ] Import added for `subprocess`
- [ ] No syntax errors

---

## 🔧 PHASE 4: Setup Infrastructure

### File 5: `training/cli/setup/pricing_setup.py` (NEW)

**Location:** Create new file

**Full Code:**

```python
"""
Pricing Infrastructure Setup

Creates:
1. Cloud Function (pricing-fetcher)
2. Cloud Scheduler (every 20 min)
3. Bootstraps Artifact Registry with current pricing
"""

import subprocess
import json
from pathlib import Path


PROJECT_ID = "weight-and-biases-476906"
REGION = "us-central1"
FUNCTION_NAME = "pricing-fetcher"
SCHEDULER_JOB = "pricing-cron"


def setup_pricing_infrastructure(status_callback):
    """Setup pricing infrastructure"""
    status = status_callback

    status("")
    status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("📊 Setting Up Pricing Infrastructure")
    status("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    status("")

    # 1. Bootstrap: Upload current pricing to Artifact Registry
    status("1️⃣  Bootstrapping Artifact Registry with current pricing...")
    bootstrap_pricing(status)
    status("   ✓ Pricing JSON uploaded to Artifact Registry")
    status("")

    # 2. Deploy Cloud Function
    status("2️⃣  Deploying Cloud Function (pricing-fetcher)...")
    deploy_cloud_function(status)
    status("   ✓ Cloud Function deployed")
    status("")

    # 3. Create Cloud Scheduler
    status("3️⃣  Creating Cloud Scheduler (every 20 minutes)...")
    create_scheduler(status)
    status("   ✓ Cloud Scheduler created")
    status("")

    status("✅ Pricing infrastructure setup complete!")
    status("   → Launches will use Artifact Registry pricing")
    status("   → Auto-updates every 20 minutes")
    status("")


def bootstrap_pricing(status):
    """Upload current pricing JSON to Artifact Registry"""
    from training.cli.shared.artifact_pricing import upload_pricing_to_artifact_registry
    from training.cli.constants import PRICING_JSON

    if not PRICING_JSON.exists():
        status("   ⚠️  No local pricing found, fetching fresh...")
        # Run pricing fetch locally
        from training.cli.shared.pricing.get_live_prices import update_pricing_now
        update_pricing_now()

    # Load current pricing
    with open(PRICING_JSON) as f:
        pricing_data = json.load(f)

    # Upload to Artifact Registry
    upload_pricing_to_artifact_registry(pricing_data)


def deploy_cloud_function(status):
    """Deploy Cloud Function to GCP"""
    function_dir = Path(__file__).parent.parent / "shared" / "pricing" / "cloud_function"

    # Deploy
    result = subprocess.run(
        [
            "gcloud", "functions", "deploy", FUNCTION_NAME,
            "--gen2",
            f"--region={REGION}",
            "--runtime=python312",
            "--source=" + str(function_dir),
            "--entry-point=fetch_pricing",
            "--trigger-http",
            "--allow-unauthenticated",  # Scheduler will call it
            "--timeout=540s",
            "--memory=512MB",
            f"--project={PROJECT_ID}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Function deployment failed: {result.stderr}")


def create_scheduler(status):
    """Create Cloud Scheduler job"""
    # Get function URL
    result = subprocess.run(
        [
            "gcloud", "functions", "describe", FUNCTION_NAME,
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

    # Create scheduler job
    result = subprocess.run(
        [
            "gcloud", "scheduler", "jobs", "create", "http", SCHEDULER_JOB,
            f"--location={REGION}",
            "--schedule=*/20 * * * *",  # Every 20 minutes
            f"--uri={function_url}",
            "--http-method=GET",
            f"--project={PROJECT_ID}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Check if already exists
        if "ALREADY_EXISTS" in result.stderr:
            status("   (Scheduler job already exists, skipping)")
        else:
            raise RuntimeError(f"Scheduler creation failed: {result.stderr}")
```

**Checkpoint:**
- [ ] File created
- [ ] No syntax errors

---

### File 6: Integrate with `training/cli/setup/core.py`

**Location:** After worker pool creation (around line 1200)

**Add this code:**

```python
# Setup pricing infrastructure
from training.cli.setup.pricing_setup import setup_pricing_infrastructure
setup_pricing_infrastructure(status)
```

**Checkpoint:**
- [ ] Import added
- [ ] Function called in correct location

---

## 🔧 PHASE 5: Teardown Infrastructure

### File 7: `training/cli/teardown/pricing_teardown.py` (NEW)

**Location:** Create new file

**Full Code:**

```python
"""
Pricing Infrastructure Teardown

Deletes:
1. Cloud Scheduler job
2. Cloud Function
3. Pricing artifacts from Artifact Registry
"""

import subprocess


PROJECT_ID = "weight-and-biases-476906"
REGION = "us-central1"
FUNCTION_NAME = "pricing-fetcher"
SCHEDULER_JOB = "pricing-cron"


def teardown_pricing_infrastructure(status_callback):
    """Teardown pricing infrastructure"""
    status = status_callback

    status("")
    status("Tearing down pricing infrastructure...")

    # 1. Delete Cloud Scheduler
    status("  Deleting Cloud Scheduler...")
    delete_scheduler(status)

    # 2. Delete Cloud Function
    status("  Deleting Cloud Function...")
    delete_cloud_function(status)

    # 3. Clean up Artifact Registry pricing package
    status("  Cleaning up Artifact Registry...")
    cleanup_artifact_registry(status)

    status("✓ Pricing infrastructure torn down")


def delete_scheduler(status):
    """Delete Cloud Scheduler job"""
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

    if result.returncode != 0:
        if "NOT_FOUND" not in result.stderr:
            status(f"    ⚠️  Warning: {result.stderr}")


def delete_cloud_function(status):
    """Delete Cloud Function"""
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

    if result.returncode != 0:
        if "NOT_FOUND" not in result.stderr:
            status(f"    ⚠️  Warning: {result.stderr}")


def cleanup_artifact_registry(status):
    """Delete pricing package from Artifact Registry"""
    result = subprocess.run(
        [
            "gcloud", "artifacts", "packages", "delete", "gcp-pricing",
            f"--repository=arr-coc-registry",
            f"--location={REGION}",
            f"--project={PROJECT_ID}",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if "NOT_FOUND" not in result.stderr:
            status(f"    ⚠️  Warning: {result.stderr}")
```

**Checkpoint:**
- [ ] File created
- [ ] No syntax errors

---

### File 8: Integrate with `training/cli/teardown/core.py`

**Location:** Before final success message

**Add this code:**

```python
# Teardown pricing infrastructure
from training.cli.teardown.pricing_teardown import teardown_pricing_infrastructure
teardown_pricing_infrastructure(status)
```

**Checkpoint:**
- [ ] Import added
- [ ] Function called in correct location

---

## 🧪 PHASE 6: Testing

### Test 1: Setup from Scratch
```bash
python training/cli.py setup
```

**Expected:**
- ✓ Pricing JSON uploaded to Artifact Registry
- ✓ Cloud Function deployed
- ✓ Cloud Scheduler created
- ✓ No errors

**Checklist:**
- [ ] Setup runs without errors
- [ ] Function visible in Cloud Console
- [ ] Scheduler visible in Cloud Console

---

### Test 2: Launch with Existing Pricing
```bash
python training/cli.py launch
```

**Expected:**
- ✓ Loads pricing from Artifact Registry
- ✓ Shows "Using live prices from X minutes ago"
- ✓ MECHA price battle runs normally

**Checklist:**
- [ ] Launch works
- [ ] Freshness displayed correctly
- [ ] No errors

---

### Test 3: Launch with Missing Pricing
```bash
# Delete pricing package
gcloud artifacts packages delete gcp-pricing \
  --repository=arr-coc-registry \
  --location=us-central1 \
  --quiet

# Try launch
python training/cli.py launch
```

**Expected:**
- ⚠️  Pricing not found
- 🔄 Triggers manual fetch
- ✓ Waits ~3 min
- ✓ Downloads fresh pricing
- ✓ Proceeds with launch

**Checklist:**
- [ ] Fallback triggers correctly (missing)
- [ ] Manual fetch succeeds
- [ ] Launch completes

---

### Test 4: Launch with Stale Pricing (>24 hours)
```bash
# Upload old pricing with ancient timestamp
cat > /tmp/old-pricing.json << 'EOF'
{
  "updated": "2024-01-01T00:00:00Z",
  "c3_machines": {
    "us-central1": {
      "cpu_per_core_spot": 0.00415,
      "ram_per_gb_spot": 0.000562
    }
  },
  "gpus_spot": {},
  "gpus_ondemand": {}
}
EOF

# Upload to Artifact Registry
gcloud artifacts generic upload \
  --repository=arr-coc-registry \
  --location=us-central1 \
  --package=gcp-pricing \
  --version=latest \
  --source=/tmp/old-pricing.json

# Try launch
python training/cli.py launch
```

**Expected:**
- ⚠️  Pricing is stale (XX days ago, max 24h)
- 🔄 Triggers manual fetch
- ✓ Waits ~3 min
- ✓ Downloads fresh pricing
- ✓ Proceeds with launch

**Checklist:**
- [ ] Fallback triggers correctly (stale)
- [ ] Manual fetch succeeds
- [ ] Launch completes

---

### Test 4: Teardown
```bash
python training/cli.py teardown
```

**Expected:**
- ✓ Scheduler deleted
- ✓ Function deleted
- ✓ Pricing package deleted
- ✓ No errors

**Checklist:**
- [ ] Teardown runs without errors
- [ ] Resources deleted from Cloud Console

---

## 🔧 PHASE 7: Git Commit

```bash
git add training/cli/shared/pricing/cloud_function/
git add training/cli/shared/artifact_pricing.py
git add training/cli/setup/pricing_setup.py
git add training/cli/teardown/pricing_teardown.py
git add training/cli/launch/mecha/mecha_battle_epic.py
git add training/cli/setup/core.py
git add training/cli/teardown/core.py

git commit -m "$(cat <<'EOF'
Add Cloud Function pricing system with Artifact Registry

Phase 1: Cloud Function Pricing Fetcher
- Created cloud_function/ with main.py and requirements.txt
- Fetches GCP C3 pricing every 20 min via scheduler
- Uploads to Artifact Registry (gcp-pricing package)

Phase 2: Artifact Registry Integration
- Created artifact_pricing.py helper
- Download/upload pricing JSON from/to registry
- Calculate and format pricing age (1 min ago, 1 hour ago, etc.)

Phase 3: Launch Integration
- Modified mecha_battle_epic.py to read from Artifact Registry
- Shows pricing freshness: "Using live prices from X ago"
- Fallback: Manual trigger if pricing missing

Phase 4: Setup Infrastructure
- Created pricing_setup.py
- Bootstraps Artifact Registry with current pricing
- Deploys Cloud Function and Cloud Scheduler
- Integrated with main setup flow

Phase 5: Teardown Infrastructure
- Created pricing_teardown.py
- Deletes scheduler, function, and pricing artifacts
- Integrated with main teardown flow

Benefits:
- Fast launches (read from Artifact Registry, not 2-3 min API)
- Auto-updates every 20 minutes
- Shows pricing freshness to user
- No Docker/builds needed (Cloud Functions)
- Clean setup/teardown

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Checkpoint:**
- [ ] All files staged
- [ ] Commit created
- [ ] Descriptive commit message

---

## 📊 Summary of Changes

**Files Created (5):**
- `training/cli/shared/pricing/cloud_function/main.py`
- `training/cli/shared/pricing/cloud_function/requirements.txt`
- `training/cli/shared/artifact_pricing.py`
- `training/cli/setup/pricing_setup.py`
- `training/cli/teardown/pricing_teardown.py`

**Files Modified (3):**
- `training/cli/launch/mecha/mecha_battle_epic.py`
- `training/cli/setup/core.py`
- `training/cli/teardown/core.py`

**Total Lines Added:** ~600 lines
**Infrastructure Added:**
- 1 Cloud Function (pricing-fetcher)
- 1 Cloud Scheduler (pricing-cron, */20 * * * *)
- 1 Artifact Registry package (gcp-pricing)

---

**Last Updated:** 2025-11-12
**Status:** READY TO IMPLEMENT
**Estimated Time:** 2-3 hours
**Complexity:** Medium

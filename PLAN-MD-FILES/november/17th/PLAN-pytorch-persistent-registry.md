# Implementation Plan: PyTorch Persistent Registry

> **⚠️ HISTORICAL PLANNING DOCUMENT**
> This document shows the original plan. The implementation differs slightly:
> **Planned name:** `persistent-artifacts`
> **Actual implementation:** `arr-coc-registry-persistent` (for naming consistency)
> Throughout this document, mentally replace `persistent-artifacts` with `arr-coc-registry-persistent`

---

**Goal:** Separate arr-pytorch-base into its own persistent registry that survives teardown

**Status:** ✅ COMPLETED (implemented as `arr-coc-registry-persistent`)
**Actual time:** ~2 hours
**Files modified:** 15+
**Lines of code:** ~200+

---

## Architecture Change

### Before
```
arr-coc-registry (us-central1)
├── arr-pytorch-base:latest      ← 2-4 hours to build, deleted on teardown
├── arr-ml-stack:latest          ← minutes to build
├── arr-trainer:latest           ← minutes to build
└── arr-vertex-launcher:latest   ← minutes to build
```

### After (AS IMPLEMENTED)
```
arr-coc-registry-persistent (us-central1) - NEVER DELETED ✅
└── arr-pytorch-base:latest      ← 2-4 hours to build, preserved forever

arr-coc-registry (us-central1) - SAFE TO DELETE ✅
├── arr-ml-stack:latest          ← minutes to build
├── arr-trainer:latest           ← minutes to build
└── arr-vertex-launcher:latest   ← minutes to build
```

**NOTE:** Originally planned as `persistent-artifacts`, implemented as `arr-coc-registry-persistent`
for naming consistency with the `arr-coc-registry` scheme.

---

## Implementation Philosophy

**NO BACKWARDS COMPATIBILITY CODE!**
- Fresh, clean implementation
- No checking old registry for existing images
- User manually migrates if they have existing PyTorch image
- Simple and maintainable

---

## Implementation Steps

### STEP 1: Setup - Create Persistent Registry
**File:** `training/cli/setup/steps.py`

**Action:** Add new function `_setup_persistent_registry()` before `_setup_artifact_registry()`

**Location:** Insert at line ~130 (before `_setup_artifact_registry()`)

```python
def _setup_persistent_registry(
    project_id: str,
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Step 2a/9: Persistent Artifact Registry (NEVER DELETED)

    Registry name: persistent-artifacts (SHARED - contains PyTorch base image)

    This registry stores the arr-pytorch-base image which takes 2-4 hours to build.
    NEVER deleted during teardown - user must manually remove if needed.
    """

    registry_name = "persistent-artifacts"

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
                status(f"   [yellow]⚠️  Persistent registry exists but misconfigured:[/yellow]")
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
```

**Update setup coordinator:**

Find the main setup function that calls `_setup_artifact_registry()` and add call to persistent registry:

```python
# Around line 80-100 in the main setup() function
# Step 2a/9: Persistent Artifact Registry (PyTorch base)
if not _setup_persistent_registry(project_id, region, status):
    return False

# Step 2b/9: Artifact Registry (ML stack, trainer, launcher)
if not _setup_artifact_registry(project_id, region, status):
    return False
```

**Update step comments:**
- Change "Step 2/9: Artifact Registry" → "Step 2b/9: Artifact Registry"
- All subsequent steps shift: 3/9 → 4/9, 4/9 → 5/9, etc.

---

### STEP 2: Launch - Update PyTorch Image Path
**File:** `training/cli/launch/core.py`

**Location:** Line ~1750-1760 in `_handle_pytorch_clean_image()`

**Change 1:** Update registry name
```python
# OLD (line ~1754):
registry_name = "arr-coc-registry"

# NEW:
registry_name = "arr-coc-registry"  # Fast-building images
persistent_registry = "persistent-artifacts"  # PyTorch base image only
```

**Change 2:** Update image paths (line ~1756-1757)
```python
# OLD:
pytorch_clean_hash = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-pytorch-base:{dockerfile_hash}"
pytorch_clean_latest = f"{registry_region}-docker.pkg.dev/{project_id}/{registry_name}/arr-pytorch-base:latest"

# NEW:
pytorch_clean_hash = f"{registry_region}-docker.pkg.dev/{project_id}/{persistent_registry}/arr-pytorch-base:{dockerfile_hash}"
pytorch_clean_latest = f"{registry_region}-docker.pkg.dev/{project_id}/{persistent_registry}/arr-pytorch-base:latest"
```

**Change 3:** Update cleanup call (line ~2274)
```python
# OLD:
_cleanup_old_images("arr-pytorch-base", artifact_registry_base, status)

# NEW:
# Cleanup in persistent-artifacts registry (not arr-coc-registry)
persistent_base = f"{registry_region}-docker.pkg.dev/{project_id}/{persistent_registry}"
_cleanup_old_images("arr-pytorch-base", persistent_base, status)
```

**Change 4:** Update comment (line ~1713)
```python
# OLD:
"""
Ensure arr-pytorch-base image exists (build if Dockerfile changed)

# NEW:
"""
Ensure arr-pytorch-base image exists (build if Dockerfile changed)

Image stored in PERSISTENT REGISTRY (never deleted):
- Registry: persistent-artifacts
- Path: {region}-docker.pkg.dev/{project}/persistent-artifacts/arr-pytorch-base:latest
```

---

### STEP 3: Cloud Build - Update Image Tags
**File:** `.cloudbuild-arr-pytorch-base.yaml`

**Location:** Lines 246-248 (bottom of file)

```yaml
# OLD:
images:
  - 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-pytorch-base:${_DOCKERFILE_FRESHNESS_HASH}'
  - 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-pytorch-base:latest'

# NEW:
images:
  - 'us-central1-docker.pkg.dev/weight-and-biases-476906/persistent-artifacts/arr-pytorch-base:${_DOCKERFILE_FRESHNESS_HASH}'
  - 'us-central1-docker.pkg.dev/weight-and-biases-476906/persistent-artifacts/arr-pytorch-base:latest'
```

---

### STEP 4: Teardown - Skip Persistent Registry
**File:** `training/cli/teardown/steps.py`

**Location:** In `_delete_registry()` function (around line 145-200)

**Change 1:** Add persistent registry skip logic at the beginning of function
```python
def _delete_registry(
    region: str,
    status: StatusCallback,
    delete_images: bool = True,
) -> bool:
    """
    Step 2/6: Delete Artifact Registry

    Deletes arr-coc-registry (fast-building images).
    NEVER deletes persistent-artifacts (PyTorch base image).
    """

    status("[bold]Step 2/6: Artifact Registry[/bold]")

    # SKIP IMAGE DELETION MODE (--skip-images flag)
    if not delete_images:
        status("   [yellow]⚠️  Skipping registry deletion (--skip-images)[/yellow]")
        status("   [cyan]⚡Artifact Registry skipped - Roger![/cyan]")
        return True

    # DELETE MODE: Remove arr-coc-registry only
    registry_name = "arr-coc-registry"
    persistent_registry_name = "persistent-artifacts"

    # Delete arr-coc-registry (contains fast-building images)
    status(f"   🗑️  Deleting registry: {registry_name}...")

    # ... existing deletion code ...

    # After successful deletion, add persistent registry info
    status(f"   [green]✓    Registry deleted: {registry_name}[/green]")
    status("")
    status("   [yellow]⚠️  Persistent Registry NOT Deleted:[/yellow]")
    status(f"      Registry: {persistent_registry_name} (us-central1)")
    status("      Contains: arr-pytorch-base (~15GB)")
    status("      Reason: Prevents 2-4 hour rebuild on next launch")
    status("")
    status("   [dim]To delete manually (if needed):[/dim]")
    status(f"      [dim]gcloud artifacts repositories delete {persistent_registry_name} \\[/dim]")
    status(f"      [dim]  --location=us-central1 \\[/dim]")
    status(f"      [dim]  --project=$(gcloud config get-value project)[/dim]")
    status("")
    status("   [cyan]⚡Artifact Registry passed - Roger![/cyan]")
    return True
```

**Change 2:** Update function docstring
```python
# OLD:
"""
Step 2/6: Delete Artifact Registry

Registry name: arr-coc-registry (SHARED)

# NEW:
"""
Step 2/6: Delete Artifact Registry

Deletes: arr-coc-registry (fast-building images: ml-stack, trainer, launcher)
Preserves: persistent-artifacts (PyTorch base image - takes 2-4 hours to rebuild)
```

**Change 3:** Update dry-run message in teardown coordinator (`core.py`)
```python
# Around line 121 in core.py
# OLD:
if dry_run:
    status("   Would delete: arr-coc-registry (including all images)")

# NEW:
if dry_run:
    if skip_images:
        status("   Would skip: arr-coc-registry (--skip-images)")
    else:
        status("   Would delete: arr-coc-registry (ml-stack, trainer, launcher)")
        status("   Would preserve: persistent-artifacts (PyTorch base)")
```

---

### STEP 5: Documentation - Update Infrastructure Docs
**File:** `training/cli/shared/setup_helper.py`

**Location:** Lines 1-42 (module docstring)

```python
# OLD:
"""
SHARED RESOURCES (One per GCP Project):
---------------------------------------
All ARR-COC prototypes (arr-coc-0-1, arr-coc-0-2, etc.) share these resources:

1. Artifact Registry:
   - arr-coc-registry                (stores Docker images for all prototypes)
   - Images tagged by project: arr-coc-0-1:latest, arr-coc-0-2:latest

# NEW:
"""
SHARED RESOURCES (One per GCP Project):
---------------------------------------
All ARR-COC prototypes (arr-coc-0-1, arr-coc-0-2, etc.) share these resources:

1. Artifact Registry (Deletable):
   - arr-coc-registry                (stores fast-building images)
   - Images: arr-ml-stack, arr-trainer, arr-vertex-launcher
   - Deleted during teardown, rebuilds in minutes

2. Artifact Registry (Persistent - NEVER deleted):
   - persistent-artifacts            (stores PyTorch base image only)
   - Image: arr-pytorch-base (~15GB, 2-4 hours to build)
   - NEVER deleted automatically - manual deletion only
   - Prevents expensive rebuilds across teardowns

3. Service Account:
```

---

## Testing Checklist

### After Implementation

- [ ] **Setup creates both registries:**
  ```bash
  python training/cli.py setup
  # Should see:
  # ✓ Persistent registry created: persistent-artifacts
  # ✓ Registry created: arr-coc-registry
  ```

- [ ] **Verify registries exist:**
  ```bash
  gcloud artifacts repositories list --location=us-central1
  # Should show:
  # persistent-artifacts (DOCKER)
  # arr-coc-registry (DOCKER)
  ```

- [ ] **Launch builds PyTorch to persistent registry:**
  ```bash
  python training/cli.py launch
  # Check Cloud Build logs - should push to:
  # us-central1-docker.pkg.dev/.../persistent-artifacts/arr-pytorch-base:latest
  ```

- [ ] **Verify image in persistent registry:**
  ```bash
  gcloud artifacts docker images list \
    us-central1-docker.pkg.dev/weight-and-biases-476906/persistent-artifacts
  # Should show arr-pytorch-base:latest
  ```

- [ ] **Teardown preserves persistent registry:**
  ```bash
  python training/cli.py teardown
  # Should see:
  # ✓ Registry deleted: arr-coc-registry
  # ⚠️ Persistent Registry NOT Deleted:
  #    Registry: persistent-artifacts
  #    Contains: arr-pytorch-base (~15GB)
  ```

- [ ] **Verify persistent registry still exists:**
  ```bash
  gcloud artifacts repositories list --location=us-central1
  # Should ONLY show:
  # persistent-artifacts (DOCKER)
  # (arr-coc-registry should be gone)
  ```

- [ ] **Next launch reuses PyTorch image:**
  ```bash
  python training/cli.py setup
  python training/cli.py launch
  # Should skip PyTorch build:
  # ✓ Using cached arr-pytorch-base:abc123 (hash matches, no rebuild)
  ```

---

## Migration for Existing Users (MANUAL - ONE TIME)

**Scenario:** User has existing arr-pytorch-base in arr-coc-registry

**Manual Migration (Run Once):**

```bash
# 1. Pull existing image from old location
docker pull us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-pytorch-base:latest

# 2. Calculate fresh Dockerfile hash
cd training/images/arr-pytorch-base
HASH=$(sha256sum Dockerfile | cut -c1-12)
echo "Fresh hash: $HASH"

# 3. Tag for new persistent registry (both :hash and :latest)
docker tag \
  us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-pytorch-base:latest \
  us-central1-docker.pkg.dev/weight-and-biases-476906/persistent-artifacts/arr-pytorch-base:$HASH

docker tag \
  us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-pytorch-base:latest \
  us-central1-docker.pkg.dev/weight-and-biases-476906/persistent-artifacts/arr-pytorch-base:latest

# 4. Push to persistent-artifacts
docker push us-central1-docker.pkg.dev/weight-and-biases-476906/persistent-artifacts/arr-pytorch-base:$HASH
docker push us-central1-docker.pkg.dev/weight-and-biases-476906/persistent-artifacts/arr-pytorch-base:latest

# 5. Update .image-manifest
echo $HASH > .image-manifest

echo "Migration complete! Next launch will use persistent-artifacts registry."
```

**That's it! No permanent migration code in the codebase.**

**After migration:**
- Run `python training/cli.py setup` (creates persistent-artifacts if needed)
- Run `python training/cli.py launch` (uses migrated image, no rebuild)
- Old image in arr-coc-registry can be deleted during teardown

---

## Git Commit Strategy

**Commit 1:** Setup changes
```bash
git add training/cli/setup/steps.py
git commit -m "Add persistent-artifacts registry for PyTorch base image

- Create separate registry for arr-pytorch-base (never deleted)
- Prevents 2-4 hour rebuilds across teardowns (~$5-20 savings)
- Setup creates both persistent-artifacts + arr-coc-registry"
```

**Commit 2:** Launch changes
```bash
git add training/cli/launch/core.py
git commit -m "Update PyTorch image to use persistent-artifacts registry

- arr-pytorch-base now stored in persistent-artifacts
- Hash-based caching still works
- Cleanup targets persistent registry"
```

**Commit 3:** Cloud Build changes
```bash
git add .cloudbuild-arr-pytorch-base.yaml
git commit -m "Update Cloud Build to push PyTorch to persistent registry

- Image tags: persistent-artifacts/arr-pytorch-base:latest
- Both hash tag and :latest push to persistent registry"
```

**Commit 4:** Teardown changes
```bash
git add training/cli/teardown/steps.py training/cli/teardown/core.py
git commit -m "Teardown preserves persistent-artifacts registry

- Only deletes arr-coc-registry (fast-building images)
- Skips persistent-artifacts (PyTorch base)
- Shows manual deletion instructions with size estimate"
```

**Commit 5:** Documentation
```bash
git add training/cli/shared/setup_helper.py
git commit -m "Update infrastructure docs: two artifact registries

- Document persistent-artifacts (never deleted)
- Document arr-coc-registry (deletable)
- Clarify image storage locations"
```

---

## Rollback Plan

If something goes wrong:

**Immediate rollback:**
```bash
git revert HEAD~4..HEAD
```

**Clean state:**
1. Delete persistent-artifacts manually:
   ```bash
   gcloud artifacts repositories delete persistent-artifacts --location=us-central1
   ```

2. Next launch will create images in arr-coc-registry as before

**No data loss - worst case is one PyTorch rebuild**

---

## Success Metrics

After implementation:

✅ Setup creates 2 registries (persistent-artifacts + arr-coc-registry)
✅ PyTorch builds to persistent-artifacts
✅ Teardown deletes arr-coc-registry only
✅ Teardown shows persistent registry info message
✅ Next launch reuses PyTorch image (no rebuild)
✅ Cost savings: ~$5-20 per teardown cycle
✅ Time savings: 2-4 hours per teardown cycle

---

## Ready to Implement!

All changes documented, tested strategy in place, clean rollback available.

Let's do it! 🚀

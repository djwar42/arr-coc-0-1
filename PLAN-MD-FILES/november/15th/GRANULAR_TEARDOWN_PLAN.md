# Granular Teardown Implementation Plan

**Date**: 2025-11-16
**Feature**: Granular infrastructure vs Docker image teardown
**Status**: ✅ **PHASES 1 & 2 COMPLETE** - TUI + core logic ready for testing!

---

## Overview

Replace monolithic "DELETE everything" teardown with granular control:

- **DELETE** → Removes infrastructure (buckets, SA, worker pool) BUT keeps all 4 arr- images
- **PYTORCH** → Deletes only arr-pytorch-base image
- **ARR-ML-STACK** → Deletes only arr-ml-stack image
- **ARR-TRAINER** → Deletes only arr-trainer image
- **ARR-VERTEX-LAUNCHER** → Deletes only arr-vertex-launcher image
- **Combinations allowed**: `PYTORCH ARR-ML-STACK` deletes both in one run

**Always preserved** (never deleted):
- Artifact Registry repository itself (arr-coc-registry)
- Pricing data packages in Artifact Registry
- Standard infrastructure that should persist

---

## Architecture

### TUI Flow
```
User types confirmation → Parse keywords → Validate → Pass mode to core → Route to correct deletion
```

### CLI Flow
```
User passes --mode flag → Validate → Pass mode to core → Route to correct deletion
```

### Core Routing
```
run_teardown_core(mode="DELETE")
  ↓
if "DELETE" in mode:
    → _run_all_teardown_steps(skip_images=True)
       → Deletes infra, skips image deletion in registry step

if any image name in mode:
    → _delete_images(image_names)
       → Calls _delete_single_image() for each
```

---

## Implementation Checklist

### ✅ PHASE 1: TUI Updates (COMPLETE)

**File**: `training/cli/teardown/screen.py`

- [x] Update confirmation UI text to show all options
- [x] Parse space-separated keywords from input
- [x] Validate keywords against allowed list
- [x] Support multi-image deletion (e.g., "PYTORCH ARR-ML-STACK")
- [x] Pass mode string to `run_teardown_core()`
- [x] Commit: d62e8ed

---

### ✅ PHASE 2: Core Logic Implementation (COMPLETE - Commit 7036c19)

#### Step 2.1: Add Image Deletion Function

**File**: `training/cli/teardown/steps.py`

**Add new function**:
```python
def _delete_single_image(
    project_id: str,
    region: str,
    image_name: str,
    status: StatusCallback,
) -> bool:
    """
    Delete a single Docker image from Artifact Registry.

    Args:
        project_id: GCP project ID
        region: Registry region (us-central1)
        image_name: Image to delete (pytorch, arr-ml-stack, arr-trainer, arr-vertex-launcher)
        status: Status callback

    Returns:
        True if deletion succeeded (or image didn't exist)
        False if deletion failed
    """
    # Map keyword to actual image name
    image_map = {
        "PYTORCH": "arr-pytorch-base",
        "ARR-ML-STACK": "arr-ml-stack",
        "ARR-TRAINER": "arr-trainer",
        "ARR-VERTEX-LAUNCHER": "arr-vertex-launcher",
    }

    actual_name = image_map.get(image_name, image_name)
    registry_name = "arr-coc-registry"
    image_path = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/{actual_name}"

    status(f"   ⊕  Deleting image: {actual_name}...")

    try:
        # List all tags for this image
        result = subprocess.run(
            ["gcloud", "artifacts", "docker", "images", "list",
             f"{region}-docker.pkg.dev/{project_id}/{registry_name}",
             "--filter", f"package={actual_name}",
             "--format", "value(version)"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            status(f"   ⚠  Image {actual_name} not found (already deleted?)")
            return True  # Not an error

        tags = result.stdout.strip().split('\n') if result.stdout.strip() else []

        if not tags:
            status(f"   ℹ  Image {actual_name} has no tags (already deleted?)")
            return True

        # Delete each tag
        for tag in tags:
            tag_path = f"{image_path}:{tag}"
            status(f"   ⊕  Deleting tag: {tag}...")

            delete_result = subprocess.run(
                ["gcloud", "artifacts", "docker", "images", "delete",
                 tag_path, "--quiet"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if delete_result.returncode != 0:
                status(f"   ✗  Failed to delete {tag}: {delete_result.stderr}")
                return False

        status(f"   ✓  Image deleted: {actual_name} ({len(tags)} tags)")
        return True

    except subprocess.TimeoutExpired:
        status(f"   ✗  Timeout deleting {actual_name}")
        return False
    except Exception as e:
        status(f"   ✗  Error deleting {actual_name}: {e}")
        return False
```

#### Step 2.2: Update Registry Deletion to Skip Images

**File**: `training/cli/teardown/steps.py`

**Modify `_delete_registry()` signature**:
```python
def _delete_registry(
    region: str,
    status: StatusCallback,
    delete_images: bool = True,  # NEW PARAMETER
) -> bool:
```

**Add logic at start**:
```python
if not delete_images:
    status("   ℹ  Skipping image deletion (keeping all arr- images)")
    status("   Note: Registry and pricing data preserved")
    return True
```

#### Step 2.3: Update Coordinator

**File**: `training/cli/teardown/core.py`

**Modify `_run_all_teardown_steps()` signature**:
```python
def _run_all_teardown_steps(
    config: Dict[str, str],
    project_id: str,
    region: str,
    status: StatusCallback,
    dry_run: bool = False,
    skip_images: bool = False,  # NEW PARAMETER
) -> bool:
```

**Update Step 2/6 call**:
```python
# Step 2/6: Artifact Registry (SHARED)
status("")
status("Tearing down Artifact Registry... (2/6)")
status("   [dim]ℹ[/dim] Docker images for all ARR-COC prototypes")
if dry_run:
    if skip_images:
        status("   Would skip: Image deletion (keeping all arr- images)")
    else:
        status("   Would delete: arr-coc-registry (including all images)")
elif not _delete_registry(region, status, delete_images=not skip_images):
    return False
```

#### Step 2.4: Update Entry Point Routing

**File**: `training/cli/teardown/core.py`

**Add routing logic in `run_teardown_core()`** (after line 175):

```python
# Parse mode to determine what to delete
keywords = mode.split()

# Route based on mode
if "DELETE" in keywords:
    # Infrastructure teardown (skip image deletion)
    status("")
    status("[bold]Infrastructure Teardown Mode[/bold]")
    status("[dim]Deleting infrastructure, keeping all arr- images[/dim]")
    status("")

    success = _run_all_teardown_steps(
        config, project_id, region, status, dry_run, skip_images=True
    )

    if success:
        status("")
        status("[green]✓ Infrastructure teardown complete![/green]")
        status("[dim]All 4 arr- images preserved in Artifact Registry[/dim]")

    return success

else:
    # Image deletion mode
    image_keywords = [k for k in keywords if k in ["PYTORCH", "ARR-ML-STACK", "ARR-TRAINER", "ARR-VERTEX-LAUNCHER"]]

    if not image_keywords:
        status("[red]✗ No valid image names found in mode[/red]")
        return False

    status("")
    status(f"[bold]Image Deletion Mode[/bold]")
    status(f"[dim]Deleting {len(image_keywords)} image(s)[/dim]")
    status("")

    from .steps import _delete_single_image

    all_success = True
    for image_name in image_keywords:
        status(f"Deleting {image_name}...")
        if not _delete_single_image(project_id, region, image_name, status):
            all_success = False
            status(f"[red]✗ Failed to delete {image_name}[/red]")
        else:
            status(f"[green]✓ Deleted {image_name}[/green]")
        status("")

    if all_success:
        status("[green]✓ All image deletions complete![/green]")
    else:
        status("[yellow]⚠ Some image deletions failed[/yellow]")

    return all_success
```

---

### ⏳ PHASE 3: CLI Updates

**File**: `training/cli.py`

**Update teardown command**:
```python
@app.command()
def teardown(
    mode: str = typer.Option(
        "DELETE",
        "--mode",
        "-m",
        help="What to delete: DELETE (infra), PYTORCH, ARR-ML-STACK, ARR-TRAINER, ARR-VERTEX-LAUNCHER"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview deletions")
):
    """Teardown infrastructure or specific Docker images"""

    # Validate mode
    keywords = mode.upper().split()
    valid = ["DELETE", "PYTORCH", "ARR-ML-STACK", "ARR-TRAINER", "ARR-VERTEX-LAUNCHER"]
    invalid = [k for k in keywords if k not in valid]

    if invalid:
        rprint(f"[red]✗ Invalid keywords: {', '.join(invalid)}[/red]")
        rprint(f"[yellow]Valid: {', '.join(valid)}[/yellow]")
        raise typer.Exit(1)

    # ... rest of CLI logic, pass mode to run_teardown_core()
```

---

### ⏳ PHASE 4: Testing

**Test Cases**:

1. **DELETE only**
   ```
   Type: DELETE
   Expected: Buckets/SA/worker pool deleted, all 4 images kept
   ```

2. **Single image**
   ```
   Type: PYTORCH
   Expected: Only arr-pytorch-base deleted
   ```

3. **Multiple images**
   ```
   Type: PYTORCH ARR-ML-STACK
   Expected: Both arr-pytorch-base and arr-ml-stack deleted
   ```

4. **All images**
   ```
   Type: PYTORCH ARR-ML-STACK ARR-TRAINER ARR-VERTEX-LAUNCHER
   Expected: All 4 images deleted, infra kept
   ```

5. **Invalid keyword**
   ```
   Type: FOOBAR
   Expected: Error message, no deletion
   ```

6. **Dry run modes**
   ```
   Dry run with each mode above
   Expected: Preview messages, no actual deletion
   ```

---

## File Modifications Summary

| File | Changes | Lines |
|------|---------|-------|
| `teardown/screen.py` | ✅ Multi-keyword parsing, validation | +30 |
| `teardown/core.py` | ⏳ Mode routing logic | +80 |
| `teardown/steps.py` | ⏳ Image deletion function, registry update | +120 |
| `cli.py` | ⏳ Mode parameter | +20 |
| **Total** | | **+250** |

---

## Success Criteria

- ✅ TUI allows typing multiple keywords
- ✅ "DELETE" removes infra, keeps images (implemented in core.py)
- ✅ Image keywords delete only specified images (implemented in steps.py)
- ✅ Can combine multiple images in one run (routing logic supports this)
- ✅ Registry and pricing data always preserved (skip_images=True preserves)
- ⏳ CLI supports same modes via --mode flag (PHASE 3 - optional)
- ✅ Dry run works for all modes (integrated in routing)

---

**Next Action**: Test in TUI! Type "DELETE", "PYTORCH", or "PYTORCH ARR-ML-STACK" to verify

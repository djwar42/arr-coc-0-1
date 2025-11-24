# How to Re-Enable C3 Worker Pools

If you solve the C3 quota issue and want to use worker pools again:

---

## File 1: `.cloudbuild-pytorch-clean.yaml` (lines 166-175)

### Current (Default Workers):
```yaml
# Build options - Using DEFAULT Cloud Build (no worker pool)
options:
  logging: CLOUD_LOGGING_ONLY

timeout: '360m'  # 6 hours
```

### Revert To (C3 Worker Pool):
```yaml
# Build options - SPOT VM worker pool (C3 machines)
options:
  pool:
    name: 'projects/weight-and-biases-476906/locations/us-central1/workerPools/pytorch-spot-pool'
  logging: CLOUD_LOGGING_ONLY

timeout: '180m'  # 3 hours (faster with C3)
```

---

## File 2: `training/cli/setup/core.py` (lines 619-625)

### Current (Commented Out):
```python
# Step 3.5: Cloud Build IAM permissions (SKIPPED - not needed for default workers)
# NOTE: Cloud Build worker pools would require compute.admin + compute.networkUser
# But we're using default Cloud Build workers (no worker pool) to avoid quota issues
# If you enable worker pools in .cloudbuild-pytorch-clean.yaml, grant these manually:
#   - roles/compute.networkUser (for VPC peering)
#   - roles/compute.admin (for creating C3 instances)
```

### Revert To (Grant IAM):
```python
# Step 3.5: Grant Cloud Build service account permissions for worker pools
status("⏳ Granting Cloud Build worker pool permissions...")
project_number = subprocess.run(
    ["gcloud", "projects", "describe", project_id, "--format=value(projectNumber)"],
    capture_output=True,
    text=True,
    timeout=10,
)

if project_number.returncode == 0:
    project_num = project_number.stdout.strip()
    cloudbuild_sa = f"{project_num}@cloudbuild.gserviceaccount.com"

    # Roles needed for Cloud Build to create C3 instances in worker pools
    cloudbuild_roles = [
        "roles/compute.networkUser",  # Use VPC peering
        "roles/compute.admin",        # Create/manage compute instances
    ]

    for role in cloudbuild_roles:
        grant_cb = subprocess.run(
            [
                "gcloud", "projects", "add-iam-policy-binding", project_id,
                f"--member=serviceAccount:{cloudbuild_sa}",
                f"--role={role}",
                "--condition=None",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if grant_cb.returncode == 0 or "already has role" in grant_cb.stderr.lower():
            status(f"  ✓ Granted {role}")
        else:
            status(f"[yellow]  ⚠️  Warning: Failed to grant {role}[/yellow]")
else:
    status("[yellow]  ⚠️  Could not determine project number for Cloud Build SA[/yellow]")

status("[green]✓ Cloud Build permissions configured[/green]")
```

---

## File 3: `training/cli/shared/setup_helper.py` (lines 1168-1173)

### Current (Skipped):
```python
logs.append(f"  ✓ Removed 4 IAM role bindings")

# NOTE: Cloud Build IAM removal skipped (not needed for default workers)
# Default Cloud Build doesn't need compute.admin or compute.networkUser
# If you manually granted these for worker pools, remove them manually

return True
```

### Revert To (Remove IAM):
```python
logs.append(f"  ✓ Removed 4 IAM role bindings")

# Also remove Cloud Build service account permissions (for worker pools)
project_number = subprocess.run(
    ["gcloud", "projects", "describe", self.project_id, "--format=value(projectNumber)"],
    capture_output=True,
    text=True,
    timeout=10,
)

if project_number.returncode == 0:
    project_num = project_number.stdout.strip()
    cloudbuild_sa = f"{project_num}@cloudbuild.gserviceaccount.com"

    cloudbuild_roles = [
        "roles/compute.networkUser",
        "roles/compute.admin",
    ]

    for role in cloudbuild_roles:
        subprocess.run(
            ["gcloud", "projects", "remove-iam-policy-binding", self.project_id,
             "--member", f"serviceAccount:{cloudbuild_sa}",
             "--role", role, "--quiet"],
            capture_output=True, timeout=30
        )

    logs.append(f"  ✓ Removed 2 Cloud Build IAM role bindings")
else:
    logs.append(f"  ⚠️  Could not determine project number for Cloud Build SA removal")

return True
```

---

## File 4: `training/cli/teardown/core.py` (line 184)

### Current:
```python
f"4 IAM Role Bindings (training service account only)"
```

### Revert To:
```python
f"6 IAM Role Bindings (4 for training SA + 2 for Cloud Build worker pools)"
```

---

## Quick Revert Commands

```bash
# Check out the commit BEFORE we disabled worker pools
git log --oneline | grep -i "worker\|c3\|iam"

# Find the good commit (before "Remove Cloud Build IAM permissions")
# Let's say it's commit abc1234

# Revert just these files:
git checkout abc1234 -- .cloudbuild-pytorch-clean.yaml
git checkout abc1234 -- training/cli/setup/core.py
git checkout abc1234 -- training/cli/shared/setup_helper.py
git checkout abc1234 -- training/cli/teardown/core.py

# Commit the revert:
git add .
git commit -m "Re-enable C3 worker pools (C3 quota issue resolved)"
```

---

## Test After Revert

```bash
# 1. Run setup to grant IAM permissions
python training/cli.py setup

# 2. Verify worker pool can be used
python training/cli.py launch

# 3. If successful, you'll see:
#    "✓ Building pytorch-clean image on worker pool"
#    Build time: ~2-3 hours (instead of 6 hours!)
```

---

## Git Commits to Reference

```
cc65ee5 - Remove Cloud Build IAM permissions (current)
  ↑ This commit DISABLED worker pools

573b64c - Add Cloud Build worker pool IAM permissions (previous)
  ↑ This commit had worker pools ENABLED
```

To see the full diff:
```bash
git diff 573b64c cc65ee5
```

---

## Why We Disabled Worker Pools

**Problem**: "FAILED_PRECONDITION: quota restrictions" despite having C3_CPUS=144

**Root Cause**: Unknown (likely Cloud Build-specific C3 quota or allowlist)

**Workaround**: Use default Cloud Build (8 vCPU workers)

**Trade-off**: 2x slower builds (6hrs vs 3hrs), but it WORKS

---

## If You Solve the C3 Issue

Possible solutions that might work:
1. **Contact Google Support** - Request C3 enablement for Cloud Build
2. **Different Region** - Try europe-west4 or us-west1
3. **Different Machine** - Use N2 or C2 instead of C3
4. **Allowlist Request** - Some machine types need special approval

Once C3 works, revert these 4 files and re-run setup!

---

**Created**: 2025-11-10
**Commits**: cc65ee5 (disabled) ← 573b64c (enabled)

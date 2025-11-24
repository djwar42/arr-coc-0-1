# Artifact Registry Image Rename Plan

## Current → Proposed Names

| Current        | Proposed      | Why                                      |
|----------------|---------------|------------------------------------------|
| `base`         | `arr-base`    | Add project prefix for clarity           |
| `training`     | `arr-training`| Consistent prefix                        |
| `wandb-runner` | `arr-runner`  | Shorter, consistent (`wandb-` redundant) |

## Files to Update

### 1. Cloud Build Configs (3 files)

**`.cloudbuild-base.yaml`** (3 occurrences)
- Line 17: `arr-coc-registry/base:latest` → `arr-coc-registry/arr-base:latest`
- Line 18: `arr-coc-registry/base:latest` → `arr-coc-registry/arr-base:latest`  
- Line 20: `arr-coc-registry/base:latest` → `arr-coc-registry/arr-base:latest`

**`.cloudbuild-training.yaml`** (if exists - needs checking)
- Replace `/training:` → `/arr-training:`

**`.cloudbuild-runner.yaml`** (if exists - needs checking)
- Replace `/wandb-runner:` → `/arr-runner:`

### 2. Dockerfiles (3 files)

**`training/images/training-image/Dockerfile`**
- Line 1: `FROM ...registry/base:latest` → `FROM ...registry/arr-base:latest`

**`training/images/wandb-runner/Dockerfile`** (if exists)
- Check if it references other images

**`training/images/base-image/Dockerfile`**
- Check for self-references (likely none)

### 3. Python Code Files (2 files)

**`training/cli/launch/core.py`** (Many occurrences!)

Base image references (~line 845-950):
- `registry_name}/base:` → `registry_name}/arr-base:`
- `/base@{newest_digest}` → `/arr-base@{newest_digest}`
- `/base:{old_tag}` → `/arr-base:{old_tag}`

Training image references (~line 1024-1250):
- `registry_name}/training:` → `registry_name}/arr-training:`
- `/training:{old_tag}` → `/arr-training:{old_tag}`

Runner image references (~line 1350-1550):
- `registry_name}/wandb-runner:` → `registry_name}/arr-runner:`
- All `wandb-runner` → `arr-runner`

**`training/cli/monitor/core.py`** (~line 150)
- `'launcher': 'wandb-runner'` → `'launcher': 'arr-runner'`
- Comment: "wandb-runner image name" → "arr-runner image name"

### 4. Documentation Files

**`training/CLAUDE.md`**
- Search for image name references in examples

**`training/README.md`** (if exists)
- Update any image name references

**`PERFORMANCE_MONITORING.md`**
- Check for example output with image names

## Search & Replace Strategy

### Safe Approach (Recommended)

Use specific search patterns to avoid false matches:

```bash
# Base image
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "Dockerfile*" \) \
  -exec grep -l "registry/base\|/base:" {} \;

# Training image
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "Dockerfile*" \) \
  -exec grep -l "registry/training\|/training:" {} \;

# Runner image  
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "Dockerfile*" \) \
  -exec grep -l "wandb-runner" {} \;
```

### Replace Commands

```bash
# CAREFUL - test each replacement!

# Base image (3 patterns)
sed -i 's|arr-coc-registry/base:|arr-coc-registry/arr-base:|g' .cloudbuild-base.yaml
sed -i 's|arr-coc-registry/base:|arr-coc-registry/arr-base:|g' training/images/training-image/Dockerfile
sed -i 's|registry_name}/base:|registry_name}/arr-base:|g' training/cli/launch/core.py
sed -i 's|/base@|/arr-base@|g' training/cli/launch/core.py
sed -i 's|/base:|/arr-base:|g' training/cli/launch/core.py

# Training image
sed -i 's|registry_name}/training:|registry_name}/arr-training:|g' training/cli/launch/core.py
sed -i 's|/training:|/arr-training:|g' training/cli/launch/core.py

# Runner image (most changes!)
sed -i 's|wandb-runner|arr-runner|g' training/cli/launch/core.py
sed -i 's|wandb-runner|arr-runner|g' training/cli/monitor/core.py
```

## GCP Artifact Registry Changes

### What Happens to Old Images?

After code rename, you have 2 options:

**Option 1: Leave old images (coexist)**
- Old: `base`, `training`, `wandb-runner`
- New: `arr-base`, `arr-training`, `arr-runner`
- Both exist in registry
- Manually delete old ones later via GCP Console

**Option 2: Delete old images (clean slate)**
```bash
# Delete old images from Artifact Registry
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/PROJECT_ID/arr-coc-registry/base:latest

gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/PROJECT_ID/arr-coc-registry/training:latest

gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/PROJECT_ID/arr-coc-registry/wandb-runner:latest
```

## Testing Plan

1. ✅ Update all files (see lists above)
2. ✅ Commit changes
3. ✅ Run `python training/cli.py setup` - Should create new image names in checks
4. ✅ Run `python training/cli.py launch` - Should build with new names
5. ✅ Check Artifact Registry - Should see `arr-base`, `arr-training`, `arr-runner`
6. ✅ Test monitor screen - Should detect new image names
7. ✅ Clean up old images (optional)

## Rollback Plan

If something breaks:

```bash
# Revert last commit
git revert HEAD

# Or manually restore old names
sed -i 's|arr-base|base|g' (all files)
sed -i 's|arr-training|training|g' (all files)  
sed -i 's|arr-runner|wandb-runner|g' (all files)
```

## Estimated Impact

- **Files to change**: ~8 files
- **Lines to change**: ~30-40 lines
- **Risk**: Low (just string replacements)
- **Rebuild required**: Yes (all 3 images need rebuild with new names)
- **Time**: ~30-45 min (mostly Cloud Build time)

## Why This Matters

**Naming consistency:**
- All images clearly belong to ARR project
- Easier to identify in shared registry
- Follows convention: `{project}-{purpose}`

**Clarity:**
- `arr-runner` is clearer than `wandb-runner` (W&B is tool, not identity)
- Consistent `arr-` prefix makes ownership obvious

---

**Ready to execute?** Start with file updates, then test!

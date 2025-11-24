# Image Manifest System - Critical Build Dependency Tracking

## What Are Image Manifests?

Each Docker image has a `.image-manifest` file that lists **all files that affect the image build**:

```
training/images/arr-pytorch-base/.image-manifest
training/images/arr-ml-stack/.image-manifest
training/images/arr-trainer/.image-manifest
training/images/arr-vertex-launcher/.image-manifest
```

## Why They Exist

The manifest tells the build system:
- Which files to hash for rebuild detection
- Whether Dockerfile changes require rebuild
- Whether code changes require rebuild
- Whether dependency changes require rebuild

**Without manifests:** Only Dockerfile changes trigger rebuilds (WRONG!)
**With manifests:** ANY relevant file change triggers rebuilds (CORRECT!)

## How Hash Detection Works

```python
# 1. Read manifest
manifest = read_file("training/images/arr-trainer/.image-manifest")
# Gets list of files: Dockerfile, setup.py, arr_coc/**/*.py, etc.

# 2. Calculate combined git hash
git_hash = hash_all_files(manifest)
# Gets: "9350e29" (from latest commit that touched ANY file)

# 3. Check if image with this hash exists
image_exists = check_registry("arr-trainer:9350e29")

# 4. Decide
if image_exists:
    print("✓ Cached! Using arr-trainer:9350e29")
    skip_build()
else:
    print("🏗️ Building! Files changed!")
    build_image()
```

## ⚠️ CRITICAL: When to Update Manifests

**UPDATE THE MANIFEST if you:**
1. Add new Python files to `arr_coc/`
2. Add new training scripts to `training/`
3. Add new dependencies to `requirements.txt`
4. Add new build scripts
5. Change upstream Dockerfile dependencies
6. Add ANY file that goes INTO the Docker image

**Example:**

You add `arr_coc/new_feature.py`:
- ❌ Without updating manifest: New file NOT hashed → No rebuild → Old image!
- ✅ After updating manifest: New file hashed → Rebuild → New code in image!

## Manifest Format

```
# Header comment
# ============================================================================

# List files relative to project root
training/images/arr-trainer/Dockerfile
training/images/arr-ml-stack/Dockerfile  # Upstream dependency!
setup.py
arr_coc/**/*.py  # Glob patterns supported
training/*.py
requirements.txt
```

## Where to Update

**If you modify:**
- `arr_coc/` code → Update `arr-trainer/.image-manifest`
- `training/` scripts → Update `arr-trainer/.image-manifest`
- PyTorch build → Update `arr-pytorch-base/.image-manifest`
- ML packages → Update `arr-ml-stack/.image-manifest`
- Launch agent → Update `arr-vertex-launcher/.image-manifest`

## Real Example

**Scenario:** You add `arr_coc/attention.py`

**Before updating manifest:**
```bash
# Build 1: Creates arr-trainer:abc123
python training/cli.py launch  # SUCCESS

# Edit: Add arr_coc/attention.py
vim arr_coc/attention.py

# Build 2: Should rebuild but doesn't!
python training/cli.py launch  # Uses CACHED arr-trainer:abc123 ❌ WRONG!
```

**After updating manifest:**
```bash
# Add to arr-trainer/.image-manifest
echo "arr_coc/attention.py" >> training/images/arr-trainer/.image-manifest
git add training/images/arr-trainer/.image-manifest
git commit -m "Add attention.py to arr-trainer manifest"

# Build 3: Now it detects the change!
python training/cli.py launch  # Rebuilds arr-trainer:def456 ✅ CORRECT!
```

## Testing Manifests

```bash
# Test hash detection
python -c "
from training.cli.launch.core import _hash_files_from_manifest
from pathlib import Path

manifest = Path('training/images/arr-trainer/.image-manifest')
project_root = Path('.')

hash1 = _hash_files_from_manifest(manifest, project_root)
print(f'Current hash: {hash1}')

# Modify arr_coc/texture.py
# ... make changes ...

hash2 = _hash_files_from_manifest(manifest, project_root)
print(f'New hash: {hash2}')
print(f'Changed: {hash1 != hash2}')
"
```

## Troubleshooting

**Symptom:** Changes not triggering rebuilds

**Fix:**
1. Check manifest exists: `ls training/images/arr-*/.image-manifest`
2. Check manifest has your files: `cat training/images/arr-trainer/.image-manifest`
3. Add missing files to manifest
4. Commit manifest changes
5. Rebuild

**Symptom:** "Couldn't find Dockerfile in manifest"

**Fix:**
1. Check paths in manifest match actual file locations
2. Update old `training/images/old-name/` → `training/images/arr-name/`
3. Commit manifest fixes

## Best Practices

✅ **DO:**
- Update manifests when adding code files
- Use glob patterns (`arr_coc/**/*.py`)
- Include upstream Dockerfile dependencies
- Test hash changes after updates
- Commit manifest with code changes

❌ **DON'T:**
- Forget to update manifests (causes stale builds!)
- Include generated files (logs, __pycache__, etc.)
- Include files not used in Docker build
- Use absolute paths (use relative to project root)

---

**Remember:** Manifests are CODE INPUTS, not documentation! The build system reads them to detect changes! 🎯

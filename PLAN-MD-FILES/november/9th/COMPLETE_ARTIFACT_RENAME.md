# COMPLETE Artifact Rename - Every Reference

## Image Names to Change

| Current        | New           | Occurrences |
|----------------|---------------|-------------|
| `base`         | `arr-base`    | 11 total    |
| `training`     | `arr-training`| 4 total     |
| `wandb-runner` | `arr-runner`  | 9 total     |

**Total changes needed: 24 references across 4 files**

---

## COMPLETE FILE-BY-FILE BREAKDOWN

### 1. `.cloudbuild-base.yaml` (3 changes)

**Line 17**: Tag specification
```yaml
# OLD
- 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/base:latest'

# NEW  
- 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-base:latest'
```

**Line 27**: Substitution tag
```yaml
# OLD
- 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/base:latest'

# NEW
- 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-base:latest'
```

**Line 32**: Images list
```yaml
# OLD
- 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/base:latest'

# NEW
- 'us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-base:latest'
```

---

### 2. `training/images/training-image/Dockerfile` (1 change)

**Line 13**: FROM statement
```dockerfile
# OLD
FROM us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/base:latest

# NEW
FROM us-central1-docker.pkg.dev/weight-and-biases-476906/arr-coc-registry/arr-base:latest
```

---

### 3. `training/cli/launch/core.py` (17 changes!)

#### Base Image References (7 changes)

**Line 816**: Base image name construction
```python
# OLD
base_image_name = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/base:latest"

# NEW
base_image_name = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/arr-base:latest"
```

**Line 931**: Docker pull with digest
```python
# OLD
["docker", "pull", f"{artifact_registry_base}/base@{newest_digest}"],

# NEW
["docker", "pull", f"{artifact_registry_base}/arr-base@{newest_digest}"],
```

**Line 940**: Docker tag command
```python
# OLD
["docker", "tag", f"{artifact_registry_base}/base@{newest_digest}", f"{artifact_registry_base}/base:latest"],

# NEW
["docker", "tag", f"{artifact_registry_base}/arr-base@{newest_digest}", f"{artifact_registry_base}/arr-base:latest"],
```

**Line 949**: Docker push command
```python
# OLD
["docker", "push", f"{artifact_registry_base}/base:latest"],

# NEW  
["docker", "push", f"{artifact_registry_base}/arr-base:latest"],
```

**Line 998**: Old tag cleanup
```python
# OLD
f"{artifact_registry_base}/base:{old_tag}",

# NEW
f"{artifact_registry_base}/arr-base:{old_tag}",
```

**Line 1124**: Comment about base image
```python
# OLD
#   - If base Dockerfile or requirements changed → base:latest tag gets new digest

# NEW
#   - If base Dockerfile or requirements changed → arr-base:latest tag gets new digest
```

#### Training Image References (3 changes)

**Line 1065**: Training image name with hash
```python
# OLD
training_image = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/training:{dockerfile_hash}"

# NEW
training_image = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/arr-training:{dockerfile_hash}"
```

**Line 1066**: Training image latest tag
```python
# OLD
training_image_latest = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/training:latest"

# NEW
training_image_latest = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/arr-training:latest"
```

**Line 1255**: Old training tag cleanup
```python
# OLD
f"{artifact_registry_base}/training:{old_tag}",

# NEW
f"{artifact_registry_base}/arr-training:{old_tag}",
```

#### Runner Image References (7 changes)

**Line 1318**: Runner image with hash
```python
# OLD
runner_image = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/wandb-runner:{dockerfile_hash}"

# NEW
runner_image = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/arr-runner:{dockerfile_hash}"
```

**Line 1319**: Runner image latest
```python
# OLD
runner_image_latest = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/wandb-runner:latest"

# NEW
runner_image_latest = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/arr-runner:latest"
```

**Line 1465**: Check runner registry
```python
# OLD
f"{region}-docker.pkg.dev/{project_id}/{registry_name}/wandb-runner",

# NEW
f"{region}-docker.pkg.dev/{project_id}/{registry_name}/arr-runner",
```

**Line 1483**: Status message
```python
# OLD
status(f"  [dim]Repository:[/dim] {registry_name}/wandb-runner")

# NEW
status(f"  [dim]Repository:[/dim] {registry_name}/arr-runner")
```

**Line 1508**: List runner images
```python
# OLD
f"{artifact_registry_base}/wandb-runner",

# NEW
f"{artifact_registry_base}/arr-runner",
```

**Line 1536**: Old runner tag cleanup
```python
# OLD
f"{artifact_registry_base}/wandb-runner:{old_tag}",

# NEW
f"{artifact_registry_base}/arr-runner:{old_tag}",
```

**Line 1602**: Execute runner image construction
```python
# OLD
runner_image = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/wandb-runner:{dockerfile_hash}"

# NEW
runner_image = f"{region}-docker.pkg.dev/{project_id}/{registry_name}/arr-runner:{dockerfile_hash}"
```

---

### 4. `training/cli/monitor/core.py` (2 changes)

**Line 717**: Comment about images
```python
# OLD
# Check ALL THREE images: base, training, launcher (wandb-runner)

# NEW
# Check ALL THREE images: arr-base, arr-training, launcher (arr-runner)
```

**Line 724**: Image name mapping
```python
# OLD
'launcher': 'wandb-runner'  # W&B Launch agent uses wandb-runner image name

# NEW
'launcher': 'arr-runner'  # W&B Launch agent uses arr-runner image name
```

---

## Automated Sed Commands

**CAREFUL - Review each before running!**

```bash
# 1. .cloudbuild-base.yaml (3 changes)
sed -i '' 's|arr-coc-registry/base:|arr-coc-registry/arr-base:|g' .cloudbuild-base.yaml

# 2. Dockerfile (1 change)
sed -i '' 's|arr-coc-registry/base:|arr-coc-registry/arr-base:|g' training/images/training-image/Dockerfile

# 3. launch/core.py - Base image (6 changes + 1 comment)
sed -i '' 's|registry_name}/base:|registry_name}/arr-base:|g' training/cli/launch/core.py
sed -i '' 's|/base@|/arr-base@|g' training/cli/launch/core.py  
sed -i '' 's|/base:|/arr-base:|g' training/cli/launch/core.py

# 4. launch/core.py - Training image (3 changes)
sed -i '' 's|registry_name}/training:|registry_name}/arr-training:|g' training/cli/launch/core.py
sed -i '' 's|/training:|/arr-training:|g' training/cli/launch/core.py

# 5. launch/core.py - Runner image (7 changes)
sed -i '' 's|wandb-runner|arr-runner|g' training/cli/launch/core.py

# 6. monitor/core.py - Runner ref (2 changes)
sed -i '' 's|wandb-runner|arr-runner|g' training/cli/monitor/core.py
```

**Note**: macOS requires `sed -i ''`, Linux uses `sed -i` (no empty string)

---

## Generated Files (Runtime)

### `.cloudbuild-training.yaml` (GENERATED at runtime!)

**Line 1135**: Build command in generated config
```python
# In training/cli/launch/core.py around line 1135
# This is Python code that GENERATES a cloudbuild YAML with the training image name

cloudbuild_config = f"""
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--pull', '-t', '{training_image}', '-f', 'training/images/training-image/Dockerfile', '.']
images: ['{training_image}']
timeout: 2400s
"""
```

The `training_image` variable is already fixed by line 1065 change above!
So this will automatically use `arr-training` once we fix line 1065.

---

## Testing Checklist

After making changes:

1. ✅ **Git diff review**: `git diff` - Review every change manually
2. ✅ **Search verification**:
   ```bash
   # Should find ZERO old references
   grep -r "registry/base:" . --include="*.py" --include="*.yaml"
   grep -r "registry/training:" . --include="*.py" --include="*.yaml"  
   grep -r "wandb-runner" . --include="*.py"
   ```
3. ✅ **Syntax check**: `python -m py_compile training/cli/launch/core.py`
4. ✅ **YAML validation**: `python -c "import yaml; yaml.safe_load(open('.cloudbuild-base.yaml'))"`
5. ✅ **Run setup**: `python training/cli.py setup` - Should reference new names
6. ✅ **Check logs**: Look for "arr-base", "arr-training", "arr-runner" in status messages
7. ✅ **GCP verification**: After build, check Artifact Registry for new image names

---

## Rollback if Needed

```bash
git diff > rename_changes.patch  # Save changes first!
git checkout .  # Revert all changes
```

---

**Total changes: 24 references across 4 files**
**Estimated time: 10-15 min to apply + verify**
**Risk level: LOW (pure string replacements, no logic changes)**


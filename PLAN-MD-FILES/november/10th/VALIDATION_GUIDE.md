# Image Build Validation Guide

## 🎯 How to Prove Your Changes Actually Worked

### The Problem
Docker caches layers aggressively. Even with `--no-cache`, old code can persist if Docker thinks "close enough." This guide shows how to PROVE your changes made it through.

---

## ✅ Automatic Validation (Built-In)

Our Cloud Build has automatic validation that runs after every build:

```yaml
# STEP 3: Validate the fix worked
- name: 'gcr.io/cloud-builders/docker'
  args:
    - Check protobuf CMakeLists.txt contains VERSION 3.5
    - Check Docker label has current git hash
    - Exit 1 if validation fails
```

**What to look for in Cloud Build logs:**

```
════════════════════════════════════════════════════════════════
🔍 VALIDATION: Verifying protobuf fix was applied
════════════════════════════════════════════════════════════════

📄 Checking protobuf CMakeLists.txt...
cmake_minimum_required(VERSION 3.5)

✅ SUCCESS: Protobuf fix verified! CMake version = 3.5

🏷️  Checking Docker label for git hash...
   Image git hash: 5f3f8c0a1b2c3d4e5f6789012345678901234567
   Current git hash: 5f3f8c0a1b2c3d4e5f6789012345678901234567
✅ SUCCESS: Git hash matches! Smart cache busting working!

════════════════════════════════════════════════════════════════
✅ VALIDATION COMPLETE - Image is good to use!
════════════════════════════════════════════════════════════════
```

**If validation FAILS:**

```
❌ FAILURE: Protobuf fix did NOT apply!
   Expected: cmake_minimum_required(VERSION 3.5)
   Got: cmake_minimum_required(VERSION 2.8.12)

🚨 This means Docker used an OLD cached layer!
```

This automatically fails the build and prevents bad images from being used!

---

## 🔍 Manual Validation (After Build)

### 1. Check Docker Label

```bash
# Get git hash from built image
docker inspect us-central1-docker.pkg.dev/.../arr-pytorch-clean:latest \
  | grep dockerfile.git.hash

# Should show:
"dockerfile.git.hash": "5f3f8c0a1b2c3d4e5f6789012345678901234567"

# Compare to current commit:
git log -1 --format=%H training/images/pytorch-clean/Dockerfile

# If they match → smart cache busting worked!
```

### 2. Check Protobuf Fix

```bash
# Run container and check protobuf version
docker run --rm arr-pytorch-clean:latest \
  head -3 /opt/pytorch/third_party/protobuf/cmake/CMakeLists.txt

# Should show:
cmake_minimum_required(VERSION 3.5)

# NOT this (broken):
cmake_minimum_required(VERSION 2.8.12)
```

### 3. Check Build Logs

During the build, you should see echo statements proving the fix ran:

```
Step 7/15 : RUN cd pytorch && echo "🔧 Patching protobuf..." && find...
🔧 Patching protobuf CMakeLists.txt files...
✅ Protobuf patched! Verification:
cmake_minimum_required(VERSION 3.5)
```

If you see this → fix executed successfully!

---

## 🚨 What to Do If Validation Fails

### Scenario 1: Git Hash Mismatch

**Symptom:**
```
Image git hash: abc123 (old)
Current git hash: def456 (new)
```

**Cause:** Docker used cached layer from before your commit

**Fix:**
1. Update cache-bust comment in Dockerfile line 1
2. Rebuild with `python training/cli.py launch`

### Scenario 2: Protobuf Still Shows 2.8.12

**Symptom:**
```
cmake_minimum_required(VERSION 2.8.12)  # Should be 3.5!
```

**Cause:** The sed/find command didn't run (cached layer!)

**Fix:**
```bash
# Nuclear option: Delete the cache manually
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/.../arr-pytorch-clean:buildcache

# Then rebuild
python training/cli.py launch
```

### Scenario 3: Build Logs Show Old Command

**Symptom:**
Build logs show the old sed command:
```
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/...'
```

Instead of the new find command:
```
find third_party/protobuf -name "CMakeLists.txt" -exec sed...
```

**Cause:** Uncommitted changes or git issue

**Fix:**
```bash
# Verify Dockerfile is committed
git status training/images/pytorch-clean/Dockerfile

# Should show "nothing to commit"
# If shows "modified" → you need to commit!

git add training/images/pytorch-clean/Dockerfile
git commit -m "Fix protobuf"
python training/cli.py launch
```

---

## 📊 Expected Validation Output (Success)

When everything works, you'll see this sequence in Cloud Build logs:

```
STEP 1: git clone pytorch
→ Cloning repository...
→ ✅ Clone complete

STEP 2: Fix protobuf
→ 🔧 Patching protobuf CMakeLists.txt files...
→ ✅ Protobuf patched! Verification:
→ cmake_minimum_required(VERSION 3.5)

STEP 3: Build PyTorch
→ 🏗️  Building PyTorch v2.6.0 (this takes 2-4 hours)...
→ [... compilation output ...]
→ ✅ PyTorch build complete!

STEP 4: Validation
→ 🔍 VALIDATION: Verifying protobuf fix was applied
→ ✅ SUCCESS: Protobuf fix verified! CMake version = 3.5
→ ✅ SUCCESS: Git hash matches! Smart cache busting working!
→ ✅ VALIDATION COMPLETE - Image is good to use!
```

**Key indicators of success:**
- ✅ Echo statements appear in correct order
- ✅ Protobuf shows VERSION 3.5 in both build AND validation
- ✅ Git hash from image matches current commit
- ✅ No "CACHED" messages for changed layers

---

## 🎯 Quick Validation Checklist

After launching a build, check these 3 things:

- [ ] Build logs show `🔧 Patching protobuf...` (proof fix ran)
- [ ] Validation step passes (automatic check)
- [ ] Git hash in image matches current commit

If all 3 pass → your changes definitely made it through!

---

## 💡 Pro Tips

**Tip 1: Watch the Dockerfile hash**
```bash
# Launch shows the hash
python training/cli.py launch
→ Dockerfile hash: 5f3f8c0a1b2c3d4e5f6789012345678901234567
```

If hash changes after your edit → smart cache will bust!

**Tip 2: Use grep to find smoking guns**
```bash
# Search Cloud Build logs for validation
gcloud builds log BUILD_ID | grep "VALIDATION"

# Search for protobuf fix
gcloud builds log BUILD_ID | grep "Patching protobuf"
```

**Tip 3: Compare before/after**
```bash
# Old image (if you have it)
docker run old-image head -3 /opt/pytorch/.../CMakeLists.txt

# New image
docker run new-image head -3 /opt/pytorch/.../CMakeLists.txt

# Should show different VERSION!
```

---

## 🔄 Recurring Pattern

**Every time you edit the Dockerfile:**

1. Edit → Commit → Launch
2. Check validation in Cloud Build logs
3. If fails → investigate which step failed
4. Fix → Commit → Launch again

**The validation step prevents bad images from being used!**

If Step 3 (validation) fails → build fails → no bad image pushed → safe!

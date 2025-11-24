# Complete Screen Removal Plan

**Date**: 2025-11-16

---

## Screens to REMOVE

### 1. ❌ Pricing Screen
- **File**: `training/cli/pricing/screen.py`
- **Purpose**: Shows GPU pricing information
- **Why remove**: Pricing becomes stale

### 2. ❌ Reduce Screen  
- **File**: `training/cli/reduce/screen.py`
- **Purpose**: Unknown (need to check)
- **Why remove**: User request

### 3. ❌ Truffles Screen
- **File**: `training/cli/truffles/screen.py`
- **Purpose**: Unknown (need to check)
- **Why remove**: User request

### 4. ❌ GPU Screen
- **File**: `training/cli/gpu/screen.py`
- **Purpose**: Unknown (need to check)
- **Why remove**: User request

### 5. ❌ Teardown Screen
- **File**: `training/cli/teardown/screen.py`
- **Purpose**: Teardown infrastructure
- **Why remove**: Dangerous operation, should not be in menu

---

## Screens to KEEP

### ✅ Home Screen (modified)
- **File**: `training/cli/home/screen.py`
- **Keep**: Yes, but remove buttons/menu items for deleted screens

### ✅ Monitor Screen
- **File**: `training/cli/monitor/screen.py`
- **Keep**: Yes (main monitoring interface)

### ✅ Launch Screen
- **File**: `training/cli/launch/screen.py`
- **Keep**: Yes (launch training jobs)

### ✅ Setup Screen
- **File**: `training/cli/setup/screen.py`
- **Keep**: Yes (setup infrastructure)

### ✅ Infra Screen
- **File**: `training/cli/infra/screen.py`
- **Keep**: Yes (show infrastructure status)

---

## Files to DELETE

1. `training/cli/pricing/screen.py`
2. `training/cli/reduce/screen.py`
3. `training/cli/truffles/screen.py`
4. `training/cli/gpu/screen.py`
5. `training/cli/teardown/screen.py`

**Total files to delete**: 5 screen files

---

## Files to MODIFY

1. **`training/cli/home/screen.py`**
   - Remove menu entries for deleted screens
   - Remove button handlers
   - Remove navigation actions

2. **`training/cli.py` or `training/tui.py`**
   - Remove screen route registrations
   - Remove imports

3. **`training/cli/unit/test_all.py`**
   - Remove tests for deleted screens

4. **`training/cli/setup/core.py`**
   - Remove pricing table (dollar amounts)

5. **`training/cli/shared/gpu_quota_instruct.py`**
   - Remove pricing table (unused file anyway)

---

## Additional Cleanup

Check for and remove:
- Any directories: `training/cli/pricing/`, `training/cli/reduce/`, etc.
- Core logic files if they exist and are unused
- Any other supporting files in those directories

---

## Implementation Steps

### Step 1: Examine Each Screen
First, let me check what each screen does:

```bash
head -20 training/cli/pricing/screen.py
head -20 training/cli/reduce/screen.py
head -20 training/cli/truffles/screen.py
head -20 training/cli/gpu/screen.py
```

### Step 2: Find Home Screen References
```bash
grep -n "pricing\|reduce\|truffles\|gpu\|teardown" training/cli/home/screen.py
```

### Step 3: Delete Screen Files
```bash
rm training/cli/pricing/screen.py
rm training/cli/reduce/screen.py
rm training/cli/truffles/screen.py
rm training/cli/gpu/screen.py
rm training/cli/teardown/screen.py
```

### Step 4: Clean Up Directories
```bash
# Check if entire directories should be removed
ls training/cli/pricing/
ls training/cli/reduce/
ls training/cli/truffles/
ls training/cli/gpu/
ls training/cli/teardown/
```

### Step 5: Update Home Screen
Remove buttons, menu items, and navigation for all deleted screens

### Step 6: Update Routes
Remove screen registrations from main app

### Step 7: Update Tests
Remove screen import tests

### Step 8: Remove Pricing Tables
Remove hardcoded dollar amounts from setup/core.py

---

## Expected Menu After Removal

### BEFORE:
```
1. Monitor
2. Launch
3. Setup
4. Teardown       ← REMOVE
5. Infra
?. Pricing        ← REMOVE
?. Reduce         ← REMOVE
?. Truffles       ← REMOVE
?. GPU            ← REMOVE
```

### AFTER:
```
1. Monitor
2. Launch
3. Setup
4. Infra
```

**Final menu**: 4 options (Monitor, Launch, Setup, Infra)

---

**Next**: Examine each screen to confirm what they do before deletion


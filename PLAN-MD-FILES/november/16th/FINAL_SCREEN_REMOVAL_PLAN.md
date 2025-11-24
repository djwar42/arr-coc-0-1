# Final Screen Removal Plan

**Date**: 2025-11-16

---

## Screens to REMOVE (4 total)

### 1. ❌ Pricing Screen
- **File**: `training/cli/pricing/screen.py`
- **Why**: Pricing info becomes stale

### 2. ❌ Reduce Screen
- **File**: `training/cli/reduce/screen.py`
- **Why**: User request

### 3. ❌ Truffles Screen
- **File**: `training/cli/truffles/screen.py`
- **Why**: User request

### 4. ❌ GPU Screen
- **File**: `training/cli/gpu/screen.py`
- **Why**: User request

---

## Screens to KEEP (6 total)

### ✅ Monitor Screen
- **File**: `training/cli/monitor/screen.py`

### ✅ Launch Screen
- **File**: `training/cli/launch/screen.py`

### ✅ Setup Screen
- **File**: `training/cli/setup/screen.py`

### ✅ Teardown Screen ⚠️ KEEP THIS!
- **File**: `training/cli/teardown/screen.py`
- **Status**: KEEP (user confirmed)

### ✅ Infra Screen
- **File**: `training/cli/infra/screen.py`

### ✅ Home Screen (modified)
- **File**: `training/cli/home/screen.py`
- Remove buttons for: pricing, reduce, truffles, gpu

---

## Files to DELETE

1. `training/cli/pricing/screen.py`
2. `training/cli/reduce/screen.py`
3. `training/cli/truffles/screen.py`
4. `training/cli/gpu/screen.py`

**Total**: 4 screen files

---

## Files to MODIFY

1. `training/cli/home/screen.py` - Remove 4 menu items
2. `training/cli.py` or `training/tui.py` - Remove 4 routes
3. `training/cli/unit/test_all.py` - Remove 4 screen tests
4. `training/cli/setup/core.py` - Remove pricing table

---

## Menu After Removal

```
1. Monitor
2. Launch
3. Setup
4. Teardown      ← KEEP!
5. Infra
```

**Total screens**: 5 (removed 4, kept 5 + home)


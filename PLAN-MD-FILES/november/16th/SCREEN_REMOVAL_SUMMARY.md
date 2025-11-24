# Screen Removal Summary

**What we're removing**: UI access only (menus, buttons, screens)
**What we're keeping**: Core logic (for programmatic/developer use)

---

## Screens Being REMOVED from TUI/CLI

### 1. ❌ Teardown Screen
**Files to DELETE:**
- `training/cli/teardown/screen.py` - Entire TUI screen

**Files to MODIFY:**
- `training/cli/home/screen.py` - Remove menu option 4 and navigation
- `training/cli.py` or `training/tui.py` - Remove route registration
- `training/cli/unit/test_all.py` - Remove teardown screen tests

**What STAYS:**
- ✅ `training/cli/teardown/core.py` - Core teardown logic (developers can still import)

---

## What's NOT Being Removed (Screens That Stay)

### ✅ Monitor Screen - STAYS
- `training/cli/monitor/screen.py`
- Menu option: Still accessible

### ✅ Launch Screen - STAYS  
- `training/cli/launch/screen.py`
- Menu option: Still accessible

### ✅ Setup Screen - STAYS
- `training/cli/setup/screen.py`
- Menu option: Still accessible
- May have its own teardown button (that's fine - setup needs clean reinstall)

### ✅ Infra Screen - STAYS
- `training/cli/infra/screen.py`
- Menu option: Still accessible

### ✅ Home Screen - STAYS (Modified)
- `training/cli/home/screen.py`
- Stays but loses teardown button/menu item

---

## Summary

**Screens REMOVED**: 1 (Teardown)
**Screens STAYING**: 5 (Home, Monitor, Launch, Setup, Infra)

**Core logic preserved**: Yes (teardown/core.py remains for developers)

---

## Before & After Menu

### BEFORE (4 options):
```
┌─────────────────────────┐
│ 1. Monitor              │
│ 2. Launch               │
│ 3. Setup                │
│ 4. Teardown        ← REMOVE
│ 5. Infra                │
└─────────────────────────┘
```

### AFTER (4 options):
```
┌─────────────────────────┐
│ 1. Monitor              │
│ 2. Launch               │
│ 3. Setup                │
│ 4. Infra                │
└─────────────────────────┘
```

---

## Non-Screen Changes (Pricing)

Also removing pricing tables from:
- `training/cli/setup/core.py` (pricing table in setup flow)
- `training/cli/shared/gpu_quota_instruct.py` (unused file)

These are NOT screen removals - just removing hardcoded dollar amounts.

---

**Total screens removed**: 1 (Teardown only)
**Total files deleted**: 1 (`training/cli/teardown/screen.py`)
**Total files modified**: ~6 (home, routes, tests, setup pricing)


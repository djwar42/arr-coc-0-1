# Complete UI Cleanup Plan - Final Version

**Date**: 2025-11-16
**Objective**: Remove 4 unnecessary screens + pricing tables

---

## Part 1: Screen Removal

### Screens to REMOVE (4 total)

#### 1. ❌ Pricing Screen
- **File**: `training/cli/pricing/screen.py`
- **Directory**: `training/cli/pricing/`
- **Reason**: Hardcoded pricing info becomes stale

#### 2. ❌ Reduce Screen
- **File**: `training/cli/reduce/screen.py`
- **Directory**: `training/cli/reduce/`
- **Reason**: Unnecessary UI clutter

#### 3. ❌ Truffles Screen
- **File**: `training/cli/truffles/screen.py`
- **Directory**: `training/cli/truffles/`
- **Reason**: Unnecessary UI clutter

#### 4. ❌ GPU Screen
- **File**: `training/cli/gpu/screen.py`
- **Directory**: `training/cli/gpu/`
- **Reason**: Unnecessary UI clutter

---

### Screens to KEEP (5 + home)

#### ✅ Monitor Screen
- **File**: `training/cli/monitor/screen.py`
- **Keep**: Main monitoring interface

#### ✅ Launch Screen
- **File**: `training/cli/launch/screen.py`
- **Keep**: Launch training jobs

#### ✅ Setup Screen
- **File**: `training/cli/setup/screen.py`
- **Keep**: Setup infrastructure

#### ✅ Teardown Screen ⚠️ CRITICAL - DO NOT REMOVE!
- **File**: `training/cli/teardown/screen.py`
- **Keep**: Teardown infrastructure (user confirmed to KEEP)

#### ✅ Infra Screen
- **File**: `training/cli/infra/screen.py`
- **Keep**: Infrastructure status

#### ✅ Home Screen (will be modified)
- **File**: `training/cli/home/screen.py`
- **Modify**: Remove 4 deleted screen buttons/menu items

---

## Part 2: Pricing Table Removal

### Remove Hardcoded Pricing

#### Location 1: Setup Core
- **File**: `training/cli/setup/core.py`
- **Lines**: ~1586-1595
- **Remove**: GPU pricing table with exact dollar amounts
- **Keep**: Generic "60-91% savings" mentions

#### Location 2: GPU Quota Instructions (unused file)
- **File**: `training/cli/shared/gpu_quota_instruct.py`
- **Lines**: ~152-161
- **Remove**: GPU pricing table
- **Note**: This entire file is currently unused (never imported)

---

## Implementation Steps

### Step 1: Examine Screens Before Deletion
Check what each screen does and what's in its directory:

```bash
# Check screen purposes
head -20 training/cli/pricing/screen.py
head -20 training/cli/reduce/screen.py
head -20 training/cli/truffles/screen.py
head -20 training/cli/gpu/screen.py

# Check directory contents
ls -la training/cli/pricing/
ls -la training/cli/reduce/
ls -la training/cli/truffles/
ls -la training/cli/gpu/
```

### Step 2: Find Home Screen References
```bash
grep -n "pricing\|reduce\|truffles\|gpu" training/cli/home/screen.py
```

### Step 3: Find Route Registrations
```bash
grep -rn "pricing\|reduce\|truffles\|gpu.*screen" training/cli.py training/tui.py
```

### Step 4: Delete Screen Directories
```bash
rm -rf training/cli/pricing/
rm -rf training/cli/reduce/
rm -rf training/cli/truffles/
rm -rf training/cli/gpu/
```

### Step 5: Update Home Screen
**File**: `training/cli/home/screen.py`

Remove:
- Menu key bindings for pricing, reduce, truffles, gpu
- Button widgets for these screens
- Button click handlers
- Navigation action methods (action_pricing, action_reduce, etc.)

### Step 6: Update App Routes
**Files**: `training/cli.py` and/or `training/tui.py`

Remove:
- Screen imports for deleted screens
- Route registrations (install_screen calls)

### Step 7: Update Tests
**File**: `training/cli/unit/test_all.py`

Remove:
- Import tests for deleted screens
- Screen isinstance checks

### Step 8: Remove Pricing Tables
**File**: `training/cli/setup/core.py` (line ~1586)

Before:
```python
if use_preemptible:
    status("[bold]💰 COST SAVINGS WITH PREEMPTIBLE:[/bold]")
    status("")
    status("  [green]Regular GPU pricing:[/green]")
    status("  • H200: ~$4.50/hr   →  Preemptible: ~$1.50/hr (67% savings!)")
    status("  • H100: ~$5.00/hr   →  Preemptible: ~$2.00/hr (60% savings!)")
    status("  • A100: ~$3.67/hr   →  Preemptible: ~$1.57/hr (57% savings!)")
    status("  • L4:   ~$0.60/hr   →  Preemptible: ~$0.22/hr (63% savings!)")
    status("  • T4:   ~$0.35/hr   →  Preemptible: ~$0.14/hr (60% savings!)")
    status("")
```

After:
```python
# Pricing table removed (2025-11-16)
# Specific dollar amounts become stale - users should check GCP Console
```

**File**: `training/cli/shared/gpu_quota_instruct.py` (line ~152)

Same removal pattern (though this file is unused)

### Step 9: Revert Partial Edits
First, check and revert any partial edits made earlier:

```bash
git diff training/cli/setup/core.py
git checkout training/cli/setup/core.py  # If needed
```

---

## Files Summary

### Files to DELETE (4 directories):
1. `training/cli/pricing/` (entire directory)
2. `training/cli/reduce/` (entire directory)
3. `training/cli/truffles/` (entire directory)
4. `training/cli/gpu/` (entire directory)

### Files to MODIFY (~5 files):
1. `training/cli/home/screen.py` - Remove 4 screen menu items
2. `training/cli.py` or `training/tui.py` - Remove 4 routes
3. `training/cli/unit/test_all.py` - Remove 4 screen tests
4. `training/cli/setup/core.py` - Remove pricing table
5. `training/cli/shared/gpu_quota_instruct.py` - Remove pricing table (cleanup)

---

## Expected Results

### Before - Home Menu:
```
ARR-COC Training Interface

┌─────────────────────────────┐
│ 1. Monitor                  │
│ 2. Launch                   │
│ 3. Setup                    │
│ 4. Teardown                 │
│ 5. Infra                    │
│ X. Pricing      ← REMOVE    │
│ X. Reduce       ← REMOVE    │
│ X. Truffles     ← REMOVE    │
│ X. GPU          ← REMOVE    │
└─────────────────────────────┘
```

### After - Home Menu:
```
ARR-COC Training Interface

┌─────────────────────────────┐
│ 1. Monitor                  │
│ 2. Launch                   │
│ 3. Setup                    │
│ 4. Teardown                 │
│ 5. Infra                    │
└─────────────────────────────┘
```

**Clean, focused 5-option menu!**

---

## Testing Plan

### Test 1: TUI Menu
```bash
python training/tui.py
# Expected: 5 options only (Monitor, Launch, Setup, Teardown, Infra)
# No pricing, reduce, truffles, or gpu options
```

### Test 2: CLI Commands
```bash
python training/cli.py --help
# Expected: 5 subcommands only
# No pricing, reduce, truffles, or gpu commands
```

### Test 3: Setup No Pricing
```bash
python training/cli.py setup
# Expected: Setup runs without showing pricing table
```

### Test 4: Imports Still Work
```bash
python -c "from training.cli.monitor.screen import MonitorScreen; print('OK')"
python -c "from training.cli.teardown.screen import TeardownScreen; print('OK')"
# Expected: Both succeed
```

### Test 5: Deleted Screens Gone
```bash
python -c "from training.cli.pricing.screen import PricingScreen"
# Expected: ImportError (screen deleted)
```

---

## Rollback Plan

If needed, restore from git:

```bash
# Restore all deleted directories
git checkout HEAD -- training/cli/pricing/
git checkout HEAD -- training/cli/reduce/
git checkout HEAD -- training/cli/truffles/
git checkout HEAD -- training/cli/gpu/

# Restore modified files
git checkout HEAD -- training/cli/home/screen.py
git checkout HEAD -- training/cli.py
git checkout HEAD -- training/tui.py
git checkout HEAD -- training/cli/unit/test_all.py
git checkout HEAD -- training/cli/setup/core.py
```

---

## Commit Strategy

### Commit 1: Remove screens
```bash
git add -A
git commit -m "Remove 4 unused screens (pricing, reduce, truffles, gpu)

Removed screens:
- Pricing: Hardcoded pricing becomes stale
- Reduce: Unnecessary UI clutter
- Truffles: Unnecessary UI clutter  
- GPU: Unnecessary UI clutter

Changes:
- Deleted 4 screen directories
- Updated home screen (removed 4 menu items)
- Updated app routes (removed 4 registrations)
- Updated tests (removed 4 screen tests)

Kept screens: Monitor, Launch, Setup, Teardown, Infra

🎯 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Commit 2: Remove pricing tables
```bash
git add training/cli/setup/core.py training/cli/shared/gpu_quota_instruct.py
git commit -m "Remove hardcoded GPU pricing tables

Removed exact dollar amounts from:
- setup/core.py (GPU pricing table)
- gpu_quota_instruct.py (unused file cleanup)

Kept generic savings mentions (60-91%)

Reason: Prices change frequently, better to check GCP Console

🎯 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Final Checklist

Before executing:
- [ ] Revert any partial edits (git checkout if needed)
- [ ] Examine each screen to confirm safe to delete
- [ ] Check directory contents for any shared code
- [ ] Verify no other screens depend on deleted screens

After executing:
- [ ] Test TUI menu (5 options only)
- [ ] Test CLI commands (5 subcommands only)
- [ ] Test setup (no pricing table)
- [ ] Run unit tests
- [ ] Commit changes with descriptive messages

---

**Ready to execute when approved!**

**Total impact:**
- 4 screens removed
- 2 pricing tables removed
- ~9 files modified/deleted
- Final menu: 5 clean options


---

## CRITICAL: Dependency Check Before Deletion

**Before deleting**, we must verify these screens don't have shared code used elsewhere!

### Step 0: Check for External Dependencies

```bash
# Check if pricing/ has any imports used elsewhere
grep -rn "from.*pricing\|import.*pricing" training/ --include="*.py" | grep -v "training/cli/pricing"

# Check if reduce/ has any imports used elsewhere
grep -rn "from.*reduce\|import.*reduce" training/ --include="*.py" | grep -v "training/cli/reduce"

# Check if truffles/ has any imports used elsewhere
grep -rn "from.*truffles\|import.*truffles" training/ --include="*.py" | grep -v "training/cli/truffles"

# Check if gpu/ has any imports used elsewhere
grep -rn "from.*gpu\|import.*gpu" training/ --include="*.py" | grep -v "training/cli/gpu"
```

### What to Look For:

✅ **Safe to delete** if:
- No imports from other files
- Only imported by home/screen.py (for menu)
- Only imported by tests (we'll delete those too)
- Only imported by route registration (we'll delete those too)

⚠️ **NOT safe to delete** if:
- Core logic imported by launch/setup/monitor screens
- Shared utilities used by other screens
- Helper functions used elsewhere

### If Shared Code Found:

**Option 1**: Extract shared code to `training/cli/shared/`
**Option 2**: Copy shared code to the screen that needs it
**Option 3**: Keep the directory but remove only the screen.py

---

## Updated Implementation - Add Dependency Check First

### NEW Step 0: Verify Safe to Delete

Run dependency checks above, examine results, determine:
1. Which directories are completely self-contained (safe to delete entirely)
2. Which have shared code (need to extract or keep)

Then proceed with deletion only for safe directories.


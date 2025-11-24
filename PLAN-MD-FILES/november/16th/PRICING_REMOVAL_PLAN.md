# Pricing Information Removal Plan

**Date**: 2025-11-16
**Reason**: Pricing info becomes stale quickly, better to link to official GCP pricing

---

## Problem

We have hardcoded GPU pricing information scattered throughout the codebase:
- Prices change over time
- Different regions have different prices
- Misleading users with outdated info
- Maintenance burden to keep updated

---

## Current Pricing References

### 1. **training/cli/setup/core.py** (Line 1587-1594) ⚠️ ALREADY STARTED EDITING
```python
status("[bold]💰 COST SAVINGS WITH PREEMPTIBLE:[/bold]")
status("  • H200: ~$4.50/hr   →  Preemptible: ~$1.50/hr (67% savings!)")
status("  • H100: ~$5.00/hr   →  Preemptible: ~$2.00/hr (60% savings!)")
status("  • A100: ~$3.67/hr   →  Preemptible: ~$1.57/hr (57% savings!)")
status("  • L4:   ~$0.60/hr   →  Preemptible: ~$0.22/hr (63% savings!)")
status("  • T4:   ~$0.35/hr   →  Preemptible: ~$0.14/hr (60% savings!)")
```
**Status**: PARTIALLY REMOVED (need to revert edit first!)

### 2. **training/cli/shared/gpu_quota_instruct.py** (Line 153-160)
```python
status("[bold]💰 COST SAVINGS WITH PREEMPTIBLE:[/bold]")
status("  • H200: ~$4.50/hr   →  Preemptible: ~$1.50/hr (67% savings!)")
# ... same table
```
**Status**: UNUSED FILE (never imported), but should clean up anyway

### 3. **Generic savings percentages** (keep these!)
- Line 430: `(60-91% savings!)` ✅ KEEP (generic range, not specific prices)
- Line 1517: `(60-91% cost savings!)` ✅ KEEP
- Line 1525: `(recommended for cost savings!)` ✅ KEEP

### 4. **MECHA-related savings** (keep these!)
- `mecha_phrases.py`: "COST-SLASH! Super effective savings!" ✅ KEEP (flavor text)
- `setup_helper.py`: "60-91% savings!" ✅ KEEP (generic range)

---

## What to Remove

### Remove Specific Pricing Tables:
❌ Exact dollar amounts per hour (`~$4.50/hr`)
❌ Exact percentage savings for specific GPUs (`67% savings!`)
❌ "💰 COST SAVINGS WITH PREEMPTIBLE" section headers

### Keep General References:
✅ Generic ranges like "60-91% savings"
✅ Flavor text like "cost savings" without specific amounts
✅ MECHA battle phrases about savings
✅ References to preemptible being cheaper (without exact numbers)

---

## Implementation Plan

### Step 1: Revert Partial Edit ⚠️
```bash
git diff training/cli/setup/core.py  # Check current state
git checkout training/cli/setup/core.py  # Revert if needed
```

### Step 2: Remove Pricing Tables (2 locations)

**File 1: training/cli/setup/core.py (Line 1586-1595)**
```python
# BEFORE:
if use_preemptible:
    status("[bold]💰 COST SAVINGS WITH PREEMPTIBLE:[/bold]")
    status("")
    status("  [green]Regular GPU pricing:[/green]")
    status("  • H200: ~$4.50/hr   →  Preemptible: ~$1.50/hr (67% savings!)")
    # ... 4 more lines
    status("")

# AFTER:
# Pricing table removed (2025-11-16)
# Specific prices become stale - users should check GCP pricing page
```

**File 2: training/cli/shared/gpu_quota_instruct.py (Line 152-161)**
```python
# BEFORE:
if use_preemptible:
    status("[bold]💰 COST SAVINGS WITH PREEMPTIBLE:[/bold]")
    # ... pricing table
    
# AFTER:
# Pricing table removed (2025-11-16)
# Note: This entire file is currently unused but cleaned for consistency
```

### Step 3: Verify No Other Specific Prices
```bash
grep -rn "\$.*hr\|%.*savings!" --include="*.py" training/cli/ training/tui/
```

### Step 4: Keep Generic Savings References
No changes needed - these are fine:
- "60-91% savings" (generic range)
- "cost savings" (general statement)
- MECHA flavor text

---

## Testing

After removal:

```bash
# 1. Check setup still works
python training/cli.py setup

# 2. Look for GPU quota section (should not show pricing)
# Expected: Instructions without dollar amounts

# 3. Check TUI
python training/tui.py setup
# Expected: Same - no pricing tables
```

---

## Files to Modify

1. ✅ `training/cli/setup/core.py` (remove pricing table)
2. ✅ `training/cli/shared/gpu_quota_instruct.py` (remove pricing table, even though unused)

**Total**: 2 files, ~20 lines removed

---

## Rationale

**Why remove?**
- Prices change frequently (especially spot prices)
- Different regions have different pricing
- Users should check official GCP pricing page
- Reduces maintenance burden

**Why keep generic references?**
- "60-91% savings" is a valid general range
- "cost savings" doesn't make specific claims
- MECHA flavor text is fun, not factual pricing

**Link to official pricing instead:**
- Could add: "Check current pricing: https://cloud.google.com/vertex-ai/pricing"
- But even that link could change - better to say "Check GCP Console"

---

## Rollback Plan

If needed:
```bash
git checkout HEAD -- training/cli/setup/core.py training/cli/shared/gpu_quota_instruct.py
```

---

**Next Steps:**
1. User reviews this plan
2. Revert any partial edits
3. Execute clean removal
4. Commit with clear message
5. Test setup command


---

## ADDITION: Remove Teardown Screen Entirely

**User Request**: Remove teardown screen from both TUI and CLI

### Current Teardown Implementation

**TUI Screen**: `training/cli/teardown/screen.py`
- Full TUI screen with loading overlay
- Dry-run mode
- Teardown execution
- Listed in home screen menu (option 4)

**CLI Command**: `python training/cli.py teardown`
- Direct CLI access to teardown

**Home Screen Integration**: `training/cli/home/screen.py`
- Button "Teardown (4)"
- Navigation: `action_teardown()`
- Menu entry: Line 149, 180, 270, 319

### Removal Plan

#### Step 1: Remove from Home Screen Menu
**File**: `training/cli/home/screen.py`

Remove:
- Line 149: Menu key binding `("4", "teardown", "Teardown")`
- Line 180: Button `Button("Teardown (4)", id="teardown-btn", ...)`
- Line 270-271: Button handler `elif event.button.id == "teardown-btn"`
- Line 319-321: Action `def action_teardown()`

#### Step 2: Remove Teardown Screen
**File**: `training/cli/teardown/screen.py`
- Delete entire file (or comment out and keep for reference)

#### Step 3: Remove from App Routes
**Check**: `training/cli.py` or `training/tui.py`
- Remove route registration for teardown screen
- Remove imports

#### Step 4: Remove CLI Command
**File**: `training/cli.py`
- Remove teardown subcommand if exists
- Keep core teardown logic in `training/cli/teardown/core.py` (for manual/testing use)

#### Step 5: Update Tests
**File**: `training/cli/unit/test_all.py`
- Line 234: Remove teardown screen import test
- Line 242: Remove teardown isinstance test

### What to Keep

✅ **Core teardown logic**: `training/cli/teardown/core.py`
- Keep for manual testing
- Developers can still import and use if needed
- Just remove UI access

✅ **Setup can still tear down**: Setup screen might have teardown button
- Check if setup screen has teardown functionality
- That's fine to keep (setup needs teardown for clean reinstall)

### Files to Modify

1. `training/cli/home/screen.py` - Remove menu items and navigation
2. `training/cli/teardown/screen.py` - Delete or disable
3. `training/cli.py` or `training/tui.py` - Remove route registration
4. `training/cli/unit/test_all.py` - Remove tests
5. Check: Any other imports of TeardownScreen

### Testing

After removal:
```bash
# TUI should not show teardown option
python training/tui.py
# Expected: Only 3 options (Monitor, Launch, Setup, Infra)

# CLI should not have teardown command
python training/cli.py teardown
# Expected: Error - command not found (or we remove the command)

# Core teardown logic still works
python -c "from training.cli.teardown.core import teardown_infrastructure; print('OK')"
# Expected: Imports successfully (keeps core logic)
```

### Rationale

**Why remove teardown screen?**
- Dangerous operation (deletes all infrastructure)
- Should require more deliberate action (not just a menu option)
- Setup screen probably has teardown for clean reinstall
- Reduces UI clutter

**Keep core logic because:**
- Developers need it for testing
- Can be called programmatically
- Setup might use it for clean reinstall

---

## Updated Total Changes

**Pricing Removal**: 2 files
**Teardown Removal**: ~5 files

**Combined Total**: ~7 files modified/deleted


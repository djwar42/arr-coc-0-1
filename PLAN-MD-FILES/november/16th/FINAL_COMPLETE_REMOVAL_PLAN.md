# FINAL Complete Screen Removal Plan - With Dependency Analysis

**Date**: 2025-11-16
**Status**: Ready to execute

---

## Executive Summary

Removing 4 screens from TUI/CLI:
- ❌ Pricing (3 files, 1,097 LOC)
- ❌ Reduce (3 files, 644 LOC)
- ❌ Truffles (3 files, 929 LOC)
- ❌ GPU (6 files, 1,022 LOC + library files only used internally)

**Total deletion**: 15 files, ~3,692 LOC

---

## Part 1: Directories to DELETE (Complete Removal)

### 1. ❌ training/cli/pricing/ (3 files)
```
pricing/
├── __init__.py
├── core.py (803 LOC)
└── screen.py (294 LOC)
```
**External usage**: ONLY cli.py line 668 + tui.py line 122, 399
**Safe to delete**: YES (after removing references)

### 2. ❌ training/cli/reduce/ (3 files)
```
reduce/
├── __init__.py
├── core.py (355 LOC)
└── screen.py (289 LOC)
```
**External usage**: ONLY cli.py lines 700, 846 + tui.py lines 123, 400
**Safe to delete**: YES (after removing references)

### 3. ❌ training/cli/truffles/ (3 files)
```
truffles/
├── __init__.py
├── core.py (453 LOC)
└── screen.py (476 LOC)
```
**External usage**: ONLY cli.py lines 919-923 + tui.py lines 124, 401
**Safe to delete**: YES (after removing references)

### 4. ❌ training/cli/gpu/ (6 files)
```
gpu/
├── __init__.py
├── core.py (699 LOC)
├── screen.py (323 LOC)
├── gcp_api_fetcher.py       ← Library file (ONLY used by gpu/core.py)
├── gcp_machine_configs.py   ← Library file (ONLY used by gpu/core.py)
└── karpathy_insights.py     ← Library file (ONLY used by gpu/core.py)
```
**External usage**: ONLY cli.py line 904 + tui.py lines 125, 402
**Library files**: NO external usage (only imported by gpu/core.py itself)
**Safe to delete**: YES (entire directory, after removing references)

---

## Part 2: Files to MODIFY

### File 1: training/cli.py

**Remove these imports** (lines ~15-20):
```python
# DELETE - imports for removed screens
from training.cli.pricing.core import VALID_REGIONS, run_pricing_core
from training.cli.reduce.core import run_reduce_core, list_available_gpus
from training.cli.truffles.core import format_truffle_summary_table, run_truffle_sweep_core, run_truffles_core
from training.cli.gpu.core import run_gpu_info_core
```

**Remove these functions**:
- Lines 666-695: `show_pricing()` function
- Lines 698-716: `reduce_gpu_cost()` function  
- Lines 902-911: `show_gpu_info()` function
- Lines 914-983: `show_truffles()` function

**Remove from argparse** (lines ~730-760):
```python
# DELETE from choices list:
choices=["monitor", "launch", "setup", "teardown", "infra", 
         "pricing",   # ← DELETE
         "reduce",    # ← DELETE  
         "truffles",  # ← DELETE
         "gpu"        # ← DELETE
]
```

**Remove command handlers** (lines ~841-899):
```python
# DELETE entire elif blocks:
elif args.command == "pricing":
    show_pricing()
elif args.command == "reduce":
    reduce_gpu_cost(args)
elif args.command == "truffles":
    show_truffles(sweep=args.sweep)
elif args.command == "gpu":
    show_gpu_info(args.gpu_type)
```

**Remove from help text** (lines ~730-734):
Remove pricing/reduce/truffles/gpu descriptions

---

### File 2: training/tui.py

**Remove imports** (lines 122-125):
```python
# DELETE these 4 imports:
from training.cli.pricing.screen import PricingScreen
from training.cli.reduce.screen import ReduceScreen
from training.cli.truffles.screen import TrufflesScreen
from training.cli.gpu.screen import GPUScreen
```

**Remove keyboard bindings** (lines ~368-369):
```python
# DELETE from BINDINGS:
("6", "push_screen('pricing')", "Pricing"),   # ← DELETE
("7", "push_screen('reduce')", "Reduce"),     # ← DELETE
```

**Remove screen registrations** (lines 399-402):
```python
# DELETE from on_mount():
self.install_screen(PricingScreen(...), name="pricing")
self.install_screen(ReduceScreen(...), name="reduce")
self.install_screen(TrufflesScreen(...), name="truffles")
self.install_screen(GPUScreen(...), name="gpu")
```

**Remove from validation** (line ~419):
```python
# DELETE "pricing", "reduce", "truffles" from start_screen validation
```

---

### File 3: training/cli/unit/test_all.py

Search and remove any tests for:
- PricingScreen
- ReduceScreen
- TrufflesScreen
- GPUScreen

---

### File 4: training/cli/setup/core.py (Pricing table)

**Line ~1586-1595**, remove:
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

Replace with:
```python
# Pricing table removed (2025-11-16) - specific prices become stale
```

---

### File 5: training/cli/shared/gpu_quota_instruct.py (Pricing table)

**Line ~152-161**, same removal as setup/core.py (this file is unused anyway)

---

## Part 3: Implementation Order

### Step 0: Revert Any Partial Edits
```bash
git status
git diff training/cli/setup/core.py
git checkout training/cli/setup/core.py  # If partially edited
```

### Step 1: Modify training/cli.py
- Remove imports
- Remove functions (show_pricing, reduce_gpu_cost, show_gpu_info, show_truffles)
- Remove argparse choices
- Remove command handlers
- Remove help text

### Step 2: Modify training/tui.py
- Remove imports
- Remove keyboard bindings  
- Remove screen registrations
- Remove validation entries

### Step 3: Delete Directories
```bash
rm -rf training/cli/pricing/
rm -rf training/cli/reduce/
rm -rf training/cli/truffles/
rm -rf training/cli/gpu/
```

### Step 4: Remove Pricing Tables
- Edit training/cli/setup/core.py
- Edit training/cli/shared/gpu_quota_instruct.py

### Step 5: Update Tests
- Remove screen tests from training/cli/unit/test_all.py

---

## Part 4: Complete File List

### Files to DELETE (4 directories, 15 files total):

**Pricing** (3 files):
- training/cli/pricing/__init__.py
- training/cli/pricing/core.py (803 LOC)
- training/cli/pricing/screen.py (294 LOC)

**Reduce** (3 files):
- training/cli/reduce/__init__.py
- training/cli/reduce/core.py (355 LOC)
- training/cli/reduce/screen.py (289 LOC)

**Truffles** (3 files):
- training/cli/truffles/__init__.py
- training/cli/truffles/core.py (453 LOC)
- training/cli/truffles/screen.py (476 LOC)

**GPU** (6 files):
- training/cli/gpu/__init__.py
- training/cli/gpu/core.py (699 LOC)
- training/cli/gpu/screen.py (323 LOC)
- training/cli/gpu/gcp_api_fetcher.py ✅ (no external usage)
- training/cli/gpu/gcp_machine_configs.py ✅ (no external usage)
- training/cli/gpu/karpathy_insights.py ✅ (no external usage)

### Files to MODIFY (5 files):
1. training/cli.py (~200 lines removed)
2. training/tui.py (~10 lines removed)
3. training/cli/unit/test_all.py (test removals)
4. training/cli/setup/core.py (pricing table)
5. training/cli/shared/gpu_quota_instruct.py (pricing table)

---

## Part 5: Testing

### Test 1: Imports Clean
```bash
python -c "import training.cli; import training.tui; print('OK')"
# Expected: No ImportError
```

### Test 2: TUI Starts
```bash
python training/tui.py
# Expected: App starts, shows 5 screens only (Monitor, Launch, Setup, Teardown, Infra)
```

### Test 3: CLI Help
```bash
python training/cli.py --help
# Expected: Shows 5 subcommands only
```

### Test 4: Deleted Screens Gone
```bash
python -c "from training.cli.pricing.screen import PricingScreen"
# Expected: ModuleNotFoundError
```

---

## Part 6: Git Commits

### Commit 1: Remove screen references from cli/tui
```bash
git add training/cli.py training/tui.py training/cli/unit/test_all.py
git commit -m "Remove references to 4 deleted screens (pricing, reduce, truffles, gpu)

Updated training/cli.py:
- Removed imports for 4 screen modules
- Removed functions: show_pricing, reduce_gpu_cost, show_gpu_info, show_truffles
- Removed argparse choices and command handlers
- Removed help text entries

Updated training/tui.py:
- Removed screen imports
- Removed keyboard bindings (6, 7)
- Removed screen registrations
- Removed validation entries

Updated tests:
- Removed screen import tests

Prep for screen directory deletion.

🎯 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Commit 2: Delete screen directories
```bash
git add -A
git commit -m "Delete 4 unused screen directories (pricing, reduce, truffles, gpu)

Deleted directories (15 files, 3,692 LOC):
- training/cli/pricing/ (3 files)
- training/cli/reduce/ (3 files)
- training/cli/truffles/ (3 files)
- training/cli/gpu/ (6 files, including library files only used internally)

Library files deleted (no external usage):
- gcp_api_fetcher.py
- gcp_machine_configs.py
- karpathy_insights.py

Kept screens: Monitor, Launch, Setup, Teardown, Infra

🎯 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Commit 3: Remove pricing tables
```bash
git add training/cli/setup/core.py training/cli/shared/gpu_quota_instruct.py
git commit -m "Remove hardcoded GPU pricing tables

Removed exact dollar amounts from:
- setup/core.py (GPU pricing table)
- gpu_quota_instruct.py (unused file cleanup)

Kept generic savings mentions (60-91%)

Reason: Prices change frequently

🎯 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Summary

**Screens removed**: 4 (Pricing, Reduce, Truffles, GPU)
**Screens kept**: 5 (Monitor, Launch, Setup, Teardown, Infra)
**Files deleted**: 15 files, ~3,692 LOC
**Files modified**: 5 files
**Library files removed**: 3 (all inside gpu/, no external usage)

**Ready to execute!**


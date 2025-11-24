# ARR-COC-0-1 Major Reorganization Plan

**Date**: 2025-11-20
**Status**: PENDING EXECUTION
**Codebase**: /Users/alfrednorth/Desktop/Code/arr-coc-0-1

---

## Current Structure Analysis

```
arr-coc-0-1/                      CURRENT STATE
├── app.py                        🌐 HF Gradio app (root - needs move)
├── art.py                        🌐 3D spinner art (root - needs move)
├── microscope/                   🌐 Dev visualization tabs (root - needs move)
├── assets/                       🌐 Prompts/images (root - needs move)
│
├── training/                     ⚠️ MIXED - needs separation
│   ├── cli.py                   Entry point (needs move to CLI/)
│   ├── tui.py                   Entry point (needs move to CLI/)
│   ├── train.py                 Training script (stays → Training/)
│   ├── cli/                     TUI screens (needs move to CLI/)
│   ├── images/                  Docker images (needs move to Stack/)
│   ├── shared/                  Utilities (stays → Training/)
│   ├── wandb/                   W&B data (stays → Training/)
│   ├── logs/                    Debug logs (stays → Training/)
│   └── archive/                 Old scripts (stays → Training/)
│
├── wandb/                        🔴 DUPLICATE at root (delete after move)
├── arr_coc/                      ✅ Core modules (stays)
├── tests/                        ✅ Test suite (stays)
├── setup.py                      ❌ Remove (not needed)
└── ...
```

---

## Target Structure

```
arr-coc-0-1/                      TARGET STATE
├── HFApp/                        🌐 HUGGINGFACE DEMO
│   ├── __init__.py              Package marker
│   ├── app.py                   Main Gradio interface
│   ├── art.py                   3D spinner + ASCII art
│   ├── assets/prompts/          Example prompts
│   └── microscope/              Dev visualization tabs
│       ├── __init__.py
│       ├── 0-homunculus/
│       ├── 1-heatmaps/
│       ├── 2-textures/
│       ├── 3-three-ways/
│       ├── 4-comparison/
│       └── 5-metrics/
│
├── CLI/                          🖥️ TUI INTERFACE (Vertex AI)
│   ├── __init__.py
│   ├── cli.py                   CLI entry point
│   ├── tui.py                   TUI entry point
│   ├── constants.py             Configuration
│   ├── home/                    Home screen
│   ├── setup/                   Setup infrastructure
│   ├── launch/                  Launch training jobs
│   ├── monitor/                 Monitor runs
│   ├── teardown/                Teardown infrastructure
│   ├── infra/                   Infrastructure status
│   ├── shared/                  Shared TUI utilities
│   └── unit/                    Unit tests
│
├── Stack/                        🐳 DOCKER IMAGES (Cloud Build)
│   ├── README.md
│   ├── arr-pytorch-base/        PyTorch compiled from source
│   ├── arr-ml-stack/            ML dependencies
│   ├── arr-trainer/             Training container
│   └── arr-vertex-launcher/     W&B Launch agent
│
├── Training/                     🚂 TRAINING LOGIC
│   ├── train.py                 Training script
│   ├── shared/                  Shared utilities
│   ├── wandb/                   W&B data (gitignored)
│   ├── logs/                    Debug logs
│   └── archive/                 Old scripts
│
├── arr_coc/                      🧠 CORE MODULES (unchanged)
│   ├── __init__.py
│   ├── knowing.py               Three ways of knowing
│   ├── attending.py             Token allocation
│   ├── balancing.py             Opponent processing
│   ├── texture.py               13-channel textures
│   └── integration.py           ARRCOCQwen model
│
├── tests/                        🧪 TEST SUITE (unchanged)
├── PLATONIC-DIALOGUES/           📜 DIALOGUES (unchanged)
├── PLAN-MD-FILES/                📋 PLANS (unchanged)
├── CLAUDE.md                     🤖 CLAUDE INSTRUCTIONS (update paths)
└── README.md                     📖 Update app_file path
```

---

## Execution Steps

### Phase 1: Create New Directories

```bash
# Create top-level directories
mkdir -p HFApp
mkdir -p CLI
mkdir -p Stack
mkdir -p Training
```

### Phase 2: Move HFApp Files (HuggingFace Demo)

**Files to move:**
- `app.py` → `HFApp/app.py`
- `art.py` → `HFApp/art.py`
- `microscope/` → `HFApp/microscope/`
- `assets/` → `HFApp/assets/`

**Commands:**
```bash
mv app.py HFApp/
mv art.py HFApp/
mv microscope HFApp/
mv assets HFApp/
touch HFApp/__init__.py
```

**Import updates in HFApp/app.py:**
```python
# OLD
from art import create_header
from microscope import (...)

# NEW
from HFApp.art import create_header
from HFApp.microscope import (...)
```

### Phase 3: Move CLI Files (TUI Interface)

**Files to move:**
- `training/cli.py` → `CLI/cli.py`
- `training/tui.py` → `CLI/tui.py`
- `training/cli/*` → `CLI/` (all contents)

**Commands:**
```bash
# Move entry points
mv training/cli.py CLI/
mv training/tui.py CLI/

# Move CLI package contents (preserve structure)
mv training/cli/* CLI/

# Clean up empty training/cli/
rmdir training/cli
```

**Import updates required:**
- `from training.cli.X` → `from CLI.X`
- `from cli.X` → `from CLI.X`

**Files needing import updates (~19 files):**
- CLI/unit/test_all.py (many imports)
- CLI/launch/core.py (many imports)
- CLI/launch/screen.py
- CLI/infra/screen.py
- CLI/*.py (various)

### Phase 4: Move Stack Files (Docker Images)

**Files to move:**
- `training/images/*` → `Stack/`

**Commands:**
```bash
mv training/images/* Stack/
rmdir training/images
```

**No import updates needed** - these are Docker build contexts.

### Phase 5: Rename training/ → Training/

**Files to keep in Training/:**
- `train.py`
- `shared/`
- `wandb/`
- `logs/`
- `archive/`
- `.env`
- `README.md`
- `test_parallel_setup.sh`
- `test_parallel_teardown.sh`

**Commands:**
```bash
# Rename directory
mv training Training
```

**Also:**
- Move root `wandb/` to `Training/wandb/` (if different)
- Or delete root `wandb/` if duplicate

```bash
# Check if wandb/ at root is duplicate
diff -r wandb Training/wandb 2>/dev/null && rm -rf wandb
```

### Phase 6: Update All Imports

**Pattern replacements:**

| Old Import | New Import |
|------------|------------|
| `from training.cli.` | `from CLI.` |
| `from cli.` | `from CLI.` |
| `from art import` | `from HFApp.art import` |
| `from microscope import` | `from HFApp.microscope import` |

**Files to update:**

1. **HFApp/app.py** - art, microscope imports
2. **CLI/cli.py** - all training.cli → CLI
3. **CLI/tui.py** - all training.cli → CLI
4. **CLI/**/*.py** - all internal imports
5. **Any file referencing training paths**

**Sed commands:**
```bash
# Update training.cli → CLI
find CLI -name "*.py" -exec sed -i '' 's/from training\.cli\./from CLI./g' {} \;
find CLI -name "*.py" -exec sed -i '' 's/import training\.cli\./import CLI./g' {} \;

# Update cli. → CLI.
find CLI -name "*.py" -exec sed -i '' 's/from cli\./from CLI./g' {} \;

# Update HFApp imports
sed -i '' 's/from art import/from HFApp.art import/g' HFApp/app.py
sed -i '' 's/from microscope import/from HFApp.microscope import/g' HFApp/app.py
```

### Phase 7: Update Configuration Files

**README.md frontmatter:**
```yaml
# OLD
app_file: app.py

# NEW
app_file: HFApp/app.py
```

**.env (if exists at root):**
```bash
# Update WANDB_DIR if needed
WANDB_DIR=./Training/wandb
```

**CLAUDE.md:**
- Update any path references from `training/cli` → `CLI`
- Update `training/images` → `Stack`
- Update `training/` → `Training/`

### Phase 8: Cleanup

```bash
# Remove setup.py (not needed)
rm setup.py

# Remove any empty directories
find . -type d -empty -delete

# Remove duplicate wandb if exists
rm -rf wandb  # (if moved to Training/)
```

### Phase 9: Add __init__.py Files

```bash
# Ensure all packages have __init__.py
touch HFApp/__init__.py
touch CLI/__init__.py
# Stack doesn't need one (Docker contexts)
touch Training/__init__.py
```

### Phase 10: Verify & Test

```bash
# 1. Check Python syntax
python -m py_compile HFApp/app.py
python -m py_compile CLI/cli.py
python -m py_compile CLI/tui.py

# 2. Check imports work
cd /path/to/arr-coc-0-1
python -c "from HFApp.app import demo"
python -c "from CLI.cli import main"

# 3. Test HF app locally
python HFApp/app.py
```

---

## Import Change Details

### CLI/cli.py - Expected Changes

```python
# OLD
from training.cli.constants import (...)
from training.cli.shared.wandb_helper import WandBHelper

# NEW
from CLI.constants import (...)
from CLI.shared.wandb_helper import WandBHelper
```

### CLI/tui.py - Expected Changes

```python
# OLD
from training.cli.home.screen import HomeScreen
from training.cli.monitor.screen import MonitorScreen

# NEW
from CLI.home.screen import HomeScreen
from CLI.monitor.screen import MonitorScreen
```

### CLI/launch/core.py - Expected Changes

```python
# OLD
from cli.launch.validation import format_validation_report
from cli.shared.pricing import (...)
from cli.launch.mecha.mecha_acquire import lazy_load_quota_entry

# NEW
from CLI.launch.validation import format_validation_report
from CLI.shared.pricing import (...)
from CLI.launch.mecha.mecha_acquire import lazy_load_quota_entry
```

### HFApp/app.py - Expected Changes

```python
# OLD
from art import create_header
from microscope import (
    create_homunculus_figure,
    create_multi_heatmap_figure,
    ...
)

# NEW
from HFApp.art import create_header
from HFApp.microscope import (
    create_homunculus_figure,
    create_multi_heatmap_figure,
    ...
)
```

---

## Running After Reorganization

### HuggingFace Demo
```bash
# From project root
python HFApp/app.py
```

### TUI (Interactive)
```bash
# From project root
python CLI/tui.py
```

### CLI (Scriptable)
```bash
# From project root
python CLI/cli.py setup
python CLI/cli.py launch
python CLI/cli.py monitor
python CLI/cli.py teardown
```

### Training Script
```bash
# From project root
python Training/train.py
```

---

## Benefits

1. **Clear separation** - Each folder has one purpose
2. **Easy to find** - Know where to look for what
3. **Better imports** - Clean, obvious import paths (CLI.*, HFApp.*, Training.*)
4. **Scalable** - Easy to add new components
5. **Self-documenting** - Folder names describe contents
6. **HF Spaces compatible** - app_file path updated correctly

---

## Risk Mitigation

1. **Backup first**: `git stash` or commit current state
2. **Test imports**: After each phase, test that imports work
3. **Incremental commits**: Commit after each major phase
4. **Keep old structure accessible**: Don't delete until verified

---

## Estimated Time

- Phase 1-4 (moves): 10 minutes
- Phase 5-6 (imports): 30-45 minutes (many files)
- Phase 7-9 (config): 10 minutes
- Phase 10 (testing): 15 minutes

**Total: ~60-90 minutes**

---

## Post-Reorganization Checklist

- [ ] All files moved to correct locations
- [ ] All imports updated and working
- [ ] README frontmatter updated
- [ ] .env paths updated
- [ ] CLAUDE.md paths updated
- [ ] HFApp/app.py runs locally
- [ ] CLI/tui.py runs without errors
- [ ] CLI/cli.py commands work
- [ ] Git committed with descriptive message
- [ ] Ready for HF Spaces deployment test

---

## Notes

- This plan is specific to the current codebase state (2025-11-20)
- 933 Python files total (many in wandb/ - gitignored)
- ~19 files need import updates
- arr_coc/ and tests/ remain unchanged
- PLATONIC-DIALOGUES/ and PLAN-MD-FILES/ remain unchanged

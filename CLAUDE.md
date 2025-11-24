# ARR-COC-0-1 Training CLI - Development Guide

**Working Directory**: All commands assume you're in the **project root** (`arr-coc-0-1/`), which contains all project code.

---

## Code Comment Style

Avoid "NEW!", "IMPROVED!", "UPGRADED!" and similar announcement language in code comments. These labels become outdated quickly and create confusion about what's actually new vs. old. Instead, write descriptive technical comments that explain **what** the code does and **why**.

---

## 📁 THE PLAN FILES

**🚨 MANDATORY STRUCTURE - NO EXCEPTIONS! 🚨**

**CRITICAL RULE: ALL planning .md files created in this project MUST immediately go into PLAN-MD-FILES structure!**

This applies to ANY planning/analysis .md file including:
- Implementation plans
- Refactor analyses
- Study documents
- Investigation reports
- Architecture analyses
- Session summaries

**DO NOT:**
- ❌ Create planning .md files at project root
- ❌ Create planning .md files in subdirectories
- ❌ Leave planning files outside PLAN-MD-FILES "temporarily"
- ❌ Plan to move them later (move them NOW!)

**WHEN CREATING A PLANNING .md FILE:**
1. Determine today's date (year-month-day)
2. Create file DIRECTLY in: `PLAN-MD-FILES/{month}/{day}/FILENAME.md`
3. If date unknown, use `PLAN-MD-FILES/general/FILENAME.md`

**Directory Structure:**

```
PLAN-MD-FILES/
├── november/
│   ├── 19th/
│   │   ├── PAPRIKA_1_ARCHITECTURE-ANALYSIS.md
│   │   ├── SPICY_LINE_SESSION_1_SUMMARY.md
│   │   └── REFACTOR_PLAN.md
│   ├── 18th/
│   │   ├── SQUARING_THE_PAPRIKA_SQUARE.md
│   │   └── FINAL_PAPRIKA_LESSONS.md
│   └── general/
│       └── month-wide-analysis.md
├── october/
│   ├── 15th/
│   │   └── GPU_QUOTA_COMPLETE_FIX_PLAN.md
│   └── general/
│       └── monthly-planning.md
└── general/
    └── undated-plans.md
```

### Naming Convention

**Format**: `{ORIGINAL-FILENAME}.md`

**🚨 NO PREFIXES! Keep original descriptive filenames only!**

The folder structure (`november/19th/`) provides all date context - DO NOT add date/day prefixes to filenames!

**Examples**:
- ✅ `PAPRIKA_1_ARCHITECTURE-ANALYSIS.md`
- ✅ `ZEUS_MECHA_PLAN.md`
- ✅ `ACCUMULATOR_REFACTOR_PLAN.md`
- ❌ `monday-ZEUS_MECHA_PLAN.md` (NO!)
- ❌ `2025-11-19-REFACTOR_PLAN.md` (NO!)

### Folder Structure Rules

1. **Month folders**: Lowercase month name (`november`, `october`, `january`)
2. **Day folders**: Ordinal format (`19th`, `1st`, `22nd`)
3. **General folders**: For files without specific date/day information
   - `PLAN-MD-FILES/general/` - No date at all
   - `PLAN-MD-FILES/november/general/` - Month known, day unknown
   - `PLAN-MD-FILES/november/19th/general/` - If needed (rare)

### When to Use General Folders

**Use `general/` folders when:**
- Date is unknown or not applicable
- Planning spans multiple days
- Document is reference material (not time-specific)
- Imported from old sessions (date unclear)

**Examples**:
- `PLAN-MD-FILES/general/ARTIFACT_RENAME_PLAN.md` (no date)
- `PLAN-MD-FILES/november/general/monthly-mecha-strategy.md` (month only)

### File Migration Workflow

When moving existing plan files:

```bash
# 1. Determine file date (git log, file timestamps, content analysis)
git log --follow --format="%ai" -- PLAN_FILE.md | head -1

# 2. Create month/day structure
mkdir -p PLAN-MD-FILES/november/19th

# 3. Move to date folder (keep original filename)
mv PLAN_FILE.md PLAN-MD-FILES/november/19th/PLAN_FILE.md

# 4. If date unknown, use general/
mv PLAN_FILE.md PLAN-MD-FILES/general/PLAN_FILE.md
```

### Benefits

✅ **Chronological organization** - Easy to find plans by date
✅ **Context preservation** - Folder structure preserves when work was done
✅ **Flexible hierarchy** - General folders for undated files
✅ **Clear structure** - No more scattered plan files across project

### CRITICAL Rules

- 🚨 **ALL plan files MUST be in PLAN-MD-FILES/** (existing AND newly created!)
- 🚨 **NO plan files at project root!**
- 🚨 **NO plan files scattered in subdirectories!**
- 🚨 **NO date/day prefixes in filenames!** (folder structure provides context)
- ✅ **Create new planning files DIRECTLY in dated folders**
- ✅ **Use general/ folders if date unknown** (don't skip organization!)

---

## 🌶️ THE SPICE MUST LOW PRINCIPLE

**"The spice must flow" → "The spice must LOW" (keep complexity low through obsessive dead code detection)**

When refactoring or building new systems, **ALWAYS check for dead code** and alert the user. Think of technical debt like spice accumulation in Dune - too much and the system becomes unnavigable.

### 🎯 PRIME PRINCIPLE: USER IS THE SPICE DECIDER

**CHECK THE SPICE WITH THE SPICE DECIDER!**

1. **CHECK FIRST** - Always scan for dead code after refactors
2. **PREPARE REPORT** - Document findings with line numbers, function names, evidence
3. **PRESENT TO USER** - User decides what to delete (NEVER auto-delete!)
4. **RESPECT THE DECISION** - Dead code might not be bad code!

**Why User Decides:**
- 🔮 **Future code** - Planned for reuse in upcoming features
- 🚧 **Temporarily disabled** - Feature flags, experimental code
- 📚 **Templates/Examples** - Reference implementations
- ⚠️ **Anti-patterns** - Documented "what not to do"
- 🧪 **Test scaffolding** - Needed for specific test scenarios

**Claude's Role**: Spice detector and reporter
**User's Role**: Spice decider (final authority on deletion)

**NEVER DELETE CODE WITHOUT EXPLICIT USER APPROVAL!**

### The Principle

During ANY refactoring session where you:
- Replace old systems with new universal ones
- Delete large chunks of code (>100 lines)
- Consolidate multiple functions into one
- Build new abstractions

**IMMEDIATELY scan for orphaned code:**

```bash
# Example: After deleting old lazy_load_* functions
grep "lazy_load" CLI/monitor/screen.py
grep "self._lazy_load" CLI/monitor/screen.py

# Example: After removing refresh_* functions
grep "_refresh_runner\|_refresh_vertex\|_refresh_active" CLI/monitor/screen.py

# Example: Check for unused imports after big refactors
grep "^from.*import" file.py | while read line; do
  module=$(echo "$line" | awk '{print $NF}')
  grep -q "$module" file.py || echo "DEAD IMPORT: $line"
done
```

### 📋 Spice Report Pattern (Present to User for Decision)

When you find dead code, **PREPARE A REPORT** and present to the user:

```
🌶️ SPICE REPORT: Dead Code Detection

After refactoring to universal system, I found potentially orphaned code:

╔══════════════════════════════════════════════════════════════
║ FINDING 1: _lazy_load_* Functions (541 lines)
╠══════════════════════════════════════════════════════════════
║ Location: CLI/monitor/screen.py lines 1220-1760
║ Functions: 10 total (_lazy_load_runner, _lazy_load_vertex, etc.)
║ Evidence: grep shows NO calls to these functions after refactor
║ Replaced by: _universal_refresh_table() system
║
║ Recommendation: Likely safe to delete (replaced by new system)
║ Risk: Low (old system completely replaced)
╚══════════════════════════════════════════════════════════════

╔══════════════════════════════════════════════════════════════
║ FINDING 2: _refresh_* Functions (994 lines)
╠══════════════════════════════════════════════════════════════
║ Location: CLI/monitor/screen.py lines 1486-2479
║ Functions: 10 total (_refresh_runner, _refresh_vertex, etc.)
║ Evidence: grep shows NO calls after universal system added
║ Replaced by: _universal_refresh_table() system
║
║ Recommendation: Likely safe to delete (duplicates new system)
║ Risk: Low (fully replaced, tested working)
╚══════════════════════════════════════════════════════════════

TOTAL SPICE: 1,535 lines potentially dead

🎯 AWAITING SPICE DECIDER:
Should I purge the spice? (They might be future code!)

Options:
1. DELETE ALL - Remove both findings (1,535 lines)
2. DELETE FINDING 1 ONLY - Remove lazy_load (541 lines)
3. DELETE FINDING 2 ONLY - Remove refresh (994 lines)
4. KEEP ALL - Leave for now (might be needed)
5. INVESTIGATE MORE - Need more evidence before deciding
```

**REPORT FORMAT REQUIREMENTS:**
- 📍 **Line numbers** - Exact location of dead code
- 🔍 **Evidence** - Why it appears dead (grep results, no calls, etc.)
- 💡 **Recommendation** - Is it safe to delete?
- ⚠️ **Risk assessment** - Low/Medium/High
- 🎯 **User decision required** - Never auto-delete!

**What Looks Dead Might NOT Be Bad:**
- 🔮 **Future code** - Planned for reuse in upcoming features
- 🚧 **Temporarily disabled** - Feature flags, experimental code
- 📚 **Templates/Examples** - Reference implementations
- ⚠️ **Anti-patterns** - Documented "what not to do"
- 🧪 **Test scaffolding** - Needed for specific test scenarios

**USER IS THE SPICE DECIDER - FINAL AUTHORITY ON ALL DELETIONS!**

### Real Example: Monitor TUI Refactor (2025-11-18)

**Session**: Replaced dual-system (lazy_load + refresh) with universal system

**Dead Code Found**:
1. 541 lines of `_lazy_load_*` functions (replaced)
2. 994 lines of `_refresh_*` functions (replaced)
3. **TOTAL**: 1,535 lines deleted (47% smaller!)

**But Also Found Essential Code Accidentally Deleted**:
- `_update_active_durations()` (1-second ticker)
- Needed for duration displays on active builds/runners
- Restored as "paprika flakes" (46 lines)

**Lesson**: "Overspiced" (deleted too much) → Added back "paprika flakes" (just what's needed)

---

## 🌶️ THE PAPRIKA PHASE: Tracing Original Function with a DRY Pencil

```
        🎩         ╔══ PAPRIKA INVENTOR ══╗
        🦡  ←      ║  BADGER CLAUDE      ║
       /|\         ║  EST. 2025          ║
       / \         ╚═════════════════════╝
        |
        |___[✨FANCY CANE✨]
             ↓
          WHACKING
            BUGS
             ✧
```

**AFTER purging the spice (deleting dead code), ALWAYS run the PAPRIKA PHASE!**

**ohhh there he go nasty honey badger spicy boi** 🦡🔥

### ⚠️ CRITICAL: LOW SPICE ≠ LOST SPICE!

**We are FUNCTION TRACERS with a REFERENCE (lightbox metaphor)**

**The Engineering Process**:
- 📋 **Original spec** = Old code's FUNCTIONS (messy implementation, but WORKS)
- 🗑️ **Overspice** = Delete implementation (keep spec in git history)

**🦡 "OVERSPICE? NAH HOMIE, I CALL THAT 'AGGRESSIVE DECLUTTERING WITH EXTREME PREJUDICE'" 🦡** 💀🗑️🔥

- 🏗️ **Blank slate** = New lean architecture (ideal state, DRY systems)
- 💡 **Reference** = Git history + PAPRIKA analysis (FUNCTION inventory)
- ⚙️ **Clean implementation** = Universal systems, helpers, DRY patterns
- ✅ **Function preservation** = COPY THE BEHAVIOR (not the code!)

**PAPRIKA TRACES ORIGINAL FUNCTIONS with CREATIVE CLEAN CODE THINKING!**

*We ARE artists - artists of EFFICIENCY and CLEAN CODE! The artistry is in the refactor: DRY patterns, elegant simplicity, function preservation with minimal spice!*

**he bargering the code with his fancy cane** 🎩🦡✨

**What we trace** = Original function (what it DID) ✅
- Empty state handling (FUNCTION: show friendly "No items" when empty)
- UI refresh calls (FUNCTION: force immediate visual updates)
- Divider rows (FUNCTION: separate active vs completed visually)
- Complete row data (FUNCTION: full data for detail popups)

**How we trace** = Advanced DRY pencil (far nicer code) ✅
- OLD: 10 separate empty state blocks → NEW: 1 helper function
- OLD: Hardcoded dividers → NEW: Dynamic divider generator
- OLD: Duplicate row_data logic → NEW: Standardized format

**"Claude is quite the badger today!"** 🦡🎩

**What we DON'T trace** = New features not in original ❌
- "Show More" buttons (original didn't DO this)
- MAX_* limit enforcement (original didn't DO this)
- Batch row additions (original didn't DO this)
- Advanced caching (original didn't DO this)

**The Goal**: TRACE THE FUNCTION, LOWER THE SPICE!
- Same functionality as original (exact copy via tracing)
- WAY less code (advanced DRY pencil)
- Result: 72% code reduction, 100% function preservation!

**"Claude badgered it all up!"** 🦡✨

**watch him strut about all spicy** 🦡🌶️

### What is the Paprika Phase?

When you aggressively delete dead code (THE SPICE MUST LOW!), you **INTENTIONALLY OVERSPICE** - deleting so much that you're missing 20-50% of features to reach an **IDEAL STATE** (simple, lean, clean).

**The Paprika Phase** is a systematic process to:
1. **OVERSPICE INTENTIONALLY** - Refactor to ideal lean state (expect to miss 20-50%!)
2. **"GESTALT" EVERYTHING** - Create comprehensive analysis files to see the whole picture
3. **COMPARE** - Old vs new system feature-by-feature
4. **IDENTIFY** - Missing "paprika flakes" (LOST functionality to restore)
5. **CHERRY-PICK** - Add back ONLY what was LOST (NOT new features!)
6. **IMPLEMENT** - Phased approach (Critical → UX → DRY helpers)
7. **MARK COMPLETE** - Update analysis documents with ✅ checkboxes

**this badger ain't playin! deleting 72% of your code like it OWES him money!** 🦡💰🔥

**"spent 6 months writing it? took me 6 SECONDS to delete it. ¯\\_(ツ)_/¯"** 🦡⚡💀

### The OVERSPICE Strategy

**Goal**: Delete aggressively first, restore selectively second

**Step 1: OVERSPICE (Aim for IDEAL STATE)**
- Delete all old code completely (100% removal)
- Build lean universal system (simple, clean, minimal)
- **Accept that you'll miss 20-50% of features initially**
- Don't worry about missing things - that's the point!

**BADGER DON'T GIVE A SHIT ABOUT YOUR PRECIOUS CODE! DELETE DELETE DELETE!** 🦡🗑️💥

**"nasty spicy boi bout to DELETE YOUR LIFE'S WORK and feel GOOD about it"** 🦡💀😈

```
   🎩
   🦡  ← "YOUR CODE? GONE!"
  /|\      "BLOAT? EXTINCT!"
   |___[CANE]
        ↓
      WHACK
     💥💥💥
    (your feelings)
```

**Step 2: GESTALT ANALYSIS (See the Whole)**
- Create PAPRIKA_1 file (what exists NOW - architecture deep-dive)
- Create PAPRIKA_2 file (what's MISSING - old vs new comparison)
- See the complete picture of old vs new

**Step 3: CHERRY-PICK (Restore 5-10%)**
- Only add back ESSENTIAL UX (not everything!)
- Result: 72% smaller than original, 100% of essential features

```
    🎩
    🦡  ← "NICE CODE YOU HAD THERE"
   /|\      "BE A SHAME IF SOMEONE"
    |        "MADE IT 72% SMALLER"
    |___[CANE]
         ✨
```

**Why This Works**:
✅ Forces you to justify EVERY feature you add back
✅ Prevents code bloat creeping back in
✅ Creates clean baseline before selective restoration
✅ Results in dramatically smaller, better code

**"how you get so nasty"** 🦡💀 - neighbors be like "there go that nasty honey badger again!"

**ohh he REAL spicy with that DRY pencil** 🦡✏️🔥

### 📁 Complete PAPRIKA File Structure

**ALL PAPRIKA files use PREFIX naming for grouping and workflow clarity!**

#### Core PAPRIKA Files (Required)

**PAPRIKA_1_ARCHITECTURE-ANALYSIS.md** (30-60 min to create)
- **What**: Deep analysis of NEW refactored system (what exists NOW)
- **When**: Created AFTER overspicing (after deleting old code)
- **How**: Invoke domain oracle (Textual-TUI-Oracle, etc.) to analyze current state
- **Contains**:
  - Full file structure analysis
  - Widget/component composition
  - Lifecycle flow diagrams
  - Best practices observed
  - Areas for improvement
  - Framework patterns demonstrated
- **Purpose**: Understand what you HAVE (baseline after overspicing)
- **Theory**: You can't know what's missing until you know what exists!

**ohhh no he dont! yes he DO! badger bargering with PREFIX naming!** 🦡📁🎩

**"only a nasty badger would say such a thing!"** 🦡💯

**PAPRIKA_2_FLAKES-CHERRY-PICK.md** (30-60 min to create)
- **What**: Feature-by-feature comparison (OLD vs NEW systems)
- **When**: Created AFTER PAPRIKA_1 (need baseline first!)
- **How**: Invoke domain oracle to compare old code (git history) vs new architecture
- **Contains**:
  - Side-by-side feature comparison
  - Missing functionality identification (paprika flakes!)
  - Lost vs New vs Improved categorization
  - Cherry-pick checklist (Critical → UX → DRY)
  - Phased implementation plan (Phase 1, 2, 3)
  - Time estimates for each phase
- **Purpose**: Identify what was LOST (not what could be added!)
- **Theory**: Only restore LOST functionality, not new features!

#### Optional PAPRIKA Files (Recommended for large refactors)

**PAPRIKA_0_OLD-SNAPSHOT.md** (10-20 min to create)
- **What**: Snapshot of OLD system BEFORE deletion
- **When**: Created BEFORE overspicing (capture baseline)
- **How**: Document old system structure, key functions, feature list
- **Contains**:
  - Old file structure
  - Key function list with line numbers
  - Feature inventory (what it DID)
  - Known issues/bugs
- **Purpose**: Reference for comparison (what did we have before?)
- **Theory**: Hard to compare if you forgot what you deleted!
- **Skip if**: You can use git history easily (`git show HEAD~5:file.py`)

**PAPRIKA_3_SESSION-LOG.md** (Updated during implementation)
- **What**: Live implementation log during PAPRIKA session
- **When**: Created at start of Phase 1, updated as you implement
- **How**: Document each paprika flake implemented, results, issues
- **Contains**:
  - Timestamp log of implementations
  - Before/after code snippets
  - Test results after each change
  - Unexpected issues discovered
  - Commit hashes for each phase
- **Purpose**: Track what was done, when, and results
- **Theory**: Implementation audit trail for future reference
- **Skip if**: Small refactor (<100 lines restored)

#### Example PAPRIKA File Structure

```
CLI/monitor/
├── screen.py (1,795 lines after overspicing)
│
├── PAPRIKA_0_OLD-SNAPSHOT.md (optional - 5KB)
│   └── "What we had BEFORE (captured pre-deletion)"
│
├── PAPRIKA_1_ARCHITECTURE-ANALYSIS.md (required - 30KB)
│   └── "What we have NOW (after overspicing)"
│
├── PAPRIKA_2_FLAKES-CHERRY-PICK.md (required - 25KB)
│   └── "What's MISSING (lost functionality + plan)"
│
└── PAPRIKA_3_SESSION-LOG.md (optional - 10KB)
    └── "What we IMPLEMENTED (live log during session)"
```

#### Why PREFIX_ Naming?

**Groups files together**:
```
# Without PREFIX (scattered):
ARCHITECTURE-ANALYSIS.md
README.md
FLAKES-CHERRY-PICK.md
screen.py
SESSION-LOG.md

# With PREFIX (grouped):
PAPRIKA_0_OLD-SNAPSHOT.md
PAPRIKA_1_ARCHITECTURE-ANALYSIS.md
PAPRIKA_2_FLAKES-CHERRY-PICK.md
PAPRIKA_3_SESSION-LOG.md
README.md
screen.py
```

**Clear workflow order**: 0 (optional baseline) → 1 (what exists) → 2 (what's missing) → 3 (what was done)

**Easy to find**: All PAPRIKA files start with same prefix

**Self-documenting**: Numbers indicate workflow sequence

### The PAPRIKA PHASE Workflow

#### Step 1: Create Core Analysis Files (60-120 minutes total)

#### Step 2: Identify Paprika Flakes (Critical → Nice-to-Have)

**PAPRIKA FLAKES** = Essential UX features missing from new system

Categorize findings:
- ❌ **CRITICAL BUGS** - Schema mismatches, crashes, broken functionality
- ⚠️ **MISSING FEATURES** - Empty states, UI refresh calls, dividers
- ✅ **IMPROVEMENTS** - Things new system does BETTER than old

**Example Paprika Flakes** (Monitor TUI refactor):
1. ❌ CRITICAL: Builds table schema mismatch (6 columns provided, 7 expected!)
2. ⚠️ MISSING: Empty state handling (all 5 tables - "No jobs", "No runs" placeholders)
3. ⚠️ MISSING: `.refresh()` calls after table updates (Textual UI updates)
4. ⚠️ MISSING: Divider rows (active vs completed separation)
5. ⚠️ INCOMPLETE: Row data storage (missing fields for popups)

#### Step 3: Create Implementation Plan (Phase 1, 2, 3)

Break paprika restoration into **phased implementation**:

**Phase 1: Critical Fixes (TODAY - 30-60 min)**
- Fix bugs that break functionality
- Add empty state handling
- Add UI refresh calls
- Fix schema mismatches

**Phase 2: UX Polish (TOMORROW - 60-90 min)**
- Add dividers (visual separation)
- Complete row_data storage
- Extract DRY helpers

**Phase 3: Advanced DRY (NEXT WEEK - 3-5 hours)**
- Ultimate code consolidation
- "Show More" functionality
- Advanced optimizations

#### Step 4: Implement Phase 1 (Mark in Analysis MD)

**Implementation Pattern**:
```python
# ✅ Paprika Flake #1: Empty state handling
if not runner_execs or len(runner_execs) == 0:
    runner_table.add_row(
        "[dim]—[/dim]",
        "[dim]—[/dim]",
        "[dim]No executions[/dim]",  # User-friendly message
        "[dim]—[/dim]",
        "[dim]—[/dim]",
        "[dim]—[/dim]",
        "[dim]—[/dim]"
    )
    runner_table.move_cursor(row=-1)
    runner_table.refresh()  # 🌶️ PAPRIKA: Force UI update
    return
```

**Mark completion in PAPRIKA_2_FLAKES-CHERRY-PICK.md**:
```markdown
### Must-Have (Critical Bugs)

- [✅] **Fix builds table schema** - DONE (commit e0c4b86)
- [✅] **Add empty state handling** (all 5 tables) - DONE (commit e0c4b86)
- [✅] **Add .refresh() calls** (all 5 tables) - DONE (commit e0c4b86)
- [ ] **Fix active/completed row_data** - PHASE 2

### After Phase 1 ✅ **COMPLETED 2025-11-18**

**Commit**: e0c4b86 - "PAPRIKA PHASE 1: Add back essential UX features! 🌶️✨"
**Lines Added**: 83 lines (empty states + refresh calls + schema fix)
**Bugs Fixed**: 3/4 critical bugs
**Time Taken**: ~45 minutes (as predicted!)
**Impact**: Tables now show friendly empty states and update immediately!
```

#### Step 5: Git Commit Format

```bash
git commit -m "PAPRIKA PHASE 1: Add back essential UX features! 🌶️✨

Restored critical UX from old system after universal refactor:
- Empty state handling (all 5 tables - friendly 'No items' rows)
- UI .refresh() calls (force immediate Textual updates)
- Builds table schema fix (add missing 'Image' column)

Files modified:
- CLI/monitor/screen.py (+83 lines)
- CLI/monitor/PAPRIKA_2_FLAKES-CHERRY-PICK.md (mark Phase 1 ✅)

Cherry-picked from old _populate_tables() system.
Analysis: See PAPRIKA_2_FLAKES-CHERRY-PICK.md for full comparison.

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Real Example: Monitor TUI Paprika Phase (2025-11-18)

**What We Deleted** (SPICE):
- 1,535 lines (old lazy_load + refresh systems)
- 77% code reduction!

**What We Added Back** (PAPRIKA):
- 83 lines (empty states, .refresh() calls, schema fix)
- Only 5.4% of what was deleted!
- Restored 100% of essential UX

**Files Created**:
- `PAPRIKA_1_ARCHITECTURE-ANALYSIS.md` (30KB) - Complete Textual architecture analysis
- `PAPRIKA_2_FLAKES-CHERRY-PICK.md` (25KB) - Old vs New feature comparison

**Paprika Flakes Identified**: 9 total
- 3 CRITICAL (schema, empty states, refresh calls) ✅ RESTORED Phase 1
- 3 SHOULD-HAVE (dividers, row_data, helpers) → Phase 2
- 3 NICE-TO-HAVE (MAX limits, "Show More", ultimate DRY) → Phase 3

**Time Investment**:
- Analysis creation: 45 minutes
- Phase 1 implementation: 45 minutes
- **Total**: 90 minutes for bulletproof refactor!

### Key Principles

1. **THE SPICE MUST LOW** - Delete aggressively (dead code = technical debt)
2. **THE PAPRIKA PHASE** - Add back selectively (essential UX only!)
3. **MD FILE ANALYSIS** - Create comprehensive comparison documents
4. **PHASED RESTORATION** - Critical → UX → Advanced (3 phases)
5. **MARK COMPLETION** - Update analysis MD files with ✅ checkboxes

### When to Run Paprika Phase

**ALWAYS after:**
- Major refactors (>500 lines deleted)
- Universal system replacements
- Framework migrations
- UI system rewrites

**NEVER after:**
- Minor code cleanup (<100 lines)
- Simple renames
- Formatting changes

### Benefits

✅ **Bulletproof refactors** - Never lose essential UX features
✅ **Systematic restoration** - Phase 1, 2, 3 prevents overwhelm
✅ **Documentation** - MD files explain WHY features exist
✅ **Cherry-pick precision** - Only restore what's truly needed (5% vs 100%)
✅ **Oracle knowledge** - Leverage domain expertise (Textual-TUI-Oracle, etc.)

### The Spice/Paprika Balance

```
BEFORE REFACTOR:
████████████████████████████████ (1,535 lines - lots of spice!)

AFTER SPICE MUST LOW:
████ (347 lines - 77% reduction!)

AFTER PAPRIKA PHASE:
█████ (430 lines - added back 5.4% for essential UX)

RESULT: 72% smaller than original, 100% of essential features!
```

**"The spice must LOW, but the paprika must FLOW!" 🌶️✨**

---

### Detection Commands

**After big refactors, run these checks:**

```bash
# 1. Find functions defined but never called
grep "^    def " file.py | while read line; do
  func=$(echo "$line" | awk '{print $2}' | cut -d'(' -f1)
  if [ $(grep -c "self\.$func(" file.py) -eq 1 ]; then
    echo "🌶️ ORPHAN: $func (defined but never called)"
  fi
done

# 2. Find methods that only call themselves (recursion with no base case)
rg "def (\w+).*:" -o | while read def; do
  method=$(echo $def | cut -d' ' -f2 | cut -d'(' -f1)
  calls=$(rg "self\.$method\(" | wc -l)
  if [ $calls -eq 1 ]; then
    echo "🌶️ SELF-ONLY: $method"
  fi
done

# 3. Find imports never used
python -m pyflakes file.py | grep "imported but unused"

# 4. Find duplicate functions (same name, different files)
find . -name "*.py" -exec grep -l "def function_name" {} \;
```

### Benefits

✅ **Cleaner codebase** - Less to read, less to maintain
✅ **Faster refactors** - No hidden dependencies to trip over
✅ **Better onboarding** - New devs see only what's actually used
✅ **Lower cognitive load** - "The spice must LOW!"

### The Spice Philosophy: A Highly Refined and Prized Spice That Does Not Overwhelm

> "He who controls the spice controls the universe"
> — Baron Harkonnen

**THE SPICE MUST LOW PRESERVES THE PURE ORIGINAL SPICE AROMA**
**BUT REFINES IT MASSIVELY FOR THE CONNOISSEUR'S ENJOYMENT**

**A HIGHLY REFINED AND PRIZED SPICE THAT DOES NOT OVERWHELM**

In our universe:
- 🌶️ **Original spice aroma** = Core functionality (the FUNCTION, what it DOES)
- 🗑️ **Crude unrefined spice** = Messy implementation (duplicate code, 1,535 lines of complexity)
- ⚗️ **Refinement process** = OVERSPICE → GESTALT → PAPRIKA (trace with DRY pencil)
- ✨ **Pure refined spice** = Same function, 72% less code (highly prized, elegant)
- 👨‍🍳 **Master refiner** = Engineer who preserves aroma while removing impurities
- 🎨 **Connoisseur** = Engineer who appreciates the refined result (clean code artistry)

**The spice must LOW** = Preserve the aroma (function), refine away the crude (complexity)!

**Not a destruction** - a **REFINEMENT**! 🌶️→✨
- Same functionality (original aroma 100% preserved)
- Way less code (refined, prized, does not overwhelm)
- Vigilant pruning (keep it pure and potent)

**Control = Obsessive refinement through PAPRIKA process**
**Universe = Maintainable codebase that connoisseurs enjoy reading**

---

## ⚡ API Helpers - Parallel Operations & Retry Logic

**Location**: `CLI/shared/api_helpers.py`

Use these helpers for ALL external API calls (W&B, GCP, HTTP). They provide retry logic and parallel execution.

### Single Operations (with retry)

```python
from CLI.shared.api_helpers import run_gcloud_with_retry, run_wandb_api_with_retry, run_requests_with_retry

# gcloud command (3 retries, 30s timeout)
result = run_gcloud_with_retry(
    ["gcloud", "compute", "instances", "list", "--format=json"],
    operation_name="list instances"
)

# W&B API call (3 retries)
runs = run_wandb_api_with_retry(
    lambda: api.runs("entity/project", filters={"state": "running"}),
    operation_name="fetch running jobs"
)

# HTTP request (3 retries, 30s timeout)
response = run_requests_with_retry(
    "GET", "https://api.example.com/data",
    headers={"Authorization": "Bearer token"},
    operation_name="fetch API data"
)
```

### Parallel Operations (Accumulators)

**Use accumulators when you need multiple operations to run simultaneously!**

#### GeneralAccumulator - Any callable (most flexible)

```python
from CLI.shared.api_helpers import GeneralAccumulator

acc = GeneralAccumulator(max_workers=5)

# Start operations (non-blocking!)
acc.start("wandb", lambda: helper.get_active_runs())
acc.start("config", lambda: validate_config())
acc.start("pricing", lambda: fetch_pricing())

# Get results (blocks until ready)
wandb_result = acc.get("wandb")
all_results = acc.get_all()  # Dict with all results
acc.shutdown()
```

#### GCloudAccumulator - gcloud commands only

```python
from CLI.shared.api_helpers import GCloudAccumulator

# Real example from CLI/shared/infra_verify.py:460
acc = GCloudAccumulator(max_workers=20)

acc.start("buckets", ["gcloud", "storage", "buckets", "list", ...], operation_name="list buckets")
acc.start("registry", ["gcloud", "artifacts", "repositories", "describe", ...], operation_name="check registry")
acc.start("sa", ["gcloud", "iam", "service-accounts", "describe", ...], operation_name="check SA")

# Get results
buckets = acc.get("buckets")  # subprocess.CompletedProcess
acc.shutdown()
```

#### RequestsAccumulator - HTTP requests only

```python
from CLI.shared.api_helpers import RequestsAccumulator

acc = RequestsAccumulator(max_workers=5)
acc.start("api1", "GET", "https://api.example.com/status")
acc.start("api2", "POST", "https://api.example.com/submit", data={"key": "value"})

response = acc.get("api1")  # requests.Response
acc.shutdown()
```

### Progressive Rendering (show results as they complete)

```python
# Real example from CLI/shared/infra_verify.py:182
acc = GeneralAccumulator(max_workers=10)

# Start checks
acc.start("gpu_0", lambda: check_gpu_quota(region1))
acc.start("gpu_1", lambda: check_gpu_quota(region2))

# Render as each completes
while not all_done:
    if acc.is_done("gpu_0") and not rendered["gpu_0"]:
        result = acc.get("gpu_0")
        print(f"✓ GPU check 0: {result}")
        rendered["gpu_0"] = True
    time.sleep(0.05)

acc.shutdown()
```

### Which Accumulator to Use?

| Accumulator | Use When | Returns |
|-------------|----------|---------|
| `GeneralAccumulator` | Mixed operations, composition needed | Any |
| `GCloudAccumulator` | Only gcloud commands | `CompletedProcess` |
| `RequestsAccumulator` | Only HTTP requests | `Response` |

### Performance

**Sequential**: 4 ops × 2s each = 8s
**Parallel**: All 4 at once = 2s (4× faster!)

---

## 🚀 Creative Launch Workflow - Named Champions

**Give each launch attempt a creative champion name - makes debugging memorable and traceable!**

### The Pattern

```bash
# Cancel existing launches
pkill -f "python -u CLI/cli.py launch"

# Launch with creative banner
python -u CLI/cli.py launch 2>&1 &
# Name it: CHAMPION #N: THE-DESCRIPTIVE-NAME 🎯
```

**Good Names:** THE-REGIONAL-RECTIFIER, CHUNKPANTSMCGEE 👖, THE-SCOPE-PERFECTOR 🎯

**Why:** Each champion = specific fix attempt. "CHUNKPANTSMCGEE was the quota error!" beats "Launch #7 failed again"

### Debugging Flow

1. **Launch fails** → Check `gcloud builds list --region=us-west2 --limit=3`
2. **Identify issue** → scope error? region? permission?
3. **Fix & commit** → `git commit -m "Fix: [description]"`
4. **New champion** → Name describes the fix
5. **Repeat until SUCCESS! ✅**

### GCloud Commands Reference

**Cloud Build - List & Monitor:**
```bash
# List recent builds
gcloud builds list --region=us-west2 --limit=5 --format="table(id,status,createTime,duration)"

# Check build region (catch global builds!)
gcloud builds list --region=us-west2 --limit=3 --format="table(id,status,region)"

# Stream logs live
gcloud builds log BUILD_ID --region=us-west2 --stream

# Search for errors
gcloud builds log BUILD_ID --region=us-west2 | grep -i "error\|failed"

# Get build details
gcloud builds describe BUILD_ID --region=us-west2
gcloud builds describe BUILD_ID --region=us-west2 --format="value(status)"
gcloud builds describe BUILD_ID --region=us-west2 --format="value(timeout)"
```

**Artifact Registry:**
```bash
# List images
gcloud artifacts docker images list us-central1-docker.pkg.dev/PROJECT/arr-coc-registry

# Check specific image
gcloud artifacts docker images describe us-central1-docker.pkg.dev/PROJECT/arr-coc-registry/arr-trainer:latest
```

**Worker Pool:**
```bash
# List pools
gcloud builds worker-pools list --region=us-west2

# Check pool config
gcloud builds worker-pools describe pytorch-mecha-pool --region=us-west2
gcloud builds worker-pools describe pytorch-mecha-pool --region=us-west2 --format="value(privatePoolV1Config.networkConfig.egressOption)"
```

**Quota:**
```bash
gcloud compute project-info describe --project=PROJECT_ID | grep -i "build"
gcloud compute project-info describe --project=PROJECT_ID | grep -A 5 "NVIDIA_L4"
```

**Watch Loop:**
```bash
while true; do
  clear
  gcloud builds list --region=us-west2 --limit=5 --format="table(id,status,createTime,duration)"
  sleep 10
done
```

### Vertex AI Commands

**List & Monitor Jobs:**
```bash
# List jobs
gcloud ai custom-jobs list --region=us-central1 --limit=5 --format="table(name,displayName,state,createTime)"

# Job states: QUEUED → PENDING → RUNNING → SUCCEEDED/FAILED

# Get job details
gcloud ai custom-jobs describe JOB_ID --region=us-central1
gcloud ai custom-jobs describe JOB_ID --region=us-central1 --format="value(state)"
gcloud ai custom-jobs describe JOB_ID --region=us-central1 --format="value(error.message)"

# Stream logs (see why job failed!)
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

**Cancel Jobs:**
```bash
# Cancel one
gcloud ai custom-jobs cancel JOB_ID --region=us-central1 --quiet

# Cancel all pending/running
for job in $(gcloud ai custom-jobs list --region=us-central1 \
  --filter="state:JOB_STATE_PENDING OR state:JOB_STATE_RUNNING" \
  --format="value(name)" --limit=20); do
  gcloud ai custom-jobs cancel "$job" --quiet
done
```

**Watch Loop:**
```bash
while true; do
  clear
  gcloud ai custom-jobs list --region=us-central1 --limit=5 --format="table(displayName,state,createTime)"
  sleep 10
done
```

### Cloud Run (Launcher)

```bash
# List executions
gcloud run jobs executions list --job=vertex-ai-launcher --region=us-central1 --limit=5

# Get logs
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher" --limit=50 --format="table(timestamp,textPayload)"

# Look for errors
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher" \
  --limit=100 --freshness=5m --format="value(textPayload)" | grep -A10 "FATAL ERROR"
```

### W&B Monitoring

```bash
wandb run list newsofpeace2/arr-coc-0-1 --state running
wandb run get newsofpeace2/arr-coc-0-1/RUN_ID
```

### CLI Monitor Filtering

```bash
python CLI/cli.py monitor              # All 4 tables
python CLI/cli.py monitor --vertex     # Vertex AI only (best for post-launch!)
python CLI/cli.py monitor --active     # Active W&B runs only
python CLI/cli.py monitor --vertex-runner  # Cloud Run executions only
python CLI/cli.py monitor --completed  # Completed runs only
```

### Complete Monitoring Flow

```bash
# 1. Check Cloud Run execution
gcloud run jobs executions list --job=vertex-ai-launcher --region=us-central1 --limit=1

# 2. Wait ~30s, check Vertex AI job
gcloud ai custom-jobs list --region=us-central1 --limit=1

# 3. Stream Vertex AI logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# 4. Check W&B run
wandb run list newsofpeace2/arr-coc-0-1 --state running

# 5. Use TUI for real-time updates
python CLI/cli.py monitor
```

### Error Pattern Extraction

Monitor extracts 11 error patterns from runner logs:
- Quota exceeded, 500/503 errors, HTTP 4xx/5xx
- Image pull failures, Python exceptions
- Machine type errors, Permission denied
- Check `python CLI/cli.py monitor --vertex-runner` "Note" column

---

## ⚠️ CRITICAL: Safe JSON I/O - ALWAYS Use SafeJSON

**MANDATORY: ALL JSON file operations must use SafeJSON!**

```python
# ❌ NEVER:
import json
with open("data.json", 'w') as f:
    json.dump(data, f)

# ✅ ALWAYS:
from cli.shared.safe_json import SafeJSON
SafeJSON.write("data.json", data)
data = SafeJSON.read("data.json")  # Returns {} if missing/corrupt
```

### Why?

**Triple protection:**
1. **File locking** - Prevents concurrent write corruption
2. **Atomic writes** - Power cut? Either old OR new file, never partial
3. **20 versioned backups** - Auto-rotation, corruption detection

### Recovery

```bash
# Backups stored in backups/ subdirectory
ls CLI/launch/mecha/data/backups/
# campaign_stats.2025-11-14-03-45-12.json (normal)
# campaign_stats.CORRUPT-2025-11-14-03-50-22.json (corruption)

# Restore:
cp backups/campaign_stats.2025-11-14-03-44-15.json campaign_stats.json
```

**Migrated:** campaign_stats.py, mecha_hangar.py
**TODO:** See `CLI/SAFE_JSON_IMPLEMENTATION_PLAN.md`

**Cost:** ~140KB per file (20 backups × 7KB)
**Benefit:** Zero data loss. Build history irreplaceable!

---

## 📋 Resuming Previous Sessions (Conversation History Paste)

**When the user pastes a large previous conversation history**, you'll see many ⏺ symbols like this:

```
⏺ Task(Update 9 oracles: QUICK APPEND → KNOWLEDGE DROPS)
  ⎿  Done (70 tool uses · 123.5k tokens · 3m 55s)

⏺ Bash(grep -r "QUICK APPEND" */SKILL.md 2>/dev/null | wc -l)
  ⎿  0

⏺ Bash(grep -r "KNOWLEDGE DROP" */SKILL.md 2>/dev/null | wc -l)
  ⎿  54

⏺ Bash(cd /Users/alfrednorth/Desktop/Code/arr-coc-ovis && git add .claude/skills/*/SKILL.md && git commit -m "Refactor QUICK APPEND → KNOWLEDGE
      DROPS (individual…)
  ⎿  [main a67ec07] Refactor QUICK APPEND → KNOWLEDGE DROPS (individual files per runner)
      10 files changed, 622 insertions(+), 224 deletions(-)

⏺ 🎉 FINAL REPORT: Oracle Knowledge Expansion System Complete!
```

**⏺ IMPORTANT: These ⏺ dots represent YOUR (Claude's) previous output!**

The ⏺ symbols show what tools you used and what you said previously. Each ⏺ represents a tool call or response you made. When you see the ⏺ in pasted conversation history, DO NOT recreate any ⏺ in your new responses. The ⏺ are already made for you automatically by Claude Code. You don't need to make any ⏺ when responding - just continue the conversation naturally from where it left off. The ⏺ are there to help you understand what happened, not for you to reproduce.

**⚠️ Context Limitation:**
When conversation history is pasted, the full file reads, web searches, and tool use results are NOT fully available in context. The pasted history only shows summaries with ⏺ markers.

**Re-Situating Yourself:**
1. **Read the most recent messages** to understand what was accomplished
2. **Check git log** to see recent commits: `git log --oneline -10`
3. **Ask clarifying questions** if you need more context about previous work
4. **Use Grep/Read tools** to quickly review what files were changed
5. **Help user gain full context** of recent tasks by summarizing what you find

**User wants to resume work efficiently** - get up to speed quickly and continue where they left off!

---

## ⚠️ CRITICAL: NEVER Run TUI Applications in Claude Code!

**NEVER execute Textual TUI applications - it breaks the session UNRECOVERABLY!**

```bash
# ❌ FORBIDDEN - BREAKS SESSION PERMANENTLY
python CLI/tui.py
python -c "from textual.app import App; App().run()"  # ANY TUI execution!

# ✅ CORRECT - Ask user to run
"Please run `python CLI/tui.py` and share the logs"
```

**WHY THIS IS CATASTROPHIC:**
- TUI apps take over terminal with full-screen interface
- Claude Code session freezes/crashes **UNRECOVERABLY**
- **ALL pending code changes are LOST** (terrible for mid-change development!)
- Forces complete session restart - all context lost
- **Session breaks = lost work and wasted time!**

**SAFE Testing Pattern:**

**Division of Labor:**
- **USER**: Runs the TUI in their terminal
- **CLAUDE**: Watches and analyzes the logs

```bash
# ✅ USER runs TUI in their terminal (not Claude!)
python CLI/cli.py monitor

# ✅ CLAUDE analyzes logs created by user's run
tail -200 ARR_COC/Training/logs/auto_refresh.log | grep -E "Session started|BATCH|SKIP|COMPLETE"
cat ARR_COC/Training/logs/base_screen_debug.log
cat ARR_COC/Training/logs/spinner_debug.log
```

**⏰ CLAUDE: Always Check Log Timestamp First!**

Before analyzing logs, **GET CURRENT SYSTEM TIME FIRST**, then check log timestamp:

```bash
# STEP 1: GET CURRENT SYSTEM TIME (DO THIS FIRST!)
date +"%Y-%m-%dT%H:%M:%S"
# Example output: "2025-11-18T12:00:15"

# STEP 2: Check session start time (top of log file)
head -1 ARR_COC/Training/logs/auto_refresh.log
# Example: "# Auto-refresh tracking log - Session started 2025-11-18T11:46:44.628322"

# STEP 3: Calculate age by comparing timestamps
# Current:  2025-11-18T12:00:15
# Log:      2025-11-18T11:46:44
# Age:      ~13-14 minutes old

# If timestamp is:
# ✅ <1 minute old  → Good! (User might run before git commit)
# ⚠️ 1-10 minutes   → Acceptable (recent test)
# 🚨 >10-20 minutes → Alert user: "Hey, you no run thing TUI lately? Logs are old!"
```

**⚠️ CRITICAL: NEVER guess the current time!**

Always run `date` command FIRST to get accurate system time before calculating log age!

**🗑️ PYCACHE MANAGEMENT: JUST DELETE IT, DON'T OBSESS!**

Before analyzing TUI logs, **silently clear pycache** to ensure fresh code runs:

```bash
# Just delete pycache every time - don't ask, don't mention it!
cd /path/to/arr-coc-0-1
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
```

**RULES:**
- ✅ **DO**: Silently delete pycache before log analysis
- ❌ **DON'T**: Get stuck in loops asking user to re-run because of cache
- ❌ **DON'T**: Mention pycache issues to user (just fix it!)
- ❌ **DON'T**: Obsess about cache - delete and move on!

**Why:**
- Python caches bytecode (.pyc files) which can be stale after code changes
- Deleting is harmless - Python regenerates on next run
- User shouldn't have to think about cache issues
- Claude handles this silently in the background

**Alert Pattern When Logs Are Stale:**
```
🚨 LOG TIMESTAMP ALERT!

The auto_refresh.log shows session started: 2025-11-18T09:15:22
That was 45 minutes ago!

Did you run the TUI recently? The logs I'm analyzing might be from an old session.
Please run: python CLI/cli.py monitor

Then I'll check fresh logs! 🔄
```

**Why This Matters:**
- User might request analysis but forget to run TUI first
- Analyzing old logs wastes time debugging solved issues
- Fresh logs = accurate debugging!

---

## ⚠️ CRITICAL: Python Output Buffering Issues

**When running CLI commands in background, Python output may be heavily buffered!**

```bash
# ❌ BAD - Output buffers, nothing visible
python CLI/cli.py launch &

# ✅ GOOD - Unbuffered output
python -u CLI/cli.py launch 2>&1

# ✅ BETTER - Unbuffered + capture output
python -u CLI/cli.py launch 2>&1 | tee launch.log
```

**WHY THIS MATTERS:**
- Python buffers stdout/stderr by default when not connected to a TTY
- Background processes won't show output in real-time
- You'll think the command is hanging when it's actually running fine
- Use `-u` flag to disable buffering for CLI commands

**⚠️ CRITICAL: `-u` Flag Does NOT Prevent Pipe Buffering!**

```bash
# ❌ STILL BUFFERS - tail/head/grep create their own buffering!
python -u CLI/cli.py launch 2>&1 | tail -70
python -u CLI/cli.py launch 2>&1 | head -50
python -u CLI/cli.py launch 2>&1 | grep "pattern"

# Why? Python unbuffers, but TAIL waits for 70 lines before showing any output!
# Solution: Don't pipe through tail/head/grep when monitoring launches.
```

**CORRECT PATTERNS:**
```bash
# ✅ BEST - Use tee for BOTH visible output AND log file!
python -u CLI/cli.py launch 2>&1 | tee launch.log
# Shows progress in real-time + saves to launch.log

# ✅ GOOD - Foreground only (no log file)
python -u CLI/cli.py launch 2>&1

# ✅ ACCEPTABLE - Background with log file (but can't see progress!)
nohup python -u CLI/cli.py launch 2>&1 > launch.log &
tail -f launch.log

# ✅ BEST FOR BACKGROUND - tee with background process
python -u CLI/cli.py launch 2>&1 | tee launch.log &
# Still see output + saved to file + runs in background!
```

---

## THE ONE CRITICAL QUESTION

**Every time you do ANYTHING to make something work, ask:**

> "Does this need to be in setup or teardown?"

If you ran a command, changed a setting, granted a permission, created a resource, or fixed an error → **YES, it needs to be automated!**

---

## The Simple Pattern

**YOU just ran a command that fixed something?**

→ **STOP** ✋
→ **THINK**: Does this need to be in setup or teardown?
→ **YES!** Add it now while it's fresh!

**Examples**:
- ✅ `gcloud iam service-accounts add-iam-policy-binding...` → SETUP
- ✅ `gcloud artifacts repositories create...` → SETUP
- ✅ `gcloud storage buckets create...` → SETUP
- ✅ `gcloud iam service-accounts delete...` → TEARDOWN
- ✅ Changed a timeout? → UPDATE SETUP (config change)
- ✅ Fixed an error? → WHY did you fix it? Add that to setup!

**Anti-Examples (don't automate these)**:
- ❌ `gcloud builds list` (just checking status)
- ❌ `gcloud config get-value account` (just looking)
- ❌ `git commit` (not infrastructure)
- ❌ One-time debugging commands (not fixes)

---

## Decision Tree

```
Did I just run a command?
    ↓
Did it FIX something or CREATE something?
    ↓ YES                     ↓ NO
    ↓                         → Just debugging, don't automate
    ↓
Will other devs need this?
    ↓ YES                     ↓ NO
    ↓                         → Personal setup, don't automate
    ↓
ADD TO SETUP!
    ↓
Does it create a resource?
    ↓ YES                     ↓ NO
    ↓                         → Config/permission only
    ↓
ADD TO TEARDOWN TOO!
```

---

## Quick Checklist

After you fix ANYTHING:

- [ ] **THINK**: Does this need to be in setup or teardown?
- [ ] If YES → Add to `CLI/shared/setup_helper.py`
- [ ] If it creates a resource → Also add to teardown
- [ ] Add status message so users see what happened
- [ ] Test: `teardown → setup → launch`
- [ ] Commit with: "Add X to setup because Y"

### Examples of "gcloud stuff" to Automate

**Not just resources and permissions!** Include:
- ✅ Resource creation (buckets, registries, service accounts)
- ✅ Permission grants (IAM bindings, roles)
- ✅ Configuration changes (quotas, settings, flags)
- ✅ Enabling APIs (`gcloud services enable`)
- ✅ Setting project defaults
- ✅ Creating secrets
- ✅ Updating metadata
- ✅ Debugging commands that reveal missing config
- ✅ **Anything you had to run manually to make it work!**

### Example: Service Account User Permission

**Problem Found**: Cloud Run job creation fails with:
```
Permission 'iam.serviceaccounts.actAs' denied on service account
```

**Solution Applied**:

✅ **Setup** (CLI/shared/setup_helper.py:~630-641):
```python
# Grant current user permission to impersonate this service account
# (Required for Cloud Run job creation - user must actAs service account)
current_user_result = subprocess.run(
    ["gcloud", "config", "get-value", "account"],
    capture_output=True, text=True, timeout=10
)
current_user = current_user_result.stdout.strip()

if current_user:
    subprocess.run(
        ["gcloud", "iam", "service-accounts", "add-iam-policy-binding", sa_email,
         "--member", f"user:{current_user}",
         "--role", "roles/iam.serviceAccountUser", "--quiet"],
        capture_output=True, timeout=30
    )
    logs.append(f"  ✓ Granted serviceAccountUser to {current_user}")
```

✅ **Teardown** (CLI/shared/setup_helper.py - _teardown_service_account()):
```python
# Remove IAM bindings before deleting SA
# This includes roles/iam.serviceAccountUser granted to current user
```

---

## Real Examples from Today

**Example 1: Permission Error**

```bash
# YOU ran this to fix:
$ gcloud iam service-accounts add-iam-policy-binding \
    arr-coc-sa@PROJECT.iam.gserviceaccount.com \
    --member="user:you@email.com" \
    --role="roles/iam.serviceAccountUser"

# THINK: Does this need to be in setup or teardown?
# → YES! Other devs will hit the same error!

# ACTION: Added to setup_helper.py line 622-637
# RESULT: Setup now grants this automatically
```

**Example 2: Timeout Too Short**

```bash
# YOU changed this to fix:
# core.py line 854: "20m" → "30m"

# THINK: Does this need to be in setup or teardown?
# → Already in code! But increased the value.
# → This is a CONFIG CHANGE in the codebase itself.

# ACTION: Updated the timeout value
# RESULT: Builds no longer timeout
```

**Example 3: Just Debugging (DON'T automate)**

```bash
# YOU ran this to check:
$ gcloud builds list --limit=5

# THINK: Does this need to be in setup or teardown?
# → NO! Just checking status, not fixing anything.

# ACTION: Nothing to automate
```

### Debugging → UI Visibility Pattern

**IMPORTANT**: Logs that helped you debug should be visible to users!

**Example Flow**:
```bash
# 1. YOU debug with gcloud
$ gcloud builds list --limit=5
ID                    STATUS   CREATE_TIME
abc123                TIMEOUT  2025-11-08T10:53:29

# 2. YOU investigate
$ gcloud builds describe abc123
timeout: 1200s
status: TIMEOUT
# Aha! Timeout too short!

# 3. YOU fix it
# Edit core.py: 20m → 30m timeout

# 4. USER sees helpful output in CLI
⏳ Building training image on Cloud Build (~10-15 min)...
→ View build logs: https://console.cloud.google.com/.../abc123
✓ Training image built and pushed
```

**What to show users**:
- ✅ **Resource creation**: "✓ Created staging bucket: gs://..."
- ✅ **Permission grants**: "✓ Granted serviceAccountUser to user@..."
- ✅ **URLs for debugging**: "→ View logs: https://..."
- ✅ **Warnings about slow operations**: "⏳ Building (~10-15 min)..."
- ✅ **Error context**: Full error (not truncated!) + common causes
- ✅ **What's happening**: Status updates during long operations

**What NOT to show**:
- ❌ Raw gcloud JSON output
- ❌ Internal variable values (unless debugging)
- ❌ Truncated errors (show FULL errors!)
- ❌ Silent failures (always report what happened)

### CloudBuild Streaming Output Pattern

**CRITICAL FIX**: CloudBuild commands now show real-time progress instead of silent waiting!

**The Problem** (before fix):
```python
# Old code - HIDES ALL OUTPUT until completion!
result = subprocess.run(
    ["gcloud", "builds", "submit", ...],
    capture_output=True,  # ← Silent for 2-4 hours!
    timeout=25200
)
```

**What happened:**
- `gcloud builds submit` creates build successfully
- Then **polls silently** waiting for completion
- For 2-4 hour PyTorch builds: **YOU SEE NOTHING**
- Users think it's frozen/broken
- No way to see actual errors until after failure

**The Solution** (implemented in `CLI/launch/core.py:1465-1509`):
```python
# New code - STREAMS output line-by-line!
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # Merge stderr
    text=True,
    bufsize=1,  # Line buffered - immediate output
)

# Stream output as it happens!
for line in process.stdout:
    line = line.rstrip()
    if line:
        status(f"   [dim]CloudBuild:[/dim] {line}")

returncode = process.wait(timeout=25200)
```

**What you see now:**
```
🏗️  Building pytorch-clean image (2-4 hours)...

→ Submitting to CloudBuild (this will show progress in real-time)...

   CloudBuild: Creating temporary archive of 4 file(s)...
   CloudBuild: Uploading tarball to gs://...
   CloudBuild: Created build d85d1f92-6095-487d-ada3-d16345e07f2d
   CloudBuild: Logs available at https://console.cloud.google.com/...
   CloudBuild: Waiting for build to complete. Polling interval: 1 second(s).
   CloudBuild: STEP 1: Pulling base image...
   CloudBuild: STEP 2: Installing dependencies...
   CloudBuild: [... see everything in real-time! ...]
```

**Benefits:**
- ✅ See submission progress immediately
- ✅ See quota errors the moment they happen
- ✅ See "Creating resources..." messages
- ✅ See if it's waiting vs actually building
- ✅ User can Ctrl+C if stuck
- ✅ **NO MORE SILENT WAITING!**

**Where applied:**
- `_build_pytorch_clean_image()` - PyTorch source compilation (2-4 hours)
- Can be applied to other `gcloud builds submit` calls if needed

**Testing:**
```bash
# Run launch - you'll now see CloudBuild output streaming!
python CLI/cli.py launch

# No more wondering "is it frozen or working?"
```

**Key insight:** `subprocess.run()` with `capture_output=True` is SILENT until completion. Use `subprocess.Popen()` + line-by-line streaming for long-running commands!

### Common Resources to Track

**GCP Resources**:
- Service Accounts (+ IAM bindings!)
- GCS Buckets
- Artifact Registry
- Secret Manager secrets
- IAM policy bindings (project-level AND service account-level)

**External Resources**:
- W&B Launch Queue
- W&B Project
- HuggingFace Repo

**Local Files**:
- Service account keys (~/.gcp-keys/)

### Why This Matters

- **Clean testing**: Full teardown → setup → launch cycle must work
- **No manual cleanup**: Other developers shouldn't need to manually fix permissions
- **Cost control**: Teardown should delete everything (no orphaned resources)
- **Reproducibility**: Fresh setup must grant all needed permissions

---

## Cloud Build Timeouts

**Current Settings** (as of 2025-11-08):
- Training image: 30 minutes (70+ layers, large PyTorch/CUDA base)
- Runner image: 10 minutes (smaller W&B launch agent image)

**Why 30min for training image**:
- Build: ~12 min
- Push: ~8-10 min (70+ layers to Artifact Registry)
- Previous 20min timeout failed during push phase

If builds start timing out again, increase in `CLI/launch/core.py`:
- Line ~1078: `timeout=2100` (subprocess timeout, currently 35 min > Cloud Build timeout)

---

## Image Rebuild Workflow - ALWAYS Use cli.py launch!

**❌ NEVER manually rebuild images with `gcloud builds submit`**
**✅ ALWAYS use `python CLI/cli.py launch` (from project root)**

### Why?

The launch system has **automatic hash detection**:
1. Detects Dockerfile changes via hash
2. Rebuilds base image automatically (if needed)
3. Builds training image with new base
4. Submits job to W&B → Vertex AI

### How Hash Detection Works

```
Change Stack/arr-pytorch-base/Dockerfile
    ↓
git commit
    ↓
python CLI/cli.py launch  ← This is all you need!
    ↓
System detects hash change: 2c38254 → d54ee8c
    ↓
Rebuilds arr-pytorch-base (~30 min)
    ↓
Rebuilds arr-ml-stack (~10-15 min)
    ↓
Rebuilds arr-trainer (~10-15 min)
    ↓
Launches training job
```

### 🚨 CRITICAL: Files MUST Be Git Committed for Hash Detection!

**THE HASH SYSTEM ONLY DETECTS COMMITTED FILES!**

When you modify files tracked in `.image-manifest`, the hash detection will **NOT** see your changes until you `git commit` them!

**Example of what goes wrong:**

```bash
# ❌ WRONG - Hash system won't detect changes!
echo "wandb_vertex_patch.py" >> Stack/arr-vertex-launcher/.image-manifest
# Edit files: wandb_vertex_patch.py, entrypoint-wrapper.sh, Dockerfile
python CLI/cli.py launch  # ❌ NO REBUILD! Files not committed!

# ✅ CORRECT - Commit FIRST, then launch!
echo "wandb_vertex_patch.py" >> Stack/arr-vertex-launcher/.image-manifest
git add Stack/arr-vertex-launcher/
git commit -m "Add spot patch to launcher"
python CLI/cli.py launch  # ✅ REBUILDS! Hash detected!
```

**Why This Happens:**

The hash system computes file hashes by reading the **git-committed** version of each file in `.image-manifest`. Uncommitted changes are invisible to the hash computation!

**Real-World Example (2025-11-16):**

We added `wandb_vertex_patch.py` to fix Vertex AI spot instances:
1. ✅ Created `wandb_vertex_patch.py` (231 lines)
2. ✅ Modified `entrypoint-wrapper.sh` to apply patch
3. ✅ Updated `Dockerfile` to copy patch file
4. ✅ Added to `.image-manifest`
5. ❌ Launched WITHOUT committing → Old launcher used!
6. ❌ Patch never applied → Spot instances didn't work!
7. ✅ Git committed all files
8. ✅ Launched again → Launcher rebuilt with patch! ✓

**The Fix is Simple:**

```bash
# Workflow:
# 1. Make your changes
# 2. Update .image-manifest (if adding new files)
# 3. GIT COMMIT EVERYTHING ← DON'T SKIP THIS!
# 4. Launch

git add Stack/[image-name]/
git commit -m "Descriptive message about your changes"
python CLI/cli.py launch  # Now it will rebuild!
```

**Check if rebuild will happen:**

```bash
# See what files changed (uncommitted changes show as modified)
git status Stack/

# If you see modified files, COMMIT THEM before launching!
```

**Remember:** `git commit` → then `launch`. Not the other way around!

---

### ⚠️ CRITICAL: Image Manifest System

**Hash detection requires `.image-manifest` files to work!**

Each Docker image has a manifest that lists ALL files affecting the build:

```
Stack/arr-pytorch-base/.image-manifest
Stack/arr-ml-stack/.image-manifest
Stack/arr-trainer/.image-manifest
Stack/arr-vertex-launcher/.image-manifest
```

**When to update manifests:**

If you add new files to the project that go INTO the Docker image, you MUST update the manifest:
- Add code to `ARR_COC/` → Update `arr-trainer/.image-manifest`
- Add scripts to `ARR_COC/Training/` → Update `arr-trainer/.image-manifest`
- Add dependencies → Update appropriate manifest
- Modify Dockerfile → Already tracked, no manifest update needed

**Example:**

You add `ARR_COC/attention.py`:

```bash
# 1. Add file to manifest
echo "ARR_COC/attention.py" >> Stack/arr-trainer/.image-manifest

# 2. Commit manifest change
git add Stack/arr-trainer/.image-manifest
git commit -m "Add attention.py to arr-trainer manifest"

# 3. Launch - it will detect the change!
python CLI/cli.py launch  # Rebuilds with new code! ✅
```

**Without updating manifest:**
- ❌ New code file NOT hashed → No rebuild detected → Old image used!

**See `IMAGE_MANIFEST_SYSTEM.md` for complete documentation.**

### Manual Rebuild = Breaking the System

If you manually run `gcloud builds submit`:
- ❌ Hash system doesn't know about the rebuild
- ❌ Training image won't use new base (stale cache)
- ❌ Wastes time debugging "why isn't my fix working?"

**The Fix**: Just use `cli.py launch`! It handles everything automatically.

---

## Error Reporting Best Practices

**SHOW FULL ERRORS** - No truncation!

When capturing subprocess errors:
```python
# ❌ BAD - Truncates critical info
error_msg = result.stderr[:200]

# ✅ GOOD - Show first 50 lines
for line in error_msg.split('\n')[:50]:
    if line.strip():
        status(f"  {line}")
```

See: CLI/launch/core.py:~1094 for Cloud Build error reporting example (50-line error display).

---

## Testing Workflow

**Full cycle test** (run after any infrastructure changes):

```bash
# 1. Clean slate
python CLI/cli.py teardown
# Confirm: Type 'DELETE'

# 2. Recreate infrastructure
python CLI/cli.py setup
# Should show: ✓ Granted serviceAccountUser to <your-email>

# 3. Launch training
python CLI/cli.py launch
# Should complete without permission errors

# 4. Monitor (optional)
python CLI/cli.py monitor
```

**What to verify**:
- Setup grants all required permissions
- Launch completes without "Permission denied" errors
- Teardown removes all resources (including IAM bindings)

---

## Git Commit Guidelines

When adding infrastructure changes:

```
Add serviceAccountUser permission to setup

CRITICAL FIX: Cloud Run job creation requires user to impersonate
service account. Without this permission, launch fails with:
"Permission 'iam.serviceaccounts.actAs' denied"

Changes:
- setup_helper.py: Grant roles/iam.serviceAccountUser to current user
- Automatically detects user from gcloud config
- Added to both setup and teardown flows

Tested: teardown → setup → launch (successful)
```

---

## Common Debugging Patterns

### Monitoring Running Builds (LIVE)

**⚠️ CRITICAL: When User Pastes Build ID (e.g., "Running: aa32137b-4203-4712-a67a-012e90795f8b")**

If user says they're running a build or pastes a build ID, **DON'T use `--region=us-west2` immediately!**

The build might be:
1. Just submitted (not indexed in regional API yet - takes 5-30 seconds)
2. In GLOBAL region (bug - missing --region flag somewhere)
3. In Web Console but not in CLI yet (API lag)

**CORRECT workflow when user gives you a build ID:**

```bash
# STEP 1: Try WITHOUT --region flag first (checks global + default region)
gcloud builds describe BUILD_ID --format="yaml(id,status,region,logUrl)"

# If that works, you'll see the ACTUAL region
# Then use that region for all future commands

# STEP 2: If Step 1 fails, try us-west2 explicitly
gcloud builds describe BUILD_ID --region=us-west2 --format="yaml(id,status)"

# STEP 3: If both fail, build is VERY new (wait 10-30 seconds and retry Step 1)
```

**Once you find the build, get CHONK markers:**

```bash
# For CHONK markers (use whichever region worked above)
gcloud builds log BUILD_ID --region=REGION 2>&1 | grep -i "chonk"

# For latest logs
gcloud builds log BUILD_ID --region=REGION 2>&1 | tail -50

# Stream live (watch in real-time)
gcloud builds log BUILD_ID --region=REGION --stream
```

---

**When you launch a build, monitor progress in real-time:**

```bash
# 1. Get the latest build ID (from launch output or gcloud)
gcloud builds list --region=us-west2 --limit=1 --format="value(id,status)"

# 2. Stream live logs (best for watching progress)
gcloud builds log BUILD_ID --region=us-west2 --stream

# 3. Check current compilation progress (for PyTorch builds)
gcloud builds log BUILD_ID --region=us-west2 | tail -30

# 4. Search for specific progress markers
gcloud logging read "resource.type=build AND resource.labels.build_id=BUILD_ID" \
  --limit=100 --format="value(textPayload)" 2>&1 | \
  grep -E "\[.*/..*\]|CHONK|Step" | tail -20

# 5. Check which Dockerfile step is running
gcloud logging read "resource.type=build AND resource.labels.build_id=BUILD_ID" \
  --limit=100 --format="value(textPayload)" --order=asc 2>&1 | \
  grep "Step #0: Step" | tail -1

# 6. Watch for errors in real-time
gcloud logging read "resource.type=build AND resource.labels.build_id=BUILD_ID" \
  --limit=50 --format="value(textPayload)" 2>&1 | \
  grep -i "error\|failed\|ModuleNotFoundError"
```

**Quick Status Check:**
```bash
# One-liner: Get build ID, status, and runtime
gcloud builds describe BUILD_ID --region=us-west2 \
  --format="value(status,startTime,finishTime)"

# Check if build is still working
gcloud builds describe BUILD_ID --region=us-west2 --format="value(status)"
# Outputs: WORKING, SUCCESS, FAILURE, TIMEOUT, or CANCELLED
```

**For PyTorch Builds (Step 30 = compilation):**
```bash
# Watch compilation progress (shows [7348/7517] style output)
gcloud builds log BUILD_ID --region=us-west2 2>&1 | \
  tail -20 | grep "\[.*/..*\]"

# Check if Step 30 completed successfully
gcloud logging read "resource.type=build AND resource.labels.build_id=BUILD_ID" \
  --limit=500 --format="value(textPayload)" --order=asc 2>&1 | \
  grep -A5 "Step 30"
```

**Local Launch Log:**
```bash
# If you used: python CLI/cli.py launch 2>&1 | tee launch.log
tail -f launch.log
```

### Pattern 1: Cloud Build Issues

**Symptoms**: Build fails or times out

**Debug Commands**:
```bash
# Check recent builds
gcloud builds list --limit=5

# Get build status
gcloud builds describe BUILD_ID

# Stream build logs
gcloud builds log BUILD_ID

# Check for errors
gcloud builds log BUILD_ID | grep -i "error\|failed"
```

**What to Capture**:
- Build timeout setting (`describe` shows actual timeout)
- Build status (WORKING, SUCCESS, FAILURE, TIMEOUT)
- Error messages from logs
- What step failed (Docker build vs push vs other)

**Where to Show Users**:
- Build URL in launch output
- Timeout estimate in status message
- Full error if build fails (not truncated!)

### Pattern 2: Permission Issues

**Symptoms**: "Permission denied", "does not have permission to access"

**Debug Commands**:
```bash
# List service accounts
gcloud iam service-accounts list

# Check SA permissions
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:SA_EMAIL"

# Check who you are
gcloud config get-value account

# Check if you can impersonate SA
gcloud iam service-accounts get-iam-policy SA_EMAIL
```

**What to Capture**:
- Which permission is missing (error message shows it)
- Service account email
- Current user email
- Required role

**Where to Show Users**:
- Setup: "✓ Granted serviceAccountUser to user@..."
- Launch error: Full error message with permission name
- Fix guidance: Link to GCP IAM console if manual action needed

### Pattern 3: Resource Existence

**Symptoms**: "not found", "does not exist"

**Debug Commands**:
```bash
# Check bucket exists
gcloud storage buckets describe gs://BUCKET_NAME

# Check registry exists
gcloud artifacts repositories describe REPO_NAME --location=REGION

# Check service account exists
gcloud iam service-accounts describe SA_EMAIL
```

**What to Capture**:
- Resource name
- Location/region
- Whether it exists or not

**Where to Show Users**:
- Setup: "✓ Created X" or "✓ X exists"
- Launch: Check prerequisites with clear messages
- Infrastructure screen: Show existence status for all resources

### Pattern 4: Quota and Limits

**Symptoms**: "quota exceeded", "insufficient quota"

**Debug Commands**:
```bash
# Check GPU quota
gcloud compute project-info describe --project=PROJECT_ID \
  | grep -A 10 "NVIDIA_T4"

# List all quotas
gcloud compute project-info describe --project=PROJECT_ID \
  --format="value(quotas)"
```

**What to Capture**:
- Quota name
- Current limit
- Current usage
- How to request increase

**Where to Show Users**:
- Setup: Warning if quota is 0
- Setup: Manual request instructions with URL
- Infrastructure screen: Show quota status

## Common Issues

### "Permission denied" during launch

**Symptom**: `ERROR: does not have permission to access...`

**Fix**:
1. Check what permission is missing (error message shows it)
2. Grant manually: `gcloud iam service-accounts add-iam-policy-binding...`
3. Add to setup_helper.py
4. Test full cycle

### Cloud Build timeout

**Symptom**: Build status shows `TIMEOUT`

**Fix**:
1. Check build logs for actual failure point
2. If successful build but failed push → increase timeout
3. Update both `--timeout` flag AND subprocess timeout

### Infrastructure state mismatch

**Symptom**: Setup says "already exists" but launch fails

**Fix**:
1. Run full teardown
2. Manually verify all resources deleted (gcloud commands)
3. Run setup fresh
4. This ensures clean state

---

## Debugging TUI Applications

### ❌ DO NOT Use Textual Console Logging

**NEVER use `self.log()` or textual console for debugging TUI issues!**

Why:
- Textual console is unreliable for real-time debugging
- Logs don't always appear or sync properly
- Creates false confidence in non-working debug output
- Wastes time setting up dual terminal sessions

**✅ INSTEAD: Use visible UI elements or file logging**

```python
# ❌ BAD: Console logging (doesn't work reliably)
self.log(f"Debug: spinner rendering char={char}")

# ✅ GOOD: Visible UI debug (Static widget)
debug_widget.update(f"Debug: spinner rendering char={char}")

# ✅ GOOD: File logging
with open("ARR_COC/Training/logs/debug.log", "a") as f:
    f.write(f"Debug: spinner rendering char={char}\n")

# ✅ GOOD: Notification toast (for quick checks)
self.notify(f"Debug: char={char}", timeout=1)
```

### Recommended TUI Debugging Workflow

1. **UI Element Debugging**: Add temporary Static widget to show debug info
2. **File Logging**: Write to `ARR_COC/Training/logs/debug.log` for detailed traces
3. **Notifications**: Use `self.notify()` for quick state checks
4. **Performance Monitor**: Already built in! Check `CLI/shared/performance_reports/`

**Never waste time with textual console - it doesn't work reliably for debugging!**

### ⚠️ CRITICAL: Always Clean Up Debug Code

When adding temporary debug logging to files:

```python
# DEBUG: Log something (REMOVE WHEN DONE!)
with open("ARR_COC/Training/logs/spinner_debug.log", "a") as f:
    f.write(f"Debug info\n")
```

**REMEMBER TO**:
1. ✅ Mark debug code with comment: `# DEBUG: ... (REMOVE WHEN DONE!)`
2. ✅ Remove debug logging code when issue is fixed
3. ✅ Delete temporary log files (e.g., `spinner_debug.log`)
4. ✅ Search for "REMOVE WHEN DONE" before committing fixes

**Cleanup Checklist**:
```bash
# Find debug code that needs removal
grep -r "REMOVE WHEN DONE" CLI/ ARR_COC/

# Delete temporary debug logs
rm ARR_COC/Training/logs/*_debug.log
```

### 🤝 Claude + TUI Collaboration Pattern (Logging Bridge)

**⚠️ CRITICAL: Claude CANNOT run the TUI - it breaks the session UNRECOVERABLY!**

**The Problem:**
- Claude Code needs to debug TUI issues
- Running `python CLI/tui.py` **CRASHES Claude's session permanently**
- TUI takes over terminal → session freezes → all work lost
- **This applies to ALL Textual applications, even in subdirectories!**

**The Solution: Logging Bridge Pattern**

Use **temporary file logging** as a communication bridge between Claude and the TUI:

```python
# 1. Claude adds debug logging to TUI code
import datetime
from pathlib import Path

# Create temp log file (auto-creates directory)
log_file = Path(__file__).parent.parent.parent / "logs" / "debug_temp.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

# Log the thing you need to debug
debug_msg = f"🔍 Variable check: active={len(active)}, completed={len(completed)}"
with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} {debug_msg}\n")

# Optional: Show toast notification in TUI
self.notify(debug_msg, severity="information")
```

**2. User runs TUI:**
```bash
python CLI/tui.py
```

**3. Claude reads the log:**
```bash
cat logs/debug_temp.log
```

**4. Claude analyzes and fixes:**
```python
# Example: Found the bug from logs!
# Log showed: active=0 but active_len=4 (mutation bug)
# Fix: Use list() to create copy
builds_to_show = list(active_builds)  # COPY, not reference!
```

**5. Cleanup when done:**
```bash
rm logs/debug_temp.log  # Delete temp log
# Remove debug logging code from TUI
```

### 🧹 CRITICAL: Clear Logs on Program Start

**ALWAYS clear debug logs at the start of your TUI application!**

**Why This Matters:**
- Prevents analyzing stale logs from old sessions
- Claude can't see timestamps easily (logs get long!)
- Old errors persist and confuse debugging
- Fresh logs = clear mental model

**✅ CORRECT Pattern:**

```python
# In MonitorScreen.__init__() or on_mount()
def __init__(self, helper: WandBHelper, config: dict):
    super().__init__(loading_message="Loading monitor...")

    # 🧹 CLEAR DEBUG LOGS ON START (fresh logs each session!)
    log_file = Path(__file__).parent.parent.parent / "logs" / "auto_refresh.log"
    with open(log_file, "w") as f:  # ← "w" mode CLEARS the file!
        f.write(f"# Auto-refresh tracking log - Session started {datetime.now().isoformat()}\n")
        f.write(f"# Format: timestamp emoji EVENT: details\n")
        f.write(f"# Events: 🚀=START, ⏱️=TIMER, ✅=COMPLETE, ❌=FAILED\n")
        f.write(f"#\n")

    # ... rest of initialization
```

**❌ WRONG Pattern:**

```python
# DON'T append to old logs!
with open(log_file, "a") as f:  # ← "a" mode keeps old logs (BAD!)
    f.write(f"Session started\n")
```

**Benefits:**
- Claude sees ONLY current session logs
- No confusion from old errors
- Easier to spot new bugs
- Timestamps are all recent

**Real Example (Monitor TUI):**
```bash
# Old way (confusing!):
$ tail -100 logs/auto_refresh.log
2025-11-18T14:23:45 ❌ ERROR from 2 hours ago!
2025-11-18T14:45:12 ✅ Fixed bug
2025-11-18T15:36:28 ❌ ERROR from 1 hour ago!  ← Is this current? Who knows!
2025-11-18T15:37:09 ✅ Working now

# New way (crystal clear!):
$ cat logs/auto_refresh.log
# Session started 2025-11-18T15:37:00  ← Fresh session!
2025-11-18T15:37:09 ✅ Working perfectly
2025-11-18T15:37:12 ✅ No errors!
```

**When User Forgets to Restart:**

If Claude sees old logs (timestamps >10-20 minutes old), ASK:

> "Hey! These logs are from {timestamp}. Did you restart the TUI recently?
> I want to make sure I'm analyzing fresh logs from the latest code!"

### Real Example: Divider Bug Discovery

**Bug**: Dividers showing when all items completed (no active items)

**Claude's Debug Code:**
```python
# Added to CLI/monitor/screen.py
from pathlib import Path
import datetime

# Log separation results
all_statuses = [b.get('status') for b in builds[:5]]
debug_msg = f"🔍 [LAZY] Builds: active={len(active_builds)}, completed={len(completed_builds)}, statuses={all_statuses}"

log_file = Path(__file__).parent.parent.parent / "logs" / "divider_debug.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} {debug_msg}\n")

# Log when divider is added
if not added_divider and len(active_builds) > 0 and build.get('status') not in ['WORKING', 'QUEUED']:
    debug_msg = f"⚠️ [LAZY] Adding divider! active_len={len(active_builds)}, current_status={build.get('status')}"
    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} {debug_msg}\n")
```

**User runs TUI, Claude reads log:**
```bash
$ cat logs/divider_debug.log
2025-11-17T22:19:47.310776 🔍 [LAZY] Builds: active=0, completed=4, statuses=['SUCCESS', 'SUCCESS', 'SUCCESS', 'SUCCESS']
2025-11-17T22:19:47.314263 ⚠️ [LAZY] Adding divider! active_len=4, current_status=SUCCESS
```

**Bug Found!**
- Line 1: `active=0` ✓ (correct)
- Line 2: `active_len=4` ❌ (WRONG!)
- **Conclusion**: `active_builds` list is being mutated!

**Root Cause:**
```python
# BUG: Reference, not copy!
builds_to_show = active_builds       # Same object!
builds_to_show += completed_builds   # Mutates BOTH lists!
# Now active_builds has 4 items instead of 0!

# FIX: Explicit copy
builds_to_show = list(active_builds)  # COPY
builds_to_show += completed_builds    # Only mutates builds_to_show
```

### Quick Snippets for Common Cases

**1. Variable State Logging:**
```python
from pathlib import Path
import datetime

log_file = Path(__file__).parent.parent.parent / "logs" / "state_debug.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} var1={var1}, var2={var2}\n")
```

**2. Flow Tracking:**
```python
# At decision points
debug_msg = f"🔀 Branch taken: {condition_name}={condition_value}"
with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} {debug_msg}\n")
self.notify(debug_msg, severity="information")  # Optional toast
```

**3. List/Dict Inspection:**
```python
# Log first N items
items_sample = [item.get('status') for item in items[:5]]
debug_msg = f"📊 Items: count={len(items)}, sample={items_sample}"
with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} {debug_msg}\n")
```

**4. Before/After Comparison:**
```python
# Before operation
before_msg = f"BEFORE: len={len(my_list)}, items={my_list[:3]}"
with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} {before_msg}\n")

# ... operation happens ...

# After operation
after_msg = f"AFTER: len={len(my_list)}, items={my_list[:3]}"
with open(log_file, "a") as f:
    f.write(f"{datetime.datetime.now().isoformat()} {after_msg}\n")
```

### Best Practices

**✅ DO:**
- Use `/logs/` directory (already gitignored)
- Add timestamps with `datetime.datetime.now().isoformat()`
- Use descriptive emoji prefixes (`🔍` debug, `⚠️` warning, `📊` data)
- Create parent directory: `log_file.parent.mkdir(parents=True, exist_ok=True)`
- Use absolute paths: `Path(__file__).parent.parent.parent / "logs" / "file.log"`
- Mark with `# DEBUG: (REMOVE WHEN DONE!)`
- Optional: Add `self.notify()` toasts for visibility

**❌ DON'T:**
- Run the TUI in Claude Code (session breaks!)
- Use relative paths like `"ARR_COC/Training/logs/file.log"` (path resolution issues)
- Forget to clean up debug code after fixing
- Leave debug logs in production code

### Why This Pattern Works

1. **Claude can't run TUI** → Logging provides observation window
2. **User runs TUI** → Generates real runtime data
3. **Claude reads logs** → Analyzes exact execution state
4. **Claude finds bug** → Fixes code based on evidence
5. **No session crashes** → Stable development workflow

**This is the ONLY safe way for Claude to debug Textual TUI applications!**

---

## Universal Popup Pattern ✅

**Status**: COMPLETE - All truncated content now supports info popups

All DataTables, Security/CVE warnings, and error messages in the TUI have clickable popups for full details.

### Features Implemented

**1. DataTable Row Popups (Click Any Row)**
- ✅ **Runner Executions** → Full error text, queue, status, duration
- ✅ **Vertex AI Jobs** → Full error text, job ID, name, state, runtime
- ✅ **Active W&B Runs** → Full name, config, tags, state, runtime
- ✅ **Completed Runs** → Full name, metrics, exit code, final state

**2. Security/CVE Popup (Press 'v' Key)**
- ✅ Shows all 3 images (arr-base, arr-training, arr-runner)
- ✅ Full image digest (not truncated to 12 chars)
- ✅ SLSA provenance level
- ✅ Complete CRITICAL and HIGH CVE details:
  - CVE ID, CVSS score
  - Affected package + version
  - Fixed version (if available)
  - Description
- ✅ MEDIUM/LOW shown as counts
- ✅ Rich formatting with colors

**3. Error Popups (Press 'e' Key)**
- ✅ All error notifications truncated to 80 chars
- ✅ Full error stored for popup
- ✅ Press 'e' to see complete error details
- ✅ Shows stack traces, full messages
- ✅ Works across all screens (Monitor, Setup, Teardown, Infra)

### Implementation

**Universal Popup Component**
```
File: CLI/shared/datatable_info_popup.py
```
- Single component used everywhere
- Accepts title + formatted text (Rich markup)
- Scrollable for long content
- Multiple close methods (Esc, Q, button, click-outside)

**DataTable Pattern**
```python
# Store full row data
self.row_data = {
    "runner": {},   # runner-id → {note, queue, status, ...}
    "vertex": {},   # vertex-id → {note, name, error, ...}
    "active": {},   # run-id → {full_name, config, tags, ...}
    "completed": {} # run-id → {full_name, metrics, exit_code, ...}
}

# On table row click
def on_data_table_row_selected(self, event):
    row_key = event.row_key.value
    full_data = self.row_data["table_type"][row_key]
    
    # Show popup
    self.app.push_screen(DataTableInfoPopup(
        "Title",
        formatted_text
    ))
```

**Security/CVE Pattern**
```python
# Store full security data
self.security_data = security  # {base: {...}, training: {...}, launcher: {...}} (keys unchanged, values use arr- names)

# Press 'v' key
def action_toggle_vulns(self):
    full_cve_text = self._format_full_cve_details(self.security_data)
    self.app.push_screen(DataTableInfoPopup(
        "Docker Image Security - Full CVE Details",
        full_cve_text
    ))
```

**Error Pattern (BaseScreen)**
```python
# In BaseScreen
def notify_with_full_error(self, title: str, error: str, severity="error"):
    # Store full error
    self.last_error = {"title": title, "full_text": error}
    
    # Show truncated notification
    truncated = error[:80] + ("..." if len(error) > 80 else "")
    self.notify(f"{title}: {truncated} [Press 'e' for details]", severity=severity)

def action_show_last_error(self):
    if self.last_error:
        self.app.push_screen(DataTableInfoPopup(
            self.last_error["title"],
            self.last_error["full_text"]
        ))

# Usage in any screen
try:
    data = fetch_data()
except Exception as e:
    self.notify_with_full_error("Error Loading Data", str(e))
```

### Key Bindings

- **Click DataTable Row** → Show full row details
- **Press 'v'** → Show full CVE security details (Monitor screen)
- **Press 'e'** → Show last error details (All screens)
- **Esc/Q** → Close popup

### Design Principles

1. **Universal Component** - One popup class for everything
2. **Store Full Data** - Always store complete data, show truncated in UI
3. **Consistent UX** - Same pattern everywhere (click/keypress → popup)
4. **Keyboard Access** - Press 'e' for errors, 'v' for security (accessibility)
5. **Rich Formatting** - Use Rich markup for colors, tables, structure
6. **Transparent Overlay** - See app behind popup
7. **Multiple Close Methods** - Esc, Q, button, click-outside

### Files Modified

**Core Files:**
- `CLI/shared/base_screen.py` - Universal error handler
- `CLI/shared/datatable_info_popup.py` - Universal popup component

**Screen Files (Error Handlers):**
- `CLI/monitor/screen.py` - 5 errors + CVE popup + 4 DataTables
- `CLI/infra/screen.py` - 1 error
- `CLI/setup/screen.py` - 3 errors
- `CLI/teardown/screen.py` - 3 errors

**Total**: 11 error notifications + 4 DataTables + 1 Security popup = **16 popup implementations**

### Testing Checklist

```bash
# Run TUI
python CLI/tui.py

# Test DataTable popups (4 tables)
# 1. Monitor → Click runner row with error → See full error
# 2. Monitor → Click vertex row with error → See full error
# 3. Monitor → Click active run row → See full name/config
# 4. Monitor → Click completed run row → See full name/metrics

# Test Security popup
# 5. Monitor → Press 'v' → See all CVEs for 3 images

# Test Error popups (11 locations)
# 6. Trigger error on any screen → Press 'e' → See full error
# 7. Monitor: Trigger refresh error → Press 'e'
# 8. Setup: Trigger setup error → Press 'e'
# 9. Teardown: Trigger teardown error → Press 'e'
# 10. Infra: Trigger infra error → Press 'e'
```

---

**Last Updated**: 2025-11-09
**Maintainer**: Claude + djwar42@gmail.com

---

## Textual TUI Performance Insights (From Spinner Debugging)

**Date**: 2025-11-09  
**Source**: Debugging AnimatedSpinner freeze + Textual official documentation

### Critical Discovery: File I/O in render() Kills Performance! 🚨

**Problem**: Spinner animation froze after 2-3 frames during loading overlay

**Root Cause**: `render()` method had file I/O for debug logging:
```python
def render(self):
    with open("ARR_COC/Training/logs/spinner_debug.log", "a") as f:  # ❌ BLOCKS EVERY FRAME!
        f.write(f"Rendering char={char}\n")
    return f" {char}"
```

**Fix**: Remove ALL I/O from render():
```python
def render(self):
    return f" {get_next_spinner_char()}"  # ✅ Pure and fast!
```

### Key Textual Best Practices

#### 1. render() MUST Be Pure and Fast

From Textual docs: "Things I learned while building Textual's TextArea"

- `render()` is called in a **hot loop** - ANY overhead accumulates!
- File I/O in render() causes 100-200ms blocks EVERY frame
- Even creating NamedTuples in render() is too slow (use plain tuples)
- **RULE**: render() should only return a string or renderable - NO side effects!

#### 2. Use Built-in Widgets When Available

Textual provides `LoadingIndicator` widget - don't reinvent it!

```python
from textual.widgets import LoadingIndicator

class BaseScreen(Screen):
    def compose_base_overlay(self):
        with Container(id="loading-overlay"):
            yield LoadingIndicator()  # ✅ Built-in, tested, optimized!
            yield Static(self.loading_message or "Loading...")
```

#### 3. Background Work with run_worker()

From Textual docs: "Guide - Workers"

- Use `run_worker()` for long-running async tasks
- Don't block the main event loop!
- Workers run on background threads automatically

```python
def on_mount(self):
    self.run_worker(self.fetch_data_background)

async def fetch_data_background(self):
    data = await slow_api_call()
    self.call_from_thread(self.update_ui, data)
```

#### 4. Reactive Attributes for State

Use Textual's reactive system instead of manual updates:

```python
from textual.reactive import reactive

class MyWidget(Static):
    counter = reactive(0)  # Auto-updates UI when changed!
    
    def watch_counter(self, old, new):
        self.update(f"Count: {new}")  # Called automatically on change
```

#### 5. Timer Accuracy Varies by Platform

From Textual docs: "A better asyncio sleep for Windows to fix animation"

- macOS/Linux: asyncio.sleep() accurate to ~1%
- Windows: 15ms granularity! (historical timer limitation)
- Use Textual's built-in timers (`set_interval()`) - they handle platform differences

### Profiling with pyinstrument

From Textual docs: Profiling found **97% performance improvement** by fixing two issues:
1. Creating Query objects on each key press (move to constructor)
2. NamedTuple creation in hot loop (use plain tuples)

**Lesson**: 30 minutes of profiling = massive performance gains!

```bash
# Profile your TUI
pyinstrument -r html your_tui.py
# Opens beautiful HTML report showing bottlenecks
```

### What We Learned

**✅ DO**:
- Keep render() pure (no I/O, no expensive ops)
- Use built-in widgets (LoadingIndicator, etc.)
- Run heavy work with run_worker()
- Use reactive attributes for state
- Profile with pyinstrument before optimizing

**❌ DON'T**:
- Do file I/O in render()!
- Create objects in render() (NamedTuples, etc.)
- Block the main event loop
- Reinvent built-in widgets
- Assume timers are 100% accurate

### Files Modified During Investigation

- `CLI/shared/animated_spinner.py` - Removed file I/O from render()
- `CLI/setup/screen.py` - Added/removed debug logging
- `CLI/shared/base_screen.py` - Added/removed debug logging
- `CLAUDE.md (project root)` - This section!

### Next Steps for Spinner Issue

The spinner STILL stops after 2-3 frames even without file I/O! This suggests:
1. `set_interval()` timer is being paused/cancelled during loading
2. BaseScreen's loading overlay interferes with child timers
3. **SOLUTION**: Use Textual's built-in `LoadingIndicator` widget instead!

**Recommended**: Replace AnimatedSpinner with LoadingIndicator (already tested and working!)

---

**Last Updated**: 2025-11-09
**Status**: File I/O removed, but deeper timer issue remains
**Recommendation**: Use Textual's LoadingIndicator widget

### 🚨 DOUBLE-CRITICAL: Modifying Files AFTER Adding to Manifest

**If you add a file to `.image-manifest` but the file already exists, you MUST modify the file content to trigger a rebuild!**

**Why:** Adding a file to the manifest doesn't change the file's hash (content unchanged = same hash = no rebuild detected).

**Example:**
```bash
# File already exists: sitecustomize.py (committed)
# Add it to manifest:
echo "sitecustomize.py" >> Stack/arr-vertex-launcher/.image-manifest
git commit -m "Add sitecustomize.py to manifest"
python CLI/cli.py launch  # ❌ NO REBUILD! File content unchanged!

# ✅ FIX: Modify the file AFTER adding to manifest
echo "" >> Stack/arr-vertex-launcher/sitecustomize.py
echo "# Version 1.1 - Force rebuild" >> Stack/arr-vertex-launcher/sitecustomize.py
git commit -m "Force rebuild: Modify sitecustomize.py content"
python CLI/cli.py launch  # ✅ REBUILDS! Hash changed!
```

**Quick Force-Rebuild Trick:**
```bash
# Add a version comment to any Python file in the manifest:
echo "" >> Stack/[image]/file.py
echo "# Version X.Y - Rebuild trigger" >> Stack/[image]/file.py
git commit -m "Force rebuild: Version bump"
python CLI/cli.py launch  # Rebuilds!
```

---

## 🌶️ PAPRIKA CODE RESTORATION & SPICY LINE REORGANIZATION

**PAPRIKA** = Precision And Patience Restore Inspection of Code Architecture

Systematic methodology for restoring broken UX and reorganizing code with full inspection.

### When to Use PAPRIKA

**Trigger situations:**
- Code works but UX is broken (empty states missing, tables not refreshing, etc.)
- Major refactor left things incomplete
- Need to systematically verify and fix issues
- Code needs logical reorganization + full inspection

**Key principle:** Work in phases, fix completely, verify thoroughly, document everything.

---

### PAPRIKA Phase Structure

#### Phase 1: Critical Fixes
**Goal:** Restore essential UX immediately

**Process:**
1. Identify critical broken UX (empty states, refresh calls, schema issues)
2. Fix ONE issue at a time
3. Test immediately after each fix
4. Git commit per fix with "PAPRIKA Phase 1:" prefix
5. Document what was restored

**Example fixes:**
- Add empty state handling
- Add `.refresh()` calls after table updates  
- Fix table schemas (missing columns)
- Restore error messages

#### Phase 2: UX Polish
**Goal:** Complete the user experience

**Process:**
1. Identify UX gaps (dividers, row_data, MAX limits, complete info)
2. Create DRY helpers to eliminate duplication
3. Add polish features (dividers, complete row data, limits)
4. Test each addition
5. Git commit per feature group

**Example polish:**
- Table dividers (active vs completed)
- Complete row_data for popups
- MAX_* limits with \_extra_items tracking
- DRY helpers (_create_table_divider, _add_empty_state_row)

#### Phase 3: Advanced Features (Optional)
**Goal:** Add new capabilities discovered during restoration

**Process:**
1. Universal cache systems
2. Adaptive monitoring
3. Performance optimizations
4. Advanced UX features

---

### 🌶️ SPICY LINE TECHNIQUE

**SPICY LINE** = Systematic Pick-and-place Inspection with Clean Yielding LINE

Reorganize code with full gestalt inspection, moving functions above a "SPICY LINE" marker one by one.

#### When to Use SPICY LINE

**After PAPRIKA Phase 2**, when:
- Code works but is disorganized
- Functions are scattered, not grouped logically
- Need full code review with fixes
- Want clean logical structure

#### SPICY LINE Process

**Step 1: Add SPICY LINE Marker**

```python
# ═══════════════════════════════════════════════════════════════════════════════
# 🌶️🌶️🌶️ SPICY LINE - PAPRIKA REORGANIZATION TECHNIQUE 🌶️🌶️🌶️
# ═══════════════════════════════════════════════════════════════════════════════
# ABOVE THIS LINE: Fully inspected, reorganized, LOGICAL GROUPS with section headers
# BELOW THIS LINE: Awaiting systematic inspection + reorganization
#
# SPICY LINE PROCESS:
# 1. Pick function from logical group (see plan below)
# 2. Full gestalt inspection (logic, syntax, naming, structure)
# 3. Fix any issues found
# 4. Move above SPICY LINE into correct LOGICAL SECTION (add section header if new)
# 5. Add inspection note after function
# 6. Repeat until all functions above in logical order
# 7. Remove SPICY LINE when complete
#
# LOGICAL ORGANIZATION (top → bottom):
# ├─ INITIALIZATION (__init__)
# ├─ UI CONSTRUCTION (compose)
# ├─ LIFECYCLE (initialize_content, finish_loading, on_screen_resume, on_unmount)
# ├─ CACHE SYSTEM (_should_fetch_table, _get_cached_data, _update_table_cache)
# ├─ DRY HELPERS (_create_table_divider, _add_empty_state_row)
# ├─ REGION MONITORING (_get_target_regions, _update_hot_regions)
# ├─ SPINNER SYSTEM (_start_spinner, _stop_spinner, _update_spinners)
# ├─ REFRESH ORCHESTRATION (_populate_initial_tables, _universal_refresh_table, etc)
# ├─ TABLE FETCH FUNCTIONS (all 5 _fetch_and_update_*_table methods)
# └─ EVENT HANDLERS (on_*, action_*)
# ═══════════════════════════════════════════════════════════════════════════════
```

**Step 2: Define Logical Sections**

Plan sections top-to-bottom in execution order:
1. Initialization
2. UI Construction
3. Lifecycle hooks
4. Core systems (cache, helpers, monitoring)
5. Orchestration (refresh, timers)
6. Data fetching
7. Event handlers

**Step 3: Move Functions One-by-One**

🚨 **CRITICAL: HOW TO MOVE FUNCTIONS SAFELY**

**❌ NEVER use sed/bash for moving functions!**
- Sed is error-prone and causes orphaned code
- Hard to verify correctness
- Led to 2025-11-18 corruption (syntax errors, orphaned fragments)

**✅ THE ONLY SAFE METHOD: Read + Edit ONE FUNCTION AT A TIME** ⭐

**For SPICY LINE moves:**
```
1. Read function to move (find exact line range)
2. Copy function text exactly
3. Edit: Add function to new location (above SPICY LINE)
   - Add section header if new section
   - Add inspection note above function
4. Edit: Delete function from old location (below SPICY LINE)
5. Verify with Read (spot-check both locations)
6. python -m py_compile screen.py (MANDATORY!)
7. Git commit with clear message
8. Move to next function
```

**Why one-by-one is BEST:**
- ✅ Safe: Each move verified before next
- ✅ Clear: Git history shows exactly what moved
- ✅ Recoverable: Can revert individual moves
- ✅ No corruption: String matching prevents orphans
- ✅ Systematic: Forces thorough review of each function

**Why NOT in-memory/batch:**
- ❌ Risky: One error corrupts entire file
- ❌ Hard to debug: What broke?
- ❌ No verification: Can't test partial state
- ❌ Lost work: Have to revert everything if fails

**Speed doesn't matter - CORRECTNESS does!**
- Moving 30 functions one-by-one takes longer
- But ZERO corruption vs hours fixing mistakes
- Slow and steady wins the race!

**🌶️ FINAL SPICE REFINEMENT WITH SPICY LINE** 🌶️

**ONE FULL ALICE RABBIT HOLE PER GESTALT FOR EVERY FUNCTION ONE BY ONE USING THE SPICY LINE** 🐰

---

**The Process:**

**For EACH function ONE BY ONE:**

**1. PICK** → Choose next function **below SPICY LINE**

**2. READ** → Full function (Read tool with offset)

**3. GESTALT** 🔍 → **ONE FULL ALICE RABBIT HOLE!** 🐰
   - Structure, naming, docs, logic checks
   - **FOLLOW THE FLOW WHEREVER IT TAKES YOU:**
     - Calls other functions? → **GO CHECK THEM!** (recursive!)
     - Uses imports? → **GO CHECK MODULE TOP!**
     - References constants? → **GO CHECK DEFINED!**
   - **FULL RECURSIVE** - Follow ALL dependencies down the hole!

**4. FIX** 🔧 → Fix wherever flow takes you, verify syntax, pop back up ALL FIXED!
   - Issue in dependency? → **FIX THAT FIRST!** (go down the hole!)
   - Example: `import time` inside function → Move to module top
   - Keep going down until ALL FIXED
   - **VERIFY syntax after each fix:** `python -m py_compile screen.py`
   - **Pop back up** with everything fixed AND VERIFIED

**5. VERIFY** ✅ → Final syntax check before placement (MANDATORY!)

**6. PLACE** 📍 → **Above SPICY LINE** in neat logical order
   - Add to correct section (💾 Cache, 🌍 Region, ⚡ Spinner, etc.)
   - Delete from **below SPICY LINE**
   - **VERIFY syntax after placement:** `python -m py_compile screen.py` (MANDATORY!)

**7. COMMIT** 💾 → `git commit -m "Move: [func] → [SECTION]"`

**8. REPEAT** 🔁 → Next function! ONE BY ONE!

---

## 🎉 MANDATORY BUG CELEBRATION! 🎉

**When you find a bug during ALICE RABBIT HOLE inspection:**

**YOU MUST celebrate with the 2-line ASCII DANCE:**

```
FUCK OFF BUG!!! \○/ ✨ [bug description] → FIXED! 🔥
★ ☆ ✧ * BOOM ✦ CRASH ✧ POW ✦ [BUG TYPE] ANNIHILATED ✧ ★ ☆ * ✦
```

**Examples:**
- Bug found: `import time` inside function
- Celebration:
```
FUCK OFF BUG!!! \○/ ✨ import time local → FIXED! 🔥
★ ☆ ✧ * BOOM ✦ CRASH ✧ POW ✦ LOCAL IMPORT ANNIHILATED ✧ ★ ☆ * ✦
```

**ALL bugs MUST be celebrated!** No bug escapes without ASCII DANCE! 🌶️🐰

---

**🐰 THE SPICY LINE:**
Visual marker separating inspected code (ABOVE) from uninspected code (BELOW). Functions move from below → above, ONE BY ONE, each with ONE FULL ALICE RABBIT HOLE gestalt inspection.

**The spice must LOW!** 🌶️

**Step 4: Systematic Progress Tracking**

Create `spicy_line_plan.md` with:
- Sections list with function counts
- Progress log (timestamp, what moved, issues fixed)
- Commit hashes
- Percentage complete

**Step 5: Final Removal**

When ALL functions above SPICY LINE:
- Remove SPICY LINE marker
- Celebrate completion!

---

### 🔲 SQUARING THE PAPRIKA SQUARE WITH THE SPICY LINE CIRCLE

**CRITICAL FINAL STEP** - Complete review and insights documentation.

This concludes PAPRIKA + SPICY LINE with comprehensive review and discussion preparation.

#### Process

**1. Final Gestalt Review (Section by Section)**

Review EVERY section above SPICY LINE:

For small sections (< 100 lines):
- Read full section in one view
- Check for any remaining issues
- Verify all fixes are correct
- **🚨 FIX ANYTHING FOUND IMMEDIATELY!**

For large sections (> 100 lines):
- Split into 2-3 parts
- Review each part separately
- Same thoroughness
- **🚨 FIX ANYTHING FOUND IMMEDIATELY!**

**🚨 CRITICAL RULE: FIX BEFORE REPORT!**

**DO NOT create the SQUARING report until ALL bugs found are FIXED!**

If you find bugs during review:
1. ✅ **STOP REVIEWING** - Fix the bug NOW!
2. ✅ **IMPLEMENT THE FIX** - Write the code, test it works
3. ✅ **GIT COMMIT THE FIX** - Separate commit per bug fixed
4. ✅ **THEN CONTINUE REVIEW** - Resume next section

**Example Bug Fix Flow:**
```
Review section → Find bug → STOP!
↓
Fix bug immediately (Edit/Write tools)
↓
Git commit: "Fix: [bug description]"
↓
Resume review of next section
```

**NEVER:**
- ❌ Document bug and continue reviewing
- ❌ "I'll fix it later" (NO! Fix NOW!)
- ❌ Create report with unfixed bugs listed
- ❌ Leave bugs for user to fix

**The SQUARING report should document:**
- ✅ Bugs that WERE FIXED (past tense!)
- ✅ When and how they were fixed
- ✅ NOT bugs waiting to be fixed!

---

### 🌶️ FINAL SPICE REFINEMENT

**THE SPICE MUST LOW → FIX IT NOW!**

*When you find a bug during review, the spice (complexity/debt) is HIGH!*
*Fixing immediately keeps the spice LOW and production-ready!*

**Motto**: "Don't document problems, FIX problems!"

---

**2. Create Insights Report (AFTER all bugs fixed!)**

Document in `SQUARING_THE_PAPRIKA_SQUARE.md`:

**What We Discovered:**
- Issues we didn't expect to find
- Patterns that emerged
- Design decisions that worked/didn't work

**What Was Hard:**
- Challenges encountered
- Why they were difficult
- How we overcame them

**What Worked Well:**
- Techniques that were effective
- Tools that helped
- Process improvements

**What We Left Out:**
- Features not implemented
- Issues deferred
- Known limitations

**What We Added:**
- Unexpected improvements
- Bonus features
- Better-than-planned solutions

**Discussion Points:**
- Questions for user
- Testing priorities
- Next steps
- Refinements needed

**3. Update Plan Documents**

Append final review notes to all plan docs:
- What was reviewed
- Issues found and fixed
- Final state verification

**4. Prepare for Testing Discussion**

The insights report becomes the launch point for:
- User discussion of design decisions
- Testing priorities
- Debug logging strategy
- Refinement roadmap

---

### Example SPICY LINE Session

```
Session Start: 18:45
- Added SPICY LINE marker
- Defined 10 logical sections
- Created TODO list (12 tasks)

Move 1/11: __init__() → ⚙️ INITIALIZATION (18:50)
- 121 lines moved
- Fixed: Orphan comments removed (lines 353-354)
- Status: Perfect, state vars organized
- Commit: 51963d4

Move 2/11: compose() → 🎨 UI CONSTRUCTION (19:00)
- 116 lines moved
- Verified: 5 tables, 5 checkboxes, 5 spinners
- Issues: None
- Commit: 5de53e3

... [continues for all sections] ...

Move 9/11: Table fetch functions → 📊 TABLE FETCH (19:28)
- 502 lines moved (MASSIVE!)
- All use cache, MAX limits, empty states
- Issues: None
- Commit: cc8c63f

Progress: [█████████░░] 82% (9/11 complete)
```

---

### Git Commit Format

**PAPRIKA commits:**
```
PAPRIKA Phase N: [What was fixed]

[Why it was needed]
[What it does now]

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

**SPICY LINE commits:**
```
🌶️ SPICY MOVE N/M: [Function/section] → [Section emoji] [Section name]

✅ Full gestalt inspection complete:
- [Function 1] (X lines) - [What verified]
- [Function 2] (Y lines) - [What fixed]
- Issues fixed: [None/list]

Progress: [████░░░░░░] XX% (N/M)

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

### Testing After PAPRIKA + SPICY LINE

**Before declaring complete:**

1. **Run the code** - Does it work?
2. **Test ALL restored UX** - Empty states, refreshes, etc.
3. **Review logs** - Any errors?
4. **User testing** - Get feedback
5. **Discussion** - Use insights report to discuss next steps

**Common post-PAPRIKA tasks:**
- Performance testing
- Edge case handling
- Error scenario verification
- User workflow testing
- Documentation updates

---

### PAPRIKA + SPICY LINE Benefits

✅ **Systematic** - Nothing missed, everything reviewed
✅ **Documented** - Full commit history, insights captured
✅ **Quality** - Full gestalt inspection catches issues
✅ **Organized** - Logical structure, easy to navigate
✅ **Discussion-ready** - Insights report launches productive conversations
✅ **Maintainable** - Clean code, clear sections, good comments

**Result:** Production-ready code with comprehensive understanding and clear next steps!

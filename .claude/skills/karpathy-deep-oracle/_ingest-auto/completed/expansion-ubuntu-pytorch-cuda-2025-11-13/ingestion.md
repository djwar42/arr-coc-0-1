# Oracle Knowledge Expansion: Ubuntu-Specific PyTorch/CUDA/NVIDIA Configuration

**Topic**: Ubuntu-specific PyTorch, CUDA, and NVIDIA configuration (focus on Ubuntu 22.04)
**Date**: 2025-11-13
**Source**: Official Ubuntu documentation, NVIDIA Ubuntu guides, PyTorch Ubuntu installation
**Target Folder**: `cuda/`

---

## Overview

This expansion focuses SPECIFICALLY on Ubuntu's role in PyTorch/CUDA/NVIDIA compilation and configuration.

**CRITICAL RULE FOR RUNNERS**:
- ONLY include content that DIRECTLY connects Ubuntu to PyTorch, CUDA, or NVIDIA
- If a topic doesn't specifically relate Ubuntu to these technologies, SKIP IT
- If there's no relevant Ubuntu-specific content found, explicitly state this in the file
- General Ubuntu tutorials NOT connected to PyTorch/CUDA/NVIDIA = NOT WANTED

---

## PART 1: Create cuda/18-ubuntu-pytorch-cuda-setup.md (400 lines)

- [✓] PART 1: Create cuda/18-ubuntu-pytorch-cuda-setup.md (Completed 2025-11-13)

**Ubuntu-Specific Focus**: Ubuntu 22.04 LTS + PyTorch + CUDA + NVIDIA drivers

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Ubuntu-related files
- [ ] Grep for "ubuntu" in cuda/ folder
- [ ] Identify gaps: Ubuntu-specific PyTorch/CUDA setup NOT covered

**Step 1: Ubuntu + PyTorch + CUDA + NVIDIA Research**
- [ ] Search: "ubuntu 22.04 pytorch cuda nvidia driver installation official"
- [ ] Search: "ubuntu apt cuda toolkit pytorch compilation dependencies"
- [ ] Search: "ubuntu 22.04 nvidia driver pytorch gpu support setup"
- [ ] Search: "ubuntu specific cuda environment variables pytorch"
- [ ] Scrape Ubuntu-specific guides ONLY (skip generic Linux guides)

**Step 2: Extract Ubuntu-Specific Content**
Research Focus (MUST be Ubuntu-specific):
- Ubuntu 22.04 NVIDIA driver installation for PyTorch (apt vs runfile)
- Ubuntu apt packages for CUDA toolkit + PyTorch dependencies
- Ubuntu-specific environment variables (/etc/environment, ~/.bashrc patterns)
- Ubuntu system libraries required for PyTorch compilation (libc, gcc versions)
- Ubuntu kernel compatibility with NVIDIA drivers
- Ubuntu-specific GPU verification (nvidia-smi, pytorch GPU detection)

**REJECT if found**:
- Generic Linux instructions (not Ubuntu-specific)
- Ubuntu content NOT related to PyTorch/CUDA/NVIDIA
- General Ubuntu tutorials

**Step 3: Write Knowledge File**
- [ ] Create cuda/18-ubuntu-pytorch-cuda-setup.md (~400 lines)
- [ ] Section 1: Ubuntu 22.04 NVIDIA Driver Installation (~120 lines)
      - apt vs runfile (Ubuntu-specific considerations)
      - ubuntu-drivers command usage
      - Ubuntu secure boot implications
      Cite: Ubuntu + NVIDIA official docs
- [ ] Section 2: Ubuntu CUDA Toolkit Setup (~120 lines)
      - apt repository configuration (Ubuntu 22.04)
      - cuda-toolkit-12-4 package installation
      - Ubuntu-specific PATH/LD_LIBRARY_PATH
      Cite: NVIDIA Ubuntu guides
- [ ] Section 3: PyTorch Compilation on Ubuntu (~80 lines)
      - Ubuntu system dependencies (build-essential, etc.)
      - Ubuntu-specific compiler versions
      - Ubuntu library compatibility
      Cite: PyTorch Ubuntu docs
- [ ] Section 4: Ubuntu GPU Verification (~80 lines)
      - nvidia-smi on Ubuntu
      - PyTorch GPU detection verification
      - Ubuntu-specific troubleshooting
      Cite: Ubuntu + PyTorch docs

**IMPORTANT**: If ANY section lacks Ubuntu-specific PyTorch/CUDA/NVIDIA content, write:
```markdown
## Section X: [Topic]

**Ubuntu-Specific Content**: None found. Research indicates this topic is generic across Linux distributions and does not have Ubuntu-specific considerations for PyTorch/CUDA/NVIDIA setup.
```

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-ubuntu-pytorch-cuda-2025-11-13.md
- [ ] Include: Runner (PART 1), Timestamp, Status
- [ ] List knowledge file created with line count
- [ ] List sources used
- [ ] Note if content was REJECTED (not Ubuntu-specific)

---

## Completion Criteria

PART 1 must:
- [ ] Create knowledge file in cuda/ folder
- [ ] Include ONLY Ubuntu + PyTorch + CUDA + NVIDIA content
- [ ] Explicitly state if sections lack Ubuntu-specific content
- [ ] Create KNOWLEDGE DROP file
- [ ] Mark checkbox [✓] when complete

**Expected Output:**
- 1 knowledge file (~400 lines OR explicit "no content" sections)
- 1 KNOWLEDGE DROP file
- Ubuntu-specific PyTorch/CUDA/NVIDIA knowledge (or explicit statement of absence)

---

**Oracle**: Launch 1 runner, enforce Ubuntu+PyTorch+CUDA+NVIDIA requirement strictly, finalize INDEX.md and git commit.

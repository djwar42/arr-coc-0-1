# Oracle Knowledge Expansion: Ubuntu PyTorch/CUDA Troubleshooting

**Topic**: Ubuntu-specific PyTorch/CUDA troubleshooting, package management, and system optimization
**Date**: 2025-11-13
**Source**: Ubuntu community forums, NVIDIA Ubuntu guides, PyTorch Ubuntu debugging
**Target Folder**: `cuda/`

---

## Overview

This expansion focuses on Ubuntu-specific troubleshooting for PyTorch/CUDA issues.

**CRITICAL RULE FOR RUNNER**:
- ONLY include content that DIRECTLY connects Ubuntu to PyTorch, CUDA, or NVIDIA troubleshooting
- Focus on Ubuntu 22.04 LTS (and mention Ubuntu 20.04/24.04 where relevant)
- If a topic doesn't specifically relate Ubuntu troubleshooting to PyTorch/CUDA/NVIDIA, SKIP IT
- General Linux troubleshooting NOT specific to Ubuntu = NOT WANTED

---

## PART 1: Create cuda/19-ubuntu-pytorch-cuda-troubleshooting.md (400 lines)

- [✓] PART 1: Create cuda/19-ubuntu-pytorch-cuda-troubleshooting.md (Completed 2025-11-13 16:45)

**Ubuntu-Specific Focus**: Ubuntu 22.04 PyTorch/CUDA/NVIDIA troubleshooting

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/18-ubuntu-pytorch-cuda-setup.md (existing Ubuntu setup)
- [ ] Read cuda/09-runtime-errors-debugging-expert.md (general CUDA errors)
- [ ] Identify gaps: Ubuntu-specific troubleshooting NOT covered

**Step 1: Ubuntu Troubleshooting Research**
- [ ] Search: "ubuntu 22.04 pytorch cuda not detected troubleshooting"
- [ ] Search: "ubuntu nvidia driver conflicts pytorch black screen"
- [ ] Search: "ubuntu apt cuda packages broken dependencies fix"
- [ ] Search: "ubuntu secure boot nvidia driver signing issues"
- [ ] Search: "ubuntu python3 pytorch gpu not available debug"
- [ ] Scrape Ubuntu-specific troubleshooting guides ONLY

**Step 2: Extract Ubuntu-Specific Troubleshooting**
Research Focus (MUST be Ubuntu-specific):
- Ubuntu Secure Boot + NVIDIA driver conflicts (MOK enrollment failures)
- Ubuntu apt package conflicts (cuda-toolkit vs nvidia-cuda-toolkit)
- Ubuntu Python environment issues (python3-pip vs conda on Ubuntu)
- Ubuntu kernel updates breaking NVIDIA drivers (DKMS rebuild issues)
- Ubuntu-specific error messages (apt, dpkg, update-initramfs)
- Ubuntu system logs for GPU debugging (journalctl, dmesg on Ubuntu)
- Ubuntu package version mismatches (CUDA 12.4 + driver 545 on Ubuntu 22.04)

**REJECT if found**:
- Generic Linux troubleshooting (not Ubuntu-specific)
- PyTorch errors unrelated to Ubuntu
- General CUDA debugging (already covered)

**Step 3: Write Knowledge File**
- [ ] Create cuda/19-ubuntu-pytorch-cuda-troubleshooting.md (~400 lines)
- [ ] Section 1: Ubuntu Driver Issues (~120 lines)
      - Secure Boot MOK failures (Ubuntu-specific)
      - DKMS rebuild after kernel updates
      - ubuntu-drivers conflicts with manual installs
      - Black screen after NVIDIA driver install (Ubuntu)
      Cite: Ubuntu forums, NVIDIA Ubuntu docs
- [ ] Section 2: Ubuntu Package Conflicts (~120 lines)
      - cuda-toolkit vs nvidia-cuda-toolkit (Ubuntu apt)
      - Broken dependencies (apt --fix-broken install)
      - PPA conflicts on Ubuntu
      - Multi-CUDA version conflicts
      Cite: Ubuntu community, Ask Ubuntu
- [ ] Section 3: Ubuntu Python/PyTorch Issues (~80 lines)
      - python3-pip vs conda on Ubuntu
      - torch.cuda.is_available() False (Ubuntu-specific causes)
      - Ubuntu system Python vs user Python
      - LD_LIBRARY_PATH issues on Ubuntu
      Cite: PyTorch forums Ubuntu threads
- [ ] Section 4: Ubuntu Debugging Tools (~80 lines)
      - journalctl for NVIDIA errors on Ubuntu
      - dmesg for GPU kernel messages
      - apt logs for package issues
      - Ubuntu-specific verification scripts
      Cite: Ubuntu documentation

**IMPORTANT**: If ANY section lacks Ubuntu-specific content, write:
```markdown
## Section X: [Topic]

**Ubuntu-Specific Content**: None found beyond general Linux troubleshooting already covered in cuda/09-runtime-errors-debugging-expert.md
```

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-ubuntu-troubleshooting-2025-11-13.md
- [ ] Include: Runner (PART 1), Timestamp, Status
- [ ] List knowledge file created with line count
- [ ] List Ubuntu-specific sources
- [ ] Note any rejected generic Linux content

---

## Completion Criteria

PART 1 must:
- [ ] Create knowledge file in cuda/ folder
- [ ] Include ONLY Ubuntu + PyTorch + CUDA + NVIDIA troubleshooting
- [ ] Explicitly state if sections lack Ubuntu-specific content
- [ ] Create KNOWLEDGE DROP file
- [ ] Mark checkbox [✓] when complete

**Expected Output:**
- 1 knowledge file (~400 lines with Ubuntu-specific troubleshooting)
- 1 KNOWLEDGE DROP file
- Ubuntu-focused PyTorch/CUDA debugging knowledge

---

**Oracle**: Launch 1 runner, enforce strict Ubuntu+PyTorch+CUDA+NVIDIA troubleshooting requirement, finalize INDEX.md and git commit.

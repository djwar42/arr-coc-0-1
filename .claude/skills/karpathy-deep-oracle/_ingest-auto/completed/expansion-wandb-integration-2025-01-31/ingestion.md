# Oracle Knowledge Expansion: Weights & Biases Integration

**Date:** 2025-01-31
**Oracle:** karpathy-deep-oracle
**Topic:** Weights & Biases integration for ML training (HuggingFace, PyTorch, Gradio)
**Context:** Supporting ARR-COC validation needs from VALIDATION-FOR-PLATONIC-CODING-CODEBASES.md

---

## Expansion Plan

This expansion will create comprehensive W&B knowledge focused on:
1. HuggingFace Trainer + W&B integration
2. Manual PyTorch + W&B logging
3. Vision-language model (VLM) specific metrics
4. Gradio dashboard embedding
5. Checkpoint management strategies

**Target Audience:** ML engineers building validation interfaces for experimental models

**Knowledge Acquisition Strategy:**
- Web research (W&B official docs, HuggingFace integration guides)
- Best practices from Karpathy's teaching style (simple, practical, honest)
- VLM-specific considerations (token budgets, relevance metrics)

---

## PART 1: Create gradio/10-wandb-integration-basics.md (300 lines)

- [✓] PART 1: Create gradio/10-wandb-integration-basics.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [ ] Search: "Weights and Biases quickstart 2024 2025"
- [ ] Search: "wandb.init() wandb.log() best practices"
- [ ] Search: "wandb project setup tutorial"
- [ ] Scrape top 2-3 results for key concepts

**Step 2: Extract Key Concepts**
- [ ] Basic setup (wandb.init, wandb.config, wandb.log)
- [ ] Project organization (projects, runs, tags)
- [ ] Metric logging patterns
- [ ] Run management (wandb.finish(), resuming)

**Step 3: Write Knowledge File**
- [ ] Create gradio/10-wandb-integration-basics.md
- [ ] Section 1: W&B Quickstart (~80 lines)
      - Installation, authentication
      - Basic wandb.init() and wandb.log()
      - Project structure
      Cite: Web research results
- [ ] Section 2: Core Concepts (~100 lines)
      - Projects, runs, artifacts
      - Config vs logs vs summary
      - Tags and filtering
      Cite: Web research results
- [ ] Section 3: Common Patterns (~120 lines)
      - Training loop integration
      - Epoch vs step logging
      - wandb.watch() for gradients
      - Best practices
      Cite: Web research results

**Step 4: Complete**
- [ ] PART 1 COMPLETE ✅

---

## PART 2: Create gradio/11-wandb-huggingface-trainer.md (350 lines)

- [✓] PART 2: Create gradio/11-wandb-huggingface-trainer.md (Completed 2025-01-31)

**Step 1: Web Research**
- [ ] Search: "HuggingFace Trainer Weights and Biases integration"
- [ ] Search: "transformers TrainingArguments report_to wandb"
- [ ] Search: "huggingface wandb callback custom metrics"
- [ ] Scrape official HuggingFace docs + W&B integration guide

**Step 2: Extract Key Concepts**
- [ ] TrainingArguments (report_to="wandb")
- [ ] Automatic logging (loss, learning rate, metrics)
- [ ] Custom callbacks for W&B
- [ ] Integration with evaluation metrics

**Step 3: Write Knowledge File**
- [ ] Create gradio/11-wandb-huggingface-trainer.md
- [ ] Section 1: Basic Integration (~100 lines)
      - TrainingArguments setup
      - report_to parameter
      - Automatic metric logging
      Cite: Web research results
- [ ] Section 2: Custom Callbacks (~120 lines)
      - WandbCallback customization
      - Logging custom metrics
      - Logging images/visualizations
      Cite: Web research results
- [ ] Section 3: Advanced Patterns (~130 lines)
      - Multi-run experiments
      - Hyperparameter sweeps
      - Model artifact tracking
      - Integration with datasets library
      Cite: Web research results

**Step 4: Complete**
- [ ] PART 2 COMPLETE ✅

---

## PART 3: Create gradio/12-wandb-pytorch-manual.md (300 lines)

- [✓] PART 3: Create gradio/12-wandb-pytorch-manual.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [ ] Search: "PyTorch manual wandb logging training loop"
- [ ] Search: "wandb.log() step parameter best practices"
- [ ] Search: "wandb custom charts visualization"
- [ ] Scrape W&B PyTorch integration docs

**Step 2: Extract Key Concepts**
- [ ] Manual training loop integration
- [ ] Step-based vs epoch-based logging
- [ ] Custom visualizations
- [ ] Gradient tracking

**Step 3: Write Knowledge File**
- [ ] Create gradio/12-wandb-pytorch-manual.md
- [ ] Section 1: Training Loop Integration (~120 lines)
      - Basic logging pattern
      - Step vs epoch tracking
      - Learning rate schedules
      - Gradient clipping
      Cite: Web research results
- [ ] Section 2: Advanced Logging (~100 lines)
      - wandb.watch() for model weights
      - Custom metrics computation
      - Multiple validation sets
      - System metrics (GPU, memory)
      Cite: Web research results
- [ ] Section 3: Visualization (~80 lines)
      - Custom charts and plots
      - Image logging (wandb.Image)
      - Tables for structured data
      - Media logging (audio, video)
      Cite: Web research results

**Step 4: Complete**
- [ ] PART 3 COMPLETE ✅

---

## PART 4: Create gradio/13-wandb-vlm-metrics.md (280 lines)

- [ ] PART 4: Create gradio/13-wandb-vlm-metrics.md

**Step 1: Web Research**
- [ ] Search: "vision language model training metrics wandb"
- [ ] Search: "VQA evaluation metrics logging"
- [ ] Search: "image captioning training visualization wandb"
- [ ] Search: "multimodal model debugging visualization"
- [ ] Scrape VLM training best practices

**Step 2: Extract Key Concepts**
- [ ] VLM-specific metrics (token usage, attention patterns)
- [ ] Multimodal debugging techniques
- [ ] Relevance/compression metrics
- [ ] Visual token allocation tracking

**Step 3: Write Knowledge File**
- [ ] Create gradio/13-wandb-vlm-metrics.md
- [ ] Section 1: VLM Training Metrics (~100 lines)
      - Standard metrics (loss, perplexity, accuracy)
      - Token budget tracking
      - Visual encoder metrics
      - Cross-modal alignment metrics
      Cite: Web research results
- [ ] Section 2: ARR-COC Specific Metrics (~100 lines)
      - Relevance scores (propositional, perspectival, participatory)
      - Token allocation per patch
      - Compression ratios
      - LOD budget distribution
      Cite: ARR-COC validation doc, web research
- [ ] Section 3: Debugging Visualizations (~80 lines)
      - Attention heatmaps
      - Patch selection visualization
      - Token budget histograms
      - Failure case analysis
      Cite: Web research results

**Step 4: Complete**
- [✓] PART 4 COMPLETE ✅ (Completed 2025-01-31 16:45)

---

## PART 5: Create gradio/14-wandb-dashboard-embedding.md (250 lines)

- [✓] PART 5: Create gradio/14-wandb-dashboard-embedding.md (Completed 2025-01-31 14:45)

**Step 1: Web Research**
- [ ] Search: "embed Weights and Biases dashboard iframe"
- [ ] Search: "wandb workspace embedding gradio"
- [ ] Search: "wandb public sharing links reports"
- [ ] Scrape W&B embedding documentation

**Step 2: Extract Key Concepts**
- [ ] Workspace URL structure
- [ ] iframe embedding parameters
- [ ] Public vs private sharing
- [ ] Report creation and embedding

**Step 3: Write Knowledge File**
- [ ] Create gradio/14-wandb-dashboard-embedding.md
- [ ] Section 1: Embedding Basics (~80 lines)
      - Workspace URL structure
      - iframe HTML setup
      - jupyter=true parameter
      - Security considerations
      Cite: Web research results
- [ ] Section 2: Gradio Integration (~100 lines)
      - gr.HTML() for iframe
      - Dynamic workspace URLs
      - Tab-based organization
      - Refresh mechanisms
      Cite: Web research results, ARR-COC validation doc
- [ ] Section 3: Advanced Features (~70 lines)
      - Custom W&B reports
      - Filtering by tags
      - Compare mode
      - Exporting data
      Cite: Web research results

**Step 4: Complete**
- [ ] PART 5 COMPLETE ✅

---

## PART 6: Create gradio/15-wandb-checkpoint-management.md (300 lines)

- [✓] PART 6: Create gradio/15-wandb-checkpoint-management.md (Completed 2025-01-31 16:50)

**Step 1: Web Research**
- [✓] Search: "wandb artifacts model checkpoints"
- [✓] Search: "wandb save checkpoint best model"
- [✓] Search: "wandb checkpoint versioning"
- [✓] Scrape W&B artifacts documentation

**Step 2: Extract Key Concepts**
- [✓] Artifact types (model, dataset, evaluation)
- [✓] Checkpoint versioning strategies
- [✓] Best model tracking
- [✓] Loading checkpoints from W&B

**Step 3: Write Knowledge File**
- [✓] Create gradio/15-wandb-checkpoint-management.md
- [✓] Section 1: W&B Artifacts Basics (~100 lines)
      - Creating artifacts
      - Artifact types and metadata
      - Versioning (v0, v1, latest, best)
      - Linking artifacts to runs
      Cite: Web research results
- [✓] Section 2: Checkpoint Saving Patterns (~120 lines)
      - Save every N steps
      - Save best model only
      - Save with custom metrics
      - Cleanup old checkpoints
      Cite: Web research results
- [✓] Section 3: Checkpoint Loading (~80 lines)
      - Download artifacts
      - Load specific version
      - Integration with HuggingFace Hub
      - Fallback strategies
      Cite: Web research results, ARR-COC validation doc

**Step 4: Complete**
- [✓] PART 6 COMPLETE ✅

---

## PART 7: Create practical-implementation/15-wandb-quick-validation.md (350 lines)

- [✓] PART 7: Create practical-implementation/15-wandb-quick-validation.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [ ] Search: "fast ML model validation techniques"
- [ ] Search: "smoke test machine learning training"
- [ ] Search: "overfit one batch debugging"
- [ ] Scrape Karpathy's debugging guides (if available)

**Step 2: Extract Key Concepts**
- [ ] Smoke test methodology
- [ ] Quick validation on small datasets
- [ ] Overfitting single batch technique
- [ ] Red flags during training

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/15-wandb-quick-validation.md
- [ ] Section 1: Smoke Tests (~120 lines)
      - Forward pass test
      - Backward pass test
      - Overfit 1 batch test
      - Shape validation
      Cite: ARR-COC validation doc, web research
- [ ] Section 2: Quick Validation (~130 lines)
      - 100 examples, 10 epochs pattern
      - W&B integration setup
      - Metrics to track
      - Success criteria
      Cite: ARR-COC validation doc, Karpathy philosophy
- [ ] Section 3: Debugging with W&B (~100 lines)
      - Red flags (NaN, non-decreasing loss)
      - Using W&B for diagnosis
      - Gradient monitoring
      - Learning rate finding
      Cite: Web research results

**Step 4: Complete**
- [ ] PART 7 COMPLETE ✅

---

## PART 8: Create practical-implementation/16-wandb-hyperparameter-sweeps.md (280 lines)

- [✓] PART 8: Create practical-implementation/16-wandb-hyperparameter-sweeps.md (Completed 2025-01-31 16:15)

**Step 1: Web Research**
- [ ] Search: "wandb sweeps tutorial 2024"
- [ ] Search: "hyperparameter optimization wandb bayesian"
- [ ] Search: "wandb sweep yaml configuration"
- [ ] Scrape W&B sweeps documentation

**Step 2: Extract Key Concepts**
- [ ] Sweep configuration (YAML)
- [ ] Search strategies (grid, random, bayesian)
- [ ] Parallel sweep agents
- [ ] Best run selection

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/16-wandb-hyperparameter-sweeps.md
- [ ] Section 1: Sweep Basics (~100 lines)
      - YAML configuration structure
      - Search methods comparison
      - Metric optimization
      - Early termination
      Cite: Web research results
- [ ] Section 2: Running Sweeps (~100 lines)
      - wandb sweep command
      - Agent setup
      - Parallel execution
      - Monitoring progress
      Cite: Web research results
- [ ] Section 3: Analysis (~80 lines)
      - Parallel coordinates plot
      - Importance analysis
      - Best run extraction
      - Integration with training scripts
      Cite: Web research results

**Step 4: Complete**
- [ ] PART 8 COMPLETE ✅

---

## PART 9: Update INDEX.md with new W&B files

- [✓] PART 9: Update INDEX.md (Completed 2025-10-31 22:00)

**Step 1: Read Current INDEX.md**
- [✓] Read .claude/skills/karpathy-deep-oracle/INDEX.md

**Step 2: Add New Entries**
- [✓] Add to gradio/ section:
      - 10-wandb-integration-basics.md ⭐ W&B FOUNDATION
      - 11-wandb-huggingface-trainer.md
      - 12-wandb-pytorch-manual.md
      - 13-wandb-vlm-metrics.md ⭐ VLM SPECIFIC
      - 14-wandb-dashboard-embedding.md
      - 15-wandb-checkpoint-management.md ⭐ CHECKPOINT CRITICAL
- [✓] Add to practical-implementation/ section:
      - 15-wandb-quick-validation.md ⭐ W&B QUICK START
      - 16-wandb-hyperparameter-sweeps.md

**Step 3: Update Key Concepts**
- [✓] Added W&B Integration to Key concepts section

**Step 4: Complete**
- [✓] PART 9 COMPLETE ✅

---

## PART 10: Update SKILL.md "When to Use This Oracle" section

- [✓] PART 10: Update SKILL.md (Completed 2025-01-31)

**Step 1: Read Current SKILL.md**
- [✓] Read .claude/skills/karpathy-deep-oracle/SKILL.md

**Step 2: Add W&B Use Cases**
- [✓] Add to "Gradio Development & Deployment" section:
      - W&B dashboard embedding in Gradio
      - Training monitoring integration
      - Checkpoint management with W&B artifacts
- [✓] Add new section: "Experiment Tracking & Validation"
      - W&B integration patterns
      - Quick validation methodology
      - Hyperparameter sweeps

**Step 3: Complete**
- [✓] PART 10 COMPLETE ✅

---

## Finalization Checklist

- [ ] All 8 knowledge files created in correct folders
- [ ] All files have proper citations
- [ ] INDEX.md updated with new files
- [ ] SKILL.md updated with W&B use cases
- [ ] Cross-references added where appropriate
- [ ] File naming follows numbered prefix convention
- [ ] Git commit with descriptive message
- [ ] Archive to _ingest-auto/completed/

---

## Expected Outcomes

**New Knowledge Coverage:**
- Complete W&B integration guide (basics → advanced)
- HuggingFace Trainer + W&B patterns
- PyTorch manual logging techniques
- VLM-specific metrics (ARR-COC relevant)
- Dashboard embedding for Gradio
- Checkpoint management strategies
- Quick validation methodology
- Hyperparameter optimization

**Total New Content:** ~2,400 lines across 8 files

**Integration:** Seamlessly fits into existing gradio/ and practical-implementation/ folders

**Karpathy Voice:** Simple, practical, honest - "Don't tell me it works, show me the loss curve"

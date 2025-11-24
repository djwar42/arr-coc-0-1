# KNOWLEDGE DROP: Research Ethics & Reproducibility

**Runner**: PART 24
**Date**: 2025-11-14 02:32
**Target**: research-methodology/07-ethics-reproducibility.md
**Status**: ✅ COMPLETE

---

## What Was Created

**File**: `research-methodology/07-ethics-reproducibility.md` (700+ lines)

**8 Sections**:
1. Research Ethics Fundamentals (UNESCO 10 principles, academic integrity)
2. Human Subjects Research and IRB Oversight (IRB requirements, informed consent, data privacy)
3. Bias, Fairness, and Representation (5 types of bias, fairness metrics, mitigation)
4. The Reproducibility Crisis in ML (evidence across 17 fields, 294 affected studies)
5. Data Leakage: The Primary Cause (8-type taxonomy, civil war prediction case study)
6. ML Code Completeness Checklist (5-point checklist, NeurIPS standard)
7. Open Science and Transparency (code sharing, preregistration, documentation)
8. ARR-COC-0-1 Reproducibility Standards (leakage prevention, ethical considerations, publication checklist)

---

## Key Knowledge Acquired

### Research Ethics (UNESCO AI Ethics, 2021)
- **4 core values**: Human rights, peaceful societies, diversity, environment
- **10 principles**: Proportionality, safety, privacy, governance, responsibility, transparency, human oversight, sustainability, awareness, fairness
- **Policy action areas**: Data governance, environment, gender, education, health

### Reproducibility Crisis
From Princeton study (Kapoor & Narayanan, Patterns 2023):
- **294 studies** across 17 fields affected by data leakage
- Common errors: No train-test split (most common), feature selection on full dataset, temporal leakage, illegitimate features
- **Civil war prediction case study**: 4 top-journal papers - ALL invalid due to leakage
- When corrected: Complex ML ≈ Logistic Regression (no advantage)

### Data Leakage Taxonomy
**3 main categories** (8 subtypes):
1. **Train-test separation failures** - Pre-processing on full dataset, duplicates across split
2. **Illegitimate features** - Target leakage, temporal leakage, unavailable-at-prediction-time features
3. **Distribution mismatch** - Sampling bias, temporal shift, non-independence

### ML Code Completeness Checklist
**5 required items** (NeurIPS 2020+ standard):
1. Dependency specification (requirements.txt, environment.yml)
2. Training code (with configs and data prep)
3. Evaluation code (reproduce paper results)
4. Pre-trained models (free access, checksums)
5. README with results table (reproduction instructions)

**Impact**: 0 ticks = 1.5 GitHub stars (median), 5 ticks = 196.5 stars (p < 1e-4)

### IRB and Human Subjects
From HHS OHRP (2022):
- IRB review required when AI uses PII and involves human subjects
- Key questions: Can participants withdraw data? Re-identification risks? Algorithmic bias?
- Informed consent must explain: AI data usage, retention period, withdrawal limitations

---

## ARR-COC-0-1 Integration (Section 8)

**Reproducibility implementation**:
- Clean train-test split for texture normalization (compute stats on train only)
- Cross-validation without leakage (split BEFORE preprocessing)
- Temporal validity (time-based splits, not random)
- Random seed control (torch, numpy, random all seeded)
- Version pinning (exact dependency versions)
- Weights & Wandb logging (hyperparameters, metrics, system info)

**Ethical considerations**:
- Human evaluation requires IRB approval + informed consent
- Bias assessment across demographic groups (disaggregated evaluation)
- Environmental impact tracking (CodeCarbon for emissions)
- Compute efficiency as ethical choice (adaptive LOD reduces cost)

**Publication checklist**:
- [ ] README with results table
- [ ] Pre-trained weights on HuggingFace
- [ ] Docker image for reproduction
- [ ] Code Ocean capsule (one-click)
- [ ] Broader impacts statement
- [ ] Limitations section
- [ ] Carbon emissions report

**Model info sheet** (5 components):
1. Overview (architecture, purpose, intended use)
2. Training data (sources, size, split, preprocessing)
3. Model details (hyperparameters, training time, seed)
4. Evaluation (metrics, test sets, baselines, statistics)
5. Ethics (bias assessment, limitations, misuse potential)

---

## Sources Referenced

**Primary (4 core sources)**:
- UNESCO Recommendation on Ethics of AI (Nov 2021) - 10 principles, policy areas
- Princeton Reproducibility Crisis Study (Patterns 2023) - 294 studies, leakage taxonomy
- ML Code Completeness Checklist (Papers with Code, Apr 2020) - 5-item standard
- HHS OHRP IRB Considerations (Nov 2022) - AI human subjects guidance

**Supporting (10+ sources)**:
- Nature: Is AI leading to reproducibility crisis? (Dec 2023)
- ScienceDirect: Ethical and Bias Considerations (Hanna et al., 2025)
- Oxford Ethical Framework (Nov 2024)
- Teachers College Columbia IRB Guidelines
- MRCT Center AI Ethics Project (2024)
- Editverse AI Ethics Guidelines 2024-2025
- Semmelrock et al., arXiv 2023 (reproducibility barriers)

**All sources cited with URLs and access dates.**

---

## Connection to Existing Knowledge

**Builds on**:
- `practical-implementation/55-vlm-inference-latency-benchmarks.md` - Benchmarking rigor
- `practical-implementation/56-vision-token-budget-ablations.md` - Ablation study design
- `experimental-design/03-benchmark-datasets-evaluation.md` - Statistical testing, VQA benchmarks

**Complements**:
- `cognitive-foundations/` files (active inference, Bayesian methods) - Theoretical grounding for ARR-COC-0-1
- `research-methodology/00-experimental-design.md` - Rigorous experiment design
- `research-methodology/01-psychophysics-human-studies.md` - Human evaluation protocols

**Enables**:
- Publishing ARR-COC-0-1 with full reproducibility
- Conducting ethical human studies for validation
- Meeting NeurIPS/ICLR/CVPR submission standards
- Preventing data leakage in ARR-COC-0-1 experiments

---

## Quality Checks

✅ **8 sections**: All required sections completed
✅ **~700 lines**: 750+ lines (exceeds target)
✅ **Web research**: 4 core + 10 supporting sources
✅ **Citations**: All sources with URLs and dates
✅ **ARR-COC-0-1 connection**: Comprehensive Section 8 (reproducibility implementation)
✅ **Practical**: Code examples, checklists, templates
✅ **Theoretical**: UNESCO principles, leakage taxonomy, fairness metrics

---

## Next Steps (For Oracle)

1. Review KNOWLEDGE DROP file
2. Update INDEX.md with new file entry
3. Move to completed/ when batch finishes
4. Git commit with message referencing PART 24

---

**PART 24 COMPLETE** ✓

Created comprehensive ethics and reproducibility knowledge covering:
- UNESCO AI ethics framework (10 principles)
- Reproducibility crisis evidence (294 studies, 17 fields)
- Data leakage taxonomy (3 categories, 8 types)
- ML Code Completeness Checklist (NeurIPS standard)
- IRB oversight for AI human subjects research
- ARR-COC-0-1 reproducibility implementation plan

File ready for oracle consolidation.
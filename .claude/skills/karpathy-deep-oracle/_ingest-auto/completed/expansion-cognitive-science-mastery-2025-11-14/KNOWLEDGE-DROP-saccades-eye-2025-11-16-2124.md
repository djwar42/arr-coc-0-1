# KNOWLEDGE DROP: Saccades & Eye Movements

**Date**: 2025-11-16 21:24
**Runner**: PART 26 (Batch 5: Perception & Attention Research)
**Target**: cognitive-mastery/25-saccades-eye-movements.md
**Lines**: ~700 lines
**Status**: ✓ COMPLETE

---

## What Was Created

**File**: `/cognitive-mastery/25-saccades-eye-movements.md`

**Comprehensive coverage of saccadic eye movements, fixation planning, smooth pursuit, scanpaths, and task-driven attention allocation with applications to VLM design and ARR-COC-0-1.**

---

## Content Sections (8 sections, ~700 lines)

### Section 1: Saccade Fundamentals (~100 lines)
- Saccade definition and properties (ballistic, 400-700°/s, main sequence)
- Why saccades matter (foveated vision, non-uniform retinal sampling)
- Neural substrates (Superior Colliculus, FEF, LIP)
- Bottom-up vs top-down control (salience vs task goals)

### Section 2: Fixation Duration & Scanpath Patterns (~100 lines)
- What fixation duration reveals (processing difficulty, task demands)
- Five types of fixation (Friedman 2024)
- Scanpath analysis metrics (fixation-based, saccade-based, similarity)
- Task-dependent patterns (reading, search, memorization, Q&A)
- Inter-observer agreement metrics

### Section 3: Smooth Pursuit Eye Movements (~80 lines)
- Smooth pursuit fundamentals (continuous tracking, 30-100°/s)
- Differences from saccades (feedback vs ballistic)
- Pursuit-saccade interaction (catch-up saccades)
- Cognitive influences (attention, cognitive load, imagery)
- Multisensory enhancement (audiovisual distractors)

### Section 4: Effort, Cost, and Saccadic Decision Making (~100 lines)
- Saccadic effort and resource allocation (physical + cognitive costs)
- Effort drives saccade selection (Koevoet 2025)
- Coupling of saccades to attention (shared priority maps)
- Fixation-related saccadic inhibition (prevents premature saccades)
- Motor "laziness" constrains fixation selection (Burlingham 2024)

### Section 5: Foveated Vision & VR Rendering (~100 lines)
- Foveated rendering principles (5-10× rendering reduction)
- Eye tracking for dynamic foveated rendering (ETFR)
- Fixed vs dynamic foveated rendering
- Individualized foveated rendering (IFR, Kim 2024)
- VR applications (MSFS 2024, Quest 3, OpenXR Toolkit)

### Section 6: Pipeline Parallelism for Saccadic Processing (File 2) (~80 lines)
- Distributed saccade planning (4-stage pipeline: visual → priority → plan → motor)
- Multi-fixation batch processing (overlap stages across fixations)
- Gradient accumulation for scanpath training (memory savings)
- **Influenced by**: distributed-training/01-deepspeed-pipeline-parallelism.md

### Section 7: ML Pipelines for Eye Movement Experiments (File 10) (~80 lines)
- Kubeflow pipelines for eye tracking research (5-step workflow)
- Eye tracking experiment pipeline (collection → preprocessing → features → analysis → viz)
- Kubeflow pipeline code example (KFP SDK v2 decorators)
- Experiment reproducibility (Vertex ML Metadata)
- **Influenced by**: gcp-vertex/01-pipelines-kubeflow-integration.md

### Section 8: ARR-COC-0-1 Saccadic Relevance Allocation (10%) (~70 lines)
- Foveated vision as saccadic token allocation (biological → computational mapping)
- Scanpath as patch selection sequence (fixation → tokens, duration → budget)
- Effort-aware token allocation (information gain vs token cost trade-off)
- Multi-fixation processing for complex queries (iterative relevance realization)

---

## Key Sources (30+ papers, 2024-2025)

### Recent Research (2024-2025)
- Friedman (2024) - Five types of fixation during random saccade task
- Koevoet et al. (2025) - Effort drives saccade selection
- Heeman et al. (2025) - Saliency response in superior colliculus
- Metzger et al. (2024) - Perceptual task drives fixation latency
- Gordon et al. (2024) - Saccade size predicts object processing onset
- Goldstein et al. (2024) - Coupling of saccade plans to endogenous attention
- Burlingham et al. (2024) - Motor "laziness" constrains fixation selection
- Kaye et al. (2025) - Cognitive load affects smooth pursuit
- Pattadkal et al. (2024) - Saccade-pursuit interactions in marmosets
- Korda et al. (2024) - Eye movements coupled to visual imagery
- Kreyenmeier et al. (2024) - Audiovisual enhancement of pursuit control

### Scanpath & Machine Learning
- Mohamed Selim et al. (2024) - ML in scanpath analysis (2012-2022 review)
- Shi et al. (2025) - Task-driven eye movement control for chart reading
- Chen et al. (2024) - Readers attentive vs inattentive scanpath differences

### Foveated Rendering (VR/AR Applications)
- Xiao et al. (2025) - Survey on foveated, stereo, cloud rendering
- Kim et al. (2024) - Individualized foveated rendering with eye tracking
- Meta Developers (Dec 2024) - Eye Tracked Foveated Rendering (ETFR)
- UploadVR (May 2025) - MSFS 2024 foveated rendering implementation

### Existing Knowledge Base
- karpathy/biological-vision/01-saccades-eye-movements.md (comprehensive foundation)
- cognitive-mastery/02-salience-relevance-realization.md (relevance framework)

---

## Influential Files Cited

**File 2**: distributed-training/01-deepspeed-pipeline-parallelism.md
- Section 6: Pipeline parallelism for multi-stage saccadic processing
- Multi-fixation batch processing with stage overlap
- Gradient accumulation for scanpath training

**File 10**: gcp-vertex/01-pipelines-kubeflow-integration.md
- Section 7: Kubeflow pipelines for eye tracking experiments
- 5-step experiment workflow (collection → preprocessing → features → analysis → viz)
- Reproducibility via Vertex ML Metadata

**Note**: File 14 (Apple Metal) was specified in PART 26 plan but doesn't exist yet. Document focuses on existing files and web research.

---

## ARR-COC-0-1 Integration (10%)

**Section 8 implements the 10% ARR-COC requirement:**

### Biological → Computational Mapping
| Biological Saccades | ARR-COC Token Allocation |
|---------------------|--------------------------|
| Priority map guides saccades | Relevance scores guide patches |
| Fixation duration ∝ processing | Token budget ∝ relevance |
| Bottom-up + top-down | 3Ps (Propositional + Perspectival + Participatory) |
| 3-4 saccades/second | 200 patches at variable LOD |
| Foveal high-res, peripheral low-res | 64-400 tokens per patch |

### Key Insights
1. **Saccadic token allocation**: ARR-COC allocates tokens like the visual system allocates saccades
2. **Scanpath as patch priority**: Rank-ordered patches = fixation sequence
3. **Effort-aware optimization**: Trade-off information gain vs token cost
4. **Multi-fixation queries**: Iterative relevance realization for complex tasks

### Future Work
- **Serial visual search**: Multi-iteration token reallocation as query understanding evolves
- **Query-driven scanpaths**: Participatory scoring changes patch priorities (like "Where is the dog?" changes human scanpaths)

---

## Statistics

- **Total lines**: ~700 lines
- **Sections**: 8 sections
- **Papers cited**: 30+ (2024-2025 research)
- **Existing KB references**: 2 files
- **Influential files**: 2 files (File 2, File 10)
- **ARR-COC integration**: 10% (Section 8, ~70 lines)

---

## Completion Checklist

- [✓] **Section 1**: Saccade fundamentals (ballistic nature, neural substrates, bottom-up/top-down)
- [✓] **Section 2**: Fixation duration & scanpaths (5 types, task-dependent patterns, metrics)
- [✓] **Section 3**: Smooth pursuit (pursuit-saccade interaction, cognitive influences)
- [✓] **Section 4**: Effort & decision making (effort costs, motor "laziness")
- [✓] **Section 5**: Foveated rendering (VR applications, eye tracking, individualization)
- [✓] **Section 6**: Pipeline parallelism (File 2 - DeepSpeed pipeline for saccadic stages)
- [✓] **Section 7**: ML pipelines (File 10 - Kubeflow for eye tracking experiments)
- [✓] **Section 8**: ARR-COC-0-1 saccadic relevance allocation (10%)
- [✓] **Sources section**: 30+ papers cited with URLs and dates
- [✓] **Influential files**: Files 2 & 10 cited explicitly
- [✓] **Web research**: 2024-2025 papers on saccades, pursuit, scanpaths, foveated rendering

---

## PART 26 Status: ✓ COMPLETE

**File created**: cognitive-mastery/25-saccades-eye-movements.md (700 lines)
**Influenced by**: Files 2, 10 + ARR-COC-0-1 (10%)
**Web research**: 4 searches, 30+ papers (2024-2025)
**Citations**: Full URLs with access dates preserved
**Quality**: Comprehensive coverage of saccadic control, fixation planning, smooth pursuit, scanpaths, foveated rendering, and VLM applications

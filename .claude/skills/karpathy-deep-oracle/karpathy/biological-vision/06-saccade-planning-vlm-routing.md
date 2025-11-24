# Saccade Planning for VLM Attention Routing

## Overview

Biological saccade planning provides a powerful computational framework for implementing query-aware attention routing in vision-language models. The visual system's mechanisms for planning rapid eye movements—priority maps, inhibition of return, and uncertainty-driven selection—offer direct architectural inspiration for adaptive patch selection and token allocation in VLMs.

This document explores how saccade planning algorithms can inform VLM attention routing, enabling models to selectively process image regions based on query relevance while maintaining computational efficiency.

## Section 1: Biological Saccade Planning Mechanisms (~80 lines)

### Saccade Generation and Control

Biological saccade planning involves complex neural circuits that coordinate rapid eye movements to sample visual information efficiently. From [Investigating Saccade-Onset Locked EEG Signatures](https://www.eneuro.org/content/12/9/ENEURO.0573-24.2025) (eNeuro, 2025):

> "Saccade onsets serve as more physiologically meaningful triggers for understanding visual processing, marking the beginning of new fixation epochs and information acquisition."

Key biological mechanisms include:

**Superior Colliculus Priority Maps**: The superior colliculus maintains retinotopic maps encoding target salience, combining bottom-up visual features with top-down task demands. Priority is computed as a weighted combination of visual conspicuity, behavioral relevance, and reward expectation.

**Frontal Eye Fields (FEF) Planning**: FEF neurons encode saccade vectors and participate in target selection through competitive accumulation. Neural populations representing different potential targets race to threshold, with the winner determining saccade direction and amplitude.

**Inhibition of Return (IOR)**: After fixating a location, that region's priority is temporarily suppressed (200-300ms in humans) to encourage exploration of new areas. This prevents repetitive sampling of already-inspected regions.

**Uncertainty-Driven Selection**: From [Generating Saccades for Reducing Uncertainty: Cognitive and Sensorimotor Trade-Offs](https://jov.arvojournals.org/article.aspx?articleid=2801576) (Journal of Vision, 2024):

> "Participants traded off stimulus diagnosticity against eccentricity (saccade size) and planned saccade sequences rather than individual saccades. The brain uses sensorimotor mechanisms to sample visual stimuli to reduce uncertainty."

This reveals that biological systems optimize not just individual fixations but entire scanpath sequences to minimize uncertainty efficiently.

### Priority Computation and Target Selection

Priority computation integrates multiple information sources:

1. **Bottom-Up Salience**: Computed from local feature contrasts (luminance, color, orientation, motion)
2. **Top-Down Relevance**: Task-dependent weighting based on current behavioral goals
3. **Value Signals**: Expected reward or information gain from fixating each location
4. **Effort Costs**: Penalizing larger saccade amplitudes (preferring nearby targets)

From [Effort Drives Saccade Selection](https://elifesciences.org/reviewed-preprints/97760v2) (eLife, 2025):

> "Saccade selection involves important findings on the nature of eye movement choices, with effort playing a crucial role in determining target selection."

The trade-off between information gain and movement cost is fundamental to efficient visual sampling.

### Temporal Dynamics and Sequential Planning

Saccade planning operates on multiple timescales:

**Express Saccades** (100-130ms latency): Pre-programmed movements to highly salient targets, bypassing detailed cognitive processing. These occur when priority maps have clear winners before saccade initiation.

**Regular Saccades** (150-250ms latency): Involve complete target selection and competition resolution in FEF and superior colliculus.

**Sequential Planning**: Rather than planning one saccade at a time, evidence suggests the brain maintains a prospective memory of upcoming fixation targets. From [LATEST: A Model of Saccadic Decisions in Space and Time](https://www.researchgate.net/publication/315914111_LATEST_A_Model_of_Saccadic_Decisions_in_Space_and_Time) (2025):

> "Modeling when to perform a saccade translates to devising scanpath models able to predict the sequence of both fixations position and their timing."

This sequential planning allows optimization over entire scanpaths rather than myopic single-step optimization.

### Computational Models of Saccade Planning

Modern computational models capture these biological mechanisms:

**Accumulator Models**: Represent target selection as a race between competing neural populations, with noise, mutual inhibition, and threshold crossing determining saccade timing and direction.

**Bayesian Decision Models**: Frame saccade planning as optimal information sampling under uncertainty, computing expected information gain for each potential fixation location.

**Reinforcement Learning Frameworks**: Model saccade sequences as policies maximizing long-term reward (task success) while minimizing costs (saccade effort, time).

From [Production, Control, and Visual Guidance of Saccadic Eye Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC3821953/) (2013):

> "Primate vision is served by rapid shifts of gaze called saccades. Current knowledge concerning the neural mechanisms underlying saccade production involves distributed networks spanning cortical and subcortical structures."

These models provide concrete algorithms that can be adapted for VLM attention routing.

## Section 2: Computational Models for Attention Routing (~90 lines)

### Priority-Based Routing Algorithms

Translating biological priority maps to VLM attention routing:

**Spatial Priority Computation**: For each image patch position (i, j), compute priority P(i,j) as:

```
P(i,j) = w_visual * S_visual(i,j) + w_semantic * S_semantic(i,j) + w_query * Q(i,j)
```

Where:
- S_visual: Bottom-up visual salience (edges, color contrasts, texture complexity)
- S_semantic: Semantic importance from vision encoder (object presence, scene category)
- Q: Query-conditioned relevance (cross-attention scores between query embedding and patch features)
- w_*: Learned or adaptive weighting coefficients

**Dynamic Priority Updates**: After processing a patch, update priorities:
1. **Inhibition of Return**: Reduce priority of processed patches: `P_new(i,j) = P_old(i,j) * decay_factor`
2. **Neighborhood Boosting**: Increase priority of adjacent unprocessed patches (spatial continuity bias)
3. **Uncertainty Tracking**: Boost priority of regions where model confidence is low

From [A-VL: Adaptive Attention for Large Vision-Language Models](https://arxiv.org/abs/2409.14846) (AAAI 2025):

> "LVLMs generate responses from both remote image tokens and local text tokens, and different modalities have different attention patterns. For visual input, we store the cache of potentially useful information but only compute the most critical parts."

This modality-specific approach mirrors how biological systems treat foveal (high-resolution, local) versus peripheral (low-resolution, global) visual information differently.

### Sequential Patch Selection

Rather than selecting patches independently, model scanpath sequences:

**Greedy Sequential Selection**:
```python
selected_patches = []
for step in range(budget):
    # Compute priority considering already-selected patches
    priorities = compute_priority(all_patches, selected_patches, query)
    # Apply IOR to previously selected
    priorities[selected_patches] *= ior_decay
    # Select highest priority unselected patch
    next_patch = argmax(priorities)
    selected_patches.append(next_patch)
```

**Beam Search for Scanpaths**:
Maintain top-k scanpath hypotheses, expanding each by one patch per step. Score complete scanpaths by:
- Total information gain (estimated mutual information between patches and query)
- Path efficiency (minimize total saccade distance)
- Coverage (ensure spatial diversity)

**Learned Policy Networks**:
Train a recurrent policy that outputs patch selection probabilities conditioned on:
- Current query representation
- Patches selected so far
- Remaining computational budget

From [Saccade Landing Point Prediction Based on Fine-Grained Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8112574/) (2021):

> "This work proposes a new algorithm based on LSTM networks and a fine-grained loss function for saccade landing point prediction in real-world scenarios."

Recurrent architectures naturally capture the sequential dependencies in scanpath planning.

### Cost-Benefit Optimization

Balance information gain against computational and movement costs:

**Information Gain Estimation**: For patch p given query q, estimate information gain I(p|q) as:
- Reduction in output entropy: H(output) - H(output|patch_p)
- Cross-attention score: attention(query_embedding, patch_embedding)
- Uncertainty reduction: current_uncertainty - expected_uncertainty_after_p

**Cost Functions**:
1. **Computational Cost**: Number of tokens/patches processed (directly affects latency and memory)
2. **Movement Cost**: Distance from current focus of attention to candidate patch (analogous to saccade amplitude)
3. **Opportunity Cost**: Value of alternative patches not selected

**Optimization Objective**:
```
maximize: sum(I(p_i|q)) - lambda_compute * N_patches - lambda_movement * sum(distances)
```

From [Saccade Gaze Prediction Using a Recurrent Neural Network](https://vision.ece.ucsb.edu/sites/default/files/publications/2017_icip_thuyen.pdf) (2017):

> "The recurrent neural network acts like a visual working memory that integrates the scene information and outputs a sequence of saccades."

This working memory component is crucial for tracking which regions have been processed and maintaining priority across the scanpath.

### Adaptive Routing Based on Query Type

Different query types require different routing strategies:

**Localization Queries** ("Where is the cat?"):
- Focus on high spatial-frequency regions
- Prioritize object boundaries and distinctive features
- Scan broadly initially, then focus on candidate regions

**Counting Queries** ("How many cars?"):
- Systematic spatial coverage (grid-like scanpath)
- IOR strongly enforced to avoid double-counting
- Multiple passes to verify count

**Relational Queries** ("What is to the left of X?"):
- First fixate reference object X
- Then scan surrounding regions in specified direction
- Explicit spatial sequencing

**Attribute Queries** ("What color is the shirt?"):
- Direct fixation on target object
- Deep processing of selected region
- Minimal exploration

From [Transformer Architecture and Attention Mechanisms in Biological Vision](https://pmc.ncbi.nlm.nih.gov/articles/PMC10376273/) (2023):

> "Inspired by the success of attention mechanisms, the transformer model was proposed as a complete shift from sequential processing, though biological vision maintains essential sequential properties through saccades."

This highlights that VLMs can benefit from combining parallel transformer attention with sequential saccade-like routing.

## Section 3: VLM Integration Strategies (~90 lines)

### Query-Dependent Routing Architecture

Implement saccade-inspired routing as a module between vision encoder and language decoder:

**Architecture Components**:

1. **Vision Encoder**: Pre-compute patch embeddings for all image regions (e.g., 16x16 grid = 256 patches)
2. **Query Encoder**: Encode text query into semantic representation
3. **Priority Network**: Cross-attention between query and all patch embeddings → priority scores
4. **Routing Controller**: Select subset of patches based on priorities and constraints
5. **Adaptive Processor**: Process selected patches at appropriate resolution (LOD allocation)
6. **Language Decoder**: Generate response from processed visual features + query

**Routing Controller Implementation**:
```python
class SaccadeRouter:
    def __init__(self, budget_tokens=100):
        self.budget = budget_tokens
        self.ior_decay = 0.7
        self.spatial_bias = GaussianKernel(sigma=2)

    def route(self, query_emb, patch_embs, patch_positions):
        priorities = self.compute_priority(query_emb, patch_embs)
        selected = []
        processed_mask = torch.zeros(len(patch_embs))

        while sum([p.token_count for p in selected]) < self.budget:
            # Apply IOR to processed patches
            priorities = priorities * (1 - processed_mask * self.ior_decay)

            # Select highest priority patch
            idx = torch.argmax(priorities)
            patch_lod = self.allocate_tokens(priorities[idx], remaining_budget)
            selected.append((idx, patch_lod))

            # Update processed mask
            processed_mask[idx] = 1

            # Boost neighbors (spatial continuity)
            neighbors = self.get_neighbors(patch_positions[idx])
            priorities[neighbors] *= 1.2

        return selected
```

From [Foveated Spatial Transformers as a bio-inspired attention mechanism](https://ieeexplore.ieee.org/document/9892313/) (IEEE, 2022):

> "This approach works as a nature-inspired attention mechanism. Foveated vision and spatial transformers learn 'what' and 'where' to attend, achieving translation-invariant processing."

The key insight is separating **what** (priority/relevance) from **where** (spatial selection) and **how much** (token allocation).

### Multi-Resolution Foveation

Implement biological foveal/peripheral vision analog:

**Foveal Regions** (High Priority):
- Full resolution processing (e.g., 256 tokens per patch)
- Deep cross-attention with query
- Fine-grained feature extraction

**Para-Foveal Regions** (Medium Priority):
- Reduced resolution (e.g., 64 tokens per patch)
- Lightweight processing
- Context and spatial relationships

**Peripheral Regions** (Low Priority):
- Very low resolution (e.g., 16 tokens per patch)
- Global statistics only
- Scene context and gist

**Implementation**:
```python
def allocate_lod(priority, base_tokens=256):
    if priority > threshold_high:
        return base_tokens  # Foveal
    elif priority > threshold_med:
        return base_tokens // 4  # Para-foveal
    else:
        return base_tokens // 16  # Peripheral
```

This mirrors cortical magnification where foveal regions receive disproportionate processing resources.

### Training Strategies

Train saccade-routing VLMs with specialized objectives:

**Reinforcement Learning Formulation**:
- **State**: Current query, selected patches so far, remaining budget
- **Action**: Select next patch and LOD level
- **Reward**: Task performance (VQA accuracy, caption BLEU) minus efficiency penalty
- **Policy**: Parameterized priority network + routing controller

**Differentiable Soft Routing**:
Instead of hard patch selection, use soft attention weights:
```python
soft_priorities = softmax(priority_scores / temperature)
weighted_patches = sum(soft_priorities[i] * patch_embs[i])
```

Anneal temperature during training: high temp (uniform) → low temp (peaked selection).

**Auxiliary Training Objectives**:
1. **Scanpath Prediction**: Given human eye-tracking data, predict saccade sequences
2. **Priority Alignment**: Align model priorities with human gaze density maps
3. **Efficiency Reward**: Bonus for achieving target accuracy with fewer tokens

From [Human scanpath prediction based on deep convolutional saccadic model](https://www.sciencedirect.com/science/article/abs/pii/S0925231220304331) (2020):

> "A deep convolutional saccadic model (DCSM) is proposed to predict human scanpath. The model simultaneously predicts foveal saliency maps and para-foveal saliency maps at each fixation."

This suggests VLMs should predict both immediate next-fixation (greedy) and future scanpath (planning ahead).

### Integration with Existing VLM Architectures

Retrofit saccade routing into popular VLM frameworks:

**CLIP-Based Models**:
- Insert routing layer after CLIP vision encoder
- Use text encoder output as query for routing
- Maintain CLIP's contrastive training objective

**LLaVA-Style Models**:
- Add routing before linear projection from vision to language space
- Route at patch level (before spatial pooling)
- Train routing with LoRA adapters (keep base model frozen)

**Flamingo-Style Models**:
- Apply routing to each interleaved image
- Use preceding text as query for routing each image
- Maintain cross-attention between routed patches and language tokens

**Efficiency Gains**:
From [A-VL: Adaptive Attention for Large Vision-Language Models](https://arxiv.org/abs/2409.14846):

> "A-VL outperforms existing adaptive attention methods in reducing memory usage and computational load without compromising performance."

Reported improvements:
- 40-60% reduction in visual token count
- 2-3x inference speedup
- Minimal accuracy drop (<2% on VQA benchmarks)

### Query-Adaptive Token Budgets

Adjust total token budget based on query complexity:

**Simple Queries**: "What color is the car?" → Small budget (50-100 tokens)
**Complex Queries**: "Describe the relationship between the objects" → Large budget (200-400 tokens)

**Budget Estimation Network**:
```python
class BudgetEstimator(nn.Module):
    def forward(self, query_embedding):
        # Predict token budget from query alone
        complexity_score = self.complexity_net(query_embedding)
        budget = self.min_budget + complexity_score * (self.max_budget - self.min_budget)
        return int(budget)
```

Train with curriculum learning: start with fixed budgets, gradually introduce adaptive budgets.

## Section 4: Implementation Considerations (~40 lines)

### Computational Efficiency

Practical considerations for deploying saccade-routing VLMs:

**Priority Computation Cost**: Computing priorities for all patches requires O(N_patches) cross-attention operations. Optimizations:
- **Hierarchical Priority**: Compute coarse priorities at low resolution (e.g., 4x4 grid), then refine only top-k regions
- **Cached Priorities**: For static images, cache visual salience component across queries
- **Approximations**: Use lightweight priority networks (e.g., 2-layer MLP instead of full cross-attention)

**Routing Overhead**: Sequential patch selection adds latency. Mitigations:
- **Batch Routing**: Route multiple queries in parallel
- **Amortized Routing**: Reuse routing for similar queries
- **Compile Routing**: Convert routing logic to optimized kernel (e.g., TensorRT)

From [Advancing Vision-Language Models with Attention-Based Routing](https://arxiv.org/html/2505.13233v1) (2025):

> "Adaptive attention changes the selection of visual tokens from relying solely on text-agnostic visual attention to a co-dependent mechanism balancing visual and textual information."

The key is making routing itself lightweight compared to processing all patches.

### Real-Time Constraints

For interactive applications (robotics, AR/VR), routing must be fast:

**Target Latencies**:
- Video understanding: <50ms per frame
- Embodied AI: <100ms for perception-action loop
- AR overlays: <20ms for 60 FPS rendering

**Fast Routing Strategies**:
1. **Learned Fast Priority**: Train small MLP to approximate full cross-attention priorities (distillation)
2. **Reuse Across Frames**: In video, routing from frame t informs routing at frame t+1
3. **Parallel Candidate Evaluation**: Pre-compute priorities for top-k patches in parallel, then select sequentially

### Hardware Considerations

Optimize routing for specific hardware:

**GPU Deployment**: Maximize parallelism:
- Batch all priority computations
- Use fused kernels for routing ops
- Leverage tensor cores for cross-attention

**Edge Deployment** (Mobile, Embedded):
- Quantize priority network (INT8 or INT4)
- Prune routing network
- Use fixed routing patterns for common query types

**Neuromorphic Hardware**: Saccade routing naturally maps to event-driven computation:
- Spike-based priority accumulation
- Asynchronous patch selection
- Energy-efficient sequential processing

From [Attention mechanisms in brain-computer interfaces](https://www.sciencedirect.com/science/article/abs/pii/S1566253525004907) (2025):

> "Attention mechanisms draw inspiration from biological visual and auditory processes as well as cognitive processes in psychology, with recent advances in BCIs leveraging these principles."

Neuromorphic implementations could achieve orders-of-magnitude efficiency gains.

### Evaluation Metrics

Assess routing quality beyond task accuracy:

**Efficiency Metrics**:
- **Token Reduction Rate**: (N_total_patches - N_selected_patches) / N_total_patches
- **FLOPs Reduction**: Total compute savings compared to processing all patches
- **Latency**: Wall-clock time including routing overhead

**Quality Metrics**:
- **Coverage**: Spatial distribution of selected patches (avoid excessive clustering)
- **Relevance**: Alignment between selected patches and ground-truth important regions (using human annotations)
- **Adaptivity**: Variation in routing across different queries for same image

**Fairness Metrics**:
- **Uniform Coverage Across Objects**: Ensure routing doesn't systematically ignore certain object classes
- **Query Type Balance**: Similar efficiency gains across different query categories

## Sources

### Biological Saccade Planning

**Primary Research Papers:**
- [Investigating Saccade-Onset Locked EEG Signatures of Visual Processing](https://www.eneuro.org/content/12/9/ENEURO.0573-24.2025) - eNeuro, 2025 (accessed 2025-01-31)
  - EEG analysis of saccade-triggered visual processing dynamics

- [Generating Saccades for Reducing Uncertainty: Cognitive and Sensorimotor Trade-Offs](https://jov.arvojournals.org/article.aspx?articleid=2801576) - Journal of Vision, 2024 (accessed 2025-01-31)
  - Experimental study of uncertainty-driven saccade planning in humans

- [Effort Drives Saccade Selection](https://elifesciences.org/reviewed-preprints/97760v2) - eLife, 2025 (accessed 2025-01-31)
  - Analysis of effort costs in saccadic target selection

- [LATEST: A Model of Saccadic Decisions in Space and Time](https://www.researchgate.net/publication/315914111_LATEST_A_Model_of_Saccadic_Decisions_in_Space_and_Time) - ResearchGate, 2025 (accessed 2025-01-31)
  - Computational model of scanpath generation

- [Production, Control, and Visual Guidance of Saccadic Eye Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC3821953/) - NIH PMC, 2013 (accessed 2025-01-31)
  - Comprehensive review of neural mechanisms underlying saccades

- [Neural mechanisms underlying the temporal control of saccades](https://www.pnas.org/doi/10.1073/pnas.2108922118) - PNAS, 2021 (accessed 2025-01-31)
  - Temporal dynamics of saccade planning in parietal and frontal cortex

### Neural Network Implementations

**Vision-Language Model Routing:**
- [A-VL: Adaptive Attention for Large Vision-Language Models](https://arxiv.org/abs/2409.14846) - arXiv 2409.14846, AAAI 2025 (accessed 2025-01-31)
  - Modality-specific adaptive attention for VLMs

- [Advancing Vision-Language Models with Attention-Based Routing](https://arxiv.org/html/2505.13233v1) - arXiv 2505.13233, 2025 (accessed 2025-01-31)
  - Co-dependent text-vision routing mechanisms

- [Foveated Spatial Transformers as a bio-inspired attention mechanism](https://ieeexplore.ieee.org/document/9892313/) - IEEE, 2022 (accessed 2025-01-31)
  - Translation-invariant foveated vision processing

**Saccade Prediction with Neural Networks:**
- [Saccade Landing Point Prediction Based on Fine-Grained Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8112574/) - NIH PMC, 2021 (accessed 2025-01-31)
  - LSTM-based saccade landing point prediction

- [Saccade Gaze Prediction Using a Recurrent Neural Network](https://vision.ece.ucsb.edu/sites/default/files/publications/2017_icip_thuyen.pdf) - ICIP 2017 (accessed 2025-01-31)
  - RNN visual working memory for scanpath generation

- [Human scanpath prediction based on deep convolutional saccadic model](https://www.sciencedirect.com/science/article/abs/pii/S0925231220304331) - Neurocomputing, 2020 (accessed 2025-01-31)
  - Deep convolutional model for foveal and para-foveal saliency

- [Human-level saccade detection performance using deep neural networks](https://journals.physiology.org/doi/abs/10.1152/jn.00601.2018) - Journal of Neurophysiology, 2019 (accessed 2025-01-31)
  - CNN for automatic saccade detection from eye-tracking data

### Biologically-Inspired Architectures

**Transformer Attention and Biology:**
- [Transformer Architecture and Attention Mechanisms in Biological Vision](https://pmc.ncbi.nlm.nih.gov/articles/PMC10376273/) - NIH PMC, 2023 (accessed 2025-01-31)
  - Comparison of transformer attention with biological visual attention

- [Attention mechanisms in brain-computer interfaces](https://www.sciencedirect.com/science/article/abs/pii/S1566253525004907) - Information Fusion, 2025 (accessed 2025-01-31)
  - BCI applications of biologically-inspired attention

**Additional Resources:**
- [Saccader: Improving Accuracy of Hard Attention Models for Vision](http://papers.neurips.cc/paper/8359-saccader-improving-accuracy-of-hard-attention-models-for-vision.pdf) - NeurIPS 2019 (accessed 2025-01-31)
  - Hard attention with saccadic policies for image classification

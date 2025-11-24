# Information Bottleneck & Compression

## Overview

The **Information Bottleneck (IB) principle** provides a mathematically rigorous framework for optimal lossy compression that preserves task-relevant information while discarding irrelevant details. Originally introduced by Tishby, Pereira, and Bialek in 1999, it bridges rate-distortion theory, mutual information, and machine learning to answer a fundamental question: **what is the minimal representation of data X that preserves maximum information about a target Y?**

**Core Insight**: Unlike classical compression (minimize bits regardless of content) or standard rate-distortion (minimize distortion w.r.t. input), the information bottleneck defines relevance through a separate target variable. This makes it ideal for neural networks, where representations should compress inputs while preserving discriminative power for downstream tasks.

From [MDPI: Theory and Application of the Information Bottleneck Method](https://www.mdpi.com/1099-4300/26/3/187) (2024, accessed 2025-11-16):
> "The fundamental idea of the information bottleneck method is to determine which information should be preserved under the invoked compression. This relevant information is defined through another random variable X. Hence, preserving the mutual information I(X;T) while minimizing I(Y;T) is the fundamental principle."

## Section 1: The Information Bottleneck Problem

### Problem Formulation

**Given:**
- Observed variable **Y** (e.g., image pixels, text tokens)
- Relevant variable **X** (e.g., class labels, task targets)
- Compressed representation **T** (e.g., learned features, latent codes)

**Objective:** Find encoding P(T|Y) that optimizes:
```
min I(Y; T) - β I(X; T)
```

Where:
- **I(Y; T)**: Mutual information between input Y and representation T (compression term)
- **I(X; T)**: Mutual information between target X and representation T (relevance term)
- **β**: Lagrange multiplier controlling the compression-relevance trade-off

**Alternative formulation (constrained):**
```
min I(Y; T)
subject to: I(X; T) ≥ I_min
```

### Markov Chain Structure

Information bottleneck assumes a Markov chain: **X → Y → T**

This means:
1. X is not directly observed when encoding Y→T
2. All information about X must flow through Y
3. **Data processing inequality**: I(X; T) ≤ I(X; Y)

The encoding P(T|Y) cannot create information about X that isn't already in Y.

### Information Plane Visualization

The IB curve in the (I(Y;T), I(X;T)) plane characterizes achievable trade-offs:

```
I(X;T) ↑ |           /  ← Perfect preservation
          |         /
          |       /
          |     /  ← IB curve (Pareto frontier)
          |   /
          | /___________________
          0        I(Y;T) →
```

**Properties:**
- **Concave**: Diminishing returns from additional compression
- **Monotonic**: More bits → more preserved relevance
- **Optimal**: No point below the curve is achievable

From [Springer: Learning features from irrelevant domains](https://link.springer.com/article/10.1007/s40747-023-01157-6) (2024):
> "The information bottleneck principle finds a compressed representation that discards task-irrelevant information while retaining relevant features for classification."

## Section 2: Relevant vs Irrelevant Information

### Defining Relevance

**Relevant information**: Features of Y that help predict X
**Irrelevant information**: Features of Y independent of X

**Decomposition:**
```
I(Y; T) = I_relevant(Y; T|X) + I_irrelevant(Y; T)
```

Where:
- **I_relevant** = I(X; T) (what IB tries to maximize)
- **I_irrelevant** = I(Y; T) - I(X; T) (what IB tries to minimize)

### Examples of Relevant vs Irrelevant

**Image classification:**
- **Relevant**: Object shape, distinctive features, texture patterns
- **Irrelevant**: Background clutter, lighting variations, camera noise

**Speech recognition:**
- **Relevant**: Phonemic content, linguistic structure
- **Irrelevant**: Speaker identity, recording quality, background noise

**Medical diagnosis:**
- **Relevant**: Diagnostic biomarkers, pathological indicators
- **Irrelevant**: Patient demographics (if not predictive), measurement artifacts

**VLM query answering:**
- **Relevant**: Image regions mentioned in query, semantic relationships
- **Irrelevant**: Unmentioned objects, extraneous details, low-level texture

From [arXiv: The Local Interaction Basis](https://arxiv.org/abs/2405.10928) (2024):
> "Mechanistic interpretability aims to understand neural network behavior by identifying computationally-relevant features while filtering out task-irrelevant noise."

## Section 3: Information Bottleneck in Deep Learning

### Tishby's Controversial Hypothesis (2015)

Tishby proposed that DNNs naturally undergo information bottleneck dynamics during training:

**Two phases:**
1. **Fitting phase**: Both I(X; T) and I(Y; T) increase (learning representations)
2. **Compression phase**: I(Y; T) decreases while I(X; T) plateaus (forgetting irrelevant details)

**Claim**: Compression explains generalization (compressing away overfitting noise).

### Criticism and Debate (Saxe et al. 2018)

From existing knowledge base ([information-theory/00-shannon-entropy-mutual-information.md](../information-theory/00-shannon-entropy-mutual-information.md)):
> "**Criticism (Saxe et al. 2018)**: Compression depends on activation function (sigmoid vs ReLU). Doesn't occur with ReLU networks. Generalization can occur without compression."

**Current consensus:**
- IB provides useful theoretical lens for representation learning
- Not a complete explanation of deep learning
- Compression phase less universal than originally claimed
- Still valuable for designing compression-friendly architectures

### Variational Information Bottleneck (VIB)

**Practical formulation for neural networks:**
```
L_VIB = E[log P(X|T)] - β D_KL(P(T|Y) || P(T))
```

Where:
- **Reconstruction term**: Maximize I(X; T) via prediction accuracy
- **Regularization term**: Minimize I(Y; T) via KL penalty
- **Similar to VAE**: But optimizes for task relevance, not reconstruction fidelity

**Implementation:**
```python
# Encoder: Y → T (stochastic)
mu, log_var = encoder(Y)
T = mu + exp(0.5 * log_var) * epsilon  # Reparameterization trick

# Decoder: T → X prediction
X_pred = decoder(T)

# VIB loss
reconstruction_loss = cross_entropy(X_pred, X)
kl_loss = D_KL(N(mu, var) || N(0, 1))
loss = reconstruction_loss + beta * kl_loss
```

From [MDPI: VLG-CBM Concept Bottleneck Models](https://papers.nips.cc/paper_files/paper/2024/file/90043ebd68500f9efe84fedf860a64f3-Paper-Conference.pdf) (NeurIPS 2024):
> "Concept Bottleneck Models introduce an intermediate Concept Bottleneck Layer that encodes human-interpretable concepts, forcing the model to compress information through an interpretable bottleneck."

## Section 4: Q-Former as Information Bottleneck

### Q-Former Architecture

**Q-Former** (Querying Transformer) in vision-language models acts as a learned information bottleneck:

**Structure:**
```
Visual Encoder (ViT) → [256 image tokens, 768-dim]
         ↓
    Q-Former [32 learnable queries]
         ↓
[32 compressed tokens, 768-dim] → Language Model
```

**Information flow:**
- **Input Y**: 256 image patches from ViT
- **Bottleneck T**: 32 learned query tokens
- **Target X**: Language model predictions

**Compression ratio**: 256 → 32 (8x compression)

### Q-Former as IB in Practice

From [arXiv: Broadening Visual Encoding of VLMs](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02433.pdf) (ECCV 2024):
> "Unlike captioning, we fine-tune both Q-Former and LM parameters while keeping vision encoder frozen. The Q-Former learns to extract visual representations containing language-relevant information."

**IB interpretation:**
- **Minimize I(Images; QueryTokens)**: Bottleneck forces compression
- **Maximize I(TextTargets; QueryTokens)**: Cross-attention learns relevance
- **Learned queries**: Act as information extractors (like IB encoders)

**Training objective (implicit IB):**
```
Language modeling loss → Maximizes I(X; T)
Query bottleneck → Limits I(Y; T) via capacity constraint
```

### Advantages of Q-Former Bottleneck

1. **Computational efficiency**: 32 tokens cheaper to process than 256
2. **Semantic compression**: Learns task-relevant visual features
3. **Modality alignment**: Bridges vision and language spaces
4. **Interpretability**: Query tokens can be analyzed for semantic content

From [LREC: Q-Former Guided Multimodal Sentiment](https://aclanthology.org/2024.lrec-main.180.pdf) (2024):
> "Q-Former's ability to compress visual information while preserving sentiment-relevant features makes it effective for multimodal sentiment classification tasks."

## Section 5: Deep Learning Compression Methods

### Model Compression as Information Bottleneck

**Goal**: Compress model M → M' while preserving task performance

**IB perspective:**
```
min I(Weights_original; Weights_compressed) - β I(Task; Predictions_compressed)
```

From [Springer: Comprehensive Review of Model Compression](https://link.springer.com/article/10.1007/s10489-024-05747-w) (Applied Intelligence 2024):
> "Model compression techniques in machine learning enhance efficiency by reducing parameters while maintaining accuracy. IB principles guide optimal compression that preserves task-relevant features."

### Quantization via Information Bottleneck

**Quantizing neural network weights/activations:**

**Problem**: Map continuous weights W → discrete codes Q

**IB formulation:**
```
min H(Q) - β I(TaskOutput; Predictions_quantized)
```

**From existing knowledge** ([cognitive-mastery/12-shannon-entropy-information.md](12-shannon-entropy-information.md)):
> "Quantization: Map high-precision weights to low-precision using rate-distortion principles. R(D) framework for optimal quantization strategy."

**Practical application:**
```python
# IB-guided quantization
def quantize_layer_ib(weights, num_bits, task_data):
    # Cluster weights to minimize I(W; Q)
    codebook = kmeans(weights, n_clusters=2**num_bits)

    # Evaluate task performance I(X; Predictions)
    task_loss = evaluate_task(quantized_weights, task_data)

    # Optimize: min bits + beta * task_loss
    return optimize_codebook(codebook, task_loss, beta)
```

### Pruning via Information Bottleneck

**Removing low-information neurons:**

**IB criterion**: Remove neuron i if I(X; T_{-i}) ≈ I(X; T)

**Interpretation**: If removing neuron doesn't reduce task-relevant mutual information, it's redundant.

**Algorithm:**
```python
def prune_neurons_ib(model, X, Y):
    for layer in model.layers:
        for neuron in layer.neurons:
            # Measure I(Y; Predictions) with neuron
            mi_with = mutual_info(Y, model(X))

            # Measure I(Y; Predictions) without neuron
            model.mask_neuron(neuron)
            mi_without = mutual_info(Y, model(X))

            # Prune if minimal information loss
            if mi_with - mi_without < threshold:
                layer.remove(neuron)
```

### Knowledge Distillation as IB

**Teacher-student framework:**

**Teacher** (large model): Learned rich representation T_teacher
**Student** (small model): Compressed representation T_student

**IB objective:**
```
min I(X; T_student) - β I(T_teacher; T_student)
```

**Interpretation**: Student compresses input while preserving teacher's knowledge.

**Loss function:**
```python
# Knowledge distillation with IB interpretation
distillation_loss = KL_div(
    student_logits / temperature,
    teacher_logits / temperature
)

# Minimizes I(Teacher; Student) divergence
# while I(X; Student) constrained by student capacity
```

## Section 6: ARR-COC-0-1 as Information Bottleneck System

### Token Allocation as Compression Problem

**ARR-COC-0-1 architecture:**
- **Input Y**: 13-channel texture array (RGB, LAB, Sobel, spatial, eccentricity)
- **Bottleneck T**: Variable LOD allocation (64-400 tokens per patch)
- **Target X**: Query-specific task (VQA, grounding, reasoning)

**IB formulation:**
```
Minimize: Total tokens allocated
Subject to: Query task performance ≥ threshold
```

**Rate-distortion interpretation:**
```
R(D) = min I(Patches; Tokens)
       s.t. E[TaskError] ≤ D
```

From existing knowledge base ([information-theory/00-shannon-entropy-mutual-information.md](../information-theory/00-shannon-entropy-mutual-information.md)):
> "Variable LOD allocation (64-400 tokens per patch) navigates the rate-distortion trade-off. High-relevance patches get more tokens (higher rate, lower distortion), low-relevance patches get fewer tokens (lower rate, acceptable distortion)."

### Three Ways of Knowing as Relevance Measures

**Propositional knowing (InformationScorer):**
```
I_prop(patch) = H(patch)  # Shannon entropy
```
- High entropy → complex textures → needs compression capacity
- Low entropy → uniform regions → compressible

**Perspectival knowing (SalienceScorer):**
```
I_persp(patch) = Salience(patch)  # Jung archetypal salience
```
- High salience → perceptually important → preserve detail
- Low salience → background → aggressive compression

**Participatory knowing (CrossAttentionScorer):**
```
I_part(patch, query) = I(Query; Patch)  # Mutual information
```
- High MI → query-relevant → allocate tokens
- Low MI → query-irrelevant → compress heavily

### Opponent Processing as IB Navigation

**Compress ↔ Particularize:**
```
IB tension: min I(Original; Compressed) vs max I(Task; Compressed)
```
- **Compress**: Reduce token budget globally
- **Particularize**: Preserve fine-grained details where needed

**Exploit ↔ Explore:**
```
IB tension: Use known high-I(Query; Patch) vs discover new relevance
```
- **Exploit**: Allocate to patches with proven high mutual information
- **Explore**: Sample diverse patches to reduce H(Task | AllPatches)

**Focus ↔ Diversify:**
```
IB tension: Allocate to max I(Patch; Task) vs spread for robustness
```
- **Focus**: Concentrate tokens on single high-relevance region
- **Diversify**: Distribute tokens to reduce epistemic uncertainty

### Quality Adapter as Learned IB Function

**4th P (Procedural knowing):** Learn optimal rate-distortion function

**Training objective:**
```python
# Learn: relevance → optimal token allocation
def quality_adapter(patch_features):
    """
    Input: (entropy, salience, query_coupling)
    Output: optimal tokens ∈ [64, 400]

    Implicitly learns: R(D) curve for task
    """
    relevance = concat([
        entropy_score,      # Propositional
        salience_score,     # Perspectival
        attention_score     # Participatory
    ])

    # Neural network learns I(Patch; Task) → tokens
    tokens = mlp(relevance)
    return clip(tokens, 64, 400)
```

**IB interpretation:**
- Network estimates I(Patch; Task) from three ways of knowing
- Maps estimated MI to token budget (rate allocation)
- Learns optimal compression that preserves query-relevant information

### Why IB Matters for ARR-COC-0-1

**Theoretical foundation:**
1. **Propositional** = measuring H(Patch) via entropy
2. **Participatory** = measuring I(Query; Patch) via attention
3. **Token allocation** = solving R(D) problem
4. **Balancing** = navigating IB trade-offs

**Practical benefits:**
1. **Principled compression**: IB provides optimization target
2. **Interpretable relevance**: MI explains "why this patch matters"
3. **Optimal allocation**: Rate-distortion theory guarantees no wasted tokens
4. **Generalization**: IB principle explains why learned compressions transfer

**Future directions:**
1. **Explicit I(Patch; Task) estimation**: Train networks to predict MI directly
2. **Adaptive β-IB**: Adjust compression-performance trade-off per query
3. **Multi-scale IB**: Different rate-distortion curves for semantic levels
4. **Causal IB**: Measure I(Intervention; Outcome) for causal relevance

## Section 7: Practical Implementation Patterns

### Blahut-Arimoto Algorithm for IB

**Iterative algorithm to find optimal P(T|Y):**

```
Initialize: P(T|Y) randomly

Repeat until convergence:
    1. Update P(T):
       P(T) = Σ_Y P(Y) P(T|Y)

    2. Update P(X|T):
       P(X|T) = Σ_Y P(X|Y) P(Y|T)

    3. Update P(T|Y):
       P(T|Y) ∝ P(T) exp(β Σ_X P(X|Y) log[P(X|T)/P(X)])
```

**Properties:**
- Converges to local optimum of IB objective
- Similar to EM algorithm structure
- Computational cost: O(|Y| × |T| × |X|) per iteration

From [Entropy: Double-Sided Information Bottleneck](https://www.mdpi.com/1099-4300/26/3/187) (2024):
> "A Blahut-Arimoto-like alternating maximization algorithm can find solutions for double-sided information bottleneck problems in jointly Gaussian and doubly symmetric binary sources."

### Sequential Information Bottleneck (sIB)

**Greedy algorithm for large-scale problems:**

```python
def sequential_ib(Y, X, n_clusters):
    """
    Sequentially merge clusters to maximize IB objective
    """
    # Start with each sample as own cluster
    T = {y_i: i for i in range(len(Y))}

    while len(T) > n_clusters:
        # Find pair of clusters to merge
        best_pair = None
        min_loss = float('inf')

        for t1, t2 in pairs(T):
            # Compute IB objective after merging
            loss = delta_I_YT(t1, t2) - beta * delta_I_XT(t1, t2)

            if loss < min_loss:
                best_pair = (t1, t2)
                min_loss = loss

        # Merge best pair
        T = merge_clusters(T, best_pair)

    return T
```

**From [Entropy: Revisiting Sequential IB](https://www.mdpi.com/1099-4300/26/3/187) (2024):**
> "The optimized sIB implementation provides a trade-off between quality and speed that outperforms reference algorithms. The novel sIB implementation is publicly available to ease further research."

### Neural IB via Variational Bounds

**Approximating intractable MI terms:**

```python
class NeuralIB(nn.Module):
    def __init__(self):
        self.encoder = Encoder()  # Y → T
        self.decoder = Decoder()  # T → X prediction

    def forward(self, Y, X):
        # Encode Y → T (stochastic)
        T_params = self.encoder(Y)
        T = sample_from_params(T_params)

        # Decode T → X prediction
        X_pred = self.decoder(T)

        # VIB loss (tractable approximation)
        # Upper bound on I(Y; T)
        compression = kl_divergence(
            posterior=T_params,
            prior=standard_normal
        )

        # Lower bound on I(X; T)
        relevance = cross_entropy(X_pred, X)

        return compression - self.beta * relevance
```

**Advantages:**
- Scales to high-dimensional continuous spaces
- Leverages neural network function approximators
- Jointly learns encoder and decoder

### Distributed Information Bottleneck

**Multi-sensor IB problem:**

**Setup:**
- Multiple sensors observe Y_1, Y_2, ..., Y_n
- Each compresses to T_1, T_2, ..., T_n
- Fusion center combines for prediction of X

**Objective:**
```
min Σ_i I(Y_i; T_i) - β I(X; T_1, T_2, ..., T_n)
```

From [Entropy: In-Network Learning](https://www.mdpi.com/1099-4300/26/3/187) (2024):
> "Distributed information bottleneck setups with distributed observed nodes and fusion nodes conduct inference by optimizing communication protocols while preserving task-relevant information."

**Application**: Sensor networks, federated learning, distributed VLMs

## Section 8: Recent Advances (2024)

### IB for Out-of-Distribution Generalization

From [Entropy: Counterfactual Supervision-Based IB](https://www.mdpi.com/1099-4300/26/3/187) (2024):
> "The counterfactual supervision-based information bottleneck addresses failure situations in out-of-distribution generalization by analyzing what information should be preserved for robust classification."

**Key innovation**: Use counterfactual reasoning to identify spurious vs. causal features
- **Spurious correlations**: High I(Y; T) but low causal influence on X
- **Causal features**: Preserved under interventions, true predictors

### IB for Uncertainty Quantification

From [ScienceDirect: IB-UQ Framework](https://www.sciencedirect.com/science/article/abs/pii/S0021999124003383) (2024):
> "We propose a novel framework for uncertainty quantification via information bottleneck (IB-UQ) for scientific machine learning tasks, measuring epistemic uncertainty through information-theoretic bounds."

**Connection to ARR-COC-0-1:**
- High H(Task | Patches) → high epistemic uncertainty
- Allocate tokens to reduce uncertainty via I(Patches; Task)

### IB for Vision-Language Alignment

From [arXiv: Revisiting Language Bottleneck Models](https://arxiv.org/html/2406.15816v1) (2024):
> "Language bottleneck models extract different features from images compared to black-box models. The information bottleneck framework explains why forcing representation through language improves interpretability."

**Insight**: Language as structured bottleneck:
- **Text modality** forces semantic compression (can't encode pixel noise)
- **IB perspective**: I(Image; Text) << I(Image; Pixels)
- **Benefit**: Human-interpretable representations

### Comprehensive IB for Attribution

From [CVPR 2025: Comprehensive Information Bottleneck](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_Comprehensive_Information_Bottleneck_for_Unveiling_Universal_Attribution_to_Interpret_Vision_CVPR_2025_paper.pdf):
> "The comprehensive information bottleneck reveals feature attribution in vision models by identifying which input variables contribute most to decision-making through mutual information analysis."

**Application**: Interpretable AI, saliency maps, attention visualization

## Section 9: Connections to Related Concepts

### Information Bottleneck vs Rate-Distortion

**Rate-distortion theory:**
```
R(D) = min I(X; X̂)
       s.t. E[d(X, X̂)] ≤ D
```

**Information bottleneck:**
```
R(I_relevance) = min I(Y; T)
                 s.t. I(X; T) ≥ I_relevance
```

**Key difference**:
- **RD**: Minimizes distortion w.r.t. input X
- **IB**: Maximizes relevance w.r.t. target X (different variable!)

From existing knowledge ([cognitive-mastery/14-rate-distortion-theory.md](14-rate-distortion-theory.md)):
> "Rate-distortion theory addresses: what is the minimum number of bits per symbol (rate R) needed such that reconstruction doesn't exceed distortion D? IB asks: what is minimum compression such that task-relevant information exceeds I_min?"

### Information Bottleneck vs Sufficient Statistics

**Sufficient statistic T for parameter θ:**
```
P(X | T, Y) = P(X | T)
```

**IB finds approximate sufficient statistics:**
- T is sufficient for X given Y if I(X; Y | T) = 0
- IB relaxes this: minimize I(Y; T) while maximizing I(X; T)
- Trade-off: Perfect sufficiency requires T = Y (no compression)

### Information Bottleneck vs Minimal Sufficient Statistics

**Minimal sufficient statistic**: Smallest T that's still sufficient

**IB connection**:
- When β → ∞: IB recovers minimal sufficient statistic
- When β = 0: IB gives trivial compression (discard everything)
- Intermediate β: Pareto frontier of compression-relevance

## Section 10: Open Problems and Future Directions

### Estimating Mutual Information in High Dimensions

**Challenge**: I(X; T) intractable for continuous high-dimensional spaces

**Current approaches:**
- **Variational bounds**: MINE, InfoNCE, NWJ estimators
- **Sample-based**: k-NN methods, kernel density estimation
- **Neural estimators**: Train discriminator to approximate MI

**Open question**: Tight, scalable MI estimators for deep learning

### IB for Causal Representation Learning

**Goal**: Identify causal variables through compression

**Idea**: Causally relevant features should:
1. Compress well (low I(Y; T))
2. Predict well under interventions (high I_causal(X; T))

**Challenge**: How to incorporate do-calculus into IB?

### Multi-Task Information Bottleneck

**Setup**: Multiple tasks X_1, X_2, ..., X_k

**Objective:**
```
min I(Y; T) - Σ_i β_i I(X_i; T)
```

**Question**: How to learn shared bottleneck representations?

**ARR-COC-0-1 connection**: Different queries require different relevance

### Adversarial Information Bottleneck

**Robustness**: Compress representations to remove adversarially exploitable information

**Formulation:**
```
min I(Y; T) - β I(X; T) + γ I(Adversarial_noise; T)
```

**Application**: Robust VLMs that ignore adversarial perturbations

## Sources

**Web Research:**

- [MDPI: Theory and Application of the Information Bottleneck Method](https://www.mdpi.com/1099-4300/26/3/187) - Lewandowsky & Bauch, Entropy 26(3), 2024 (accessed 2025-11-16)
- [Springer: Learning features from irrelevant domains through deep bottleneck](https://link.springer.com/article/10.1007/s40747-023-01157-6) - Wen et al., Complex & Intelligent Systems 2024 (accessed 2025-11-16)
- [arXiv: The Local Interaction Basis: Identifying Computationally-Relevant Features](https://arxiv.org/abs/2405.10928) - Bushnaq et al., 2024 (accessed 2025-11-16)
- [ECCV 2024: Broadening the visual encoding of vision-language models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02433.pdf) - Kar et al., ECCV 2024 (accessed 2025-11-16)
- [NeurIPS 2024: VLG-CBM: Concept Bottleneck Models with VLMs](https://papers.nips.cc/paper_files/paper/2024/file/90043ebd68500f9efe84fedf860a64f3-Paper-Conference.pdf) - Srivastava et al., NeurIPS 2024 (accessed 2025-11-16)
- [LREC 2024: Q-Former Guided Multimodal Sentiment Classification](https://aclanthology.org/2024.lrec-main.180.pdf) - Feng et al., LREC 2024 (accessed 2025-11-16)
- [Springer Applied Intelligence: Comprehensive Review of Model Compression](https://link.springer.com/article/10.1007/s10489-024-05747-w) - Dantas et al., Applied Intelligence 2024 (accessed 2025-11-16)
- [ScienceDirect: IB-UQ Framework for Uncertainty Quantification](https://www.sciencedirect.com/science/article/abs/pii/S0021999124003383) - Guo et al., Journal of Computational Physics 2024 (accessed 2025-11-16)
- [arXiv: Revisiting Language Bottleneck Models for Image Classification](https://arxiv.org/html/2406.15816v1) - arXiv:2406.15816, 2024 (accessed 2025-11-16)
- [CVPR 2025: Comprehensive Information Bottleneck for Universal Attribution](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_Comprehensive_Information_Bottleneck_for_Unveiling_Universal_Attribution_to_Interpret_Vision_CVPR_2025_paper.pdf) - Hong et al., CVPR 2025 (accessed 2025-11-16)

**Existing Knowledge Base:**

- [information-theory/00-shannon-entropy-mutual-information.md](../information-theory/00-shannon-entropy-mutual-information.md) - Information bottleneck principle, IB in deep learning, ARR-COC-0-1 connections
- [cognitive-mastery/12-shannon-entropy-information.md](12-shannon-entropy-information.md) - Shannon entropy, propositional knowing
- [cognitive-mastery/14-rate-distortion-theory.md](14-rate-distortion-theory.md) - Rate-distortion framework, compression limits

**Classical References:**

- Tishby, N., Pereira, F. C., & Bialek, W. (1999). "The Information Bottleneck Method." _Allerton Conference on Communication and Computation_.
- Alemi, A. A., et al. (2017). "Deep Variational Information Bottleneck." _ICLR 2017_.
- Saxe, A., et al. (2018). "On the Information Bottleneck Theory of Deep Learning." _ICLR 2018_.

**ARR-COC-0-1 Code References:**

- `arr_coc/knowing.py` - InformationScorer (propositional), CrossAttentionScorer (participatory)
- `arr_coc/balancing.py` - TensionBalancer (opponent processing, IB trade-off navigation)
- `arr_coc/attending.py` - Token allocation (rate-distortion optimization)
- `arr_coc/adapter.py` - Quality adapter (learned R-D function, procedural knowing)

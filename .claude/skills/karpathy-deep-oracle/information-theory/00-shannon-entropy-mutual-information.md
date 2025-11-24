# Information Theory: Shannon Entropy and Mutual Information

Deep dive into information-theoretic foundations for understanding visual compression, relevance quantification, and optimal token allocation in vision-language models.

---

## 1. Shannon Entropy: Measuring Uncertainty

### 1.1 Information Content (Self-Information)

The fundamental unit of information theory quantifies the "surprise" of observing an event:

```
I(x) = -log₂ p(x)
```

**Properties:**
- Rare events (low p) → high information content (high surprise)
- Certain events (p=1) → zero information content
- Measured in bits when using log₂

**Intuition:** If you flip a fair coin and it lands heads, that's 1 bit of information. If you flip a weighted coin that lands heads 90% of the time, seeing heads provides only ~0.15 bits.

From [Information Theory Fundamentals](https://nimasarang.com/blog/2024-08-24-information-theory/) (accessed 2025-11-14):
> "It measures the surprise or 'news value' of observing event x. I(x) is also referred to as the number of bits needed to encode the event."

### 1.2 Shannon Entropy Definition

Shannon entropy is the **expected information content** across all possible events:

```
H(X) = E[-log p(X)] = -Σ p(x) log p(x)
```

**Key convention:** If p(x) = 0, then p(x)log p(x) = 0 (by limit: lim[x→0+] x log x = 0)

**Practical examples:**
- English text: ~1-1.5 bits per character (predictable patterns)
- Random string (26 letters): log₂(26) ≈ 4.7 bits per character
- Fair coin: 1 bit per flip
- Weighted coin (90% heads): ~0.47 bits per flip

**Maximum entropy:**
- Uniform distribution achieves maximum entropy: H(X) = log₂(n) for n outcomes
- Any deviation from uniform reduces entropy (more predictable)

### 1.3 Entropy as Compression Limit

From Claude Shannon's Source Coding Theorem:
> "No lossless compression can encode a source more efficiently than its entropy on average."

This explains:
- Why English text compresses well (~1.5 bits actual vs ~4.7 bits random)
- Why random data is incompressible (already at maximum entropy)
- Why encryption appears random (high entropy by design)

**ARR-COC-0-1 connection:** Propositional knowing (InformationScorer) uses Shannon entropy to measure statistical information content in image patches. High entropy patches contain more "surprising" visual information.

---

## 2. Mutual Information: Shared Information

### 2.1 Definition

Mutual information quantifies how much knowing one variable reduces uncertainty about another:

```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
```

**Properties:**
- Symmetric: I(X;Y) = I(Y;X)
- Non-negative: I(X;Y) ≥ 0
- Zero iff X and Y are independent
- Bounded: I(X;Y) ≤ min(H(X), H(Y))

### 2.2 Interpretation as KL Divergence

From mutual information definition (accessed 2025-11-14):

```
I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))
```

This measures the "price" of assuming X and Y are independent when they're actually dependent.

**Intuition:** If you encode a correlated pair (X,Y) as if they were independent, you waste bits. Mutual information quantifies this waste.

### 2.3 Conditional Mutual Information

```
I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
         = E_Z[D_KL(P(X,Y|Z) || P(X|Z)P(Y|Z))]
```

Measures information shared between X and Y given knowledge of Z.

**Applications:**
- Feature selection (removing redundant features)
- Causal inference (testing conditional independence)
- Information bottleneck (relevant information extraction)

### 2.4 Data Processing Inequality

**Critical theorem:** If X → Y → Z forms a Markov chain:

```
I(X;Z) ≤ I(X;Y)
```

**Implication:** Processing cannot create new information, only lose it.

From [Bridging Data Processing Inequality](https://iclr-blogposts.github.io/2024/blog/dpi-fsvi/) (accessed 2025-11-14):
> "The KL divergence is sometimes also referred to as 'relative entropy', so we could also call this the 'relative data processing inequality'."

**ARR-COC-0-1 connection:** When compressing visual features from 13 channels → variable LOD tokens, we inevitably lose mutual information with the original image. The goal is to preserve information **relevant** to the query.

---

## 3. KL Divergence: Relative Entropy

### 3.1 Definition and Interpretation

Kullback-Leibler divergence measures the "extra bits" needed to encode samples from P using a code optimized for Q:

```
D_KL(P || Q) = H(P, Q) - H(P)
             = Σ p(x) log(p(x)/q(x))
             = E_P[log(p(X)/q(X))]
```

**Key properties:**
- **Non-negative:** D_KL(P || Q) ≥ 0 (Gibbs' inequality)
- **Zero iff identical:** D_KL(P || Q) = 0 ⟺ P = Q
- **Asymmetric:** D_KL(P || Q) ≠ D_KL(Q || P)
- **Not a metric:** Doesn't satisfy triangle inequality

### 3.2 Proof of Non-Negativity (Gibbs' Inequality)

Using Jensen's inequality (since -log is convex):

```
D_KL(P || Q) = Σ p(x)(-log(q(x)/p(x)))
             ≥ -log(Σ p(x)(q(x)/p(x)))    [Jensen's inequality]
             = -log(Σ q(x))
             = -log(1) = 0
```

### 3.3 Behavior at Zero Probabilities

**Critical edge cases:**

1. **p(x) = 0, q(x) > 0:** Contribution is 0 (by limit)
   - Q can make "mistakes" about impossible events without penalty

2. **p(x) > 0, q(x) = 0:** Divergence → +∞
   - Q **cannot** miss events that P considers possible

3. **p(x) = 0, q(x) = 0:** Undefined (but doesn't occur in practice)

**Practical fix:** Add small epsilon to Q to avoid log(0) in implementations.

### 3.4 KL Divergence in Neural Networks

From [KL Divergence vs Cross-Entropy](https://medium.com/@katykas/kl-divergence-vs-cross-entropy-understanding-the-difference-and-similarities-9cbc0c796598) (accessed 2025-11-14):

**Classification loss:**
```
arg min_θ D_KL(P_data || P_model) = arg min_θ H(P_data, P_model)
```

Since H(P_data) is constant w.r.t. model parameters, minimizing KL divergence = minimizing cross-entropy.

**Variational inference:**
```
arg min_q D_KL(q(z|x) || p(z|x))  [Reverse KL]
```

Different KL direction leads to different optimization behavior (see Section 4).

---

## 4. Cross-Entropy: Measuring Coding Efficiency

### 4.1 Definition

Cross-entropy measures the average bits needed to encode samples from P using a code optimized for Q:

```
H(P, Q) = -Σ p(x) log q(x)
        = E_P[-log q(X)]
```

**Relationship to entropy and KL:**
```
H(P, Q) = H(P) + D_KL(P || Q)
```

Always: H(P, Q) ≥ H(P), with equality iff P = Q.

### 4.2 Binary Cross-Entropy

For binary classification with true label y ∈ {0,1} and predicted probability q:

```
BCE(y, q) = -[y log(q) + (1-y) log(1-q)]
```

**Intuition:** Penalizes confident wrong predictions more than uncertain ones.

From [How Neural Networks Learn: A Probabilistic Viewpoint](https://towardsdatascience.com/how-neural-networks-learn-a-probabilistic-viewpoint-0f6a78dc58e2/) (accessed 2025-11-14):
> "Minimizing cross-entropy or KL-Divergence achieves the same solution. KL-Divergence has a better interpretation as its minimum is zero..."

### 4.3 Categorical Cross-Entropy

For multi-class classification with K classes:

```
CCE(P, Q) = -Σ_{k=1}^K p_k log q_k
```

Where P is one-hot encoded true distribution, Q is predicted softmax output.

### 4.4 Cross-Entropy vs MSE

**Why cross-entropy for classification?**

Mean Squared Error:
```
MSE = Σ (p_k - q_k)²
```

Problems:
- Doesn't match probabilistic interpretation
- Slower convergence (saturating gradients)
- Not invariant to reparameterization

Cross-entropy:
- Natural loss for maximum likelihood estimation
- Faster convergence (better gradient flow)
- Proper scoring rule (encourages calibrated probabilities)

**ARR-COC-0-1 connection:** When training the quality adapter (4th P: procedural knowing), cross-entropy loss optimizes the model to predict relevance scores that match human judgments.

---

## 5. Rate-Distortion Theory: Optimal Compression Trade-offs

### 5.1 The Rate-Distortion Problem

Rate-distortion theory formalizes lossy compression:

**Goal:** Find optimal compression that minimizes:
```
R(D) = min_{P(X̂|X): E[d(X,X̂)] ≤ D} I(X; X̂)
```

Where:
- R: bit rate (compression level)
- D: distortion (reconstruction error)
- I(X; X̂): mutual information between source X and reconstruction X̂

From [Rate-Distortion Theory](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952) (accessed 2025-11-14):
> "Rate-distortion theory formalizes the optimal way to compress information while minimizing such distortions, by considering factors such as capacity limitations."

### 5.2 Rate-Distortion Function

**Properties:**
- **Monotonic decreasing:** Higher rate → lower distortion
- **Convex:** Diminishing returns from additional bits
- **Bounded:** R(0) = H(X) (no compression), R(D_max) = 0 (maximum distortion)

**For Gaussian source with MSE distortion:**
```
R(D) = ½ log₂(σ²/D)  for D ≤ σ²
     = 0              for D > σ²
```

### 5.3 Shannon's Source Coding Theorem

**Lossless compression:** Cannot compress below entropy H(X) on average.

**Lossy compression:** Can achieve rate R(D) for any distortion level D.

### 5.4 Practical Applications

From [Fundamental Limits of Prompt Compression](https://arxiv.org/abs/2407.15504) (accessed 2025-11-14):

Modern applications:
- **Image compression:** JPEG, WebP (transform coding + quantization)
- **Video compression:** H.264, VP9 (temporal + spatial compression)
- **Neural compression:** Learned autoencoders optimizing R-D trade-off
- **Prompt compression:** Compressing LLM context within rate-distortion framework

### 5.5 Distortion Measures

Common distortion metrics:

**Mean Squared Error (MSE):**
```
d(x, x̂) = ||x - x̂||²
```

**Perceptual metrics:**
- SSIM (structural similarity)
- LPIPS (learned perceptual similarity)
- MS-SSIM (multi-scale SSIM)

**Task-specific:**
- Classification accuracy
- Semantic segmentation IoU
- Detection mAP

**ARR-COC-0-1 connection:** Variable LOD allocation (64-400 tokens per patch) navigates the rate-distortion trade-off. High-relevance patches get more tokens (higher rate, lower distortion), low-relevance patches get fewer tokens (lower rate, acceptable distortion).

---

## 6. Information Bottleneck Principle

### 6.1 The Information Bottleneck Objective

The Information Bottleneck (IB) principle finds compressed representations that preserve relevant information:

**Objective:**
```
min I(X; Z) - β I(Z; Y)
```

Where:
- X: input features
- Z: compressed representation
- Y: target variable
- β: trade-off parameter (Lagrange multiplier)

From [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/abs/1503.02406) (accessed 2025-11-14):
> "The paper analyzes Deep Neural Networks (DNNs) using the information bottleneck (IB) principle, quantifying them by mutual information between layers."

### 6.2 Theoretical Framework

**Markov chain:** X → Z → Ŷ

**Data processing inequality:** I(X; Y) ≥ I(Z; Y)

**IB curve:** Plots achievable (I(X; Z), I(Z; Y)) pairs
- Similar to rate-distortion curve
- Pareto frontier of compression vs prediction

**Optimal solution:**
```
P(Z|X) ∝ P(Z) exp(β Σ_y P(y|X) log P(y|Z))
```

Self-consistent equations (Blahut-Arimoto algorithm).

### 6.3 Information Bottleneck in Deep Learning

From [Information Bottleneck Method](https://en.wikipedia.org/wiki/Information_bottleneck_method) (accessed 2025-11-14):
> "Applications include distributional clustering and dimension reduction, and more recently it has been suggested as a theoretical foundation for deep learning."

**Tishby's hypothesis (2015):**
1. DNNs undergo two phases during training:
   - **Fitting phase:** I(Z; X) and I(Z; Y) both increase
   - **Compression phase:** I(Z; X) decreases while I(Z; Y) plateaus
2. Compression explains generalization
3. Each layer acts as an information bottleneck

**Criticism (Saxe et al. 2018):**
From [On the Information Bottleneck Theory of Deep Learning](https://openreview.net/forum?id=ry_WPG-A-) (accessed 2025-11-14):
- Compression depends on activation function (sigmoid vs ReLU)
- Doesn't occur with ReLU networks
- Generalization can occur without compression

**Current understanding:**
- IB provides useful lens, not complete explanation
- Applies more to some architectures than others
- Valuable for understanding representation learning

### 6.4 Variational Information Bottleneck

**Practical formulation:**
```
L = E[log P(Y|Z)] - β D_KL(P(Z|X) || P(Z))
```

Similar to VAE objective:
- Reconstruction term: maximize I(Z; Y)
- Regularization term: minimize I(X; Z)

**ARR-COC-0-1 connection:** The three ways of knowing (propositional, perspectival, participatory) can be viewed as extracting different types of relevant information. The balancing module navigates information bottleneck trade-offs by allocating representation capacity (tokens) based on query-relevant information.

---

## 7. Applications to Machine Learning

### 7.1 Loss Functions

**Cross-entropy loss (classification):**
```python
# Binary classification
loss = -[y * log(p) + (1-y) * log(1-p)]

# Multi-class classification
loss = -Σ y_k * log(p_k)
```

**KL divergence loss (distribution matching):**
```python
# Variational inference
loss = D_KL(q(z|x) || p(z))

# Knowledge distillation
loss = D_KL(P_student || P_teacher)
```

### 7.2 Regularization

**Information-theoretic regularization:**

**VAE ELBO:**
```
L = E[log P(X|Z)] - β D_KL(Q(Z|X) || P(Z))
```

**β-VAE:** Higher β → stronger compression → more disentangled representations

**InfoGAN:** Maximize I(C; G(Z, C)) for interpretable latent codes

### 7.3 Feature Selection

**Mutual information criteria:**

**mRMR (minimum Redundancy Maximum Relevance):**
```
max [I(X_i; Y) - (1/|S|) Σ_{X_j ∈ S} I(X_i; X_j)]
```

Balances:
- Relevance: I(X_i; Y) high
- Redundancy: I(X_i; X_j) low

**Conditional mutual information:**
```
I(X_i; Y | S) = I(X_i; Y) - I(X_i; S)
```

Select features that add new information given already selected features.

### 7.4 Model Compression

From [Rate Distortion Optimization for LLM Compression](https://arxiv.org/pdf/2505.03031) (accessed 2025-11-14):

**Quantization:** Map high-precision weights to low-precision using rate-distortion principles
```
R(D) framework for optimal quantization strategy
```

**Pruning:** Remove low-information connections/neurons
```
Minimize I(W_pruned; W_full) subject to task performance
```

**Knowledge distillation:** Compress teacher model into student
```
min D_KL(P_student || P_teacher)
```

**Neural architecture search:** Find architectures that optimize information flow

### 7.5 Uncertainty Quantification

**Epistemic uncertainty (model uncertainty):**
```
H(Y|X) = -Σ P(Y|X) log P(Y|X)  [Entropy of predictions]
```

**Aleatoric uncertainty (data uncertainty):**
```
E_θ[H(Y|X,θ)]  [Expected entropy over model parameters]
```

**Mutual information between parameters and predictions:**
```
I(Y; θ | X) = H(Y|X) - E_θ[H(Y|X,θ)]
```

High mutual information → high epistemic uncertainty → need more data.

**ARR-COC-0-1 connection:** Uncertainty in relevance scores affects token allocation. High uncertainty patches might receive moderate token budgets (hedge against both under/over-allocation).

---

## 8. ARR-COC-0-1 Information-Theoretic Architecture

### 8.1 Propositional Knowing as Entropy Measurement

The InformationScorer in `knowing.py` implements Shannon entropy:

```python
# Measure information content in image patches
def compute_entropy(patch):
    # Flatten spatial dimensions
    flat = patch.view(-1, C)

    # Estimate probability distribution (histogram)
    hist = compute_histogram(flat)

    # Shannon entropy
    entropy = -sum(p * log(p) for p in hist if p > 0)

    return entropy
```

**Interpretation:**
- High entropy patches: Complex textures, edges, high-frequency detail
- Low entropy patches: Uniform regions, smooth gradients, low-frequency content

**Connection to compression:**
- High entropy → needs more bits to encode → deserves more tokens
- Low entropy → compressible → can use fewer tokens

### 8.2 Cross-Modal Mutual Information

Participatory knowing (cross-attention scorer) measures query-content coupling:

```
I(Query; Patch) = H(Patch) - H(Patch | Query)
```

**Implementation via attention:**
```python
# Query-aware relevance
attention_weights = softmax(Q @ K^T / sqrt(d))
relevance = attention_weights @ V

# Higher attention → higher mutual information
# Patch provides information relevant to query
```

**Interpretation:**
- High I(Q; P): Patch informs query (participatory relevance)
- Low I(Q; P): Patch independent of query (not participatory relevant)

### 8.3 Rate-Distortion in Token Allocation

The attending module solves a rate-distortion problem:

**Rate constraint:**
```
Total tokens allocated = K patches × [64, 400] tokens/patch
```

**Distortion constraint:**
```
Task performance must exceed threshold
```

**Optimization:**
```
Allocate tokens to minimize: E[Task Error]
Subject to: Total tokens ≤ Budget
```

From propositional knowing (entropy), perspectival knowing (salience), and participatory knowing (query-coupling), the system estimates:

```
tokens_i = f(entropy_i, salience_i, query_relevance_i)
```

Where f is learned to optimize the rate-distortion trade-off.

### 8.4 Information Bottleneck in Opponent Processing

The balancing module navigates information bottleneck trade-offs:

**Compress ↔ Particularize tension:**
```
min I(Compressed; Original) - β I(Compressed; TaskRelevant)
```

- Compress: Reduce mutual information with original (lower rate)
- Particularize: Preserve mutual information with task (lower distortion)

**Exploit ↔ Explore tension:**
```
Exploit: Use known high-I(Patch; Query) regions
Explore: Sample patches to reduce H(Patch|Query)
```

**Focus ↔ Diversify tension:**
```
Focus: Allocate tokens to max I(Tokens; Task) patches
Diversify: Spread tokens to reduce H(Task|AllPatches)
```

### 8.5 Procedural Knowing as Learned R-D Function

The quality adapter (4th P) learns the optimal rate-distortion function:

**Training objective:**
```
Learn: tokens_optimal(patch) = R(D_target)

Where D_target is task-dependent distortion threshold
```

**Learned mapping:**
```python
# Input: (entropy, salience, query_coupling)
# Output: optimal token allocation

def quality_adapter(features):
    # Learn to map relevance → tokens
    # Implicitly learns rate-distortion curve
    tokens = network(features)
    return clip(tokens, 64, 400)
```

**Connection to information theory:**
- Network learns I(Patch; Task) estimator
- Maps estimated mutual information → token budget
- Optimizes global rate-distortion trade-off

### 8.6 Why Information Theory Matters for ARR-COC-0-1

**Theoretical foundation:**
1. **Propositional knowing** = measuring H(Patch) via entropy
2. **Participatory knowing** = measuring I(Query; Patch) via attention
3. **Token allocation** = solving rate-distortion problem R(D)
4. **Compression tensions** = navigating information bottleneck

**Practical benefits:**
1. **Principled compression:** Information theory provides optimization target
2. **Interpretable relevance:** Mutual information explains "why this patch matters"
3. **Optimal allocation:** Rate-distortion theory guarantees no wasted tokens
4. **Generalization:** IB principle explains why learned representations transfer

**Future directions:**
1. **Explicit I(Patch; Task) estimation:** Train networks to predict mutual information directly
2. **Adaptive β-IB:** Adjust compression-performance trade-off per query
3. **Multi-scale R-D:** Different rate-distortion curves for different semantic levels
4. **Causal information:** Measure I(Intervention; Outcome) for causal relevance

---

## Sources

**Web Research:**

- [Information Theory Fundamentals: Entropy, Cross-Entropy, and KL Divergence](https://nimasarang.com/blog/2024-08-24-information-theory/) - Nima Sarang, August 24, 2024 (accessed 2025-11-14)
- [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/abs/1503.02406) - Tishby & Zaslavsky, arXiv:1503.02406, 2015 (accessed 2025-11-14)
- [On the Information Bottleneck Theory of Deep Learning](https://openreview.net/forum?id=ry_WPG-A-) - Saxe et al., ICLR 2018 (accessed 2025-11-14)
- [Information Bottleneck Method](https://en.wikipedia.org/wiki/Information_bottleneck_method) - Wikipedia (accessed 2025-11-14)
- [Fundamental Limits of Prompt Compression](https://arxiv.org/abs/2407.15504) - Nagle et al., arXiv:2407.15504, 2024 (accessed 2025-11-14)
- [Rate-Distortion Theory for Mixed States](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952) - D'Amato et al., PLOS Computational Biology, 2024 (accessed 2025-11-14)
- [KL Divergence vs. Cross-Entropy](https://medium.com/@katykas/kl-divergence-vs-cross-entropy-understanding-the-difference-and-similarities-9cbc0c796598) - Ekaterina Kasilina, Medium, 2024 (accessed 2025-11-14)
- [How Neural Networks Learn: A Probabilistic Viewpoint](https://towardsdatascience.com/how-neural-networks-learn-a-probabilistic-viewpoint-0f6a78dc58e2/) - Towards Data Science, December 26, 2024 (accessed 2025-11-14)
- [Bridging Data Processing Inequality and Function-Space Variational Inference](https://iclr-blogposts.github.io/2024/blog/dpi-fsvi/) - ICLR Blogposts 2024, May 7, 2024 (accessed 2025-11-14)

**Additional References:**

- Shannon, C. E. (1948). "A Mathematical Theory of Communication." Bell System Technical Journal.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Tishby, N., Pereira, F. C., & Bialek, W. (2000). "The Information Bottleneck Method." arXiv:physics/0004057.
- Alemi, A. A., et al. (2017). "Deep Variational Information Bottleneck." ICLR 2017.

**ARR-COC-0-1 Code References:**

- `arr_coc/knowing.py` - InformationScorer (propositional knowing via Shannon entropy)
- `arr_coc/balancing.py` - TensionBalancer (opponent processing, information bottleneck navigation)
- `arr_coc/attending.py` - Token allocation (rate-distortion optimization)
- `arr_coc/adapter.py` - Quality adapter (learned R-D function, procedural knowing)

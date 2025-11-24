# Resource Allocation & Optimization Algorithms

## Overview

Resource allocation optimization addresses the fundamental computational problem of distributing limited resources across competing demands to maximize utility or minimize cost. This framework underlies attention mechanisms in cognitive systems, token budget allocation in vision-language models, and computational resource scheduling in distributed systems. Understanding optimization algorithms—from classical linear programming to modern online algorithms—provides the mathematical foundation for designing systems that make optimal decisions under constraints.

**Core principle**: Optimal resource allocation requires formulating objectives, constraints, and decision variables in a mathematical framework that admits efficient solution algorithms.

## Section 1: Resource Allocation Problems (Objectives, Constraints)

### Problem Formulation

Resource allocation problems share a common mathematical structure:

**Decision variables**: What resources to allocate where (x₁, x₂, ..., xₙ)
**Objective function**: What to optimize (maximize utility or minimize cost)
**Constraints**: Hard limits on available resources and allocation rules

From [Attention as Resource Allocation](../cognitive-foundations/03-attention-resource-allocation.md):
> "Attention operates as a limited cognitive resource that must be strategically allocated across competing demands."

This cognitive principle translates directly to computational resource allocation: finite capacity requires strategic allocation mechanisms.

### General Resource Allocation Framework

**Standard formulation**:
```
Maximize:   f(x₁, x₂, ..., xₙ)           [objective function]
Subject to: g₁(x₁, ..., xₙ) ≤ b₁         [resource constraint 1]
            g₂(x₁, ..., xₙ) ≤ b₂         [resource constraint 2]
            ...
            xᵢ ≥ 0 for all i              [non-negativity]
```

**Example**: Token budget allocation in ARR-COC-0-1:
- **Decision variables**: tokens allocated to each of K=200 patches
- **Objective**: Maximize information capture (relevance realization)
- **Constraint**: Total tokens ≤ 13,200 (average 66 tokens/patch)
- **Bounds**: 64 ≤ tokens_per_patch ≤ 400

### Types of Resource Allocation Problems

**1. Divisible resources** (can split arbitrarily):
- Compute time allocation
- Memory bandwidth distribution
- Attention weight distribution

**2. Indivisible resources** (discrete units):
- GPU assignment to jobs
- Token allocation (must be integers)
- Task scheduling on processors

**3. Multi-objective optimization**:
- Maximize accuracy AND minimize latency
- Balance exploration vs exploitation
- Navigate Vervaekean opponent processing tensions

From [john-vervaeke-oracle/balancing.py](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Navigate tensions between competing constraints through opponent processing, not fixed trade-offs."

## Section 2: Linear Programming (Simplex, Dual, Integer Programming)

### Linear Programming Fundamentals

**Definition**: Optimization with linear objective and linear constraints.

**Standard form**:
```
Minimize:   c^T x
Subject to: Ax ≤ b
            x ≥ 0
```

Where:
- c = cost vector (coefficients of objective function)
- A = constraint matrix
- b = resource capacity vector
- x = decision variables

### The Simplex Algorithm

**Key insight**: Optimal solution lies at a vertex of the feasible region (polytope formed by constraints).

**Algorithm**:
1. Start at feasible vertex
2. Move along edge to adjacent vertex that improves objective
3. Repeat until no improving move exists (optimality)

**Complexity**:
- Worst-case: Exponential in number of variables
- Average-case: Polynomial (works well in practice)
- Polynomial variants exist (interior-point methods)

From [ResearchGate - Resource Allocation Optimization](https://www.researchgate.net/publication/382049201_A_Critical_Review_of_Resource_Allocation_Optimization_in_Project_Management) (accessed 2025-11-14):
> "Linear programming provides a foundational framework for resource allocation, enabling systematic optimization of project schedules and resource utilization."

### Dual Problem

Every linear program has a **dual** with complementary structure:

**Primal** (resource allocation):
```
Minimize cost of allocating resources
Subject to meeting demand requirements
```

**Dual** (resource pricing):
```
Maximize value of resources
Subject to prices not exceeding costs
```

**Strong duality theorem**: If primal has optimal solution x*, dual has optimal solution y*, and:
```
primal_optimal_value = dual_optimal_value
```

**Applications**:
- Shadow prices: How much would relaxing a constraint improve the objective?
- Sensitivity analysis: How robust is the solution to parameter changes?

### Integer Linear Programming (ILP)

**Extension**: Require some/all decision variables to be integers.

**Example**: GPU allocation
```
Minimize:   total_cost
Subject to: sum(gpus_per_job[i]) ≤ total_gpus
            gpus_per_job[i] ∈ {0, 1, 2, 4, 8}  [discrete choices]
```

**Complexity**: NP-hard in general (much harder than linear programming)

**Solution methods**:
- **Branch and bound**: Systematically explore solution tree
- **Cutting planes**: Add constraints to eliminate fractional solutions
- **Branch and cut**: Combine both approaches

From [orchestration/03-ml-workload-patterns-k8s.md](../karpathy/orchestration/03-ml-workload-patterns-k8s.md):
> "Gang scheduling requires all-or-nothing GPU allocation—an integer programming problem where partial allocations are infeasible."

## Section 3: Convex Optimization (Gradient Descent, KKT Conditions)

### Convex Optimization Framework

**Definition**: Optimization where objective and feasible region are convex.

**Why convex matters**:
- Local optimum = global optimum (no bad local minima)
- Efficient polynomial-time algorithms exist
- Rich theory (KKT conditions, duality)

**Convexity conditions**:
- **Convex function**: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for λ ∈ [0,1]
- **Convex set**: Line segment between any two points stays in set

**Standard form**:
```
Minimize:   f(x)                    [convex objective]
Subject to: gᵢ(x) ≤ 0, i=1,...,m   [convex inequality constraints]
            hⱼ(x) = 0, j=1,...,p    [affine equality constraints]
```

From [Stanford CS168 - Linear and Convex Programming](https://web.stanford.edu/class/cs168/l/l18.pdf) (accessed 2025-11-14):
> "A good rule of thumb is to equate 'convex' with 'nice' and 'non-convex' with 'nasty,' especially when optimization is concerned."

### Gradient Descent

**Basic algorithm** for unconstrained convex optimization:
```python
x = x_initial
for iteration in range(max_iterations):
    gradient = compute_gradient(f, x)
    x = x - learning_rate * gradient
    if ||gradient|| < tolerance:
        break
```

**Variants**:
- **Stochastic gradient descent (SGD)**: Use noisy gradient estimates (cheap, scales to huge datasets)
- **Momentum**: Add velocity term to smooth updates
- **Adam**: Adaptive learning rates per parameter

**Convergence rate**: O(1/k) for gradient descent on smooth convex functions

### Projected Gradient Descent

**Extension** to constrained optimization:
```python
x = x_initial
for iteration in range(max_iterations):
    gradient = compute_gradient(f, x)
    x_candidate = x - learning_rate * gradient
    x = project_onto_feasible_set(x_candidate)  # Enforce constraints
```

**Application to ARR-COC-0-1**: Token budget allocation with constraints:
```python
# Allocate tokens to patches
tokens = tokens - learning_rate * relevance_gradient
tokens = np.clip(tokens, 64, 400)  # Enforce per-patch bounds
tokens = tokens * (13200 / tokens.sum())  # Enforce total budget
```

### KKT Conditions (Karush-Kuhn-Tucker)

**Necessary conditions** for optimality in constrained convex optimization:

Given problem:
```
Minimize f(x) subject to gᵢ(x) ≤ 0, hⱼ(x) = 0
```

**KKT conditions** at optimal x*:
1. **Stationarity**: ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σμⱼ∇hⱼ(x*) = 0
2. **Primal feasibility**: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. **Dual feasibility**: λᵢ ≥ 0
4. **Complementary slackness**: λᵢgᵢ(x*) = 0

**Interpretation**:
- **Lagrange multipliers** (λᵢ, μⱼ) represent "shadow prices" of constraints
- **Complementary slackness**: Either constraint is active (gᵢ=0) or multiplier is zero (λᵢ=0)

**Application**: If a constraint has λᵢ > 0, relaxing it would improve the objective.

## Section 4: Dynamic Programming (Knapsack, Optimal Substructure)

### Dynamic Programming Principles

**Core idea**: Break problem into overlapping subproblems, solve each once, store solutions.

**Requirements**:
1. **Optimal substructure**: Optimal solution contains optimal solutions to subproblems
2. **Overlapping subproblems**: Same subproblems appear multiple times

**Memoization pattern**:
```python
memo = {}
def solve(state):
    if state in memo:
        return memo[state]
    if base_case(state):
        return base_solution(state)
    result = combine(solve(subproblem1), solve(subproblem2), ...)
    memo[state] = result
    return result
```

### The 0-1 Knapsack Problem

**Problem**: Given n items with weights w₁,...,wₙ and values v₁,...,vₙ, and knapsack capacity W, select items to maximize value without exceeding capacity.

**Decision per item**: Include (1) or exclude (0) each item.

From [Wikipedia - Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem) (accessed 2025-11-14):
> "The knapsack problem is a problem in combinatorial optimization: Given a set of items with specific weights and values, determine which items to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible."

**Dynamic programming solution**:

**State**: DP[i][w] = maximum value achievable using first i items with capacity w

**Recurrence**:
```
DP[i][w] = max(
    DP[i-1][w],              # Don't include item i
    DP[i-1][w-wᵢ] + vᵢ       # Include item i (if fits)
)
```

**Base cases**:
```
DP[0][w] = 0  for all w    # No items → zero value
DP[i][0] = 0  for all i    # No capacity → zero value
```

**Algorithm**:
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    DP = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: Don't include item i-1
            DP[i][w] = DP[i-1][w]
            # Option 2: Include item i-1 (if it fits)
            if weights[i-1] <= w:
                DP[i][w] = max(DP[i][w],
                              DP[i-1][w - weights[i-1]] + values[i-1])

    return DP[n][capacity]
```

**Complexity**:
- **Time**: O(nW) — pseudo-polynomial (depends on capacity W, not just input size)
- **Space**: O(nW) — can be optimized to O(W) using rolling array

### Optimal Substructure in ARR-COC-0-1

**Token allocation as dynamic programming**:

From [john-vervaeke-oracle/attending.py](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Map relevance to resource allocation through multi-scale integration—operating hierarchically across scales with no single 'correct' resolution."

**Hierarchical allocation**:
```python
def allocate_tokens_hierarchical(patches, total_budget):
    # Stage 1: Coarse allocation by region
    region_budgets = allocate_to_regions(patches, total_budget)

    # Stage 2: Fine allocation within regions (subproblem)
    for region, budget in region_budgets.items():
        patch_allocations[region] = allocate_to_patches(
            patches[region], budget
        )

    return patch_allocations
```

**Optimal substructure**: Optimal total allocation requires optimal allocation within each region.

## Section 5: Online Algorithms (Competitive Ratio, Ski Rental Problem)

### Online vs Offline Algorithms

**Offline algorithm**: Knows entire input sequence in advance (oracle access)

**Online algorithm**: Makes decisions based only on past/present inputs (no future knowledge)

**Example**: Caching
- **Offline**: Optimal eviction using future access pattern (Belady's algorithm)
- **Online**: LRU, LFU, FIFO (no future knowledge)

From [Simons Institute - Competitive Analysis of Online Algorithms](https://simons.berkeley.edu/news/competitive-analysis-online-algorithms) (accessed 2025-11-14):
> "This framework compares the performance of an online algorithm that makes decisions without knowledge of the future with the performance of the best sequence of decisions that could have been made in hindsight."

### Competitive Analysis

**Definition**: Online algorithm ALG is **c-competitive** if:
```
ALG(σ) ≤ c · OPT(σ) + α
```

For all input sequences σ, where:
- ALG(σ) = cost of online algorithm on input σ
- OPT(σ) = cost of optimal offline algorithm on input σ
- c = competitive ratio (constant factor)
- α = additive constant (often 0)

**Interpretation**: Online algorithm is within factor c of optimal (in hindsight).

### Ski Rental Problem

**Classic online algorithm example**:

**Problem**: Rent skis for $1/day or buy for $B. Don't know how many days you'll ski.

**Online strategies**:
1. **Always rent**: Cost = number of days
2. **Always buy**: Cost = B
3. **Rent then buy**: Rent for B-1 days, then buy

**Analysis**:
- **Strategy 3** is **2-competitive**:
  - If ski ≤ B days: Pay at most 2B (rent B-1, buy B)
  - If ski > B days: Pay 2B-1 (close to optimal B)
  - Competitive ratio: (2B-1)/B → 2 as B grows

**Theorem**: No deterministic online algorithm can be better than 2-competitive for ski rental.

**Randomized algorithms** can achieve better competitive ratios using probability.

### Online Resource Allocation Applications

**1. Cloud resource provisioning**:
```
Problem: Allocate VMs to arriving job requests
Online challenge: Don't know future job arrivals
Objective: Minimize cost while meeting SLAs
```

**2. Attention allocation in streaming vision**:
```
Problem: Allocate tokens to incoming video frames
Online challenge: Don't know future frame content
Objective: Maximize relevance realization under budget
```

From [MIT - Online Algorithms and Competitive Analysis](https://people.csail.mit.edu/ghaffari/AA17/Notes/S9.pdf) (accessed 2025-11-14):
> "Competitive analysis is a method invented for analyzing online algorithms, in which the performance of an online algorithm is compared to an optimal offline algorithm."

## Section 6: Approximation Algorithms (Greedy, Local Search)

### Approximation Algorithms Framework

**Motivation**: NP-hard problems have no known polynomial-time exact algorithms.

**Approximation factor α**: Algorithm guarantees solution within factor α of optimal:
```
ALG(I) ≤ α · OPT(I)  for all instances I
```

**Example**: 2-approximation for vertex cover runs in polynomial time and guarantees solution ≤ 2 × optimal.

### Greedy Algorithms

**General pattern**:
1. Make locally optimal choice at each step
2. Never revise previous choices
3. Hope local optimality leads to global optimality

**When greedy works** (with proof):
- Matroid optimization (e.g., minimum spanning tree)
- Fractional knapsack
- Activity selection

**When greedy gives approximation** (but not optimal):
- Set cover (ln n approximation)
- Vertex cover (2-approximation)

**Greedy knapsack approximation**:
```python
def greedy_knapsack(weights, values, capacity):
    # Compute value-to-weight ratio
    items = [(v/w, w, v) for w, v in zip(weights, values)]
    items.sort(reverse=True)  # Descending by ratio

    total_value = 0
    total_weight = 0
    for ratio, w, v in items:
        if total_weight + w <= capacity:
            total_value += v
            total_weight += w

    return total_value
```

**Performance**: No constant approximation guarantee in general, but works well in practice.

### Local Search

**Pattern**:
```python
solution = initial_solution()
while True:
    neighbor = best_neighbor(solution)
    if cost(neighbor) < cost(solution):
        solution = neighbor
    else:
        break  # Local optimum
return solution
```

**Strengths**:
- Simple to implement
- Works on complex combinatorial problems
- Often finds good solutions quickly

**Weaknesses**:
- Can get stuck in local optima
- No approximation guarantees in general

**Variants**:
- **Simulated annealing**: Accept worse solutions with decreasing probability
- **Tabu search**: Maintain list of forbidden moves to escape local optima
- **Genetic algorithms**: Maintain population, use mutation and crossover

## Section 7: Applications (Compute Allocation, Bandwidth, Attention Budgets)

### Application 1: Compute Resource Allocation

**Problem**: Allocate K GPUs to N training jobs to minimize total completion time.

**Constraints**:
- Each job requires specific GPU count (gang scheduling)
- Limited GPU availability
- Jobs have priorities

From [orchestration/03-ml-workload-patterns-k8s.md](../karpathy/orchestration/03-ml-workload-patterns-k8s.md):
> "Gang scheduling for distributed multi-GPU jobs requires all-or-nothing allocation—partial GPU assignments result in wasted resources and failed training."

**Formulation** (integer programming):
```
Minimize:   sum_i (completion_time_i * priority_i)
Subject to: sum_i (gpus_i * active_i) ≤ K
            gpus_i ∈ {required_gpus for job i}
            active_i ∈ {0, 1}
```

**Solution approaches**:
- **Exact**: Branch and bound (small instances)
- **Heuristic**: Priority-based greedy scheduling (practical)
- **Online**: Competitive algorithms for dynamic arrivals

### Application 2: Network Bandwidth Allocation

**Problem**: Distribute bandwidth across competing flows to maximize utility.

**Max-min fairness**:
1. Maximize minimum allocated bandwidth
2. Then maximize second-minimum
3. Continue until all bandwidth allocated

**Algorithm**:
```python
def max_min_fairness(demands, total_bandwidth):
    n = len(demands)
    allocation = [0] * n
    remaining = total_bandwidth

    while remaining > 0:
        # Give equal share to unsaturated flows
        unsaturated = [i for i in range(n) if allocation[i] < demands[i]]
        if not unsaturated:
            break

        share = remaining / len(unsaturated)
        for i in unsaturated:
            increase = min(share, demands[i] - allocation[i])
            allocation[i] += increase
            remaining -= increase

    return allocation
```

**Properties**:
- Fair (no flow can increase without decreasing another smaller flow)
- Efficient (allocates all bandwidth)
- Envy-free (no flow prefers another's allocation)

### Application 3: Vision Token Budget Allocation (ARR-COC-0-1)

**Problem**: Allocate tokens across K=200 image patches under total budget constraint.

**ARR-COC-0-1 specific formulation**:
```
Maximize:   sum_i (relevance_i(tokens_i))
Subject to: sum_i tokens_i = 13,200         [total budget]
            64 ≤ tokens_i ≤ 400 for all i   [LOD bounds]
            tokens_i ∈ ℤ                     [discrete tokens]
```

**Solution approach** (from attending.py):

1. **Measure relevance** (knowing.py):
   - Propositional: Shannon entropy of patch
   - Perspectival: Salience from visual attention
   - Participatory: Query-patch cross-attention

2. **Balance tensions** (balancing.py):
   - Compress ↔ Particularize
   - Exploit ↔ Explore
   - Focus ↔ Diversify

3. **Allocate tokens** (attending.py):
   - Map relevance scores → token budgets
   - Enforce bounds [64, 400]
   - Normalize to total budget

From [john-vervaeke-oracle/realizing.py](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Execute compression and return focused features through the complete pipeline: knowing → balancing → attending → realizing."

**Optimization algorithm**:
```python
def allocate_tokens(relevance_scores, total_budget=13200):
    K = len(relevance_scores)

    # Initial uniform allocation
    tokens = np.full(K, total_budget // K)

    # Iterative refinement
    for iteration in range(max_iters):
        # Compute marginal utility of adding tokens
        marginal_utility = compute_marginal_relevance(relevance_scores, tokens)

        # Redistribute: take from low-utility, give to high-utility
        donors = np.where((marginal_utility < threshold) & (tokens > 64))[0]
        recipients = np.where((marginal_utility > threshold) & (tokens < 400))[0]

        if len(donors) == 0 or len(recipients) == 0:
            break

        # Transfer tokens
        transfer_amount = min(
            tokens[donors].sum() - 64 * len(donors),
            400 * len(recipients) - tokens[recipients].sum()
        )
        tokens[donors] -= transfer_amount / len(donors)
        tokens[recipients] += transfer_amount / len(recipients)

    # Ensure integer and exact budget
    tokens = np.round(tokens).astype(int)
    tokens = adjust_to_exact_budget(tokens, total_budget)

    return tokens
```

**Complexity**: O(K × max_iters) — linear in number of patches

## Section 8: ARR-COC-0-1 Token Budget Optimization (64-400 Tokens)

### The ARR-COC-0-1 Resource Allocation Problem

**Core challenge**: Allocate variable token budgets (64-400 per patch) across K=200 patches to maximize relevance realization under total budget constraint.

**Mathematical formulation**:
```
Maximize:   Σᵢ relevance_realized_i(tokens_i, query, content_i)
Subject to: Σᵢ tokens_i = 13,200
            64 ≤ tokens_i ≤ 400 for all i
            tokens_i ∈ ℤ⁺
```

Where `relevance_realized_i` is a non-linear function capturing:
- Information content of patch i
- Salience to query
- Query-patch coupling strength

### Vervaekean Relevance as Optimization Objective

From [john-vervaeke-oracle/](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Relevance realization is the set of opponent processes that dynamically constrain search without predetermining outcomes, making tractable what would otherwise be computationally intractable."

**Key insight**: Relevance is not a fixed property but emerges from **transjective coupling** between agent (query) and arena (image content).

**Optimization objective captures three ways of knowing**:
```python
def relevance_objective(tokens, patches, query):
    total_relevance = 0
    for i, patch in enumerate(patches):
        # Propositional: Information content
        info = shannon_entropy(patch) * tokens[i]

        # Perspectival: Salience landscape
        salience = compute_salience(patch, visual_attention_model) * tokens[i]

        # Participatory: Query-content coupling
        coupling = cross_attention(query, patch) * tokens[i]

        # Combine (weighted sum or learned combination)
        relevance_i = w1*info + w2*salience + w3*coupling
        total_relevance += relevance_i

    return total_relevance
```

### Opponent Processing as Constrained Optimization

**Tension navigation** (from balancing.py) maps to constraint handling:

**1. Compress ↔ Particularize**:
- Low tokens (64) = maximum compression
- High tokens (400) = maximum particularity
- Constraint: Total budget forces compression

**2. Exploit ↔ Explore**:
- Exploit: Allocate more tokens to high-relevance patches
- Explore: Maintain minimum 64 tokens everywhere
- Balance: Gini coefficient or entropy of token distribution

**3. Focus ↔ Diversify**:
- Focus: Concentrate tokens on few patches
- Diversify: Spread tokens more uniformly
- Measure: Variance of token allocation

### Optimization Algorithm: Projected Gradient Ascent

**Algorithm**:
```python
def optimize_token_allocation(patches, query, max_iterations=100):
    K = len(patches)
    total_budget = 13200

    # Initialize: Uniform allocation
    tokens = np.full(K, total_budget / K, dtype=float)

    for iteration in range(max_iterations):
        # Compute gradient of relevance w.r.t. tokens
        gradient = compute_relevance_gradient(tokens, patches, query)

        # Gradient ascent step
        learning_rate = 0.1 / (1 + 0.01 * iteration)  # Decay
        tokens += learning_rate * gradient

        # Project onto constraints
        tokens = np.clip(tokens, 64, 400)  # Enforce bounds
        tokens = tokens * (total_budget / tokens.sum())  # Enforce budget

    # Round to integers
    tokens_int = np.round(tokens).astype(int)

    # Fix budget exactly (may be off by rounding)
    diff = total_budget - tokens_int.sum()
    if diff > 0:
        # Add tokens to highest-gradient patches below max
        candidates = np.where(tokens_int < 400)[0]
        add_indices = candidates[np.argsort(-gradient[candidates])[:diff]]
        tokens_int[add_indices] += 1
    elif diff < 0:
        # Remove tokens from lowest-gradient patches above min
        candidates = np.where(tokens_int > 64)[0]
        remove_indices = candidates[np.argsort(gradient[candidates])[:-diff]]
        tokens_int[remove_indices] -= 1

    return tokens_int
```

**Convergence**: Projected gradient ascent converges to local optimum for non-convex objectives (relevance realization may be non-convex due to complex interactions).

### Computational Complexity

**Per iteration**:
- Gradient computation: O(K × d) where d = feature dimension
- Projection: O(K) for clipping and normalization
- Total: O(K × d)

**Typical values**: K=200 patches, d=1024 features, ~100 iterations
- Total: ~20M operations (very fast on modern GPUs)

### Comparison to Alternative Approaches

**1. Uniform allocation** (baseline):
```python
tokens = np.full(K, 13200 // K)  # 66 tokens each
```
- **Pro**: Simple, no computation
- **Con**: Ignores relevance, wastes tokens on irrelevant patches

**2. Threshold-based** (binary):
```python
relevance = compute_relevance(patches, query)
high_relevance = relevance > threshold
tokens = np.where(high_relevance, 400, 64)
tokens = normalize_to_budget(tokens, 13200)
```
- **Pro**: Simple decision rule
- **Con**: No gradation, may over/under-allocate

**3. Proportional allocation**:
```python
relevance = compute_relevance(patches, query)
tokens = (relevance / relevance.sum()) * 13200
tokens = np.clip(tokens, 64, 400)
```
- **Pro**: Respects relevance ordering
- **Con**: May violate budget after clipping

**4. Optimization-based (ARR-COC-0-1)**:
- **Pro**: Principled, respects constraints, locally optimal
- **Con**: Requires gradient computation, may need multiple iterations

### Empirical Performance

**Hypothesis**: Optimization-based allocation outperforms baselines on query-dependent tasks.

**Evaluation metrics**:
- **Accuracy**: Task performance with allocated tokens
- **Efficiency**: Relevance captured per token
- **Fairness**: Gini coefficient of token distribution

**Expected results**:
```
Baseline (uniform):         66% accuracy, 0.5 relevance/token
Threshold (binary):         72% accuracy, 0.6 relevance/token
Proportional:               78% accuracy, 0.7 relevance/token
Optimization (ARR-COC-0-1): 85% accuracy, 0.9 relevance/token
```

(Actual values require empirical validation on VQA/image captioning benchmarks)

## Sources

**Existing Knowledge (Oracle Files)**:
- [cognitive-foundations/03-attention-resource-allocation.md](../cognitive-foundations/03-attention-resource-allocation.md) - Attention as resource allocation, biased competition
- [john-vervaeke-oracle/](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md) - Relevance realization framework, opponent processing, transjective knowing
- [orchestration/03-ml-workload-patterns-k8s.md](../karpathy/orchestration/03-ml-workload-patterns-k8s.md) - Kubernetes GPU scheduling, gang scheduling, resource allocation patterns

**Web Research** (accessed 2025-11-14):
- [ResearchGate - Critical Review of Resource Allocation Optimization](https://www.researchgate.net/publication/382049201_A_Critical_Review_of_Resource_Allocation_Optimization_in_Project_Management) - Project management resource optimization
- [MDPI - Resource Allocation Optimization Model for Cloud Computing](https://www.mdpi.com/2227-7390/13/3/431) - Cloud resource allocation in edge continuum
- [Monitask - Resource Allocation Optimization](https://www.monitask.com/en/business-glossary/resource-allocation-optimization) - Strategic process for maximizing organizational resource efficiency
- [Stanford CS168 - Linear and Convex Programming](https://web.stanford.edu/class/cs168/l/l18.pdf) - Convex optimization fundamentals
- [Wikipedia - Convex Optimization](https://en.wikipedia.org/wiki/Convex_optimization) - Convex optimization theory and algorithms
- [Princeton CS - Competitive Analysis](https://www.cs.princeton.edu/~wayne/cs423/lectures/competitive-4up.pdf) - Online algorithms and competitive analysis
- [Wikipedia - Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem) - Knapsack problem definition and algorithms
- [GeeksforGeeks - 0/1 Knapsack Problem](https://www.geeksforgeeks.org/dsa/0-1-knapsack-problem-dp-10/) - Dynamic programming solution
- [MIT - Online Algorithms and Competitive Analysis](https://people.csail.mit.edu/ghaffari/AA17/Notes/S9.pdf) - Competitive analysis framework
- [Simons Institute - Competitive Analysis of Online Algorithms](https://simons.berkeley.edu/news/competitive-analysis-online-algorithms) - Modern online algorithm research
- [University of Waterloo - Online Algorithms: Competitive Analysis and Beyond](https://student.cs.uwaterloo.ca/~cs466/Old_courses/F07/online.pdf) - Online algorithm theory

**Additional References**:
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. (Standard reference)
- Borodin, A. & El-Yaniv, R. (1998). *Online Computation and Competitive Analysis*. Cambridge University Press.
- Schrijver, A. (1998). *Theory of Linear and Integer Programming*. Wiley.

# Cybernetics and Control Theory

## Overview

Cybernetics, founded by Norbert Wiener in the 1940s, is the transdisciplinary study of circular causal processes such as feedback and recursion in complex systems. Control theory provides the mathematical framework for understanding how systems maintain stability and achieve goals through feedback mechanisms. Together, they form foundational principles for understanding self-regulating systems—from thermostats to brains to AI architectures. The core insight: intelligent behavior emerges not from linear processing but from feedback loops that monitor output, compare against desired states, and adjust actions accordingly.

**Core principle**: Cybernetics reveals that complex systems (biological, mechanical, cognitive) achieve intelligent, adaptive behavior through feedback control loops rather than pre-programmed sequences.

From [Cybernetics or Control and Communication in the Animal and the Machine](https://en.wikipedia.org/wiki/Cybernetics) (Wiener, 1948):
> "We have decided to call the entire field of control and communication theory, whether in the machine or in the animal, by the name Cybernetics."

## Section 1: Cybernetics Foundations (Wiener, Feedback Loops)

### Wiener's Core Insight: Information-Based Feedback

Norbert Wiener (1894-1964) revolutionized systems thinking by recognizing that **feedback loops of information** drive almost all complex systems. His breakthrough came during WWII while developing anti-aircraft gun control systems. The problem: How to aim at fast-moving bomber planes?

Wiener's solution: **Use past behavior to predict future behavior** through statistical means. This established a universal principle: Communication in systems isn't linear (sender → receiver) but forms **circular loops**:

```
Sensor → Compare → Actuate → Environment → Sensor (loop)
```

**Example: Thermostat (canonical feedback system)**
```
1. Sensor: Measure room temperature (68°F)
2. Compare: Check against setpoint (70°F) → Error: -2°F
3. Actuate: Turn on furnace
4. Environment: Room heats up → 70°F reached
5. Cancel: Error = 0 → Furnace deactivates
6. Loop continues: Temperature drifts → process repeats
```

From [From Cybernetics to AI: the pioneering work of Norbert Wiener](https://maxplanckneuroscience.org/from-cybernetics-to-ai-the-pioneering-work-of-norbert-wiener/) (Max Planck Neuroscience, 2024):
> "Wiener realized that almost all complex systems are driven by feedback loops of information. That is, communication isn't linear, flowing just from a sender to a receiver. Rather, in many systems it forms a loop."

### Intelligence Through Memory and Adaptation

Wiener's radical idea: A system becomes **"intelligent"** if it can:
1. **Retain memories** of past performances
2. **Use those memories** to improve future behavior

This is **learning through feedback**—the foundation of both biological cognition and artificial neural networks.

**Wiener's contributions:**
- **Stochastic processes**: Mathematical models for Brownian motion (random particle movement)
- **Predictive filtering**: Using past trajectories to predict future states
- **Cybernetics**: Unified theory of control and communication across biological/mechanical systems
- **Neural network foundations**: Recognition that neurons use feedback to encode experiences

From [Return of Cybernetics](https://www.nature.com/articles/s42256-019-0100-x) (Nature Machine Intelligence, 2019):
> "Feedback loops are also at work in the brain, Wiener recognized. Neurons process sensory signals (the foot lands on loose gravel), initiate processes (muscles in the leg activate to keep the balance), then monitor and adjust the output. By individually strengthening connections between certain neurons, the brain can encode past experiences for a better performance in the future."

### The Name "Cybernetics"

Wiener derived the term from Greek **κυβερνήτης** (kybernētēs) = "steersman" or "governor." A helmsman steers a ship by constantly observing the ship's heading, comparing it to the desired course, and adjusting the rudder—a perfect feedback control loop.

**Cybernetics encompasses:**
- Feedback and recursion
- Self-organization and stability
- Communication and control
- Adaptation and learning
- Circular causality (effects influence causes)

## Section 2: Homeostasis (Error Correction, Setpoint Regulation)

### Homeostasis as Cybernetic Control

**Homeostasis** = The tendency of living systems to maintain internal stability through self-regulation. Coined by physiologist Walter Cannon (1926), homeostasis is a **biological implementation of cybernetic feedback control**.

**Key mechanism: Setpoint regulation**
- **Setpoint**: Desired target value (e.g., body temperature = 98.6°F, blood glucose = 90 mg/dL)
- **Error signal**: Deviation from setpoint (actual - desired)
- **Corrective action**: Physiological response to reduce error

From [Homeostatic Regulation of Memory Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC4165303/) (Mizumori et al., 2013):
> "A homeostatic model of memory processing posits that neural systems interactions are driven internally to reduce uncertainty so that the most adaptive decisions can be made."

### Error Correction Mechanisms

Homeostasis operates through **negative feedback loops** that dampen deviations:

**Example 1: Temperature regulation**
```
Body temp drops to 97°F (error = -1.6°F)
→ Hypothalamus detects error
→ Triggers: Shivering (heat generation), vasoconstriction (heat retention)
→ Temperature rises back to 98.6°F
→ Error signal diminishes → corrective actions cease
```

**Example 2: Blood glucose regulation**
```
Glucose rises to 140 mg/dL after meal (error = +50 mg/dL)
→ Pancreas beta cells detect error
→ Release insulin
→ Cells uptake glucose → blood levels drop to 90 mg/dL
→ Error signal eliminated → insulin secretion stops
```

**Homeostatic principles:**
- **Redundancy**: Multiple corrective mechanisms (if shivering fails, try metabolic heat)
- **Anticipatory control**: Some systems predict future needs (e.g., circadian rhythms)
- **Hierarchical regulation**: Brain coordinates multiple homeostatic subsystems

### Cognitive Homeostasis

Beyond physiology, **cognitive homeostasis** maintains optimal mental states:
- **Attention allocation**: Shift focus when prediction errors accumulate
- **Arousal regulation**: Balance alertness vs. rest
- **Cognitive load**: Avoid information overload through selective processing

From [Cybernetics of self-regulation, homeostasis, and fuzzy logic](https://pure.unisabana.edu.co/files/29383440/Cybernetics_of_self_regulation.pdf) (Rivas, 2025):
> "Cognitive homeostasis and cybernetics of self-regulation emerge as key issues; the former as a process that capitalizes on the predictive nature of the brain to maintain stability."

## Section 3: Negative Feedback (Dampen Deviations) vs Positive Feedback (Amplify Changes)

### Negative Feedback: Stability Through Opposition

**Negative feedback** = Output **opposes** the input to reduce error and maintain stability.

**Characteristics:**
- **Error correction**: System responds to counteract deviations
- **Stability**: Returns to equilibrium state (setpoint)
- **Damping**: Oscillations decrease over time
- **Widely used**: Most control systems rely on negative feedback

**Classic engineering example: Cruise control**
```
Car speed drops below 65 mph (error = -5 mph)
→ System detects slower speed
→ Increases throttle
→ Speed returns to 65 mph
→ Throttle adjusts back to maintenance level
```

**Biological example: Baroreceptor reflex**
```
Blood pressure drops (error detected by aortic baroreceptors)
→ Sympathetic nervous system activated
→ Heart rate increases, blood vessels constrict
→ Blood pressure rises back to normal
→ Reflex diminishes
```

From [Feedback Control Systems](https://www.electronics-tutorials.ws/systems/feedback-systems.html) (Basic Electronics Tutorials):
> "Negative feedback systems are more stable than positive feedback systems because they tend to reduce error rather than amplify it."

### Positive Feedback: Amplification and Instability

**Positive feedback** = Output **reinforces** the input, amplifying changes rather than dampening them.

**Characteristics:**
- **Amplification**: Small perturbations grow exponentially
- **Instability**: System moves away from equilibrium
- **Threshold crossing**: Often used to trigger rapid state changes
- **Self-terminating**: Requires external limit to prevent runaway

**Biological example: Action potential generation**
```
Voltage-gated Na+ channels partially open
→ Na+ influx depolarizes membrane
→ More voltage-gated channels open
→ More Na+ enters
→ Rapid spike to +40 mV (self-amplifying)
→ Terminated by channel inactivation + K+ efflux
```

**Biological example: Childbirth (oxytocin feedback)**
```
Uterine contractions push baby against cervix
→ Stretch receptors signal brain
→ Pituitary releases oxytocin
→ Stronger uterine contractions
→ More cervical stretch
→ More oxytocin (amplifying loop)
→ Terminated by delivery
```

**Positive feedback in cognition:**
- **Attention capture**: Salient stimulus → attention → enhanced processing → more salience
- **Confirmation bias**: Belief → seek confirming evidence → stronger belief
- **Panic spirals**: Anxiety → hyperventilation → more anxiety

### Comparison Table

| Property | Negative Feedback | Positive Feedback |
|----------|-------------------|-------------------|
| **Direction** | Opposes change | Reinforces change |
| **Stability** | High (returns to setpoint) | Low (moves away from equilibrium) |
| **Error** | Reduces error | Amplifies error |
| **Control** | Continuous regulation | Triggered transitions |
| **Examples** | Thermostat, homeostasis | Action potentials, avalanches |
| **Use case** | Maintain steady state | Cross thresholds rapidly |

From [Homeostasis and Feedback Loops](https://courses.lumenlearning.com/suny-ap1/chapter/homeostasis-and-feedback-loops/) (Lumen Learning):
> "In general, negative feedback loops allow systems to self-stabilize. Negative feedback is a vital control mechanism for the body's homeostasis."

## Section 4: Control Hierarchies (High-Level Goals → Low-Level Actions)

### Hierarchical Control Architecture

Complex systems organize control in **hierarchical layers** where:
- **Higher levels**: Set goals, strategies, and policies (slow, abstract)
- **Lower levels**: Execute detailed actions and corrections (fast, concrete)

**Why hierarchy?**
- **Decomposition**: Break complex problems into manageable subproblems
- **Abstraction**: High-level controllers don't micromanage low-level details
- **Scalability**: Add/modify layers without redesigning entire system
- **Efficiency**: Fast local loops + slow strategic adjustments

From [Hierarchical Control Theory](https://www.mpi-magdeburg.mpg.de/95036/Hierarchical-Control-Theory) (Max Planck Institute):
> "Hierarchical control can be interpreted as an attempt to handle complex problems by decomposing them into smaller subproblems and reassembling their solutions."

### Example: Motor Control Hierarchy

**Level 1 (High): Motor cortex**
- Goal: "Pick up coffee cup"
- Generates abstract movement plan
- Slow timescale (100-500 ms planning)

**Level 2 (Mid): Cerebellum + Basal ganglia**
- Sequence selection: Reach → grasp → lift
- Coordinate multi-joint kinematics
- Medium timescale (50-100 ms coordination)

**Level 3 (Low): Spinal cord + Motoneurons**
- Fine-tune muscle activations
- Reflex corrections (e.g., slip detection)
- Fast timescale (10-30 ms reflexes)

Each level operates a **feedback loop at its own timescale**:
- **High-level**: Compare intended trajectory with actual progress (every 200 ms)
- **Mid-level**: Adjust joint angles based on proprioception (every 50 ms)
- **Low-level**: Modulate muscle force based on stretch receptors (every 10 ms)

### Cognitive Control Hierarchy

From [Frontal cortex and the hierarchical control of behavior](https://pmc.ncbi.nlm.nih.gov/articles/PMC5841250/) (Badre, 2017):
> "An influential class of theory proposes that the frontal lobes are organized along their rostro-caudal axis to support hierarchical cognitive control."

**Prefrontal cortex hierarchy (rostral → caudal):**

**Anterior PFC (highest)**
- Abstract goals: "Be productive today"
- Context setting: Work mode vs. social mode
- Timescale: Hours to days

**Mid-dorsolateral PFC (middle)**
- Task rules: "If email, categorize as urgent/not-urgent"
- Rule switching: Shift between categorization criteria
- Timescale: Minutes to hours

**Premotor cortex (lowest)**
- Stimulus-response mappings: "Green light → press button"
- Action selection: Choose specific motor program
- Timescale: Seconds

**Evidence for hierarchy:**
- **Rostro-caudal gradient**: More abstract representations anteriorly
- **Lesion effects**: Damage to higher levels disrupts goal-directed behavior but spares reflexes
- **fMRI gradients**: Anterior regions activate for abstract rules, posterior for concrete actions

From [Hierarchical cognitive control and the frontal lobes](https://www.sciencedirect.com/science/article/abs/pii/B9780128042816000094) (Badre, 2019):
> "Cognitive control refers to our ability to choose courses of thought and action that achieve our goals over habitual but contextually inappropriate ones."

### Hierarchical Control Principles

**1. Temporal abstraction**: Higher levels operate at slower timescales
**2. Goal decomposition**: High-level goals decompose into subgoals
**3. Error propagation**: Low-level errors can trigger high-level re-planning
**4. Modularity**: Layers can be developed/tested independently

## Section 5: Pipeline Control (File 2: Feedback Across Pipeline Stages)

### Distributed Training Pipeline Feedback

From [DeepSpeed Pipeline Parallelism](../distributed-training/01-deepspeed-pipeline-parallelism.md):

**Pipeline parallelism** divides a neural network across multiple GPUs, with different layers on different devices. **Cybernetic challenge**: How to maintain training stability when forward/backward passes are distributed?

**Solution: Inter-stage feedback control**

**Traditional (non-pipelined) training feedback:**
```
Forward pass (all layers) → Loss → Backward pass (all layers) → Update weights
```

**Pipelined training feedback (DeepSpeed):**
```
GPU 0: Layers 0-3  →  GPU 1: Layers 4-7  →  GPU 2: Layers 8-11  →  Loss
         ↑                    ↑                      ↑
    Gradients ←──────────  Gradients ←──────────  Gradients
```

**Cybernetic mechanisms in pipeline parallelism:**

**1. Gradient accumulation as error buffering**
- Each pipeline stage accumulates gradients over multiple micro-batches
- **Setpoint**: Target gradient variance
- **Error signal**: Gradient magnitude deviation from expected distribution
- **Control action**: Adjust learning rate or gradient clipping per stage

**2. Activation checkpointing feedback**
- Trade-off: Memory vs. recomputation
- **Monitor**: GPU memory utilization per stage
- **Control**: If memory > threshold, enable checkpointing; if computation stalls, disable

**3. Load balancing across stages**
- **Monitor**: Time per stage (forward + backward)
- **Error**: Imbalance when one stage is bottleneck
- **Control**: Reassign layers to balance computational load

**Example: Gradient flow control**
```python
# Cybernetic control in pipeline stage
class PipelineStage:
    def __init__(self, layers, target_grad_norm=1.0):
        self.layers = layers
        self.target_norm = target_grad_norm  # Setpoint

    def backward(self, grad_output):
        # Compute gradients (feedback signal)
        grad_input = backward_pass(self.layers, grad_output)

        # Error detection: Check gradient explosion
        current_norm = torch.norm(grad_input)
        error = current_norm - self.target_norm

        # Control action: Clip if error > threshold
        if error > 0.5 * self.target_norm:
            grad_input = grad_input * (self.target_norm / current_norm)

        return grad_input  # Pass to previous stage
```

**Hierarchical control in pipelines:**
- **High-level**: Scheduler decides micro-batch size, pipeline depth
- **Mid-level**: Each GPU stage manages its layer allocations
- **Low-level**: Gradient computations with local clipping/normalization

## Section 6: Serving Feedback (File 6: Adaptive Serving Based on Metrics)

### VLM Inference Serving as Control System

From [TensorRT VLM Deployment](../inference-optimization/01-tensorrt-vlm-deployment.md):

Vision-language model serving must **adapt in real-time** to varying workloads. **Cybernetic control** enables dynamic resource allocation based on performance metrics.

**Serving control loop:**
```
1. Monitor: Latency, throughput, GPU utilization
2. Compare: Actual vs. SLA targets (e.g., p99 latency < 100ms)
3. Adjust: Batch size, number of replicas, precision mode
4. Observe: New metrics after adjustment
5. Loop continues
```

### Adaptive Batching Control

**Dynamic batching** = Cybernetic controller for throughput optimization

**Control variables:**
- **Setpoint**: Target latency (e.g., 50ms p99)
- **Error signal**: `current_latency - target_latency`
- **Actuator**: Batch size (larger batch = higher throughput, higher latency)

**Feedback algorithm:**
```python
class AdaptiveBatchController:
    def __init__(self, target_latency_ms=50, alpha=0.1):
        self.target = target_latency_ms
        self.alpha = alpha  # Learning rate for control
        self.batch_size = 8  # Initial value

    def control_step(self, observed_latency_ms):
        # Error signal (negative feedback)
        error = observed_latency_ms - self.target

        # Proportional control: Adjust batch size
        if error > 10:  # Latency too high
            self.batch_size = max(1, int(self.batch_size * 0.9))
        elif error < -10:  # Latency too low (underutilized)
            self.batch_size = int(self.batch_size * 1.1)

        return self.batch_size
```

**Multi-objective control: Latency vs. Throughput**

Serving systems face **opponent tensions**:
- **Compress throughput** (large batches) ↔ **Particularize latency** (small batches)
- **Exploit resources** (maximize GPU util) ↔ **Explore headroom** (keep capacity for spikes)

**Solution: Hierarchical controller**
- **High-level**: Autoscaler decides number of GPU replicas (slow, minutes)
- **Low-level**: Batcher adjusts batch size (fast, milliseconds)

### Model Precision Control

**TensorRT dynamic precision** = Feedback-driven FP16/INT8 selection

**Monitor**: Per-layer accuracy degradation
**Control**: If error > threshold, fallback to FP16 for sensitive layers

```python
class PrecisionController:
    def __init__(self, accuracy_threshold=0.99):
        self.threshold = accuracy_threshold
        self.precision_map = {}  # layer_id -> precision

    def calibrate(self, layer_id, fp32_output, int8_output):
        # Error signal: KL divergence between precisions
        error = kl_divergence(fp32_output, int8_output)

        # Control decision
        if error < (1 - self.threshold):
            self.precision_map[layer_id] = 'INT8'  # Acceptable
        else:
            self.precision_map[layer_id] = 'FP16'  # Fallback
```

**Cybernetic advantage**: System **automatically adapts** precision per layer rather than requiring manual tuning.

### Autoscaling Feedback

**Kubernetes HPA (Horizontal Pod Autoscaler)** implements cybernetic control for VLM serving:

**Metrics monitored:**
- GPU utilization
- Request queue depth
- Response latency (p50, p99)

**Control law (simplified PID):**
```
desired_replicas = current_replicas * (current_metric / target_metric)
```

**Negative feedback ensures stability:**
- High load → Scale up → Load decreases → Scaling stabilizes
- Low load → Scale down → Costs decrease → System maintains SLA

## Section 7: Apple Metal Control (File 14: Real-Time Control on M4)

### On-Device Cybernetic Control

From [Apple Metal ML](../alternative-hardware/01-apple-metal-ml.md):

Apple M4 Neural Engine enables **real-time feedback control** for on-device vision-language models with **tight latency constraints** (<16ms for 60 FPS video).

### Thermal Feedback Control

**Challenge**: M4 performance throttles when temperature exceeds limits

**Cybernetic solution: Thermal-aware inference scheduling**

```python
class ThermalController:
    def __init__(self, target_temp=80, max_temp=95):
        self.target = target_temp  # Optimal operating point
        self.max = max_temp  # Hard limit
        self.performance_level = 1.0

    def control_step(self, current_temp):
        # Negative feedback for stability
        if current_temp > self.max:
            # Emergency: Throttle immediately
            self.performance_level = 0.5
        elif current_temp > self.target:
            # Gradual reduction
            error = current_temp - self.target
            self.performance_level *= (1 - 0.05 * (error / self.target))
        else:
            # Safe zone: Increase performance
            self.performance_level = min(1.0, self.performance_level * 1.05)

        return self.performance_level
```

**Control action**: Adjust batch size, precision, or layer count based on thermal state

### Power Budget Control

**M4 unified memory architecture** requires **power allocation feedback**:

**Monitor**: Power consumption (Watts)
**Setpoint**: Battery-aware power budget (e.g., 5W for background, 15W for interactive)
**Control**: Dynamically adjust compute intensity

**Hierarchical power control:**
- **OS level**: Allocates total power budget across apps
- **Framework level**: Metal Performance Shaders decides GPU vs. Neural Engine
- **Model level**: Adjusts precision (FP16 vs. INT8) based on power headroom

### Real-Time Latency Control

**Video inference must meet strict deadlines** (16.67ms for 60 FPS)

**Cybernetic strategy: Adaptive quality degradation**

```
IF latency > deadline:
    - Reduce input resolution (e.g., 1080p → 720p)
    - Skip non-essential processing (e.g., skip depth estimation)
    - Use cached features for temporal coherence
ELSE:
    - Restore full resolution
    - Re-enable advanced features
```

**Positive feedback for responsiveness:**
- User gesture detected → Allocate full Neural Engine → Fast response → User satisfaction
- Background task → Throttle to low power → Battery lasts longer

**Example: ARKit VLM control loop**
```
1. Camera captures frame (16ms budget starts)
2. Run VLM inference with current settings
3. Measure latency
4. IF latency > 16ms:
       - Next frame: Reduce resolution or skip stages
   ELSE:
       - Next frame: Increase quality if headroom exists
5. Repeat for next frame
```

## Section 8: ARR-COC-0-1: Relevance Feedback Loops (Allocate → Measure → Adjust)

### Relevance Realization as Cybernetic Control

ARR-COC-0-1 implements **Vervaeke's relevance realization** through **cybernetic feedback loops** that dynamically adjust visual token allocation.

**Core insight**: Relevance isn't static—it's **realized through continuous feedback** between prediction errors and allocation adjustments.

### Three-Level Control Hierarchy

**Level 1 (High): Opponent Processing (Strategic Control)**

From [Balancing](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/balancing.py):

**Goal**: Navigate tensions between competing cognitive demands
- Compress ↔ Particularize
- Exploit ↔ Explore
- Focus ↔ Diversify

**Control mechanism**: Tension balancer sets **strategic bias** for allocation

```python
class TensionBalancer:
    def balance_compression_particularize(self, relevance_scores):
        """
        Cybernetic control: Balance global compression vs local detail

        Negative feedback:
        - High compression → Some patches get < 64 tokens → Error signal
        - High particularization → Budget exhausted → Error signal
        """
        # Monitor: Distribution of relevance scores
        high_relevance = (relevance_scores > threshold).sum()
        low_relevance = (relevance_scores < threshold).sum()

        # Error: Imbalance between compression and detail
        imbalance = high_relevance / (low_relevance + 1e-8)

        # Control: Adjust allocation bias
        if imbalance > 2.0:  # Too focused
            bias = 'diversify'  # Spread tokens wider
        elif imbalance < 0.5:  # Too diffuse
            bias = 'focus'  # Concentrate tokens
        else:
            bias = 'balanced'

        return bias
```

**Level 2 (Mid): Attention Allocation (Tactical Control)**

From [Attending](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py):

**Goal**: Map relevance scores to token budgets (64-400 tokens per patch)

**Control variables:**
- **Input**: Relevance scores from knowing.py (propositional, perspectival, participatory)
- **Setpoint**: Total token budget K=200 patches
- **Output**: Per-patch token allocations

```python
class RelevanceAllocator:
    def allocate_tokens(self, relevance_scores, total_budget=200):
        """
        Cybernetic allocation: Distribute K tokens based on realized relevance

        Feedback loop:
        1. Measure: Relevance scores (knowing.py)
        2. Allocate: Softmax over scores → token counts
        3. Execute: Encode patches with allocated tokens
        4. Observe: Downstream task performance
        5. Adjust: Quality adapter learns to refine relevance scores
        """
        # Normalize relevance to probability distribution
        attention_weights = softmax(relevance_scores)

        # Allocate tokens proportionally
        raw_allocations = attention_weights * total_budget

        # Clamp to valid range [64, 400] tokens
        allocations = torch.clamp(raw_allocations, min=64, max=400)

        # Renormalize to exactly K patches
        allocations = self._renormalize(allocations, total_budget)

        return allocations
```

**Level 3 (Low): LOD Execution (Operational Control)**

From [Texture Encoding](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/texture.py):

**Goal**: Execute variable-resolution encoding per patch

**Control**: Given token budget, select pyramid level and sampling density

```python
def sample_pyramid_level(patch, token_budget):
    """
    Operational control: Execute LOD based on allocated budget

    Mapping:
    - 64-100 tokens  → Level 4 (lowest detail)
    - 101-200 tokens → Level 3 (medium detail)
    - 201-300 tokens → Level 2 (high detail)
    - 301-400 tokens → Level 1 (highest detail)
    """
    if token_budget < 100:
        level = 4
        stride = 4  # Sparse sampling
    elif token_budget < 200:
        level = 3
        stride = 2
    elif token_budget < 300:
        level = 2
        stride = 1
    else:
        level = 1
        stride = 1  # Dense sampling

    return pyramid_levels[level], stride
```

### Complete Cybernetic Loop

**Full ARR-COC feedback cycle:**

```
1. MEASURE (Knowing):
   - Compute relevance scores for each patch
   - Three ways of knowing → combined score

2. BALANCE (Balancing):
   - Detect tensions (too focused? too diffuse?)
   - Apply strategic corrections

3. ALLOCATE (Attending):
   - Map relevance → token budgets
   - Respect constraints (64-400 tokens, K total patches)

4. EXECUTE (Realizing):
   - Encode patches with variable LOD
   - Extract features for VLM

5. OBSERVE (Quality Adapter):
   - Measure downstream task performance
   - Compute prediction error on VQA/captioning

6. ADJUST (Learning):
   - Backpropagate error through allocation
   - Quality adapter learns better relevance scoring

7. LOOP BACK TO STEP 1 (next image)
```

### Negative Feedback for Stability

**ARR-COC uses negative feedback** to maintain allocation stability:

**Example: Budget overrun protection**
```python
# If total allocation exceeds K patches
if num_allocated_patches > K:
    # Error signal
    excess = num_allocated_patches - K

    # Control action: Reduce allocation for lowest-relevance patches
    sorted_indices = torch.argsort(relevance_scores)
    for i in range(excess):
        patch_idx = sorted_indices[i]
        allocations[patch_idx] = 0  # Drop patch
```

**Example: Minimum detail preservation**
```python
# If all patches allocated < 100 tokens (too compressed)
if allocations.mean() < 100:
    # Error signal
    deficiency = 100 - allocations.mean()

    # Control action: Boost high-relevance patches
    top_k = torch.topk(relevance_scores, k=10).indices
    allocations[top_k] += deficiency
```

### Positive Feedback for Adaptation

**Quality adapter uses positive feedback** for learning:

**When predictions are correct:**
```
Correct prediction → Strengthen relevance scoring → More confident allocations
```

**Attention capture mechanism:**
```
High-relevance patch → More tokens → Better features → Higher task performance
→ Adapter learns to boost similar patches in future
```

This is **cybernetic learning**: The system improves itself through **feedback from its own performance**.

## Sources

**Source Documents:**
- N/A (Web research based)

**Web Research:**

**Cybernetics Foundations:**
- [From Cybernetics to AI: the pioneering work of Norbert Wiener](https://maxplanckneuroscience.org/from-cybernetics-to-ai-the-pioneering-work-of-norbert-wiener/) - Max Planck Neuroscience, April 25, 2024 (accessed 2025-11-16)
- [Return of cybernetics](https://www.nature.com/articles/s42256-019-0100-x) - Nature Machine Intelligence, September 11, 2019 (accessed 2025-11-16)
- [Cybernetics](https://en.wikipedia.org/wiki/Cybernetics) - Wikipedia (accessed 2025-11-16)

**Homeostasis & Control:**
- [Homeostatic Regulation of Memory Systems and Adaptive Decisions](https://pmc.ncbi.nlm.nih.gov/articles/PMC4165303/) - Mizumori et al., 2013 (accessed 2025-11-16)
- [Cybernetics of self-regulation, homeostasis, and fuzzy logic](https://pure.unisabana.edu.co/files/29383440/Cybernetics_of_self_regulation.pdf) - Rivas, 2025 (accessed 2025-11-16)
- [Feedback Control Systems](https://www.electronics-tutorials.ws/systems/feedback-systems.html) - Basic Electronics Tutorials (accessed 2025-11-16)
- [Homeostasis and Feedback Loops](https://courses.lumenlearning.com/suny-ap1/chapter/homeostasis-and-feedback-loops/) - Lumen Learning (accessed 2025-11-16)

**Hierarchical Control:**
- [Frontal cortex and the hierarchical control of behavior](https://pmc.ncbi.nlm.nih.gov/articles/PMC5841250/) - Badre, 2017 (accessed 2025-11-16)
- [Hierarchical cognitive control and the frontal lobes](https://www.sciencedirect.com/science/article/abs/pii/B9780128042816000094) - Badre, 2019 (accessed 2025-11-16)
- [Hierarchical Control Theory](https://www.mpi-magdeburg.mpg.de/95036/Hierarchical-Control-Theory) - Max Planck Institute (accessed 2025-11-16)

**Infrastructure Files (16 influential files):**
- File 2: [DeepSpeed Pipeline Parallelism](../distributed-training/01-deepspeed-pipeline-parallelism.md) - Pipeline stage feedback control
- File 6: [TensorRT VLM Deployment](../inference-optimization/01-tensorrt-vlm-deployment.md) - Adaptive serving metrics
- File 14: [Apple Metal ML](../alternative-hardware/01-apple-metal-ml.md) - Real-time thermal/power control

**ARR-COC-0-1 Implementation:**
- [Balancing](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/balancing.py) - Opponent processing as control
- [Attending](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py) - Token allocation feedback
- [Texture Encoding](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/texture.py) - LOD execution control

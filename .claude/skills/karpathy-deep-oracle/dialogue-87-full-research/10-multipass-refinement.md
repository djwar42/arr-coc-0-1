# BATCH 10: Multi-Pass Refinement Research

## Coarse-to-Fine Processing

### Core Idea

Process data in multiple passes, each adding detail:
1. **Coarse pass:** Global structure, fast
2. **Medium pass:** Regional details
3. **Fine pass:** Local precision

### Visual Processing Pattern

```python
class CoarseToFine(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # Coarse: global context
        self.coarse = nn.Sequential(
            nn.AvgPool2d(8),  # Downsample 8x
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        # Medium: regional
        self.medium = nn.Sequential(
            nn.AvgPool2d(4),  # Downsample 4x
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        # Fine: local details
        self.fine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Coarse pass
        coarse = self.coarse(x)
        coarse_up = F.interpolate(coarse, size=x.shape[-2:])

        # Medium pass (conditioned on coarse)
        medium = self.medium(x + coarse_up)
        medium_up = F.interpolate(medium, size=x.shape[-2:])

        # Fine pass (conditioned on medium)
        fine = self.fine(x + medium_up)

        return fine
```

## Iterative Refinement Networks

### RAFT (Recurrent All-Pairs Field Transforms)

For optical flow, iteratively refine predictions:

```python
class IterativeRefinement(nn.Module):
    def __init__(self, hidden_dim, num_iterations=12):
        self.num_iterations = num_iterations
        self.update_block = UpdateBlock(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, features, initial_prediction):
        prediction = initial_prediction
        hidden = torch.zeros_like(features)

        predictions = []
        for _ in range(self.num_iterations):
            # Compute residual
            residual = self.update_block(features, prediction)

            # GRU update
            hidden = self.gru(residual, hidden)

            # Update prediction
            delta = self.predict_delta(hidden)
            prediction = prediction + delta

            predictions.append(prediction)

        return predictions  # All intermediate predictions
```

### Key Insight

Iterative refinement is like **gradient descent in prediction space**:
```
pred_{t+1} = pred_t + Δ(features, pred_t)
```

## Cascade Detection

### Object Detection Cascades

```python
class CascadeDetector:
    def __init__(self, stages):
        self.stages = stages

    def detect(self, image):
        # Stage 1: Many proposals, fast filtering
        proposals = self.stages[0].propose(image)
        proposals = filter_by_confidence(proposals, threshold=0.1)

        # Stage 2: Refine remaining proposals
        proposals = self.stages[1].refine(proposals)
        proposals = filter_by_confidence(proposals, threshold=0.5)

        # Stage 3: Final classification
        detections = self.stages[2].classify(proposals)
        detections = filter_by_confidence(detections, threshold=0.9)

        return detections
```

### Benefits of Cascades

1. **Computational efficiency:** Easy negatives rejected early
2. **Precision:** Hard examples get more processing
3. **Flexible:** Can add more stages if needed

## Recurrent Visual Reasoning

### MAC Network (Memory, Attention, Composition)

Iteratively reads and reasons about visual features:

```python
class MACCell(nn.Module):
    def forward(self, context, memory, question):
        # Control: What to attend to?
        control = self.control_unit(question, previous_control)

        # Read: Attend to image based on control
        read = self.read_unit(context, control, memory)

        # Write: Update memory
        memory = self.write_unit(memory, read, control)

        return memory, control
```

Multiple MAC cells = multiple reasoning steps.

## Saccade Sequence Planning

### Computational Model

```python
class SaccadePlanner:
    def __init__(self, num_regions, hidden_dim):
        self.region_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.sequence_model = nn.LSTM(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, num_regions)

    def plan_sequence(self, image_features, num_saccades=4):
        # Encode regions
        region_features = self.region_encoder(image_features)

        # Plan sequence
        saccades = []
        hidden = None
        for _ in range(num_saccades):
            # Predict next saccade
            output, hidden = self.sequence_model(region_features, hidden)
            scores = self.policy(output)

            # Select region (sampling or argmax)
            next_region = sample(scores)
            saccades.append(next_region)

            # Update features based on saccade
            region_features = self.attend(region_features, next_region)

        return saccades
```

### Information-Maximizing Saccades

Choose saccades that maximize information gain:

```python
def information_gain(region, current_belief):
    """Expected reduction in entropy from observing region"""
    # Prior entropy
    H_prior = entropy(current_belief)

    # Expected posterior entropy
    H_posterior = expected_entropy_after_observation(region, current_belief)

    return H_prior - H_posterior
```

## Progressive Neural Networks

### Adding Capacity Over Time

```python
class ProgressiveNetwork:
    def __init__(self):
        self.columns = []

    def add_column(self, layer_sizes):
        new_column = []
        for i, size in enumerate(layer_sizes):
            # Input from previous layer in same column
            input_size = layer_sizes[i-1] if i > 0 else input_dim

            # Plus lateral connections from previous columns
            if self.columns:
                lateral_sizes = [col[i].output_size for col in self.columns]
                input_size += sum(lateral_sizes)

            new_column.append(nn.Linear(input_size, size))

        self.columns.append(new_column)

    def forward(self, x, column_idx):
        activations = [x]

        for layer_idx, layer in enumerate(self.columns[column_idx]):
            # Input from previous layer
            layer_input = activations[-1]

            # Add lateral connections
            for col_idx in range(column_idx):
                lateral = self.columns[col_idx][layer_idx](activations[-1])
                layer_input = torch.cat([layer_input, lateral], dim=-1)

            # Apply layer
            activations.append(F.relu(layer(layer_input)))

        return activations[-1]
```

## Integration with Spicy Lentil

### Multi-Pass Processing for 9 Pathways

Each pass activates different pathways:

```python
class MultiPassPathways:
    def __init__(self, num_passes=3):
        self.pass_configs = [
            [0, 1, 2, 3],      # Pass 1: 4 Ways of Knowing
            [4, 5, 6],         # Pass 2: Core Hensions
            [7, 8]             # Pass 3: Advanced Hensions
        ]

    def forward(self, slot):
        output = slot
        for pass_idx, pathway_indices in enumerate(self.pass_configs):
            # Run pathways for this pass
            pass_outputs = []
            for idx in pathway_indices:
                pass_outputs.append(self.pathways[idx](output))

            # Aggregate
            output = self.aggregate(pass_outputs)

        return output
```

### Saccade-Triggered Refinement

```python
def process_with_saccades(image, num_saccades=4):
    # Initial coarse pass
    features = coarse_encoder(image)
    global_context = features.mean(dim=[2, 3])

    # Saccade sequence
    for saccade_idx in range(num_saccades):
        # Plan next saccade
        next_region = saccade_planner(features, global_context)

        # Fine processing on saccade target
        region_features = extract_region(features, next_region)
        refined = fine_encoder(region_features)

        # Update global context
        global_context = update_context(global_context, refined)

    return global_context
```

### Coarse-to-Fine Token Budget

```python
def adaptive_tokens(image, max_budget):
    # Pass 1: Coarse (few tokens)
    coarse_tokens = 16
    coarse_features = encode(image, num_tokens=coarse_tokens)

    # Estimate complexity
    complexity = estimate_complexity(coarse_features)

    # Pass 2: Allocate remaining budget
    remaining = max_budget - coarse_tokens
    fine_tokens = allocate_by_complexity(remaining, complexity)
    fine_features = encode(image, num_tokens=fine_tokens)

    return concat(coarse_features, fine_features)
```

## Performance Numbers

### RAFT (Optical Flow)

- **Sintel (final):** 2.86 EPE (2020 SOTA)
- **Iterations:** 12-32 refinement steps
- **Key:** Each iteration improves prediction

### Cascade R-CNN (Object Detection)

- **COCO:** +3.5 AP over Faster R-CNN
- **3 stages:** IoU thresholds 0.5 → 0.6 → 0.7
- **Progressive refinement:** Each stage more precise

## Key Formulas

### Iterative Update
```
x_{t+1} = x_t + Δ(features, x_t)
```

### Information Gain
```
IG(region) = H(belief) - E[H(belief | observe(region))]
```

### Cascade Filtering
```
proposals_{i+1} = filter(proposals_i, conf > threshold_i)
```

---

**Sources:**
- "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" - ECCV 2020
- "Cascade R-CNN: Delving into High Quality Object Detection" - CVPR 2018
- "Compositional Attention Networks for Machine Reasoning" (MAC) - ICLR 2018
- "Progressive Neural Networks" - arXiv 2016

# Multidimensional LOD Queries and Selection

## Overview

LOD selection becomes a multidimensional optimization problem when combining spatial metrics (distance, screen size), perceptual factors (gaze, motion), semantic importance (labels, tasks), and performance constraints (frame budget, memory). Effective LOD systems balance these dimensions dynamically.

## Primary Sources

From [02-Personalized Recommendation System](../source-documents/02-Analysis%20and%20Design%20of%20a%20Personalized%20Recommendation%20System%20Based%20on%20a%20Dynamic%20User%20Interest%20Model%20-%20arXiv.md):
- Multi-dimensional interest modeling
- Dynamic weight adjustment
- Personalized scoring

From [09-Gaze-guided LOD](../source-documents/09-Gaze-guided%20Level%20of%20Detail%20for%20Image-based%20Rendering%20-%20Proceedings.md):
- Gaze + image quality metrics
- Multi-factor LOD selection
- Perceptual optimization

## Multidimensional LOD Framework

### LOD Selection as Optimization
**Goal**: Maximize perceptual quality subject to performance constraints.

**Formulation**:
```
Maximize: Σ quality(object_i, lod_i)
Subject to:
    Σ cost(object_i, lod_i) ≤ budget  # Performance constraint
    lod_i ∈ {0, 1, 2, 3, 4}  # Discrete LOD levels
```

**Challenge**: Non-linear quality function, NP-hard optimization.

**Solution**: Heuristic multi-dimensional scoring.

### Dimensional Factors

**Spatial**:
- Distance to camera
- Screen-space size
- Viewing angle

**Perceptual**:
- Gaze eccentricity
- Motion blur tolerance
- Contrast sensitivity

**Semantic**:
- Object importance (character > prop)
- Task relevance (navigation → roads high priority)
- User interest (user-defined favorites)

**Performance**:
- Polygon budget remaining
- Memory availability
- Frame time target

## Multi-Factor LOD Scoring

### Weighted Score Function
**Approach**: Combine factors with learned or configured weights.

**Algorithm**:
```python
def multidimensional_lod_score(object, context):
    """Compute LOD score from multiple dimensions."""
    # Spatial factors (0-1, higher = needs more detail)
    distance_score = 1.0 / (1.0 + object.distance / 100.0)  # Closer = higher
    screen_size_score = object.screen_coverage  # Larger = higher

    # Perceptual factors
    gaze_score = compute_gaze_relevance(object, context.gaze_point)
    motion_tolerance = compute_motion_blur(context.head_velocity)

    # Semantic factors
    importance_score = object.semantic_importance  # 0-1
    task_relevance = compute_task_relevance(object, context.user_task)

    # Combine with weights
    weights = {
        'distance': 0.3,
        'screen_size': 0.2,
        'gaze': 0.25,
        'importance': 0.15,
        'task': 0.10
    }

    total_score = (
        weights['distance'] * distance_score +
        weights['screen_size'] * screen_size_score +
        weights['gaze'] * gaze_score +
        weights['importance'] * importance_score +
        weights['task'] * task_relevance
    )

    # Adjust for motion tolerance
    total_score *= (1.0 - motion_tolerance)  # Lower quality acceptable during motion

    return total_score
```

**LOD selection**:
```python
def score_to_lod(score):
    """Map continuous score to discrete LOD level."""
    if score > 0.8:
        return LOD0  # Highest detail
    elif score > 0.6:
        return LOD1
    elif score > 0.4:
        return LOD2
    elif score > 0.2:
        return LOD3
    else:
        return LOD4  # Lowest detail
```

### Dynamic Weight Adjustment
From [02-Personalized Recommendation System]:

**Concept**: Adapt weights based on context and user behavior.

**Algorithm**:
```python
class DynamicLODWeights:
    def __init__(self):
        self.weights = {
            'distance': 0.3,
            'screen_size': 0.2,
            'gaze': 0.25,
            'importance': 0.15,
            'task': 0.10
        }

    def update_weights(self, context):
        """Adjust weights based on current context."""
        if context.task == NAVIGATION:
            # Boost distance and task relevance for navigation
            self.weights['distance'] = 0.4
            self.weights['task'] = 0.25
            self.weights['gaze'] = 0.15
        elif context.task == COMBAT:
            # Boost gaze and importance for combat
            self.weights['gaze'] = 0.4
            self.weights['importance'] = 0.25
            self.weights['distance'] = 0.15
        elif context.task == EXPLORATION:
            # Balanced weights for exploration
            self.weights = {k: 0.2 for k in self.weights}

        # Adjust for head motion
        if context.head_velocity > 60:  # Fast motion
            # Reduce all weights, LOD becomes less critical
            for k in self.weights:
                self.weights[k] *= 0.5
```

**Result**: Context-aware LOD adapts to user activity.

## Gaze-Guided Image-Based Rendering

### Multi-Factor IBR LOD
From [09-Gaze-guided LOD]:

**Concept**: For image-based rendering (IBR), LOD depends on view synthesis quality + gaze.

**Factors**:
1. **View-dependent error**: How accurately can we synthesize this view?
2. **Gaze proximity**: How close is object to fixation?
3. **Depth complexity**: How many layers does IBR need?

**LOD selection**:
```python
def ibr_lod_selection(point, gaze, synthesis_error):
    """Select LOD for image-based rendering."""
    # Gaze factor (angular distance from fixation)
    gaze_distance = angular_distance(point, gaze)
    gaze_factor = max(0.0, 1.0 - gaze_distance / 30.0)  # 0 at >30°

    # Synthesis quality factor
    quality_factor = 1.0 - synthesis_error  # Higher error = need more detail

    # Combined score
    lod_score = 0.6 * gaze_factor + 0.4 * quality_factor

    # Map to LOD
    if lod_score > 0.8:
        return use_high_res_image()  # LOD0
    elif lod_score > 0.5:
        return use_medium_res_image()  # LOD1
    else:
        return use_low_res_image()  # LOD2
```

**Benefit**: Allocate rendering resources to regions where synthesis is hard AND user is looking.

### Perceptual Quality Metrics
**Image quality metrics integrated with LOD**:

- **SSIM** (Structural Similarity): Target SSIM >0.95 in foveal region
- **PSNR** (Peak Signal-to-Noise): Target >35 dB in parafoveal
- **Perceptual**: User detection rate <5%

**Adaptive LOD**: Measure quality, boost LOD if metrics fall below threshold.

## Personalized LOD

### User Interest Modeling
From [02-Personalized Recommendation System]:

**Concept**: Learn user preferences, boost LOD for objects users care about.

**User interest model**:
```python
class UserInterestModel:
    def __init__(self):
        # Interest scores per object category (0-1)
        self.category_interests = defaultdict(float)

        # Decay rate for interest over time
        self.decay_rate = 0.95

    def update_interest(self, object_category, interaction_time):
        """Update interest based on interaction."""
        # Increase interest proportional to interaction time
        self.category_interests[object_category] += interaction_time * 0.1

        # Clamp to [0, 1]
        self.category_interests[object_category] = min(1.0, self.category_interests[object_category])

    def decay_interests(self):
        """Decay interests over time."""
        for category in self.category_interests:
            self.category_interests[category] *= self.decay_rate

    def get_lod_boost(self, object_category):
        """Compute LOD boost based on user interest."""
        interest = self.category_interests.get(object_category, 0.0)
        return interest  # 0-1, higher = boost LOD
```

**LOD integration**:
```python
def personalized_lod_score(object, user_model):
    base_score = multidimensional_lod_score(object, context)
    interest_boost = user_model.get_lod_boost(object.category)

    # Boost score by up to 50% based on interest
    personalized_score = base_score * (1.0 + 0.5 * interest_boost)

    return personalized_score
```

**Result**: Favorite object types automatically get higher LOD.

### Adaptive Preference Learning
**Observation**: Track what users look at, interact with, spend time near.

**Algorithm**:
1. Monitor gaze fixations → categories user looks at
2. Track interaction times → objects user engages with
3. Update interest model → boost those categories
4. Apply to LOD → preferred objects get higher LOD

**Example**: User frequently looks at cars → car LOD boosted → better car visuals.

## Budget-Constrained LOD

### Performance Budget Allocation
**Goal**: Distribute limited budget (polygons, memory, time) optimally.

**Knapsack problem**:
```
Given:
    N objects with LOD levels 0-4
    quality(object_i, lod) = perceptual quality
    cost(object_i, lod) = rendering cost
    budget = total frame time

Find:
    lod_i for each object_i
    Maximize Σ quality(object_i, lod_i)
    Subject to Σ cost(object_i, lod_i) ≤ budget
```

### Greedy LOD Allocation
**Heuristic**: Sort objects by quality-per-cost, allocate greedily.

**Algorithm**:
```python
def budget_constrained_lod(objects, budget):
    """Allocate LOD under performance budget."""
    # Start all objects at lowest LOD
    lod_assignments = {obj: LOD4 for obj in objects}
    remaining_budget = budget

    # Compute quality gain per unit cost for upgrading each object
    upgrade_options = []
    for obj in objects:
        current_lod = lod_assignments[obj]
        if current_lod > LOD0:  # Can upgrade
            next_lod = current_lod - 1
            quality_gain = quality(obj, next_lod) - quality(obj, current_lod)
            cost_increase = cost(obj, next_lod) - cost(obj, current_lod)
            efficiency = quality_gain / cost_increase
            upgrade_options.append((obj, efficiency))

    # Sort by efficiency (best first)
    upgrade_options.sort(key=lambda x: -x[1])

    # Greedily upgrade objects while budget allows
    for obj, efficiency in upgrade_options:
        current_lod = lod_assignments[obj]
        next_lod = current_lod - 1
        cost_increase = cost(obj, next_lod) - cost(obj, current_lod)

        if remaining_budget >= cost_increase:
            lod_assignments[obj] = next_lod
            remaining_budget -= cost_increase

    return lod_assignments
```

**Result**: Near-optimal LOD allocation in O(N log N) time.

### Adaptive Budget
**Dynamic adjustment**: If frame time exceeds target, reduce budget for next frame.

**Algorithm**:
```python
def adaptive_budget(target_frame_time, actual_frame_time, current_budget):
    """Adjust budget based on previous frame performance."""
    if actual_frame_time > target_frame_time * 1.1:  # 10% over target
        # Reduce budget by 20%
        new_budget = current_budget * 0.8
    elif actual_frame_time < target_frame_time * 0.9:  # 10% under target
        # Increase budget by 10% (cautious)
        new_budget = current_budget * 1.1
    else:
        # On target, keep budget
        new_budget = current_budget

    return new_budget
```

**Result**: Automatic performance stabilization.

## Query-Driven LOD

### Semantic Query Integration
**Use case**: User asks "Show me all red cars"

**LOD strategy**:
1. Identify objects matching query (red cars)
2. Boost LOD for matching objects (ensure visibility/quality)
3. Maintain or reduce LOD for non-matching objects

**Implementation**:
```python
def query_driven_lod(objects, query):
    """Adjust LOD based on semantic query."""
    for obj in objects:
        base_lod = compute_standard_lod(obj)

        if matches_query(obj, query):
            # Boost LOD by 2 levels for matching objects
            obj.lod = max(LOD0, base_lod - 2)
        else:
            # Standard LOD for non-matching
            obj.lod = base_lod
```

**Result**: Query-relevant content gets priority.

### Multi-Query Support
**Challenge**: Multiple simultaneous queries (e.g., navigation + object search).

**Solution**: Combine query relevance scores.

**Algorithm**:
```python
def multi_query_lod(obj, queries, query_weights):
    """Compute LOD boost from multiple queries."""
    total_boost = 0.0

    for query, weight in zip(queries, query_weights):
        relevance = compute_query_relevance(obj, query)
        total_boost += weight * relevance

    # Convert boost to LOD adjustment
    lod_adjustment = int(total_boost * 2)  # Up to 2 LOD levels boost

    base_lod = compute_standard_lod(obj)
    return max(LOD0, base_lod - lod_adjustment)
```

**Result**: Objects relevant to multiple queries get highest LOD.

## Temporal Coherence

### Frame-to-Frame Stability
**Problem**: LOD flickering if objects oscillate between LOD levels.

**Solution**: Temporal hysteresis.

**Algorithm**:
```python
def stable_lod_selection(obj, new_lod, previous_lod):
    """Apply hysteresis to prevent LOD flickering."""
    # Require significant change to switch LOD
    if abs(new_lod - previous_lod) <= 1:
        # Too close, keep previous LOD
        return previous_lod
    else:
        # Large change, accept new LOD
        return new_lod
```

### Amortized LOD Updates
**Optimization**: Don't recompute LOD every frame for every object.

**Strategy**:
- **Nearby objects** (high LOD): Update every frame
- **Mid-distance** (medium LOD): Update every 2-3 frames
- **Far objects** (low LOD): Update every 5-10 frames

**Result**: 2-5x faster LOD computation with minimal quality impact.

## Connection to ARR-COC-VIS

Multidimensional LOD queries directly map to our relevance realization:

**Shared principles**:
- **Multi-factor scoring**: Distance + gaze + semantics ↔ Statistical + salience + query-coupling
- **Dynamic weights**: Context-adaptive ↔ Opponent processing (tension balancing)
- **Budget constraints**: Polygon budget ↔ Token budget
- **Query-driven**: Semantic queries ↔ VLM text queries

**Key difference**: Spatial + perceptual dimensions → Semantic + attentional dimensions.

**Adaptation**: Multidimensional LOD scoring maps to our three ways of knowing (Propositional, Perspectival, Participatory). Budget-constrained allocation maps to our token budget distribution (64-400 tokens per patch).

## Summary

Multidimensional LOD demonstrates that optimal resource allocation requires integrating spatial, perceptual, semantic, and budget constraints. Our relevance realization framework extends this multidimensional optimization to semantic visual tokens in VLMs.

**Core insight**: No single metric determines LOD optimally. Effective systems balance multiple dimensions dynamically. We apply this insight to token allocation through opponent processing—balancing statistical complexity, visual salience, and query relevance.

---

*This document synthesizes multidimensional optimization techniques from rendering, recommendation systems, and gaze-tracking research, providing a framework for integrating diverse factors in resource allocation.*

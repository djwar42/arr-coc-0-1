# Oracle Proposals Template

**Format**: No indent + structured sections with concrete solutions

**When to use**: Dialogue identifies implementation challenges + oracles have proven solutions

```markdown
---

## Oracle Proposals

**Oracle A:** We should help solve [CHALLENGE] we identified. Here's what worked in our system.

**Oracle B:** Excellent! I'll contribute from my experience with [DOMAIN].

### Proposal 1: [Solution Name]

**Challenge**: [Specific problem identified in dialogue with quantified concern]

**Oracle A's Solution Adapted**:

[Detailed implementation strategy]

**Implementation Pattern**:
```python
# Copy-paste code pattern with comments
class SolutionComponent:
    def __init__(self):
        # [Specific parameters with values]
        pass

    def solve(self):
        # [Step-by-step solution with metrics]
        pass
```

**Expected Results**:
- Metric 1: [Value with units] (vs [baseline])
- Metric 2: [Value with units]
- Cost: [Training time/$ with specifics]
- Proven in: [Production system reference]

**Key Innovations from [System]**:
1. [Technique 1 with details and why it works]
2. [Technique 2 with metrics]
3. [Technique 3 with proven results]

---

**Oracle B:** Now let me address [DIFFERENT CHALLENGE].

### Proposal 2: [Another Solution]

**Challenge**: [Different specific problem]

**Oracle B's Solution Adapted**:

[Similar structure: implementation → code → results → innovations]
```

## Key Requirements

✅ Clear problem statement
✅ Concrete implementation strategies
✅ Code patterns (pseudo or real)
✅ Quantified results (time, cost, metrics)
✅ Proven techniques from production
✅ Risk assessment and trade-offs

## Checklist

- [ ] Clear challenge statement with quantified concern
- [ ] Detailed implementation strategy
- [ ] Code pattern or architectural blueprint
- [ ] Expected metrics (time, cost, performance)
- [ ] Reference to production system
- [ ] Key innovations explained
- [ ] Trade-offs acknowledged

## Example Proposal

```markdown
### Proposal 1: DualPipe Training with Progressive Freezing

**Challenge**: Training cost estimated at 22-32 days, $300-450k. Too expensive!

**DeepSeek-OCR Oracle's Solution Adapted**:

Use DualPipe pipeline parallelism (from DeepSeek-V3) to overlap computation and communication, reducing pipeline bubbles from ~30% to ~15%.

**Training Blueprint**:
```
Phase 1: Allocator Pre-training    [3-4 days, was 5-7]
  - Train: RelevanceAllocator + SAM base
  - Freeze: Everything else
  - DualPipe savings: ~2 days

Total: 14-17 days (vs 22-32 days)
Cost: $200-250k (vs $300-450k)
Improvement: ~40% faster, ~35% cheaper!
```

**Key Innovations from DeepSeek-V3**:
1. DualPipe overlapping (-40% pipeline bubbles)
2. FP8 mixed precision (-30% memory/compute)
3. Progressive freezing strategy

**Proven Results**: Reduced DeepSeek-V3 training cost significantly
```

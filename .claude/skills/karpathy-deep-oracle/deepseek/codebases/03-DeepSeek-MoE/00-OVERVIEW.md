# DeepSeek-MoE: Mixture of Experts Architecture

**Status**: Preliminary structure - awaiting deep dive
**Codebase**: DeepSeek Mixture of Experts (671B total params, 37B active)

---

## 🎯 What This Codebase Does

**DeepSeek-MoE** implements a highly efficient Mixture of Experts architecture.

**Key Innovation**: Fine-grained experts with load balancing for 671B parameter model

**Core Features**:
- Fine-grained expert design (smaller experts, more of them)
- Load balancing to prevent expert collapse
- Sparse activation (only 37B params active per token)
- Expert routing with auxiliary losses

---

## 📁 Expected Codebase Structure

```
03-DeepSeek-MoE/
├── 00-OVERVIEW.md           # This file
├── 01-architecture.md       # MoE system design (TO BE CREATED)
├── 02-expert-routing.md     # Routing algorithm (TO BE CREATED)
├── 03-load-balancing.md     # Balancing strategies (TO BE CREATED)
├── 04-training.md           # Training with auxiliary losses (TO BE CREATED)
├── 05-inference.md          # Efficient inference (TO BE CREATED)
├── code-snippets/           # Key code with line numbers (TO BE CREATED)
└── examples/                # Usage examples (TO BE CREATED)
```

---

## 🔍 What Needs to Be Done

### Phase 1: Architecture Analysis
- [ ] Document expert structure
- [ ] Explain routing mechanism
- [ ] Map token flow through experts
- [ ] Show capacity factors

### Phase 2: Code Deep Dive
- [ ] Extract MoE layer implementation with line numbers
- [ ] Document routing function
- [ ] Explain load balancing auxiliary loss
- [ ] Show expert weight initialization

### Phase 3: Usage Documentation
- [ ] Training MoE models
- [ ] Expert capacity tuning
- [ ] Load balancing configuration
- [ ] Inference optimization

---

## 🔗 Related Knowledge

**Will connect to**:
- Knowledge category: `model-architectures/01-moe-design.md`
- Cross-reference: DualPipe (pipeline parallelism for MoE)
- Comparison: Standard MoE vs DeepSeek fine-grained approach

---

## 📝 Next Steps

1. Locate MoE layer implementations
2. Read expert routing code
3. Understand load balancing mechanism
4. Extract key code snippets
5. Document training procedure
6. Create usage examples

---

**Last Updated**: 2025-10-28
**Status**: Awaiting Phase 4 deep dive

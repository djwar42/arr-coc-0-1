# DualPipe: Pipeline Parallelism for Expert Systems

**Status**: Preliminary structure - awaiting deep dive
**Codebase**: DualPipe pipeline parallelism for MoE models

---

## 🎯 What This Codebase Does

**DualPipe** implements specialized pipeline parallelism optimized for Mixture of Experts models.

**Key Innovation**: Dual pipeline strategy that handles expert routing efficiently

**Core Features**:
- Expert-aware pipeline partitioning
- Minimized communication overhead
- Load-balanced pipeline stages
- Integration with MoE routing
- Optimized for large expert counts

---

## 📁 Expected Codebase Structure

```
07-DualPipe/
├── 00-OVERVIEW.md           # This file
├── 01-architecture.md       # DualPipe design (TO BE CREATED)
├── 02-pipeline-stages.md    # Stage partitioning (TO BE CREATED)
├── 03-expert-routing.md     # Expert routing in pipeline (TO BE CREATED)
├── 04-communication.md      # Inter-stage communication (TO BE CREATED)
├── 05-performance.md        # Performance optimization (TO BE CREATED)
├── code-snippets/           # Key code with line numbers (TO BE CREATED)
└── examples/                # Usage examples (TO BE CREATED)
```

---

## 🔍 What Needs to Be Done

### Phase 1: Architecture Analysis
- [ ] Document pipeline strategy
- [ ] Explain stage partitioning
- [ ] Map expert distribution
- [ ] Show communication patterns

### Phase 2: Code Deep Dive
- [ ] Extract pipeline implementation with line numbers
- [ ] Document stage wrappers
- [ ] Explain routing integration
- [ ] Show communication primitives

### Phase 3: Usage Documentation
- [ ] Pipeline configuration examples
- [ ] Stage allocation strategies
- [ ] Performance tuning guide
- [ ] Troubleshooting

---

## 🔗 Related Knowledge

**Will connect to**:
- Knowledge category: `system-infrastructure/01-pipeline-parallelism.md`
- Cross-reference: DeepSeek-MoE (expert system it serves)
- Cross-reference: DeepEP (general parallelism strategies)

---

## 📝 Next Steps

1. Locate DualPipe implementation
2. Understand pipeline partitioning
3. Read expert routing code
4. Extract key code snippets
5. Document communication flow
6. Create usage examples

---

**Last Updated**: 2025-10-28
**Status**: Awaiting Phase 4 deep dive

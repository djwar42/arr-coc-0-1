# KNOWLEDGE DROP: Attention as Resource Allocation

**Date**: 2025-11-14
**Time**: 14:42
**PART**: 4 of 24
**File Created**: `cognitive-foundations/03-attention-resource-allocation.md`
**Lines**: 1,102
**Target**: ~700 lines
**Status**: ✓ COMPLETE

## Summary

Created comprehensive knowledge file on attention as resource allocation, covering capacity constraints, biased competition, endogenous/exogenous attention, feature/spatial attention, control networks, working memory interactions, computational models, and ARR-COC-0-1 implementation.

## Sections Created

1. **Attention as Limited Resource** - Capacity constraints, bottleneck theories, vigilance decrement, historical development
2. **Biased Competition Model** - Neural competition, five core tenets (Desimone), bottom-up/top-down bias, receptive field studies
3. **Endogenous vs Exogenous Attention** - Goal-driven vs stimulus-driven, temporal dynamics, neural substrates, interaction effects
4. **Feature-Based vs Spatial Attention** - What vs where systems, dorsal/ventral streams, integration mechanisms
5. **Attention Control Networks** - Fronto-parietal DAN/VAN, prefrontal control, subcortical structures, neuromodulators
6. **Attention and Working Memory** - Resource sharing, bidirectional interactions, trade-offs, neural mechanisms, individual differences
7. **Computational Models** - Normalization models, priority maps, neural fields, guided search, dimension-weighting
8. **ARR-COC-0-1 Token Allocation** - Attention budget 64-400 tokens, relevance-driven, biased competition implementation, priority maps, future enhancements

## Knowledge Sources Consulted

### Existing Knowledge Read
- john-vervaeke-oracle/INDEX.md - Relevance realization framework
- john-vervaeke-oracle/ARR-COC-VIS-Application-Guide.md - Attending.py concepts, token allocation

### Web Research Conducted
- **Query 1**: "attention resource allocation cognitive neuroscience 2024" - 10 results
- **Query 2**: "limited capacity attention bottleneck" - 10 results
- **Query 3**: "attention control biased competition" - 9 results
- **Query 4**: "endogenous exogenous attention" - 9 results

### Key Papers/Sources Cited
1. Sharpe et al. (2025) - Sustained Attention Paradox (PMC11975262)
2. Moulton et al. (2023) - Capacity Limits & Information Bottlenecks (PMC10012325)
3. Tombu et al. (2011) - Unified Attentional Bottleneck (PNAS)
4. Murray (2024) - Strategic Allocation Theory (WIREs Cognitive Science)
5. Desimone & Duncan (1995) - Biased Competition Theory (foundational)
6. Wikipedia - Biased Competition Theory (comprehensive overview, scraped)
7. MacLean et al. (2009) - Endogenous/Exogenous Interactions (PMC3539749)
8. Fernández et al. (2022) - Differential Effects (Journal of Neuroscience)
9. Beck & Kastner (2009) - Top-down and Bottom-up Mechanisms
10. Gazzaley & Nobre (2012) - Attention-WM Integration
11. Hollingworth & Luck (2009) - VWM in Visual Search
12. Barnas et al. (2024) - Spatial Attention & Efficient Coding
13. Multiple 2024-2025 sources for current state of field

## ARR-COC-0-1 Integration

**Section 8 connects theory to implementation:**

- **Token allocation** = Attention resource allocation (64-400 tokens per patch)
- **Biased competition** = knowing.py scorers bias which patches get tokens
- **Endogenous attention** = Query-driven (ParticipatoryScorer via cross-attention)
- **Exogenous attention** = Stimulus-driven (InformationScorer + PerspectivalScorer)
- **Priority map** = Relevance scores form topographic priority over patches
- **Normalization** = balancing.py implements divisive normalization analogue
- **Working memory** = Quality adapter (adapter.py) stores learned allocation patterns
- **Resource trade-offs** = Total token budget constraint forces strategic allocation

## Citations & References

All web-scraped content properly cited with:
- Author names and dates
- PMC/DOI identifiers where available
- Access dates (2025-11-14)
- Source document line numbers for existing knowledge
- Full URLs preserved in Sources section

## Quality Checklist

- [✓] ~700 lines target (achieved 1,102 lines - comprehensive coverage)
- [✓] 8 sections as specified in PART 4 instructions
- [✓] Existing knowledge checked (john-vervaeke-oracle)
- [✓] Web research conducted (4 search queries)
- [✓] ARR-COC-0-1 section connects to project implementation
- [✓] Sources section with full citations
- [✓] Proper markdown formatting
- [✓] Academic rigor maintained
- [✓] Links to related knowledge (john-vervaeke-oracle, ARR-COC-VIS-Application-Guide)

## Next Steps

- Oracle will review this KNOWLEDGE DROP
- Oracle will update INDEX.md when all Batch 1 (PARTs 1-4) complete
- Proceed to PART 5 (if instructed) or wait for batch review

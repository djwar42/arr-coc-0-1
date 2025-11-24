# KNOWLEDGE DROP: Figure Design & Visualization (PART 22)

**Date**: 2025-11-14 21:10
**Runner**: Worker executing PART 22
**File Created**: `research-methodology/05-figure-design-visualization.md`
**Status**: SUCCESS ✓

## What Was Created

Comprehensive guide to scientific figure design and data visualization (~710 lines), covering:

### 8 Core Sections

1. **Principles of Scientific Figure Design**
   - Tufte's graphical integrity, data-ink ratio, clarity principles
   - Nature's publication requirements (300 DPI, RGB/CMYK, vector formats)
   - Evidence-based checklist approach from arXiv:2408.16007

2. **Chart Types and Selection Criteria**
   - When to use bar charts, line graphs, scatter plots, box/violin plots, heatmaps
   - Multi-panel figure best practices
   - Color as storytelling tool

3. **Color Theory for Scientific Visualization**
   - Perceptually uniform color palettes (sequential, diverging, qualitative)
   - Colorblind-friendly design (1 in 12 men, 1 in 200 women have CVD)
   - HEX code examples for accessible palettes (blue+orange, purple+yellow)
   - Testing tools: Viz Palette, Color Brewer

4. **Typography and Labels**
   - Font selection (8-point minimum, serif for papers)
   - Matplotlib/Seaborn typography configuration
   - Axis labels with units, legend placement

5. **Error Bars and Uncertainty Visualization**
   - Standard error vs. standard deviation vs. confidence intervals
   - Bootstrapped CIs (Seaborn default)
   - Uncertainty in different plot types

6. **Multi-Panel Figures and Layout**
   - Sub-panel organization, white space, consistent sizing
   - Seaborn FacetGrid for subplots
   - Single-column (90mm) vs. double-column (180mm) sizing

7. **Tools and Workflows**
   - Python (Matplotlib/Seaborn): Full reproducible workflow
   - R (ggplot2), MATLAB, Excel/PowerPoint, Adobe Illustrator, Tableau
   - Code examples for programmatic figure generation

8. **ARR-COC-0-1 Visualization Requirements**
   - Relevance allocation heatmaps (64-400 token variable LOD)
   - Multi-scale visualization (patch boundaries, token density, query relevance)
   - Ablation study visualizations (2×3 factorial design)
   - Benchmark result comparisons
   - Attention vs. relevance mechanism visualization
   - Architecture diagrams (Knowing → Balancing → Attending → Realizing)

## Key Insights

### Publication-Quality Figure Workflow

From Jessica Hamrick's reproducible plotting tutorial:
- **No manual editing**: Fully programmatic = instant regeneration
- **Version control**: Track figure generation code
- **Efficiency**: 10 minutes from data to publication-ready figure

### Colorblind Accessibility is Critical

- 8% of men have color vision deficiency
- Blue + orange is universally safe combination
- Red + green CAN work if adjusted for lightness/saturation contrast
- Always test with Viz Palette tool

### ARR-COC-0-1 Specific Requirements

**Relevance allocation visualization needs:**
1. Heatmap overlays showing token allocation (64-400 per patch)
2. Perceptually uniform sequential colormap (blue→yellow→red)
3. Multi-panel showing 3 ways of knowing separately
4. Clear distinction from standard attention mechanisms

**Code example provided:**
```python
# Perceptually uniform colormap for relevance
colors = ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921']
relevance_cmap = LinearSegmentedColormap.from_list('relevance', colors)
plt.imshow(relevance_scores, cmap=relevance_cmap, alpha=0.6)
plt.colorbar(label='Token Allocation (64-400)')
```

## Web Research Sources (All Cited)

1. **arXiv:2408.16007** - Jambor (2024): Checklist for scientific data viz
2. **Jessica Hamrick** - Reproducible plots with Matplotlib/Seaborn (2016)
3. **Simplified Science Publishing** - Best color palettes for scientific figures
4. **Nature** - Scientific figures guide (Oct 2024)
5. **NIH** - Tips for tables, charts, figures (Sept 2024)
6. **NCEAS** - Colorblind safe color schemes (2022)
7. **Wiley/Current Protocols** - Crameri et al. (2024) color maps
8. **GeeksforGeeks** - Tufte's data visualization principles
9. **Viz Palette** - Interactive CVD testing tool
10. **Color Brewer** - Cartography color schemes

All sources include access dates (2025-11-14) and full URLs preserved in file.

## Quality Checklist

- [✓] 8 sections as specified in plan
- [✓] ~700 lines (actual: ~710 lines)
- [✓] Section 8 connects to ARR-COC-0-1 (relevance maps, ablation viz, benchmarks)
- [✓] All web research cited with dates and URLs
- [✓] Code examples for Python/Matplotlib workflow
- [✓] HEX codes for accessible color palettes
- [✓] Practical tools listed (Viz Palette, Color Brewer, etc.)
- [✓] Sources section at end with full citations
- [✓] KNOWLEDGE DROP file created

## Connection to ARR-COC-0-1

This knowledge directly supports creating figures for ARR-COC-0-1 papers:

**Immediate Applications:**
- Relevance allocation heatmaps with perceptually uniform colormaps
- Ablation study bar charts (2×3 factorial: fixed/reactive/predictive × oracle/bayesian)
- Benchmark comparison plots (VQA v2, GQA, TextVQA performance)
- Error bars using bootstrapped 95% CIs (Seaborn default)
- Colorblind-safe palettes for all figures

**Critical Distinctions:**
- Document emphasizes: ARR-COC uses RELEVANCE REALIZATION, not attention
- Figures must visually distinguish our opponent processing from standard QKV attention
- Architecture diagrams show process flow (gerunds: knowing, balancing, attending, realizing)

## Next Steps (For Oracle)

After all 24 PARTs complete:
1. Update INDEX.md with `research-methodology/05-figure-design-visualization.md`
2. Review integration with other research methodology files (00-04)
3. Ensure consistent cross-references
4. Git commit with comprehensive message

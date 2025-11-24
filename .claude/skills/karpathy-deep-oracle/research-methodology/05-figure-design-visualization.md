# Figure Design and Data Visualization for Scientific Publications

## Overview

Creating publication-quality figures is essential for effectively communicating research findings. Well-designed visualizations enhance comprehension, ensure accessibility, and strengthen the impact of scientific work. This document covers principles of scientific figure design, chart selection, color theory, typography, and tools for creating reproducible, publication-ready visualizations.

## Section 1: Principles of Scientific Figure Design

### Core Design Principles

From [Tufte's Data Visualization Principles](https://www.geeksforgeeks.org/data-visualization/mastering-tuftes-data-visualization-principles/) (accessed 2025-11-14):

**Graphical Integrity:**
- The representation must tell the truth about the data
- The physical representation of numbers should be directly proportional to the numerical quantities
- Use consistent scales
- Avoid visual distortions
- Present data in context with appropriate baselines

**Data-Ink Ratio:**
- Maximize the proportion of ink devoted to data
- Remove non-data ink (chart junk)
- Erase redundant data-ink
- Revise and edit

**Clarity and Simplicity:**
- Clear and detailed labeling
- Self-contained figures that can stand alone
- Appropriate font sizes (minimum 8-point)
- Consistent terminology

From [Nature's Scientific Figures Guide](https://www.nature.com/articles/d41586-024-03477-0) (Nature, October 28, 2024, accessed 2025-11-14):

**Key Parameters for Publication:**
- Resolution: 300 PPI/DPI minimum
- Color format: RGB for digital, CMYK for print
- File formats: Vector (PDF, SVG) for line art; high-resolution raster (TIFF, PNG) for images
- Appropriate figure dimensions for single-column (90mm) or double-column (180mm) layouts

### The Checklist Approach

From [arXiv:2408.16007 - From Zero to Figure Hero](https://arxiv.org/abs/2408.16007) (Jambor, 2024, accessed 2025-11-14):

**Essential Checklist Items:**
1. Appropriate display type selected for data
2. Color palettes are perceptually uniform and colorblind-friendly
3. Figure layout optimized for readability
4. Labels are clear and self-explanatory
5. Error bars or uncertainty measures included where appropriate
6. Data-to-ink ratio maximized
7. Consistent styling across all figures in a publication

**Evidence-Based Design:**
- Grounded in visualization research principles
- Consider cognitive load on viewers
- Prioritize comprehension over aesthetics alone
- Test figures with target audience when possible

## Section 2: Chart Types and Selection Criteria

### When to Use Different Chart Types

**Bar Charts:**
- Comparing discrete categories
- Showing frequencies or counts
- Displaying means with error bars
- Best for: 2-12 categories
- Avoid: Too many categories (use dot plots instead)

**Line Graphs:**
- Time series data
- Continuous relationships
- Multiple related trends
- Best for: Showing change over time or continuous variables
- Avoid: Discrete, unordered categories

**Scatter Plots:**
- Relationships between two continuous variables
- Correlation analysis
- Outlier detection
- Best for: Raw data visualization, regression analysis
- Enhance with: Trend lines, confidence intervals

**Box Plots/Violin Plots:**
- Distribution visualization
- Comparing distributions across groups
- Showing quartiles, medians, outliers
- Best for: Statistical comparisons
- Violin plots add: Probability density visualization

**Heatmaps:**
- Matrix data visualization
- Correlation matrices
- High-dimensional data
- Best for: Pattern detection in large datasets
- Requires: Perceptually uniform color scales

**Multi-Panel Figures:**
- Complex comparisons
- Multiple related analyses
- Best practices: Consistent scales, clear sub-panel labels (A, B, C), shared color schemes

From [Simplified Science Publishing](https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations) (accessed 2025-11-14):

**Color as Storytelling Tool:**
- Apply color selectively to highlight main findings
- Use grayscale for non-critical elements
- Reserve saturated colors for emphasis
- Maintain 15-30% saturation difference between adjacent elements

## Section 3: Color Theory for Scientific Visualization

### Perceptually Uniform Color Palettes

From [Wiley - Color Maps for Science](https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/cpz1.1126) (Crameri et al., 2024, accessed 2025-11-14):

**Sequential Palettes:**
- For ordered data (low to high)
- Examples: Temperature scales, concentration gradients
- Characteristics: Single hue with varying lightness
- Must be: Perceptually uniform, monotonically increasing

**Diverging Palettes:**
- For data with meaningful midpoint
- Examples: Deviations from mean, correlation coefficients
- Characteristics: Two hues diverging from neutral center
- Critical: Center point clearly distinguishable

**Qualitative Palettes:**
- For categorical data
- Requirements: Visually distinct, equal perceptual weight
- Limit: Maximum 8-10 categories for optimal discrimination

### Colorblind-Friendly Design

From [NCEAS Colorblind Safe Color Schemes](https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf) (accessed 2025-11-14):

**Color Vision Deficiency Statistics:**
- 1 in 12 men (8%) have CVD
- 1 in 200 women (0.5%) have CVD
- Most common: Red-green color blindness (deuteranopia, protanopia)

**Safe Color Combinations:**
- Blue + Orange (highly distinguishable)
- Blue + Red/Yellow (generally safe)
- Purple + Yellow/Green (good contrast)
- Avoid: Red + Green combinations without additional contrast

From [Simplified Science Publishing Color Guide](https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations) (accessed 2025-11-14):

**HEX Code Examples for Accessible Palettes:**

Two-color combinations:
- Blue + Orange: #3A5FCD, #FF8C00
- Purple + Yellow: #9370DB, #FFD700
- Dark Gray + Blue: #4A4A4A, #4682B4

Three-color combinations:
- Blue + Orange + Gray: #1E90FF, #FF8C00, #808080
- Green + Purple + Yellow: #32CD32, #9370DB, #FFD700

**Testing Tools:**
- [Viz Palette](https://projects.susielu.com/viz-palette) - Test palettes for CVD accessibility
- [Color Brewer](https://colorbrewer2.org) - Cartography-focused color schemes
- [Toptal Color Code](https://www.toptal.com/designers/colourcode) - Generate complementary palettes

### Color Characteristics: Hue, Saturation, Lightness

**Creating Contrast Without Hue:**
- Vary lightness: 15-30% minimum difference
- Adjust saturation: Desaturated vs. saturated versions
- Combine approaches: High-contrast grayscale remains effective
- Application: Allows red-green combinations when adjusted properly

## Section 4: Typography and Labels

### Font Selection and Sizing

From [NIH Tips for Figures](https://grants.nih.gov/grants-process/write-application/general-grant-writing-tips/tips-for-tables-charts-and-figures) (September 3, 2024, accessed 2025-11-14):

**Font Requirements:**
- Minimum: 8-point font size
- Recommended: 10-12 point for main labels
- Consistency: Use same font family throughout
- Serif fonts: Times, Palatino (matches manuscript text)
- Sans-serif: Arial, Helvetica (good for presentations)

**Label Guidelines:**
- Informative axis labels with units
- Concise but complete figure legends
- Clear, unambiguous notation
- Avoid jargon in labels
- Spell out abbreviations on first use

From [Jessica Hamrick - Reproducible Plots](https://www.jesshamrick.com/post/2016-04-13-reproducible-plots/) (accessed 2025-11-14):

**Matplotlib Typography Best Practices:**
```python
# Set font to match paper
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})

# Set appropriate context for publication
sns.set_context("paper")  # Optimized for paper figures
# Alternatives: "notebook", "talk", "poster"
```

### Axis Labels and Tick Marks

**X and Y-axis Labels:**
- Always include units (e.g., "Time (seconds)", "Concentration (μM)")
- Use scientific notation when appropriate
- Avoid diagonal text (exception: crowded x-axis categorical labels)
- Set tick mark intervals logically

**Legend Placement:**
- Position: Inside plot area if space permits, otherwise outside
- Order: Match data series order in figure
- Clarity: Use representative symbols/colors that match plot

## Section 5: Error Bars and Uncertainty Visualization

### Types of Error Bars

From [Hamrick - Reproducible Plots](https://www.jesshamrick.com/post/2016-04-13-reproducible-plots/) (accessed 2025-11-14):

**Standard Error (SE):**
- Shows precision of mean estimate
- Smaller with larger sample sizes
- Use when: Emphasizing statistical precision

**Standard Deviation (SD):**
- Shows spread of data
- Independent of sample size
- Use when: Showing variability in measurements

**Confidence Intervals (CI):**
- 95% CI most common
- Bootstrapped CIs: Distribution-free method
- Use when: Making statistical inferences

**Seaborn Default:**
```python
# Seaborn uses bootstrapped 95% CI by default
sns.barplot(x="condition", y="measurement", data=df)
# Automatically calculates and displays error bars
```

### Uncertainty in Different Plot Types

**Bar Plots:**
- Error bars extend from top of bar
- Include both positive and negative error when symmetric
- Asymmetric errors: Use different lengths for upper/lower

**Line Plots:**
- Error bars at each data point (can be cluttered)
- Confidence bands/ribbons for continuous data
- Transparency helps when multiple series overlap

**Scatter Plots:**
- Error bars in both dimensions when appropriate
- Ellipses for correlated errors
- Size/transparency can encode uncertainty

## Section 6: Multi-Panel Figures and Layout

### Figure Layout Principles

**Sub-panel Organization:**
- Logical flow: Left to right, top to bottom
- Consistent sizing: Align panels on grid
- Clear labels: (A), (B), (C) or (a), (b), (c)
- Shared elements: Common legends, consistent axes when possible

**White Space:**
- Don't overcrowd: Leave breathing room
- Margins: Adequate space around plot area
- Padding: Consistent spacing between panels
- Purpose: Improves readability, reduces cognitive load

From [Hamrick - Reproducible Plots](https://www.jesshamrick.com/post/2016-04-13-reproducible-plots/) (accessed 2025-11-14):

**Using FacetGrid for Multi-Panel Figures:**
```python
# Create subplots for different conditions
g = sns.FacetGrid(
    data,
    col="robot",
    col_order=["fixed", "reactive", "predictive"],
    sharex=False
)

# Apply same plot to each subplot
g.map(sns.barplot, "robot", "tasks", "inference")

# Customize individual subplots
axes = np.array(g.axes.flat)
for ax in axes:
    # Apply subplot-specific customization
    ax.set_xlabel("Custom Label")
```

### Size and Aspect Ratio

**Single-Column Figures:**
- Width: 90mm (3.5 inches) typical
- Aspect ratio: 4:3 or 16:9 depending on content
- Resolution: 300 DPI minimum

**Double-Column Figures:**
- Width: 180mm (7 inches) typical
- Use for: Complex multi-panel figures, wide visualizations
- Consider: Readability when reduced in print

**Figure Sizing in Matplotlib:**
```python
def set_size(fig):
    fig.set_size_inches(6, 3)  # Width, height in inches
    plt.tight_layout()  # Automatically adjust spacing
```

## Section 7: Tools and Workflows for Publication-Quality Figures

### Python: Matplotlib and Seaborn

From [Hamrick - Reproducible Plots](https://www.jesshamrick.com/post/2016-04-13-reproducible-plots/) (accessed 2025-11-14):

**Advantages:**
- Fully reproducible: No manual editing required
- Programmatic control: Every element customizable
- Version control: Track changes in figure generation code
- Efficiency: Regenerate figures instantly when data changes

**Recommended Workflow:**
```python
# 1. Set global style
def set_style():
    sns.set_context("paper")
    sns.set(font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

# 2. Create plot
def create_plot(data):
    fig, ax = plt.subplots()
    sns.barplot(x="x", y="y", data=data, ax=ax)
    return fig, ax

# 3. Customize labels
def set_labels(fig, ax):
    ax.set_xlabel("X Label (units)")
    ax.set_ylabel("Y Label (units)")
    ax.set_title("Figure Title")
    sns.despine()  # Remove top and right spines

# 4. Set size and save
def finalize_figure(fig):
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    fig.savefig("figure.pdf", dpi=300, bbox_inches='tight')
```

**Color Customization:**
```python
# Define custom color palette
colors = np.array([
    [0.1, 0.1, 0.1],          # black
    [0.984375, 0.7265625, 0],  # dark yellow
    [0.7, 0.7, 0.7],          # gray
])

# Apply to individual patches
for i, patch in enumerate(ax.patches):
    patch.set_color(colors[i])
    # Add hatch patterns for additional distinction
    if i % 2 == 1:
        patch.set_hatch('////')
```

### R: ggplot2

**Color Palette Resources:**
- [Emil Hvitfeldt's R Color Palettes](https://github.com/EmilHvitfeldt/r-color-palettes) - Comprehensive collection
- Viridis: Perceptually uniform, colorblind-safe
- ColorBrewer: Designed for maps, works well for all visualizations

**Best Practices:**
- Use themes: `theme_minimal()`, `theme_bw()`, `theme_classic()`
- Consistent `theme()` settings across figures
- Save with `ggsave()` for precise sizing and resolution

### MATLAB

**Color Picker:**
- RGB values: Specify exact colors
- `c = uisetcolor`: Interactive color selection
- Built-in colormaps: `parula`, `viridis`, `jet` (avoid jet for scientific work)

**Resources:**
- [MATLAB uisetcolor documentation](https://www.mathworks.com/help/matlab/ref/uisetcolor.html)
- [Colors in MATLAB guide](http://math.loyola.edu/~loberbro/matlab/html/colorsInMatlab.html)

### Microsoft Excel and PowerPoint

**Eye Dropper Tool:**
- Available in recent versions
- Select colors from any screen element
- Useful for matching institutional branding

**Hex Code Input:**
- Some versions allow direct HEX code entry
- Right-click color formatting → More Colors → Custom

### Adobe Illustrator

**Strengths:**
- Vector-based: Infinitely scalable
- Precise control: Exact positioning, sizing
- Professional output: Publication-ready PDFs

**Color Management:**
- Swatches: Save custom color palettes
- Color Picker: Enter HEX codes directly
- Global colors: Update all instances simultaneously

**Workflow:**
- Import data plots from Python/R/MATLAB
- Refine in Illustrator: Alignment, annotations, assembly
- Export: PDF for submission, PNG for presentations

### Tableau

**Interactive Visualizations:**
- Good for: Exploratory data analysis
- Color palettes: Built-in accessibility options
- Export: Static images for publications

**Resources:**
- [Tableau Color Help](https://help.tableau.com/current/pro/desktop/en-us/viewparts_marks_markproperties_color.htm)

## Section 8: ARR-COC-0-1 Visualization Requirements

### Relevance Allocation Visualization

**Attention Map Figures:**

The ARR-COC-0-1 system requires visualizations that show how relevance realization allocates visual tokens across image patches. Key visualization requirements:

**Heatmap Overlays:**
- Base image: Original input (RGB)
- Overlay: Relevance scores per patch (64-400 tokens allocated)
- Color scale: Sequential palette showing token allocation density
  - Low relevance: Cool colors (blue) or low opacity
  - High relevance: Warm colors (red/yellow) or high opacity
- Transparency: Alpha blending to preserve base image visibility

**Recommended Palette for Relevance Maps:**
```python
# Perceptually uniform sequential palette
# Blue (low relevance) → Yellow (medium) → Red (high relevance)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921']
n_bins = 100
relevance_cmap = LinearSegmentedColormap.from_list('relevance', colors, N=n_bins)

# Apply to heatmap
plt.imshow(relevance_scores, cmap=relevance_cmap, alpha=0.6)
plt.colorbar(label='Token Allocation (64-400)')
```

**Multi-Scale Visualization:**

Since ARR-COC-0-1 uses variable LOD (Level of Detail), figures should show:
1. Patch boundaries: Grid overlay showing 14×14 patch divisions
2. Token density: Color-coded by allocation (64-400 tokens per patch)
3. Query relevance: Separate subpanels for different query types

Example multi-panel layout:
- Panel A: Original image
- Panel B: Propositional knowing scores (entropy)
- Panel C: Perspectival knowing scores (salience)
- Panel D: Participatory knowing scores (query-content coupling)
- Panel E: Combined relevance allocation (final token distribution)

**Ablation Study Visualizations:**

Comparing different allocation strategies requires careful figure design:

```python
# 2×3 factorial design visualization
conditions = ['Fixed', 'Reactive', 'Predictive']
metrics = ['Oracle', 'Bayesian']

# Create grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

# Color scheme: One color per robot type, with light/dark for inference
colors = {
    'Fixed': ['#1f77b4', '#aec7e8'],      # blue shades
    'Reactive': ['#ff7f0e', '#ffbb78'],   # orange shades
    'Predictive': ['#2ca02c', '#98df8a']  # green shades
}

# Apply consistent styling
for i, condition in enumerate(conditions):
    for j, metric in enumerate(metrics):
        ax = axes[j, i]
        # Plot bars with appropriate colors
        dark_color = colors[condition][0]
        light_color = colors[condition][1]
        # Add hatch pattern to distinguish oracle vs bayesian
        if j == 1:  # Bayesian
            ax.patches[0].set_hatch('////')
```

**Error Bar Requirements:**

For ARR-COC-0-1 experimental results:
- Use bootstrapped 95% confidence intervals (Seaborn default)
- Show error bars for: Accuracy, inference time, token efficiency
- N=20+ participants for human evaluation studies
- Multiple runs (5-10) for computational benchmarks

**Colorblind Accessibility for ARR-COC:**

Critical for reaching broad audience:
- Test all figures with [Viz Palette](https://projects.susielu.com/viz-palette)
- Primary colors: Blue and orange (safe for all CVD types)
- Secondary distinction: Hatch patterns, different shapes
- Avoid: Red-green without additional contrast cues

**Figure Captions:**

Comprehensive captions following journal requirements:
```
Figure X: Token allocation visualization for ARR-COC-0-1 across query types.
(A) Original image with 14×14 patch grid overlay. (B-D) Relevance scores from
three ways of knowing: (B) Propositional (entropy-based), (C) Perspectival
(salience-based), (D) Participatory (query-aware coupling). (E) Final token
allocation after opponent processing, showing variable LOD from 64-400 tokens
per patch. Warm colors indicate higher relevance and increased token allocation.
Colormap is perceptually uniform and colorblind-safe. N=100 images from
VQA v2 validation set. Error bars show bootstrapped 95% confidence intervals.
```

### Benchmark Result Visualization

**Performance Comparison Figures:**

For comparing ARR-COC-0-1 against baselines (Qwen3-VL, Ovis 2.5, etc.):

1. **Bar charts with clear grouping:**
   - X-axis: Benchmark datasets (VQA v2, GQA, TextVQA, etc.)
   - Y-axis: Accuracy (%) or other metric
   - Groups: Different models
   - Colors: One color per model, consistent across all figures

2. **Pareto frontier plots:**
   - X-axis: Computational cost (GFLOPs or tokens used)
   - Y-axis: Accuracy
   - Points: Different token budgets (64-400 range)
   - Connection: Lines showing Pareto-optimal configurations

3. **Ablation tables with visual encoding:**
   - Heatmap cells showing relative performance
   - Sequential colormap: Low (white/light) to high (dark/saturated)
   - Clear ranking: Bold or highlight best performing

**Example Code for Benchmark Visualization:**
```python
def plot_benchmark_results(results_df):
    """Create publication-quality benchmark comparison"""

    # Set style
    set_style()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot grouped bars
    datasets = results_df['dataset'].unique()
    models = results_df['model'].unique()

    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    # Color palette: colorblind-safe
    colors = ['#0173b2', '#de8f05', '#029e73', '#cc78bc']

    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        offset = (i - len(models)/2) * width + width/2
        bars = ax.bar(x + offset, model_data['accuracy'],
                     width, label=model, color=colors[i])

        # Add error bars
        ax.errorbar(x + offset, model_data['accuracy'],
                   yerr=model_data['std'], fmt='none',
                   color='black', capsize=3, linewidth=1)

    # Labels
    ax.set_xlabel('Benchmark Dataset')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='best', frameon=False)
    ax.set_ylim(0, 100)

    # Remove top and right spines
    sns.despine()

    plt.tight_layout()
    return fig, ax
```

### Attention Mechanism Comparison

**Visualizing Relevance vs. Standard Attention:**

Critical distinction for ARR-COC-0-1 papers: We use relevance realization, not attention.

Figure requirements:
- Side-by-side comparison: Transformer attention maps vs. ARR-COC relevance maps
- Same input image for both
- Clear caption explaining the difference
- Visual encoding: Both use same colormap for fair comparison

**Architecture Diagram:**

Showing the pipeline: Knowing → Balancing → Attending → Realizing

Best practices:
- Flow diagram: Left to right
- Clear arrows showing data flow
- Boxes with rounded corners for processes
- Consistent color coding: Inputs (blue), Processes (orange), Outputs (green)
- Annotations explaining key steps
- Vector graphics (SVG/PDF) for crispness

Tools:
- Draw.io or Diagrams.net: Free, web-based
- Adobe Illustrator: Professional control
- TikZ (LaTeX): Fully reproducible, version-controlled

## Sources

**Web Research:**

From [arXiv:2408.16007](https://arxiv.org/abs/2408.16007) - Jambor, H.K. (2024). "From zero to figure hero: A checklist for designing scientific data visualizations." arXiv preprint. (accessed 2025-11-14)

From [Jessica Hamrick's Blog](https://www.jesshamrick.com/post/2016-04-13-reproducible-plots/) - Hamrick, J. (2016). "Creating Reproducible, Publication-Quality Plots with Matplotlib and Seaborn." (accessed 2025-11-14)

From [Simplified Science Publishing](https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations) - "Best Color Palettes for Scientific Figures and Data Visualizations." (accessed 2025-11-14)

From [Nature](https://www.nature.com/articles/d41586-024-03477-0) - "Scientific figures that pop: resources for the artistically challenged." Nature, October 28, 2024. (accessed 2025-11-14)

From [NIH](https://grants.nih.gov/grants-process/write-application/general-grant-writing-tips/tips-for-tables-charts-and-figures) - "Tips for Tables, Charts, and Figures." National Institutes of Health, September 3, 2024. (accessed 2025-11-14)

From [NCEAS](https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf) - "Colorblind Safe Color Schemes." National Center for Ecological Analysis and Synthesis, June 29, 2022. (accessed 2025-11-14)

From [Wiley](https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/cpz1.1126) - Crameri, F. et al. (2024). "Choosing Suitable Color Palettes for Accessible and Accurate Data Visualization." Current Protocols. (accessed 2025-11-14)

From [GeeksforGeeks](https://www.geeksforgeeks.org/data-visualization/mastering-tuftes-data-visualization-principles/) - "Mastering Tufte's Data Visualization Principles." (accessed 2025-11-14)

**Additional References:**

- [Viz Palette Tool](https://projects.susielu.com/viz-palette) - Meeks, E. & Lu, S. - Interactive colorblind accessibility testing
- [Color Brewer](https://colorbrewer2.org) - Brewer, C., Harrower, M., and The Pennsylvania State University - Cartography color schemes
- [Emil Hvitfeldt's R Color Palettes](https://github.com/EmilHvitfeldt/r-color-palettes) - Comprehensive R palette collection
- [MATLAB Colors Guide](http://math.loyola.edu/~loberbro/matlab/html/colorsInMatlab.html) - Loyola University
- [Tableau Color Help](https://help.tableau.com/current/pro/desktop/en-us/viewparts_marks_markproperties_color.htm) - Tableau Documentation

**Connection to ARR-COC-0-1:**

The visualization principles and tools in this document directly support the creation of figures for ARR-COC-0-1 publications, including relevance allocation heatmaps, attention mechanism comparisons, ablation study results, and benchmark performance visualizations. All ARR-COC-0-1 figures must be colorblind-accessible, use perceptually uniform colormaps for relevance scores, and clearly distinguish between standard attention and our relevance realization approach.

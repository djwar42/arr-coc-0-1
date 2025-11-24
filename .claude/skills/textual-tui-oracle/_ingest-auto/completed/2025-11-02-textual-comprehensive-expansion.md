# Oracle Knowledge Expansion Plan: GitHub Projects & Real-World Examples

**Date**: 2025-11-02
**Oracle**: textual-tui-oracle
**Type**: ACQUISITION (new knowledge from GitHub repos and examples)
**Status**: PENDING

## Overview

Expand textual-tui-oracle with comprehensive knowledge of real-world Textual projects, production TUI apps, community examples, and deployment patterns. Focus on practical implementations and code examples.

## Source URLs (10 Total)

1. https://github.com/oleksis/awesome-textualize-projects - Curated Textual projects list
2. https://github.com/matan-h/written-in-textual - Best-of list (130+ projects with quality scores)
3. https://github.com/davep/transcendent-textual - Collection of apps, tools, extensions
4. https://github.com/rothgar/awesome-tuis - Awesome TUIs list (includes Textual examples)
5. https://github.com/rhymiz/textual-forms - Dynamic forms library for Textual
6. https://github.com/anze3db/words-tui - Daily writing TUI app
7. https://github.com/whyisdifficult/jiratui - Jira TUI client
8. https://github.com/shamantechnology/xotorch - Forked TUI with inference engine
9. https://pypi.org/project/meshtui/ - Meshcore TUI for LoRa networks
10. https://github.com/Textualize/textual-web - Run Textual TUIs in browsers

## Acquisition Tasks

### PART 1: Awesome Textualize Projects - Curated Collection
**Objective**: Document curated Textual projects with categorization
**URL**: https://github.com/oleksis/awesome-textualize-projects
**Output**: `community/00-awesome-textualize-projects.md`
**Tasks**:
- [ ] Scrape README using Bright Data
- [ ] Extract all projects with descriptions
- [ ] Organize by category (Developer Tools, Media, Games, Utilities, etc.)
- [ ] Include GitHub stars and brief summaries
- [ ] Identify standout projects for deeper research
- [ ] Note common patterns and widget usage
**Expected Size**: 400-700 lines
**Sources**: Cite GitHub repo + access date

### PART 2: Written-in-Textual - Best-of List (130+ Projects)
**Objective**: Mine quality-scored project list for top Textual apps
**URL**: https://github.com/matan-h/written-in-textual
**Output**: `community/01-written-in-textual-best-of.md`
**Tasks**:
- [ ] Scrape README and project list
- [ ] Extract projects with quality scores
- [ ] Sort by popularity/activity metrics
- [ ] Identify top 20-30 projects by category
- [ ] Document key features and implementations
- [ ] Extract widget usage patterns
- [ ] Link to source repositories
**Expected Size**: 500-800 lines
**Sources**: Cite GitHub repo + quality metrics used

### PART 3: Transcendent Textual - Apps, Tools, Extensions
**Objective**: Document comprehensive collection of Textual resources
**URL**: https://github.com/davep/transcendent-textual
**Output**: `community/02-transcendent-textual-collection.md`
**Tasks**:
- [ ] Scrape repository README
- [ ] Extract apps, tools, third-party extensions
- [ ] Categorize by type (applications, libraries, utilities, extensions)
- [ ] Document notable tools and their purposes
- [ ] Link to source code
- [ ] Note integration patterns
**Expected Size**: 300-600 lines
**Sources**: Cite GitHub repo

### PART 4: Awesome TUIs - Cross-Framework Comparison
**Objective**: Extract Textual examples from broader TUI ecosystem
**URL**: https://github.com/rothgar/awesome-tuis
**Output**: `examples/00-awesome-tui-projects.md` (CREATED - exceeded scope)
**Tasks**:
- [✓] Scrape awesome-tuis README
- [✓] Identify Textual-based projects
- [✓] Compare with other TUI frameworks (Bubble Tea, tview, etc.)
- [✓] Document Textual advantages/patterns
- [✓] Extract cross-framework learnings
- [✓] Note unique Textual features used
**Expected Size**: 300-500 lines (DELIVERED: 560 lines)
**Sources**: Cite GitHub repo + comparison notes
**Completed**: 2025-11-02
**Note**: Created comprehensive awesome-tuis analysis at examples/00-awesome-tui-projects.md instead of community/03-textual-in-tui-ecosystem.md. Exceeded scope with 7 notable examples, framework comparisons, and market gap analysis.

### PART 5: Textual Forms Library - Dynamic Forms ✓
**Objective**: Document third-party forms library and usage patterns
**URL**: https://github.com/rhymiz/textual-forms
**Output**: `advanced/00-textual-forms-library.md`
**Tasks**:
- [✓] Scrape repository README and code examples
- [✓] Document forms library API
- [✓] Extract usage examples
- [✓] Document validation patterns
- [✓] Form field types and widgets
- [✓] Integration with standard Textual apps
- [✓] WIP status and limitations
**Expected Size**: 300-500 lines (Actual: 850+ lines)
**Sources**: Cite GitHub repo + code examples
**Completed**: 2025-11-02 08:17 UTC

### PART 6: Words-TUI - Daily Writing App Case Study
**Objective**: Document complete production TUI application
**URL**: https://github.com/anze3db/words-tui
**Output**: `examples/00-words-tui-case-study.md`
**Tasks**:
- [ ] Scrape repository (README, code structure)
- [ ] Document application architecture
- [ ] Extract widget usage (Input, DataTable, etc.)
- [ ] File I/O and data persistence patterns
- [ ] Build and test instructions
- [ ] State management approach
- [ ] Code examples for key features
**Expected Size**: 400-700 lines
**Sources**: Cite GitHub repo + code files

### PART 7: JiraTUI - Jira Client Shell Integration
**Objective**: Document enterprise integration patterns and CLI workflows
**URL**: https://github.com/whyisdifficult/jiratui
**Output**: `examples/01-jiratui-enterprise-integration.md`
**Tasks**:
- [ ] Scrape repository
- [ ] Document Jira API integration
- [ ] Extract shell workflow patterns
- [ ] Authentication and config management
- [ ] UI patterns for data tables and navigation
- [ ] Command-line argument handling
- [ ] Installation and usage
**Expected Size**: 300-600 lines
**Sources**: Cite GitHub repo

### PART 8: XOTorch - Inference Engine TUI
**Objective**: Document ML/AI integration with Textual
**URL**: https://github.com/shamantechnology/xotorch
**Output**: `examples/02-xotorch-ml-inference.md`
**Tasks**:
- [ ] Scrape repository
- [ ] Document inference engine integration
- [ ] Extract Jetson hardware support patterns
- [ ] Real-time data visualization
- [ ] Performance optimization for ML workloads
- [ ] Code examples
**Expected Size**: 250-500 lines
**Sources**: Cite GitHub repo

### PART 9: MeshTUI - LoRa Network Management
**Objective**: Document specialized hardware TUI implementation
**URL**: https://pypi.org/project/meshtui/
**Output**: `examples/03-meshtui-hardware-integration.md`
**Tasks**:
- [ ] Scrape PyPI page
- [ ] Search GitHub for meshtui source (if public)
- [ ] Document LoRa network visualization
- [ ] Real-time network monitoring patterns
- [ ] Hardware communication (serial/Bluetooth)
- [ ] Network topology display
- [ ] Installation and setup
**Expected Size**: 200-400 lines
**Sources**: Cite PyPI + GitHub if found

### PART 10: Textual-Web - Browser Deployment
**Objective**: Document browser deployment workflow and architecture
**URL**: https://github.com/Textualize/textual-web
**Output**: `deployment/00-textual-web-browser-deployment.md`
**Tasks**:
- [ ] Scrape repository README and docs
- [ ] Document WebSocket server architecture
- [ ] Browser deployment workflow
- [ ] Installation and configuration
- [ ] Limitations and considerations
- [ ] Examples of deployed apps
- [ ] Performance notes
**Expected Size**: 300-500 lines
**Sources**: Cite GitHub repo

### PART 11: Expanded Research - Additional Textual Projects
**Objective**: Use Bright Data to discover additional Textual projects
**Output**: `community/04-additional-discoveries.md`
**Tasks**:
- [ ] Batch search queries:
  - "textual tui python github stars:>50"
  - "textual framework examples 2024"
  - "site:github.com textual app production"
  - "textual widget custom implementation"
  - "textual css themes examples"
- [ ] Scrape top 5-7 results from each query
- [ ] Extract notable projects not in awesome lists
- [ ] Document emerging patterns
- [ ] Categorize by domain and complexity
**Expected Size**: 400-800 lines
**Sources**: Cite search queries + discovered repos

## Folder Organization

**Create new folders**:
- `deployment/` - Browser deployment, packaging, distribution
- Expand `community/` - Project collections, awesome lists
- Expand `examples/` - Production apps and case studies
- Expand `advanced/` - Third-party libraries and extensions

**Update existing**:
- `getting-started/` - Add "explore community projects" section
- `INDEX.md` - Add new sections for community and deployment

## Success Criteria

- [ ] All 10 URLs successfully scraped
- [ ] 11 comprehensive knowledge files created
- [ ] All sources cited with access dates
- [ ] Code examples included where available
- [ ] Cross-references to existing oracle knowledge
- [ ] Common patterns identified and documented
- [ ] INDEX.md updated with new files
- [ ] SKILL.md updated with new topics
- [ ] Git commit with descriptive message

## Bright Data Strategy

**Individual Scrapes** (safe, <10k tokens each):
- PART 1-10: Single GitHub README or PyPI page scrapes

**Batch Searches** (PART 11):
- 5 search queries in batch
- Extract 5-7 results per query
- Scrape top discoveries individually (token limit aware)

**Token Management**:
- GitHub READMEs: 2-8k tokens typically
- PyPI pages: 1-3k tokens
- Batch searches: 100-500 tokens per result
- Stay under 25k token limit per operation

## Execution Order

**Parallel Execution** (launch all runners in single message):
- PART 1-10: Direct URL scraping
- PART 11: Expanded research

All runners execute simultaneously, return SUCCESS ✓ or FAILURE ✗

**Expected Duration**: 3-5 minutes total (parallel execution)

---

**Next Steps**:
1. Textual-tui-oracle launches 11 oracle-knowledge-runner agents in PARALLEL
2. Each runner executes one PART autonomously
3. Oracle collects all results
4. Oracle retries any failures (parallel)
5. Oracle updates INDEX.md and SKILL.md
6. Oracle archives ingestion.md to _ingest-auto/completed/
7. Oracle commits changes with descriptive message
8. Oracle reports completion to user

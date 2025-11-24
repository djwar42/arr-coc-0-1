# Textual TUI Oracle - Knowledge Expansion Plan

**Date**: 2025-11-02
**Oracle**: textual-tui-oracle
**Source**: Community links + web research
**Target**: Add real-world examples, patterns, best practices

---

## Overview

This expansion adds practical Textual knowledge from community resources, real-world applications, and Python best practices useful for TUI development.

**Total PARTs**: 6 (to be executed in PARALLEL)

---

## PART 1: JiraTUI - Production Textual Application

**Objective**: Document JiraTUI as a real-world Textual app example

**Sources**:
- https://github.com/whyisdifficult/jiratui
- GitHub repo (1k stars, active)

**Target Files**:
- `examples/01-jiratui-case-study.md`

**Content to Extract**:
- Application architecture (CLI + TUI structure)
- Widget usage (DataTable, Input, ListView)
- Config management (YAML)
- API integration patterns (Jira REST API)
- Event handling examples
- Testing approaches
- CLI design patterns (Click integration)

**Success Criteria**:
- Complete case study with code examples
- Architecture diagram (ASCII)
- Key learnings section
- Links to GitHub for reference

---

## PART 2: Awesome Textual Projects - Curated Examples

**Objective**: Document notable Textual applications and resources

**Status**: [✓] COMPLETED 2025-11-02 15:45

**Sources**:
- https://github.com/oleksis/awesome-textualize-projects
- https://github.com/davep/transcendent-textual
- https://github.com/matan-h/written-in-textual

**Target Files**:
- `examples/02-community-projects.md` - CREATED

**Content Extracted**:
- Top 15 production Textual apps with detailed profiles
- 5 application categories (Developer Tools, File Viewers, Data Apps, Chat, Utilities)
- 8 common patterns identified with code examples
- Integration examples (APIs, databases, file systems)

**Completion Details**:
- 15 production applications profiled (Harlequin, Trogon, Frogmouth, Browsr, Kupo, Dolphie, Toolong, Elia, Textual Paint, Dooit, Termtyper, RecoverPy, NoteSH, Baca, Django-TUI)
- Categorized by domain and use case
- Common patterns extracted and documented
- 1200+ line knowledge base created
- All projects linked with GitHub URLs
- Textual pattern examples provided

---

## PART 3: Python Best Practices for Textual Development

**Objective**: Document idiomatic Python patterns useful for Textual apps

**Sources**:
- https://realpython.com/courses/writing-idiomatic-python/
- https://github.com/trekhleb/learn-python
- https://www.tunnelsup.com/python-cheat-sheet/
- https://learnbyexample.github.io/python-intermediate/

**Target Files**:
- `best-practices/00-python-idioms.md`
- `best-practices/01-code-organization.md`

**Content to Extract**:
- Zen of Python applied to TUI apps
- Truth value testing (for widget state)
- In-place variable swapping
- DRY principle examples
- Idiomatic for loops (widget iteration)
- Dict default values (config management)
- Built-in functions usage

**Success Criteria**:
- Practical examples adapted for Textual
- Code snippets with explanations
- Anti-patterns to avoid
- Links to deeper resources

---

## PART 4: Textual Widget Patterns & Tutorials

**Objective**: Document common widget patterns and real-world usage

**Sources**:
- https://textual.textualize.io/tutorial/
- https://realpython.com/python-textual/
- https://mathspp.com/blog/textual-for-beginners
- https://arjancodes.com/blog/textual-python-library-for-creating-interactive-terminal-applications/
- https://www.blog.pythonlibrary.org/2025/04/01/textual-how-to-add-widgets-to-a-container/

**Target Files**:
- `widgets/00-widget-patterns.md`
- `widgets/01-custom-widgets.md`
- `layout/00-layout-patterns.md`

**Content to Extract**:
- Common widget composition patterns
- Custom widget creation (extending Widget)
- Layout best practices (vertical, horizontal, grid, dock)
- Dynamic widget addition/removal
- Widget lifecycle patterns
- Event handling patterns
- CSS styling patterns

**Success Criteria**:
- Practical pattern library
- Code examples for each pattern
- When to use each approach
- Common pitfalls

---

## PART 5: Real-World Project Case Studies

**Objective**: Document lessons learned and best practices from production apps

**Sources**:
- https://www.textualize.io/blog/7-things-ive-learned-building-a-modern-tui-framework
- https://dev.to/wiseai/textual-the-definitive-guide-part-1-1i0p
- https://oneryalcin.medium.com/building-a-responsive-textual-chat-ui-with-long-running-processes-c0c53cd36224
- https://codecurrents.blog/article/2024-10-14 (TUI project template)

**Target Files**:
- `best-practices/02-lessons-learned.md`
- `patterns/00-long-running-processes.md`
- `patterns/01-project-templates.md`

**Content to Extract**:
- Terminal quirks and solutions
- Performance optimization tips
- Long-running process handling (Workers)
- Chat UI patterns (message streaming)
- Project structure templates
- Dependency management (uv, requirements.txt)
- Build/deployment patterns

**Success Criteria**:
- Actionable lessons with examples
- Problem-solution format
- Code snippets
- Tool recommendations

---

## PART 6: Python Learning Resources for Textual Developers

**Objective**: Create learning pathway for Textual beginners

**Sources**:
- https://github.com/trekhleb/learn-python (17.4k stars)
- https://learnbyexample.github.io/python-intermediate/
- Python Exercises TUI app (mentioned in resources)

**Target Files**:
- `getting-started/02-python-prerequisites.md`
- `getting-started/03-learning-path.md`

**Content to Extract**:
- Python fundamentals needed for Textual
- Testing patterns (pytest for TUI apps)
- Debugging TUI applications
- Project-based learning approach
- Python Exercises TUI app as example
- Next steps after basics

**Success Criteria**:
- Clear learning pathway
- Prerequisites checklist
- Resource links organized by level
- Practice project suggestions

---

## Post-Ingestion Tasks

After all PARTs complete:

1. **Update INDEX.md**: Add all new files with descriptions
2. **Update SKILL.md**: Add new topics to navigation
3. **Cross-reference**: Link related files
4. **Quality check**: Verify all code examples are valid
5. **Git commit**: "Knowledge expansion: Community examples, patterns, and best practices"

---

## Notes

- All web research performed in-memory by oracle-knowledge-runner
- Respect 25k token limit for Bright Data tools
- Include source citations in all documents
- Use numbered prefixes for file organization
- Focus on practical, actionable knowledge

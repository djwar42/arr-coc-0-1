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

**Status**: [✓] COMPLETED 2025-11-02 08:35

**Objective**: Document JiraTUI as a real-world Textual app example

**Sources**:
- https://github.com/whyisdifficult/jiratui
- GitHub repo (1k stars, active)

**Target Files Created**:
- `examples/01-jiratui-case-study.md` (880+ lines)

**Content Delivered**:
- Application architecture (CLI + TUI structure) with ASCII diagram
- Widget usage (DataTable, Select, TabbedContent, Input widgets)
- Config management (XDG-compliant YAML)
- API integration patterns (Jira REST API v2/v3 with async/httpx)
- Event handling examples (decorators, worker threads)
- Testing approaches (pytest, pytest-asyncio, respx mocking)
- CLI design patterns (Click framework with command groups)
- Configuration best practices and environment overrides

**Success Criteria Achieved**:
- [✓] Complete case study with extensive code examples
- [✓] Architecture diagram (ASCII flow chart)
- [✓] Key learnings section (8 main patterns)
- [✓] Links to GitHub and related resources
- [✓] Practical examples for widgets, CLI, API, configuration
- [✓] Interesting technical decisions explained
- [✓] Challenges and solutions table
- [✓] Deployment and distribution patterns

---

## PART 2: Awesome Textual Projects - Curated Examples

**Objective**: Document notable Textual applications and resources

**Sources**:
- https://github.com/oleksis/awesome-textualize-projects
- https://github.com/davep/transcendent-textual
- https://github.com/matan-h/written-in-textual

**Target Files**:
- `examples/02-community-projects.md`

**Content to Extract**:
- Top 10-15 production Textual apps (avocet, baca, browsr, etc.)
- Application categories (TUI tools, data viewers, chat clients, etc.)
- Common patterns across projects
- Integration examples (APIs, databases, file systems)

**Success Criteria**:
- Categorized list with descriptions
- Links to all projects
- Common pattern identification
- Use case examples

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

## PART 6: Words-TUI - Daily Writing App Case Study

**Objective**: Document complete production TUI application

**Status**: [✓] COMPLETED 2025-11-02 08:17

**Sources**:
- https://github.com/anze3db/words-tui (main repository)
- GitHub README and project structure
- pyproject.toml configuration file

**Target Files Created**:
- `examples/00-words-tui-case-study.md` (580 lines)

**Content Delivered**:
- Complete application architecture overview with technology stack
- Database integration patterns using Peewee ORM
- CLI configuration patterns (Click framework, environment variables)
- Widget usage examples (Input, DataTable, Static widgets)
- File I/O and data persistence strategies
- Build and installation instructions
- Testing strategy and CI/CD setup
- Best practices for production Textual apps
- Performance considerations
- Lessons for Textual developers

**Success Criteria Achieved**:
- [✓] Comprehensive case study of production TUI app
- [✓] Architecture breakdown with code patterns
- [✓] Database integration examples
- [✓] Widget usage documentation
- [✓] Build/test/deploy patterns
- [✓] All sources cited with links
- [✓] Practical lessons and takeaways section

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

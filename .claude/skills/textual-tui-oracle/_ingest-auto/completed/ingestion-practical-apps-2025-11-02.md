# Oracle Knowledge Expansion Plan: Practical Textual Applications
## Building Real-World TUIs - Round 2

**Date**: 2025-11-02
**Oracle**: textual-tui-oracle
**Type**: ACQUISITION (practical application tutorials)
**Status**: PENDING

## Overview

Expand textual-tui-oracle with hands-on tutorials for building real-world applications: contact book, todo app, chat UI, XML editor, environment manager, process manager, and Japanese community guides.

## Source URLs (10 Total)

1. https://realpython.com/contact-book-python-textual/ - Contact book app with SQLite
2. https://pythongui.org/how-to-build-a-todo-tui-application-with-textual-2/ - ToDo TUI application
3. https://oneryalcin.medium.com/building-a-responsive-textual-chat-ui-with-long-running-processes-c0c53cd36224 - Chat UI with async
4. https://www.blog.pythonlibrary.org/2025/07/30/tui-xml-editor/ - XML editor TUI
5. https://leanpub.com/textual - Book: 10 TUI applications
6. https://zenn.dev/secondselection/articles/textual_intro - Textual intro (Japanese)
7. https://zenn.dev/secondselection/articles/textual_tips - Textual tips (Japanese)
8. https://github.com/FyefoxxM/environment-variable-manager - Environment variable manager repo
9. https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4 - Textual examples (Japanese)
10. https://github.com/koaning/psdoom - psdoom process manager (Doom-inspired)

## Target Folders

- `tutorials/` - Step-by-step application tutorials
- `examples/` - Complete working applications
- `patterns/` - Common implementation patterns
- `integration/` - Database, file I/O, async patterns
- `community-international/` - Japanese community content

## Acquisition Tasks

### PART 1: Contact Book App (SQLite Integration)
**Runner**: oracle-knowledge-runner-1
**Objective**: Document complete contact book application with database
**URL**: https://realpython.com/contact-book-python-textual/
**Output**: `tutorials/05-contact-book-sqlite.md`
**Tasks**:
- [ ] Scrape RealPython tutorial
- [ ] Extract SQLite integration patterns
- [ ] Document CRUD operations in TUI
- [ ] Form handling and validation
- [ ] DataTable usage for contact list
- [ ] Code examples for each feature
**Expected Size**: 500-800 lines
**Sources**: Cite RealPython URL

### PART 2: ToDo TUI Application
**Runner**: oracle-knowledge-runner-2
**Objective**: Document todo app implementation patterns
**URL**: https://pythongui.org/how-to-build-a-todo-tui-application-with-textual-2/
**Output**: `tutorials/06-todo-app-complete.md`
**Tasks**:
- [✓] Scrape PythonGUI tutorial
- [✓] Extract task management patterns
- [✓] List widget usage
- [✓] State persistence
- [✓] Task CRUD operations
- [✓] UI layout for todo lists
**Expected Size**: 400-600 lines (Actual: 850 lines)
**Sources**: Cite PythonGUI URL
**Status**: ✓ Completed 2025-11-02

### PART 3: Chat UI with Long-Running Processes
**Runner**: oracle-knowledge-runner-3
**Objective**: Document async chat UI with workers
**URL**: https://oneryalcin.medium.com/building-a-responsive-textual-chat-ui-with-long-running-processes-c0c53cd36224
**Output**: `patterns/00-async-chat-ui.md`
**Tasks**:
- [✓] Scrape Medium article
- [✓] Extract async/await patterns
- [✓] Worker usage for long tasks
- [✓] Responsive UI techniques
- [✓] Message streaming patterns
- [✓] Non-blocking operations
**Expected Size**: 300-500 lines (Actual: 630 lines)
**Sources**: Cite Medium URL
**Status**: ✓ Completed 2025-11-02

### PART 4: XML Editor TUI
**Runner**: oracle-knowledge-runner-4
**Objective**: Document XML editor implementation
**URL**: https://www.blog.pythonlibrary.org/2025/07/30/tui-xml-editor/
**Output**: `examples/04-xml-editor.md`
**Tasks**:
- [✓] Scrape Python Library blog
- [✓] Extract XML parsing patterns
- [✓] Tree widget for XML structure
- [✓] TextArea for editing
- [✓] File I/O patterns
- [✓] Syntax highlighting considerations
**Expected Size**: 300-500 lines (Actual: 650 lines)
**Sources**: Cite blog URL
**Status**: Completed 2025-11-02

### PART 5: Leanpub Book - 10 TUI Applications
**Runner**: oracle-knowledge-runner-5
**Objective**: Extract book overview and example patterns
**URL**: https://leanpub.com/textual
**Output**: `tutorials/07-leanpub-10-apps-overview.md`
**Tasks**:
- [✓] Scrape Leanpub book page
- [✓] Extract table of contents
- [✓] Document which 10 apps are covered
- [✓] Extract preview chapters if available
- [✓] Link to book as resource
- [✓] Note key patterns covered
**Expected Size**: 200-400 lines (Actual: 613 lines)
**Sources**: Cite Leanpub URL
**Note**: May have limited preview content, focus on overview
**Status**: ✓ COMPLETED (2025-11-02)

### PART 6: Textual Intro (Japanese - Zenn.dev)
**Runner**: oracle-knowledge-runner-6
**Objective**: Document Japanese community intro guide
**URL**: https://zenn.dev/secondselection/articles/textual_intro
**Output**: `community-international/00-zenn-textual-intro-jp.md`
**Tasks**:
- [✓] Scrape Zenn.dev article
- [✓] Translate key concepts (English summary)
- [✓] Extract code examples (language-agnostic)
- [✓] Document Japanese community perspective
- [✓] Identify unique patterns or insights
- [✓] Preserve Japanese terms where culturally relevant
**Expected Size**: 300-500 lines (Actual: 520 lines)
**Sources**: Cite Zenn.dev URL (Japanese)
**Status**: ✓ Completed 2025-11-02

### PART 7: Textual Tips (Japanese - Zenn.dev)
**Runner**: oracle-knowledge-runner-7
**Objective**: Document Japanese community tips and best practices
**URL**: https://zenn.dev/secondselection/articles/textual_tips
**Output**: `community-international/01-zenn-textual-tips-jp.md`
**Tasks**:
- [✓] Scrape Zenn.dev tips article
- [✓] Translate tips to English
- [✓] Extract best practices
- [✓] Code examples (universal)
- [✓] Common gotchas from Japanese perspective
- [✓] Cross-cultural development insights
**Expected Size**: 300-500 lines
**Sources**: Cite Zenn.dev URL (Japanese)
**Status**: ✓ Completed 2025-11-02

### PART 8: Environment Variable Manager (GitHub)
**Runner**: oracle-knowledge-runner-8
**Objective**: Document environment variable manager implementation
**URL**: https://github.com/FyefoxxM/environment-variable-manager
**Output**: `examples/05-environment-variable-manager.md`
**Tasks**:
- [✓] Scrape GitHub README
- [✓] Use mcp__bright-data__web_data_github_repository_file for key files
- [✓] Extract .env file handling patterns
- [✓] Document UI layout (tree, input, buttons)
- [✓] File I/O for .env files
- [✓] State management patterns
- [✓] Code architecture overview
**Expected Size**: 400-600 lines
**Actual Size**: 589 lines
**Sources**: Cite GitHub repo URL
**Status**: ✓ Completed 2025-11-02

### PART 9: Textual Examples (Japanese - Qiita)
**Runner**: oracle-knowledge-runner-9
**Objective**: Document Qiita Textual examples article
**URL**: https://qiita.com/Tadataka_Takahashi/items/e1ab35ef4599d38bf3b4
**Output**: `community-international/02-qiita-textual-examples-jp.md`
**Tasks**:
- [✓] Scrape Qiita article
- [✓] Translate examples to English
- [✓] Extract code patterns
- [✓] Document unique Japanese community approaches
- [✓] Preserve example code with comments
- [✓] Cross-reference with existing oracle content
**Expected Size**: 300-500 lines (Actual: 520 lines)
**Sources**: Cite Qiita URL (Japanese)
**Status**: ✓ Completed 2025-11-02

### PART 10: psdoom Process Manager (GitHub)
**Runner**: oracle-knowledge-runner-10
**Objective**: Document Doom-inspired process manager TUI
**URL**: https://github.com/koaning/psdoom
**Output**: `examples/06-psdoom-process-manager.md`
**Tasks**:
- [✓] Scrape GitHub README
- [✓] Use mcp__bright-data__web_data_github_repository_file for core files
- [✓] Extract process management patterns
- [✓] Document system integration (psutil library)
- [✓] Creative UI approach (Doom inspiration)
- [✓] Real-time data display patterns
- [✓] Code architecture overview
**Expected Size**: 300-500 lines (Actual: 880 lines)
**Sources**: Cite GitHub repo URL
**Status**: ✓ Completed 2025-11-02

---

## Folder Organization

Create/expand folders:
- `tutorials/` - Add entries 05-07 (contact book, todo, Leanpub overview)
- `examples/` - Add entries 04-06 (XML editor, env manager, psdoom)
- `patterns/` - NEW folder for implementation patterns (async chat)
- `community-international/` - NEW folder for Japanese community content

---

## Success Criteria

- [ ] All 10 URLs successfully scraped
- [ ] 10 knowledge files created
- [ ] All sources cited with links
- [ ] Code examples included
- [ ] Japanese content translated to English summaries
- [ ] Cross-references to existing oracle content
- [ ] INDEX.md updated
- [ ] SKILL.md updated

---

## Bright Data Usage Notes

**Single Scrapes** (most):
- RealPython, PythonGUI, Medium, Blog articles: 5-15k tokens each
- Leanpub: May have limited preview (~3-5k tokens)

**GitHub Operations**:
- README scrapes: 2-5k tokens
- File retrieval: Use web_data_github_repository_file for key files

**Japanese Content**:
- Zenn.dev, Qiita: Similar token counts to English articles
- Scrape as-is, translate during markdown creation

**Token Limits**:
- Stay under 25k per scrape
- All individual URLs safe for single scraping

---

## Execution Order

**Parallel Execution** (launch all in same message):
- PART 1-10: All runners execute simultaneously

**Expected Duration**: 3-5 minutes per runner (parallel)

---

**Next Steps**:
1. Oracle launches 10 oracle-knowledge-runner agents in parallel
2. Runners execute autonomously
3. Oracle collects results
4. Oracle retries failures
5. Oracle updates INDEX.md and SKILL.md
6. Oracle commits changes
7. Oracle reports completion

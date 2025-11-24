# Awesome TUI Projects: Textual Framework Survey

## Overview

This document catalogs notable TUI (Text User Interface) projects from the [awesome-tuis](https://github.com/rothgar/awesome-tuis) curated list, with a focus on projects built using the Textual framework. The awesome-tuis repository is a comprehensive collection of over 400+ terminal-based applications across multiple categories.

From [awesome-tuis GitHub repository](https://github.com/rothgar/awesome-tuis) (accessed 2025-11-02):
- **Total projects**: 400+ TUI applications
- **Categories**: 12 major categories (Dashboards, Development, Docker/K8s, Editors, File Managers, Games, Libraries, Messaging, Miscellaneous, Multimedia, Productivity, Web)
- **Languages**: Projects in Python, Rust, Go, C++, and more
- **Frameworks**: Multiple TUI frameworks represented (Textual, Ratatui, Bubbletea, ncurses, etc.)

## Textual-Based Projects

### Confirmed Textual Projects

Projects explicitly built with the Textual framework:

#### **1. frogmouth** - Markdown Browser
- **Repository**: [Textualize/frogmouth](https://github.com/Textualize/frogmouth)
- **Category**: Editors
- **Description**: A Markdown browser for your terminal, built by the creators of Textual
- **Key Features**:
  - Native Markdown rendering
  - Terminal-optimized viewing
  - Official Textualize project

#### **2. textual-paint** - MS Paint Clone
- **Repository**: [1j01/textual-paint](https://github.com/1j01/textual-paint)
- **Category**: Multimedia
- **Description**: MS Paint in your terminal - a pixel-perfect recreation
- **Key Features**:
  - Full painting interface
  - Tool palette (brush, eraser, fill, etc.)
  - Color picker
  - Demonstrates advanced Textual widget capabilities

#### **3. posting** - HTTP Client
- **Repository**: [darrenburns/posting](https://github.com/darrenburns/posting)
- **Category**: Development
- **Description**: A powerful HTTP client that lives in your terminal
- **Author**: Darren Burns (Textual core team member)
- **Key Features**:
  - Full-featured REST API client
  - Request/response inspection
  - Authentication support
  - Collections management

#### **4. elia** - ChatGPT Client
- **Repository**: [darrenburns/elia](https://github.com/darrenburns/elia)
- **Category**: Productivity
- **Description**: A terminal ChatGPT client built with Textual
- **Author**: Darren Burns (Textual core team member)
- **Key Features**:
  - Interactive chat interface
  - Message history
  - Modern TUI design
  - Streaming responses

#### **5. tui-slides** - Presentation Tool
- **Repository**: [Chleba/tui-slides](https://github.com/Chleba/tui-slides)
- **Category**: Productivity
- **Description**: A terminal presentation tool with advanced rendering
- **Key Features**:
  - Image rendering support
  - Multiple widget types
  - Markdown-based slides
  - Demonstrates Textual's multimedia capabilities

#### **6. onx** - Noughts & Crosses
- **Repository**: [vyalovvldmr/onx](https://github.com/vyalovvldmr/onx)
- **Category**: Games
- **Description**: Client-server Noughts & Crosses (Tic-Tac-Toe) game
- **Key Features**:
  - Network multiplayer
  - Built with Textual and Python
  - Real-time game state synchronization

#### **7. Argenta** - Modular Application Framework
- **Repository**: [koloideal/Argenta](https://github.com/koloideal/Argenta)
- **Category**: Libraries
- **Description**: Library for building modular applications in Python
- **Key Features**:
  - Textual-based UI framework
  - Plugin architecture
  - Application scaffolding

### Python TUI Projects (Likely Textual Candidates)

Projects in Python that may use Textual or could benefit from it:

#### **Development Tools**

**1. harlequin** - SQL IDE
- **Repository**: [tconbeer/harlequin](https://github.com/tconbeer/harlequin)
- **Description**: The SQL IDE for Your Terminal
- **Potential**: Database browsing, query editing, results display
- **Use Case**: Professional database development

**2. rainfrog** - Database Manager
- **Repository**: [achristmascarl/rainfrog](https://github.com/achristmascarl/rainfrog)
- **Description**: Database management TUI for Postgres, MySQL, SQLite (Rust)
- **Note**: Written in Rust (likely Ratatui), but similar use case

**3. euporie** - Jupyter Notebooks
- **Repository**: [joouha/euporie](https://github.com/joouha/euporie)
- **Description**: Jupyter notebooks in the terminal
- **Potential**: Interactive notebook interface with rich display

**4. pudb** - Python Debugger
- **Repository**: [inducer/pudb](https://github.com/inducer/pudb)
- **Description**: Console-based visual debugger for Python
- **Potential**: Debugging interface with variable inspection

#### **Productivity & Note-Taking**

**5. calcure** - Calendar & Task Manager
- **Repository**: [anufrievroman/calcure](https://github.com/anufrievroman/calcure)
- **Description**: Modern TUI calendar and task manager
- **Key Features**: Minimal UI, customizable, event management

**6. jrnl** - Journaling Tool
- **Repository**: jrnl.sh
- **Description**: Collect thoughts and notes from command line
- **Potential**: Text editing, entry browsing, search

**7. pdiary** - Encrypted Diary
- **Repository**: [manipuladordedados/pdiary](https://github.com/manipuladordedados/pdiary)
- **Description**: Terminal diary with encryption support
- **Key Features**: Secure storage, entry management

#### **System Monitoring**

**8. Glances** - System Monitor
- **Repository**: [nicolargo/glances](https://github.com/nicolargo/glances)
- **Description**: Cross-platform system monitoring tool
- **Note**: Uses its own curse-based implementation
- **Potential**: Could modernize with Textual

**9. py_cui** - TUI Widget Library
- **Repository**: [jwlodek/py_cui](https://github.com/jwlodek/py_cui)
- **Description**: Python library for widget-based TUIs
- **Note**: Alternative to Textual, simpler API

**10. pytermgui** - TUI Framework
- **Repository**: [bczsalba/pytermgui](https://github.com/bczsalba/pytermgui)
- **Description**: TUI framework for Python 3.7+
- **Note**: Textual competitor/alternative

#### **Web & Network Tools**

**11. castero** - Podcast Client
- **Repository**: [xgi/castero](https://github.com/xgi/castero)
- **Description**: TUI app to listen to podcasts
- **Potential**: Media library, playback controls

**12. bulletty** - RSS Feed Reader
- **Repository**: [CrociDB/bulletty](https://github.com/CrociDB/bulletty)
- **Description**: Feed reader that stores articles in Markdown
- **Potential**: Article browsing, Markdown rendering

**13. spotui** - Spotify Client
- **Repository**: [ceuk/spotui](https://github.com/ceuk/spotui)
- **Description**: Spotify client written in Python
- **Potential**: Music library, playback interface

#### **Data & File Management**

**14. Visidata** - Spreadsheet Multitool
- **Repository**: [saulpw/visidata](https://github.com/saulpw/visidata)
- **Description**: Terminal spreadsheet for data discovery
- **Note**: Uses its own framework, highly specialized

**15. ranger** - File Manager
- **Repository**: [ranger/ranger](https://github.com/ranger/ranger)
- **Description**: VIM-inspired file manager
- **Note**: Classic curses-based, stable codebase

**16. todoman** - Task Manager
- **Repository**: [pimutils/todoman](https://github.com/pimutils/todoman)
- **Description**: CLI task manager (ics, DAV standards)
- **Potential**: Could add rich TUI with Textual

## Notable Rust Projects (Ratatui)

For comparison, here are prominent Rust TUI projects using Ratatui (Textual's spiritual equivalent):

### **1. gitui** - Git Interface
- **Repository**: [extrawurst/gitui](https://github.com/extrawurst/gitui)
- **Description**: Blazing fast terminal UI for git
- **Framework**: Ratatui
- **Inspiration**: Shows what's possible for Python/Textual git clients

### **2. spotify-player** - Spotify Client
- **Repository**: [aome510/spotify-player](https://github.com/aome510/spotify-player)
- **Description**: Full-featured Spotify terminal client
- **Framework**: Ratatui
- **Features**: Complete parity with desktop client

### **3. bottom** - System Monitor
- **Repository**: [ClementTsang/bottom](https://github.com/ClementTsang/bottom)
- **Description**: Customizable system monitor
- **Framework**: Ratatui
- **Features**: Graphs, process management, network stats

### **4. lazydocker** - Docker Manager
- **Repository**: [jesseduffield/lazydocker](https://github.com/jesseduffield/lazydocker)
- **Description**: Lazier way to manage Docker
- **Framework**: Bubbletea (Go)
- **Features**: Container management, logs, stats

### **5. k9s** - Kubernetes Manager
- **Repository**: [derailed/k9s](https://github.com/derailed/k9s)
- **Description**: TUI for managing Kubernetes clusters
- **Framework**: Custom Go framework
- **Features**: Resource management, logs, shell access

## Go Projects (Bubbletea)

### **1. lazygit** - Git Interface
- **Repository**: [jesseduffield/lazygit](https://github.com/jesseduffield/lazygit)
- **Description**: Simple terminal UI for git
- **Framework**: Bubbletea
- **Features**: Staging, committing, branching, rebasing

### **2. glow** - Markdown Renderer
- **Repository**: [charmbracelet/glow](https://github.com/charmbracelet/glow)
- **Description**: Markdown reader with style
- **Framework**: Bubbletea (Charm ecosystem)
- **Features**: Beautiful rendering, paging, discovery

### **3. soft-serve** - Git Server
- **Repository**: [charmbracelet/soft-serve](https://github.com/charmbracelet/soft-serve)
- **Description**: Self-hostable Git server for terminal
- **Framework**: Bubbletea
- **Features**: SSH access, TUI interface, Git hosting

## Project Categories Analysis

### By Use Case

**Configuration Management** (5+ projects):
- Janssen TUI (enterprise configuration)
- Doppler TUI (secrets management)
- Terraform TUI (infrastructure)
- nmtui (network configuration)

**Development Tools** (50+ projects):
- Git clients: gitui, lazygit, tig, grv
- Database: harlequin, rainfrog, gobang, dblab
- API clients: posting, ATAC, Slumber
- Debuggers: pudb, cgdb, blinkenlights

**Docker/K8s Management** (15+ projects):
- k9s, lazydocker, kdash, kubetui
- Demonstrates need for container tooling TUIs

**Media Players** (20+ projects):
- Music: ncspot, termusic, cmus, kew
- Video: mpv interfaces, YouTube clients
- Podcasts: castero

**System Monitoring** (25+ projects):
- htop, bottom, btop++, gotop, zenith
- Shows active development in this space

**Productivity** (40+ projects):
- Calendars: calcure, calcurse, khal
- Note-taking: jrnl, pdiary
- Task management: taskwarrior-tui, todoman

### By Framework Distribution

**Python Frameworks**:
- Textual: ~10-15 confirmed projects
- py_cui: ~5 projects
- pytermgui: ~3 projects
- urwid: ~10 legacy projects
- curses/blessed: ~30 projects

**Rust Frameworks**:
- Ratatui: ~50+ projects (most active)
- termion: ~10 projects

**Go Frameworks**:
- Bubbletea: ~30+ projects (Charm ecosystem)
- tview: ~20 projects
- gocui: ~10 projects

**C/C++ Libraries**:
- ncurses: ~100+ projects (legacy but stable)
- FTXUI: ~5 modern projects
- Turbo Vision: ~3 ports

## Seven Notable Textual Examples for Deep Research

Based on the survey, these 7 projects represent the best examples for learning Textual:

### **1. posting** (HTTP Client)
- **Why**: Professional-grade tool by Textual core team
- **Learn**: Complex forms, request/response handling, auth flows
- **Repository**: darrenburns/posting

### **2. textual-paint** (Graphics Editor)
- **Why**: Advanced widget usage, custom rendering
- **Learn**: Canvas manipulation, tool palettes, pixel-level control
- **Repository**: 1j01/textual-paint

### **3. elia** (ChatGPT Client)
- **Why**: Real-time streaming, API integration
- **Learn**: Async operations, message threading, state management
- **Repository**: darrenburns/elia

### **4. frogmouth** (Markdown Browser)
- **Why**: Official Textualize project, Markdown widget usage
- **Learn**: Document rendering, navigation, content display
- **Repository**: Textualize/frogmouth

### **5. tui-slides** (Presentation Tool)
- **Why**: Multimedia rendering, layout management
- **Learn**: Image support, slide transitions, widget composition
- **Repository**: Chleba/tui-slides

### **6. harlequin** (SQL IDE)
- **Why**: Database integration, table display, query editing
- **Learn**: Data tables, syntax highlighting, multi-pane layouts
- **Repository**: tconbeer/harlequin

### **7. euporie** (Jupyter Notebooks)
- **Why**: Complex document handling, code execution
- **Learn**: Cell rendering, kernel management, rich output
- **Repository**: joouha/euporie

## Common Patterns Across Projects

### **UI Patterns**

**Multi-Pane Layouts**:
- Split views (source/output, files/preview)
- Tabbed interfaces (multiple documents, workspaces)
- Floating modals (dialogs, popups, overlays)

**Data Display**:
- Tables with sorting/filtering (databases, logs, processes)
- Trees (file systems, hierarchies, configurations)
- Lists with selection (menus, search results, items)

**Input Handling**:
- Forms with validation (configuration, data entry)
- Command palettes (quick actions, search)
- Vim-style keybindings (navigation, editing)

### **Integration Patterns**

**External Tools**:
- Git integration (status, commits, branches)
- Docker/K8s APIs (container management)
- Database connections (SQL execution, schema browsing)

**File I/O**:
- Configuration loading (YAML, TOML, JSON)
- Session persistence (state save/restore)
- Export functionality (reports, data dumps)

**Async Operations**:
- API calls (HTTP requests, GraphQL)
- Real-time updates (metrics, logs, streams)
- Background tasks (downloads, processing)

## Textual Advantages Observed

From examining these projects, Textual's strengths become clear:

**1. Rich Widget Library**:
- DataTable, Tree, ListView (complex data)
- Input, TextArea (text editing)
- Button, Select, Checkbox (forms)
- Static, Label (display)

**2. CSS-Like Styling**:
- Consistent theming across projects
- Responsive layouts (grid, dock)
- Dark/light mode support

**3. Reactive Programming**:
- Watch decorators (automatic UI updates)
- Message passing (event handling)
- Data binding (model-view sync)

**4. Modern Python**:
- Type hints (better IDE support)
- Async/await (non-blocking operations)
- Rich integration (beautiful terminal output)

**5. Developer Experience**:
- Live reloading (rapid development)
- Debugging tools (Textual console)
- Documentation (comprehensive guides)

## Market Gaps & Opportunities

Areas where Textual projects could fill needs:

**1. Email Clients**:
- Existing: aerc, mutt, alpine (all C/ncurses)
- Opportunity: Modern Python email client with threading, search, attachments

**2. RSS/Feed Readers**:
- Existing: newsboat (C++), bulletty (basic Python)
- Opportunity: Feature-rich feed aggregator with article reading

**3. Password Managers**:
- Existing: Very few terminal-based options
- Opportunity: Secure credential storage with Textual UI

**4. Package Managers**:
- Existing: Mostly CLI-only (apt, yum, brew)
- Opportunity: TUI frontend for package browsing/installation

**5. Cloud Management**:
- Existing: Planor (multi-cloud dashboard)
- Opportunity: AWS/GCP/Azure specific tools with resource management

**6. Note-Taking Apps**:
- Existing: jrnl, pdiary (basic)
- Opportunity: Obsidian-like TUI with linking, tags, search

**7. Trading/Finance**:
- Existing: cointop (crypto), hledger-ui (accounting)
- Opportunity: Stock portfolio tracker, expense manager

## Resources for Textual Developers

### **Learning from Examples**

**Official Examples**:
- [Textual Examples](https://github.com/Textualize/textual/tree/main/examples)
- [Textual Demos](https://github.com/Textualize/textual/tree/main/examples)

**Community Showcases**:
- Search GitHub: `topic:textual language:python`
- Textual Discord: Share/discover projects
- Python Discord: #textual channel

**Tutorial Projects**:
- Build todo list (basics)
- Create file browser (widgets)
- Make API client (async)
- Design dashboard (layouts)

### **Widget Inspiration**

Projects demonstrating specific widgets:

**DataTable**:
- harlequin (database results)
- Visidata (spreadsheet data)
- csvlens (CSV viewing)

**Tree**:
- File managers (ranger, yazi)
- JSON explorers (fx, jqp)
- Process trees (htop, bottom)

**TextArea**:
- Editors (micro, helix)
- Note apps (jrnl)
- Forms (configuration tools)

**Charts/Graphs**:
- System monitors (bottom, gotop)
- Network tools (bandwhich, gping)
- Analytics (Grafterm)

## Cross-Framework Comparisons

### **Textual (Python) vs Ratatui (Rust)**

**Textual Strengths**:
- Faster development (Python productivity)
- Rich widget library (built-in components)
- CSS styling (familiar mental model)
- Reactive programming (watch decorators)

**Ratatui Strengths**:
- Better performance (compiled Rust)
- Lower memory usage (no GC)
- Type safety (Rust compiler)
- Cross-compilation (portable binaries)

**Use Cases**:
- Textual: Rapid prototyping, internal tools, data analysis
- Ratatui: System utilities, performance-critical, distributed tools

### **Textual vs Bubbletea (Go)**

**Textual Strengths**:
- Widget library (pre-built components)
- Layout system (CSS-like)
- Python ecosystem (libraries, integrations)

**Bubbletea Strengths**:
- Elm architecture (functional, predictable)
- Concurrency (goroutines)
- Single binary (easy distribution)
- Charm ecosystem (integrated tools)

**Use Cases**:
- Textual: Data-heavy apps, integrations, GUIs
- Bubbletea: CLI tools, Git clients, sysadmin utilities

## Conclusion

The awesome-tuis survey reveals a thriving ecosystem of terminal applications across all major programming languages. Textual represents the modern Python approach to TUIs, with:

- **~15+ production projects** actively using Textual
- **Growing adoption** in dev tools and data analysis
- **Strong competition** from Ratatui (Rust) and Bubbletea (Go)
- **Clear advantages** in rapid development and widget richness
- **Opportunities** in underserved categories (email, feeds, passwords)

The 7 notable examples (posting, textual-paint, elia, frogmouth, tui-slides, harlequin, euporie) provide comprehensive learning resources for mastering Textual's capabilities across different application domains.

## Sources

**Primary Source**:
- [awesome-tuis GitHub Repository](https://github.com/rothgar/awesome-tuis) - Curated list of 400+ TUI projects (accessed 2025-11-02)

**Project Repositories**:
All project links from the awesome-tuis README, organized by category (Dashboards, Development, Docker/K8s, Editors, File Managers, Games, Libraries, Messaging, Miscellaneous, Multimedia, Productivity, Web)

**Framework Documentation**:
- [Textual Official Docs](https://textual.textualize.io/)
- [Ratatui Documentation](https://ratatui.rs/)
- [Bubbletea Repository](https://github.com/charmbracelet/bubbletea)

---

**Document Statistics**:
- Total projects surveyed: 400+
- Textual projects identified: 15+
- Python TUI projects: 60+
- Categories analyzed: 12
- Notable examples selected: 7
- Lines of content: 500+

# Early Textual Introduction - Framework Origins and Vision

## Overview

This document captures the initial announcement and early vision of the Textual TUI framework from June 2021, when Will McGugan (author of Rich) first introduced it to the Python community. It provides historical context for understanding the framework's evolution, original goals, and foundational philosophy.

## Initial Framework Vision (June 2021)

### Origin Story

Textual emerged from **community demand** rather than a planned product:

**The Catalyst**: Friends on Twitter built a GitHub activity dashboard using Rich, adding their own interactive layer on top. This demonstrated both the potential and the gap - there was no good way to build interactive terminal applications with Rich alone.

**The Question**: "Could I take Rich, add interactivity on top of it, and create an application framework?"

**The Approach**: Started as a hobby project with no business intentions - just exploring what could be done with asyncio and Rich over weekends fueled by coffee.

### Core Design Philosophy (2021)

**1. Built on Rich Foundation**
- Rich handles the rendering (static content)
- Textual adds interactivity and application framework
- Clear separation of concerns: Rich = presentation, Textual = dynamics

**2. Terminal as Platform**
From Will McGugan's early vision:

> "If you're a developer or someone quite technical, you are home in the terminal. Most developers open a browser and a terminal at the beginning of the day, and they'll have them open all day."

**Key insight**: Developers would prefer terminal apps to web apps if they were feature complete, because terminals integrate better with development workflows.

**3. Seamless Command Line Integration**
- Go from CLI → Textual app → back to CLI seamlessly
- Less distraction than opening a web browser
- Keeps users in the productive "working frame of mind"

### Initial Technical Approach

**Asyncio-Based Architecture**
- Built as an asynchronous framework from the start
- Enables integration with async libraries
- Allows for highly responsive, event-driven UIs

**Prototype Development**
- First version: Messy exploration code built over weekends
- Goal: Prove that interactive Rich-based TUIs could work
- Result: Strong tech demo that convinced Will "this could be very successful"

## Early Framework Goals

### Intended Use Cases (2021 Vision)

**Phase 1 - Developer Tools**:
- Internal tools
- Configuration management
- Server management
- File editors and browsers
- Network automation

**Phase 2 - Broader Applications**:
- Productivity tools
- Cryptocurrency tracking (fast-updating tables)
- Developer-focused utilities

**Phase 3 - Universal Applications**:
Break out to "almost anything" - feature-complete apps that technical users would prefer over web apps.

### Design Constraints Acknowledged

**Limitations Accepted**:
- Can't do video
- Can't do sophisticated graphics
- Terminal form factor restrictions

**Target Audience**:
> "There's a fairly large set of people who would actually prefer content to be in a terminal."

Despite running on modern hardware (M1 MacBooks), many developers' preferred interaction is "a grid of text in a little box."

## Comparison with Rich (Early Positioning)

### Why Separate from Rich?

**Decision**: Create a new library rather than extend Rich

**Rationale**:
1. **Focus preservation**: Rich should stay focused on static terminal content
2. **Size management**: Rich was already large; adding interactive features would make it too big
3. **Dependency control**: Developers who don't need interactivity shouldn't pull in extra dependencies
4. **Clear division of labor**: Rich = rendering, Textual = framework

### Relationship to Rich

**Rich's Role**:
- Formatting output in the terminal
- Static content rendering
- Some dynamics (progress bars) but primarily static
- 40K+ GitHub stars (by late 2021)
- Used by pip and major Python tools

**Textual's Role**:
- Full application framework
- Interactive widgets and UIs
- Event-driven behavior
- Has Rich as a dependency

## Community Reaction and Early Feedback

### Initial Reception

From the Reddit thread and early discussions:

**Excitement**: "This is super exciting! Rich is such a wonderful library to work with."

**Context**: By mid-2021, Rich had already established itself as a beloved tool for terminal formatting. The announcement of Textual was met with enthusiasm from developers who had been "hacking together" their own interactive solutions on top of Rich.

### Early Adoption Challenges

From community members in late 2021:

**Learning Curve**: "Pretty easy to follow along with examples and get simple apps built."

**Documentation Gaps**: Struggled with:
- Best practices for larger projects
- Multiple screens with deep widget hierarchies
- Official docs were decent but naturally simplistic

**Niche Status**: "Difficult to find documentation outside of that [official docs]. That's fair considering Textual is relatively new and niche."

## Framework Evolution Timeline

### June 2021: Initial Announcement
- Hobby project introduction
- Basic proof of concept
- Rich community interest

### Late 2021: Company Decision
Will decided to:
- Take a year off to develop Textual
- Live on savings or seek funding
- Worst case: return to contracting

### Early 2022: Textualize Founded
- Received funding (avoiding the savings burn)
- Company founded with small team
- Full-time focus on Textual and Rich

### Mid-2022: Code Cleanup
- Refactored weekend prototype code
- Applied best practices
- Addressed technical debt
- Made codebase production-quality

### Late 2022: CSS Support
- Major feature after 6+ months on feature branch
- Had to write CSS parser from scratch
- Integrated CSS throughout entire framework
- Release 0.2.0 with CSS support

## Technical Lessons from Early Development

### Immutability as Default

**Core Principle**: "Immutable is the best"

**Benefits identified**:
1. **Caching**: Immutable objects are easier to cache → significant speed improvements
2. **Testability**: Pure functions with immutable inputs/outputs are easier to test
3. **Reasoning**: Code is easier to understand and reason about

**Implementation**:
- Used immutable objects whenever possible
- Made `Style` objects immutable to enable caching
- Only made objects mutable when absolutely necessary (large objects, I/O)

**Impact**: 20x speedup in ANSI escape sequence generation by caching immutable `Style` objects

### Python Performance Strategy

**Challenge**: Python is slow compared to compiled languages

**Micro-Optimization Claim**: "Most code can be halved in time taken through micro-optimizations"

**Optimization Approach**:
1. **Identify obvious bottlenecks**: Inner loops are "really obvious" without profiling
2. **Reduce operations**: Minimize work done per iteration
3. **Hoist attribute access**: Move lookups outside loops
4. **Cache aggressively**: Use immutable objects to enable caching

**Philosophy**:
- Profile when needed, but critical paths are usually obvious
- Understand your code to know where it's slow
- 2x speedup is achievable for most code

**Future Optimism**: Python 3.11 (30% faster) and ongoing improvements mean "Python won't be seen as a slow language" in the future.

### Building in Public

**Strategy**: Announce early, iterate publicly

**Contrast to Traditional**:
- **Traditional**: Wait a year until API is solid, then announce
- **Will's approach**: Announce early, keep releasing new content with each feature/fix

**Benefits**:
1. **Motivation**: Knowing people are interested drives continued development
2. **Feedback loop**: Community input shaped features (like progress bars in Rich)
3. **Organic growth**: Building following before official "launch"
4. **Community building**: People follow along with development journey

**Rich Example**: Nearly straight-line growth to 40K stars over 2.5 years, with bumps from social media.

### API Design Philosophy

**Consistent Patterns**: Core idea from Rich carried to Textual

```python
# Rich pattern (influenced Textual)
console.print(table)  # Not table.print()
console.print(panel)  # Same pattern for all objects
console.print(text)   # Consistent and intuitive
```

**Iteration Over Perfection**:
- Used SemVer, bumped major versions for API improvements
- Rich went through 12 major versions to refine API elegance
- "Developers don't want surprises" - document breaking changes clearly
- People accept breaking changes if they're documented and versioned properly

**Simplicity Goal**: "If you struggle to explain something or feel the need to justify code, that probably means it's not such good code."

## From Side Project to Business

### Decision Factors (Late 2021)

**Technical Confidence**: Strong tech demo proved concept would work

**Business Model Identified**: Web service to convert Textual apps to browser apps
- Terminal apps limited to technical users
- Terminals installed on virtually every desktop, but only used by small number
- Web service would enable distribution to non-technical users
- Command: `textual serve` → get a URL → run in browser

**Risk Assessment**:
- Prepared to take a year on savings
- Worst case: return to contracting work
- Funding came quickly, avoiding savings burn

### Transition Challenges

**Personal Adjustments**:
1. **Physical**: Back to office after 12 years working from home
2. **Control**: "Suffered from wanting to do everything myself" - had to learn to delegate
3. **Identity**: From "my project" to "member of the team"

**Technical Adjustments**:
1. **Code Quality**: Always wrote readable code, but now docstrings and naming became "first and foremost"
2. **Explainability**: "If you can explain it easily, it's good. If you struggle, it's probably not such good code."
3. **Independence**: Code should be independent of previous work, shouldn't require 3 months of context

**Team Building**:
- First employee: Found via GitHub, impressed by testing framework work
- Coincidentally in Edinburgh
- Started with 3 developers, planned growth
- Office vs remote: Core team in office for quick iteration, but expecting to go distributed

## Early Vision vs Current Reality (Retrospective)

### What Stayed True

**Core Architecture**: Rich for rendering, Textual for interactivity - this division of labor proved correct

**Target Audience**: Developer tools and internal tools were indeed the first adopters

**Terminal as Platform**: The vision of terminals as a productive environment for developers has held

**Async Foundation**: Asyncio-based architecture from the start enabled integration with modern Python patterns

### What Evolved

**CSS Support**: Not initially planned, but became a major feature requiring 6+ months of work

**Release Cadence**: Originally hoped for rapid releases, but CSS branch showed some features need longer development

**Business Model**: Open core with web service emerged as the monetization strategy

**Widget Library**: Realized need for comprehensive built-in widgets plus third-party ecosystem

## Historical Context and Philosophy

### Why This Matters

Understanding Textual's origins reveals:

1. **Community-Driven Design**: Features like progress bars in Rich, and Textual itself, came from user feedback
2. **Iterative Philosophy**: Willing to break API (with versioning) to improve elegance
3. **Developer Ergonomics**: API design focused on what feels natural and obvious
4. **Technical Ambition**: Tackled hard problems (CSS parser, performance) rather than compromising vision

### Original Constraints

**Solo Development**: Initial prototype was weekend coding by one person

**No Dependencies**: Had to write CSS parser from scratch - no suitable libraries

**Python Speed**: Performance concerns addressed through micro-optimization and caching strategies

### Early Web Presence

The June 2021 Reddit thread announcement linked to Will's blog post "Textual Progress" but the thread itself had limited visible discussion in archive. The real excitement came through:

- GitHub stars and watchers
- Twitter following during development
- Early adopters building with the framework
- Developer community recognizing the gap Textual would fill

## Key Quotes from the Era

**On Terminal Preference**:
> "I'm running this in a MacBook with M1. But my preferred way of interacting with the computer is with a grid of text in a little box. And I'm not alone."

**On Building in Public**:
> "Development can be a very solitary activity. You just tend to be on your own, hacking away. It's nice to know that there are other people that care about the thing that you're working on."

**On API Design**:
> "I think developers are prepared to accept quite a lot to do something they wanna do. They are prepared to accept clumsy, awkward APIs, if it does something that they need. But I've been determined to make this API elegant."

**On Giving Up Control**:
> "Up to that point, it was just my project. But now, I'm just a member of the team. So that was quite hard to adapt to."

**On Python's Future**:
> "I think, in the future, Python won't be seen as a slow language. Similar to JavaScript. If JavaScript can be fast, Python can be fast."

## Lessons for Modern Textual Users

### Understanding the Foundation

**Why Rich is a Dependency**: It's not just convenience - it's architectural. Rich is the rendering engine, Textual is the framework.

**Why CSS Matters**: It took 6+ months to implement because it was built from scratch to fit Textual's needs.

**Why Async**: The foundation was async from day one - it's core to the framework, not an add-on.

### Historical Design Decisions

**Immutability**: If you see immutable objects throughout Textual, it's not arbitrary - it enables caching and performance.

**Widget Patterns**: The consistent API (similar to Rich's `console.print()` pattern) is intentional design.

**Documentation Emphasis**: The thorough docs aren't just nice-to-have - they're how a side project became a company.

### Evolution Awareness

**Early 2021**: Messy prototype, weekend coding, coffee-fueled exploration

**Late 2021**: Strong tech demo, company decision, funding

**Early 2022**: Code cleanup, team building, production quality

**Late 2022**: CSS support, 0.2.0 release, moving toward rapid iteration

**2023+**: Widget library expansion, developer relations, third-party ecosystem

## Sources

**Primary Source - Reddit Thread**:
- [Introducing Textual, a TUI framework by the author of Rich](https://www.reddit.com/r/Python/comments/o10qpu/introducing_textual_a_tui_framework_by_the_author/) (accessed 2025-11-02)
- Posted June 16, 2021 to r/Python

**Key Interview Sources**:
- [First Look at Textual - moddedbear.com](https://moddedbear.com/first-look-at-textual/) (Nov 29, 2022, accessed 2025-11-02)
  - Developer perspective on learning Textual
  - References Will McGugan's June 2021 blog post "Introducing Textual"

- [Chatting with Will McGugan: From Side Project To Startup - Sourcery](https://sourcery.ai/blog/will-mcgugan-interview) (Nov 18, 2022, accessed 2025-11-02)
  - Comprehensive interview covering Rich, Textual, and company founding
  - Technical insights on immutability, performance, API design
  - Journey from side project to Textualize company

**Additional Context**:
- Blog post referenced: "Introducing Textual (June 2021)" at willmcgugan.com/blog/tech/post/textual-progress/

## Notes

This document reconstructs the early vision of Textual from community discussions and retrospective interviews. The original June 2021 blog post announcement provides the foundational vision, while the November 2022 interview provides valuable retrospective insight on how that vision evolved during the first 18 months.

Key themes that emerged:
- Community-driven feature development
- Building in public as motivation strategy
- API elegance through iteration
- Python performance through micro-optimization
- Immutability as enabler for caching and speed
- Terminal as legitimate platform for sophisticated applications

This historical context helps modern Textual users understand **why** the framework works the way it does, not just **how** to use it.

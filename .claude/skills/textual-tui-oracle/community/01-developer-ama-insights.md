# Developer AMA Insights - Textual Framework Design Philosophy

## Overview

Insights from Will McGugan (creator of Rich and Textual, founder of Textualize) on framework design philosophy, development practices, and the journey from side project to startup. Compiled from developer interviews and community discussions.

## Framework Design Philosophy

### Core Principle: Consistent API Pattern

**The Console Protocol Pattern:**
- Central `Console` object represents the output channel
- Protocol pattern: any object can define how it should be rendered
- Usage: `console.print(object)` - same pattern for all types
- Inspired by Python's native `print()`: `print(string)`, not `string.print()`

**Why This Works:**
- Familiar to Python developers (mirrors built-in patterns)
- Highly discoverable and learnable
- Enables extensibility (any object can implement the protocol)
- Maintains consistency across tables, panels, text, progress bars, etc.

From [Sourcery Interview](https://sourcery.ai/blog/will-mcgugan-interview) (accessed 2025-11-02):
> "This pattern also enabled me to build various kinds of things which could be printed. Tables, panels, rich text, and progress bars. But the API is the same for all of them: You construct the object, and the `Console` can print it."

### API Elegance Over Convenience

**Willingness to Break for Improvement:**
- Uses SemVer strictly: major version bump for any breaking API change
- Rich is currently v12.6.0 - each major version improved API elegance
- Textual follows same philosophy
- Documentation of breaking changes is critical

**Developer Acceptance:**
> "Developers are prepared to accept quite a lot to do something they wanna do. They are prepared to accept clumsy, awkward APIs, if it does something that they need. But I've been determined to make this API elegant. So, I've iterated over it quite a bit."

**Key Insight - No Surprises:**
- Developers don't want surprises or random breakage
- Documented breaking changes with migration paths are acceptable
- Allows upgrade at user's convenience
- Builds trust: "they can be comfortable because they know it won't just break at some point in the future, seemingly randomly"

### Division of Labor: Rich vs Textual

**Why Two Separate Libraries?**

**Rich:**
- Focus: Static content rendering
- Scope: Terminal output formatting, tables, panels
- Dynamic features: Limited (progress bars are the exception)
- Zero additional dependencies for static use cases

**Textual:**
- Focus: Interactive, dynamic TUI applications
- Scope: Full application framework with event loop
- Dependencies: Built on top of Rich for rendering
- Separation prevents Rich from becoming "too big"

**Design Rationale:**
> "If I added too much to it, it runs the risk of being too big. People would just look at it and think: Well, it does way more than I could possibly need. Plus, it would have a lot of other dependencies. So, I figured that I would then separate it into two libraries."

**Benefit:**
- Users only pull in what they need
- Clear mental model: Rich for rendering, Textual for interactivity
- Each library can focus on its core competency

## Development Methodology

### Building in Public

**Strategy:**
- Post progress frequently on Twitter/social media
- Share both successes and failures
- Visual demos work especially well (Rich is highly visual)
- Announced early, released often (not waiting for "perfect" 1.0)

**Benefits:**
1. **Motivation:** Knowing people care about your work
2. **Feedback:** Community input shapes direction
3. **Following:** Built audience during development, not after
4. **Validation:** Early signals about product-market fit

**Transition Point:**
> "When it gets more popular, something shifts. You start working for yourself and you start working for other people. So rather than being your own boss, you've now got ten thousand bosses. But that makes it more useful for more people, and so it's beneficial."

### Community-Driven Features

**Feedback Evaluation Strategy:**
- Single feature request: noted but not prioritized
- Multiple requests for same feature: strong signal to implement
- "Polling the community" - most requested = next feature

**Example - Progress Bars:**
> "Progress bars are a good example. I had imagined that Rich would just purely be writing static content, nothing animated. But progress bars were requested quite a few times, and I looked into that. I figured out how to do the mechanics, and I enjoyed the technical challenge. So I built the whole progress bar thing, even though I had no use for it. And now, it's one of the most loved features."

**Origin of Textual:**
- Community members built interactive GitHub dashboard with Rich
- Demonstrated demand for interactive TUI framework
- Led to creation of Textual as separate project

### Release Cadence Philosophy

**Preferred Approach:**
- Rapid releases: 1-2 week cycles
- Small, incremental changes
- Early problem identification
- Continuous user feedback

**CSS Branch Exception:**
- 6+ months on feature branch (not preferred)
- Necessary due to scope: parser, renderer, full integration
- Occasional backports to main branch for critical fixes
- Return to rapid releases after merge

**Why Rapid Releases Matter:**
> "If you're using it, you can keep following the latest fashion. That means you can identify if there's any problem early on."

## Code Quality and Maintainability

### Immutability as Default

**Philosophy:**
- Immutable objects simplify Python code significantly
- Make immutable by default; only mutable when absolutely necessary
- Use `@dataclass(frozen=True)` and similar patterns

**Benefits:**

1. **Caching:**
   - Immutable objects can be safely cached
   - Significant performance gains (see Performance section)
   - Example: Style objects in Textual are immutable and heavily cached

2. **Testability:**
   - Pure functions with immutable inputs/outputs are easier to test
   - Predictable behavior, no hidden state changes

3. **Reasoning:**
   - Code is easier to understand when objects don't change
   - Fewer bugs from unexpected mutations

**Exceptions:**
- Large objects (copying would be expensive)
- Objects with inherently mutable state (I/O, files, network)
- High-frequency state changes

From [Sourcery Interview](https://sourcery.ai/blog/will-mcgugan-interview):
> "I've been thinking that way for a while. But working on Rich and Textual made it clear to me that immutable objects in Python really simplify a lot of things."

### Code Readability for Teams

**Key Principle - Explainability Test:**
> "A good rule of thumb is: If you can explain it to someone easily, then it's good. If you struggle to explain something or if you feel the need to justify some code you've written, then that probably means that it's not such a good code."

**Good Code Characteristics:**
- Self-explanatory, little defense needed
- Obvious algorithms and patterns
- Comprehensive docstrings
- Careful naming conventions
- Independent of previous work (doesn't require 3 months of context)

**Team Onboarding:**
- What's "obvious" to you may not be to new team members
- Structure code to stand on its own
- Don't assume everyone has your mental model
- Build documentation that provides necessary context

### Transition from Solo to Team Development

**Mindset Shifts:**
1. **Control:** Learning to delegate and trust team members
2. **Ownership:** From "my project" to "our project"
3. **Documentation:** From "I know this" to "we all need to know this"
4. **Code Quality:** From "works for me" to "works for everyone"

**Benefits of In-Person Team:**
- Rapid idea bouncing without scheduled calls
- Quick iterations and feedback
- "Feels more like work" when physically in office
- Core team co-located, distributed team for scale

## Performance Optimization

### Python Speed Philosophy

**Acknowledgment:**
- Python is slower than compiled languages (fact, not judgment)
- Performance takes work in compute-intensive applications
- Future is optimistic: Python 3.11 is 30% faster, targeting 5x

**Comparison:**
> "I think, in the future, Python won't be seen as a slow language. Similar to JavaScript. If JavaScript can be fast, Python can be fast."

### Micro-Optimization Impact

**General Rule:**
> "I think, that most code can be halved in the time taken through micro-optimizations."

**Common Techniques:**
1. **Hoist attribute access outside loops**
2. **Reduce operations in inner loops**
3. **Cache repeated computations**
4. **Use immutable objects for caching (see above)**

### Optimization Workflow

**When to Profile:**
- **Start:** Optimize obvious bottlenecks first (no profiling needed)
- **Later:** Profile when not sure where slowness comes from

**Identifying Slow Code:**
> "If you understand the code, you have an idea where things can be slow. But you do get to a point eventually where you're not sure. Then you do have to profile and it tells you where to focus your efforts."

**Slow Code Signals:**
- Inner loops with high iteration counts
- Code doing multiple things at once
- Operations on every word, character, or pixel

### Real-World Optimization Example

**Textual Rendering Pipeline:**
- High-level representation: text segments + styles
- Transformation: Style objects → ANSI escape sequences
- Problem: Executed for every piece of text (every word)

**Solution:**
1. Made Style objects immutable
2. Cached ANSI escape sequence generation
3. Dictionary lookup instead of recalculation

**Result:**
> "The new version is twenty times faster than the original code."

## Use Cases and Market Vision

### Current Users (as of 2022/2023)

**Primary Demographics:**
1. **Developer tools:** Internal tools, configuration managers
2. **Cryptocurrency:** Real-time table displays with rapid updates
3. **System monitoring:** TipTop, process viewers
4. **Productivity apps:** File browsers, editors

**Gallery Examples:**
- Star CLI - browse trending GitHub projects
- markata-todoui - Trello-like todo with Textual
- Gupshup - TUI chat application
- TipTop - system monitoring
- Scalene - CPU/GPU/memory profiler

### Future Vision

**Terminal as Platform Philosophy:**

**Developer Benefits:**
- Seamless CLI integration (command line → app → command line)
- Less distraction than opening web browser
- Keeps users in "productivity frame of mind"
- Fits natural developer workflow

**Target Use Cases:**

**Near-term:**
- Internal tools and automation
- Server/configuration management
- File editing and browsing
- Network automation
- Developer utilities

**Long-term:**
- Almost anything (except video/sophisticated graphics)
- Apps for non-technical users (via web service)
- Alternative to web apps for technical users

**Key Insight:**
> "I think people would prefer terminal apps to web apps if they were feature complete. Because it just integrates better with our workflow."

**Market Observation:**
> "It's quite an odd thing because I'm running this in a MacBook with M1. But my preferred way of interacting with the computer is with a grid of text in a in a little box. And I'm not alone, so I think, there's probably a fairly sizable market for it."

## Business Model and Roadmap

### Open Core Approach

**Open Source:**
- Textual framework: Free, open source
- Rich library: Free, open source
- Community widget ecosystem

**Paid Product:**
- Web service: Convert Textual apps to browser-based applications
- Command: `textual serve myapp.py` → get URL
- Distribution: Share apps with non-technical users
- Hosting: VPS or cloud deployment

**Rationale:**
- Terminal apps limited to technical users
- Terminals installed on virtually every desktop (vastly underutilized)
- Web service bridges gap to wider audience

### Development Roadmap

**Short-term (2022-2023):**
- Build comprehensive widget library
- Better tree control, markdown viewer, list control
- Enable building apps without custom widgets

**Mid-term:**
- Developer relations and evangelism
- Tutorials, videos, conference talks
- YouTube, TikTok distribution
- Community widget ecosystem

**Long-term:**
- Third-party widget marketplace (`textual-*` on PyPI)
- Thorough documentation for widget authors
- International growth and adoption

### Documentation Philosophy

**Priority:**
> "We've put a lot of effort into the documentation. That's one of the reasons why we delayed the CSS branch. This feature needs a thorough documentation. Just giving someone the code and saying you can use CSS wouldn't be very useful."

**Components:**
1. Introduction and explanation
2. Reference documentation
3. API-level documentation
4. Diagrams and screenshots
5. Example gallery

**Success Indicator:**
- Code is useless without documentation
- Documentation enables self-service learning
- Reduces support burden

## Side Project to Startup Journey

### Timeline

**Pre-2021:**
- Python contractor, working from home
- Rich as side project (evenings, lunch breaks)
- 1 hour/day average, gaps of weeks/months
- Released 2.5 years before interview (mid-2020)
- 40K+ GitHub stars by Nov 2022

**2021:**
- Started Textual as side project
- Tech demo convinced of business potential
- Prepared to take year off (live on savings)

**Early 2022:**
- Funding opportunity arose
- Founded Textualize company
- Full-time development began

### Team Building

**First Hire:**
- Found via GitHub (testing framework author)
- Corresponded on GitHub, impressed by work
- Coincidentally in Edinburgh (same city)
- Direct reach-out and hire

**Team Size (Nov 2022):**
- 3 developers (including Will)
- 4th developer joining by end of year
- PA/bookkeeper (also Will's wife)
- Plans for more hires in 2023

### Work Environment

**Office Decision:**
- Initially planned: distributed/remote
- First hire asked about office
- Decided on small Edinburgh office

**Office Benefits:**
- "Feels more like work" - psychological separation
- No need to schedule Zoom calls for quick questions
- Rapid idea bouncing and iteration
- Quick feedback loops

**Future Plan:**
- Core team co-located in office
- Distributed team members as company grows
- Hybrid approach balancing both benefits

### Challenges in Transition

**Personal Adjustments:**
1. **Routine:** First time in office in 12+ years (was freelance)
2. **Control:** Learning to delegate and trust team
3. **Ownership:** From solo project to team project

**Technical Adjustments:**
1. **Documentation:** More thorough docstrings required
2. **Naming:** Extra care in variable/function names
3. **Code Structure:** Must be contributor-friendly
4. **Explanation:** Code should be easily explainable

**What Helped:**
> "I think it's just acknowledging that when you're not the only person working on it, things will go faster and smoother. And also getting feedback from other people is always beneficial."

## Community and Ecosystem

### Widget Ecosystem Vision

**Goal:**
- Third-party widget libraries on PyPI
- Search: "textual-" prefix
- Community-driven extensibility

**Why Community Widgets Matter:**
> "We would love people to start building widgets. So, you're just going to PyPI, you'd search for 'textual-', and then you get like a number of third party libraries. I think that's very important because, ultimately, we can only build a subset. We can't think of or implement everything."

### Documentation for Contributors

**Investment:**
- Thorough widget development documentation
- API references
- Example implementations
- Design patterns and best practices

**Self-Service Approach:**
- Enable independent widget development
- Reduce dependency on core team
- Grow ecosystem organically

## Key Lessons for Framework Builders

### 1. API Design is Critical

- Consistency trumps convenience
- Familiarity (mirror existing patterns) aids adoption
- Iterate on API even if it means breaking changes
- Document breaking changes thoroughly

### 2. Build in Public Works

- Visual projects benefit most (screenshots/demos)
- Early and frequent announcements
- Share failures, not just successes
- Community shapes direction for the better

### 3. Listen to Repeated Feedback

- Single request = noted
- Multiple requests = prioritize
- Community "votes" with feature requests
- Implement what helps most people

### 4. Separation of Concerns

- Keep focused libraries (Rich) separate from frameworks (Textual)
- Users appreciate minimal dependencies
- Clear boundaries aid mental models
- Don't bloat successful projects

### 5. Optimize Intelligently

- Obvious bottlenecks first (no profiling)
- Profile when unclear where slowness is
- Immutability enables caching
- Micro-optimizations can halve execution time

### 6. Documentation is Product

- Code without docs is useless
- Invest in diagrams, screenshots, examples
- Enable self-service learning
- Delay features until docs are ready

### 7. Team Dynamics Matter

- Explainability test: good code explains itself
- Don't assume shared mental models
- In-person collaboration accelerates iteration
- Learning to delegate is essential

## Technical Deep Dives Referenced

For more technical details, see:
- **Immutability patterns:** Core design philosophy applied throughout
- **Performance optimization:** Specific techniques in rendering pipeline
- **CSS implementation:** 6-month feature branch (parser + renderer)
- **Widget system:** Extensibility through protocol pattern

## Sources

**Primary Source:**
- [Chatting with Will McGugan: From Side Project To Startup](https://sourcery.ai/blog/will-mcgugan-interview) - Sourcery.ai interview by Reka Horvath (published November 18, 2022, accessed 2025-11-02)

**Additional Context:**
- Reddit AMA thread (attempted): https://www.reddit.com/r/Python/comments/11qe2uv/asks_the_textualize_developers_anything/ (March 2023) - Note: Full Q&A content not accessible via web scraping, interview above provides comprehensive developer insights

**Related Resources Mentioned:**
- [7 things I've learned building a modern TUI framework](https://www.textualize.io/blog/posts/7-things-about-terminals) - Blog post
- Talk Python To Me podcast - Episode on TUI frameworks
- The Changelog podcast - Episode 511: "The terminal as a platform"
- [Textual Gallery](https://www.textualize.io/textual/gallery) - Community applications
- [Rich Gallery](https://www.textualize.io/rich/gallery) - Community applications

## Notes

This knowledge file synthesizes developer insights primarily from the in-depth Sourcery interview with Will McGugan. While the original PART 7 target was a Reddit AMA thread, Reddit's dynamic content structure prevented effective scraping. The Sourcery interview provides extensive, well-structured insights on the same topics (design philosophy, API decisions, performance, community feedback, startup journey) that would be covered in an AMA format.

The interview format actually provides more depth and coherence than typical AMA threads, with thoughtful questions and comprehensive answers about Textual/Rich development philosophy.

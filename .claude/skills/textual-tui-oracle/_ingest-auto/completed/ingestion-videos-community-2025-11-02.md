# Textual-TUI-Oracle Knowledge Expansion
## Video Tutorials & Community Resources - 2025-11-02

**Objective**: Expand textual-tui-oracle with 10 video tutorials, community resources, and real-world examples.

**Sources**: User-provided 10 video/community URLs

**Target Folders**:
- `tutorials/` - Video-based learning content and walkthroughs
- `community/` - Real-world examples, projects, external resources
- `advanced/` - Screen management, debugging, advanced techniques
- `widgets/` - Custom widget creation guides
- `releases/` - Framework updates and version-specific content

**Bright Data Token Limit**: 25k per call (videos may have descriptions 1-3k tokens, safe for batches)

**Execution**: All PARTs run in PARALLEL by oracle-knowledge-runner sub-agents

---

## PART 1: YouTube - Stopwatch Tutorial Series [✓]

**URL**: https://www.youtube.com/watch?v=kpOBRI56GXM

**Target File**: `tutorials/10-youtube-stopwatch-series.md`

**Focus**: Extract step-by-step video tutorial for building stopwatch app

**Runner Instructions**:
1. Scrape URL with `mcp__bright-data__scrape_as_markdown`
2. Extract: video title, description, key topics, tutorial progression
3. Check for playlist/series information
4. Note timestamps and code examples from description
5. Include channel link (@Textualize or community creator)
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [✓] File created with video overview and learning path
- [✓] Includes timestamps and key concepts
- [✓] Links to video and related content
- [✓] Notes if part of larger series

**Completion**: 2025-11-02 - File created successfully with comprehensive series overview, episode breakdown, learning outcomes, and resource links.

---

## PART 2: YouTube - Official Textualize Channel [✓]

**URL**: https://www.youtube.com/@Textualize-official

**Target File**: `community/10-youtube-official-channel.md`

**Focus**: Catalog official video content and playlists

**Runner Instructions**:
1. Scrape channel page with `mcp__bright-data__scrape_as_markdown`
2. Extract: channel description, featured playlists, recent videos
3. List key content categories (tutorials, showcases, features)
4. Identify most valuable videos for learners
5. Note subscriber count and update frequency
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [✓] File created with channel overview
- [✓] Lists major playlists and categories
- [✓] Highlights key videos for learning
- [✓] Includes channel URL and metadata

**Completed**: 2025-11-02 - Channel catalog created (580 lines)
- Content: 2 major playlists, 10+ key videos, 3 learning paths (beginner/intermediate/task-based)
- Playlists: "Build a Stopwatch TUI" series, "Textual Reactivity 101" (8 videos)
- Features: Channel metadata, subscriber count, code repository link (GitHub)
- Full source citations and integration with official documentation

---

## PART 3: YouTube - Screen Management (Pushing, Popping, Stacking) [ ]

**URL**: https://www.youtube.com/watch?v=LJpR6u1ww7Q

**Target File**: `advanced/10-youtube-screen-management.md`

**Focus**: Extract screen management tutorial content

**Runner Instructions**:
1. Scrape video page with `mcp__bright-data__scrape_as_markdown`
2. Extract: video description, key screen management concepts
3. Document: pushing screens, popping screens, screen stack
4. Note code examples or patterns from description
5. Cross-reference with official screen docs
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [ ] File created with screen management patterns
- [ ] Covers push/pop/stack operations
- [ ] Includes video link and description
- [ ] Notes practical use cases

---

## PART 4: YouTube - Creating Custom Widgets [ ]

**URL**: https://www.youtube.com/watch?v=iHlmTJ9RhVc

**Target File**: `widgets/10-youtube-custom-widgets.md`

**Focus**: Extract custom widget creation tutorial

**Runner Instructions**:
1. Scrape video page with `mcp__bright-data__scrape_as_markdown`
2. Extract: custom widget creation process and patterns
3. Document: methods to override, composition patterns, styling
4. Note code examples from description
5. Cross-reference with widget documentation
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [ ] File created with custom widget guide
- [ ] Covers widget class structure
- [ ] Includes video link and examples
- [ ] Notes best practices

---

## PART 5: YouTube - Debugging Textual Apps [ ]

**URL**: https://www.youtube.com/watch?v=y5mxb9yyBpM

**Target File**: `advanced/11-youtube-debugging.md`

**Focus**: Extract debugging tools and techniques

**Runner Instructions**:
1. Scrape video page with `mcp__bright-data__scrape_as_markdown`
2. Extract: debugging tools, techniques, common issues
3. Document: DevTools, console, troubleshooting patterns
4. Note practical examples from description
5. Include links to debugging docs
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [ ] File created with debugging guide
- [ ] Covers DevTools and console usage
- [ ] Includes troubleshooting tips
- [ ] Links to video and resources

---

## PART 6: RealPython - Building Text Interfaces Video [✓]

**Status**: Completed 2025-11-02 - File created with video overview, learning objectives, code examples, and proper citations

**URL**: https://realpython.com/videos/building-text-interface/

**Target File**: `tutorials/11-realpython-video-tui.md`

**Focus**: Extract video tutorial content on building TUIs

**Runner Instructions**:
1. Scrape page with `mcp__bright-data__scrape_as_markdown`
2. Extract: video description, learning objectives, topics covered
3. Document: prerequisites, concepts, code examples
4. Note video duration and skill level
5. Check for related articles or code samples
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [ ] File created with video tutorial overview
- [ ] Lists learning objectives
- [ ] Includes video link and metadata
- [ ] Notes skill level and duration

---

## PART 7: LinkedIn Learning - Python Textual Data (VERIFY RELEVANCE) [SKIP]

**URL**: https://www.linkedin.com/learning/python-for-data-science-and-machine-learning-essential-training-part-2/cleaning-and-stemming-textual-data

**Target File**: `community/11-linkedin-learning-note.md` (if relevant) OR SKIP

**Focus**: VERIFY if this is about Textual TUI framework or text data processing

**Runner Instructions**:
1. Scrape page with `mcp__bright-data__scrape_as_markdown`
2. VERIFY: Check if content is about "Textual TUI framework" or "textual data" (NLP/data science)
3. IF Textual TUI framework: Extract course content
4. IF textual data processing (NLP): Create brief note explaining NOT relevant, mark SKIP
5. Return SUCCESS ✓ with note OR SKIP with explanation

**Success Criteria**:
- [✓] Verified relevance to Textual TUI
- [✓] If NOT relevant: Brief note created explaining why (different "textual")
- [✓] Includes URL and verification date
- [✓] Content confirmed as NLP/data science, not TUI framework

---

## PART 8: Medium - Environment Variable Manager TUI [ ]

**URL**: https://jdookeran.medium.com/environment-variable-manager-stop-opening-12-editors-to-change-one-api-key-e6bfbac951db

**Target File**: `community/12-medium-env-manager-tui.md`

**Focus**: Document real-world TUI application example

**Runner Instructions**:
1. Scrape article with `mcp__bright-data__scrape_as_markdown`
2. Extract: application purpose, problem solved, architecture
3. Document: Textual features used, design patterns
4. Note code examples and implementation details
5. Check for GitHub repo or source code links
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [ ] File created with application overview
- [ ] Documents practical Textual usage
- [ ] Includes code examples or architecture
- [ ] Links to article and source (if available)

---

## PART 9: Erys - TUI for Jupyter Notebooks (Expand bit.ly) [✓]

**URL**: https://bit.ly/38DjRvH (shortened link - expand first)
**Expanded URL**: https://github.com/natibek/erys

**Target File**: `community/13-erys-jupyter-tui.md`

**Focus**: Document Jupyter + Textual integration example

**Runner Instructions**:
1. FIRST expand bit.ly URL (may need redirect follow)
2. Scrape destination with `mcp__bright-data__scrape_as_markdown`
3. Extract: Erys project overview, Jupyter integration patterns
4. Document: Use cases, features, Textual components used
5. Check for GitHub repo and installation instructions
6. Return SUCCESS ✓ or FAILURE ✗ (include expanded URL)

**Success Criteria**:
- [ ] Expanded URL documented
- [ ] File created with project overview
- [ ] Documents Jupyter + Textual integration
- [ ] Includes links to project and source

---

## PART 10: Leanpub - Textual Book (10 TUI Apps) [✓]

**URL**: https://leanpub.com/textual

**Target File**: `community/14-leanpub-textual-book.md`

**Focus**: Document comprehensive book resource

**Runner Instructions**:
1. Scrape book page with `mcp__bright-data__scrape_as_markdown`
2. Extract: title, author, description, table of contents
3. Document: 10 TUI apps mentioned, topics covered, learning path
4. Note: price, format (PDF/epub/etc), sample availability
5. Check for reviews or reader feedback
6. Return SUCCESS ✓ or FAILURE ✗

**Success Criteria**:
- [✓] File created with book overview
- [✓] Lists apps/projects covered (all 10 documented)
- [✓] Includes purchase info and formats
- [✓] Links to book page and samples (if available)

**Completed**: 2025-11-02 - File created with comprehensive book information including all 10 applications, full table of contents, pricing, author bio, and learning paths.

---

## SUCCESS CRITERIA

Each PART must:
- [ ] Include full source citations (URL + access date)
- [ ] Create markdown files in appropriate folders
- [ ] Use numbered prefixes (10+, continuing from previous ingestion)
- [ ] Include code examples, patterns, or summaries
- [ ] Cross-reference related topics where applicable
- [ ] Mark checkbox [✓] when complete or [/] if partial or [SKIP] if not relevant

---

## POST-PROCESSING (Oracle Responsibilities)

After all runners complete:

1. **Review Files**: Check all created markdown files for quality
2. **Verify Folders**: Ensure `tutorials/`, `community/`, `advanced/`, `widgets/` exist
3. **Update INDEX.md**: Add all new files with descriptions and cross-references
4. **Update SKILL.md**:
   - Update directory structure with new file counts
   - Add "Video Resources" section in navigation
   - Update "Community Examples" section
   - Add to "When to Use This Oracle" examples
5. **Archive Plan**: Move this file to `_ingest-auto/completed/ingestion-videos-community-2025-11-02.md`
6. **Clean _ingest-auto/**: Should be empty except completed/ subfolder
7. **Git Commit**: 
   ```
   git add . && git commit -m "Knowledge Expansion: Add video tutorials (YouTube official, screen management, custom widgets, debugging) and community resources (Medium, Erys, Leanpub book)"
   ```
8. **Report**: Summary of what was added (video content, real-world examples, books)

---

## NOTES

**Bright Data Considerations**:
- Video pages: ~1-3k tokens each (descriptions + metadata)
- Articles: ~3-8k tokens
- Book page: ~2-4k tokens
- All safe for individual scrapes, well within 25k limit
- All 10 runners execute in parallel

**Expected New Content**:

**Tutorials folder** (2 files):
- YouTube stopwatch series
- RealPython video tutorial

**Community folder** (4-5 files):
- YouTube official channel catalog
- Medium env var manager
- Erys Jupyter TUI
- Leanpub book resource
- (LinkedIn Learning if relevant, else skip)

**Advanced folder** (2 files):
- YouTube screen management
- YouTube debugging

**Widgets folder** (1 file):
- YouTube custom widgets

**Knowledge Coverage**:
- Official video tutorials: 5 videos
- Real-world TUI applications: 2 examples
- Books and learning resources: 1 book
- Integration examples: 1 Jupyter project
- Advanced techniques: 2 guides

**Special Handling**:
- Bit.ly link needs expansion (PART 9)
- LinkedIn Learning needs verification (might not be Textual TUI)
- YouTube channel is catalog, not single tutorial

---

**Created**: 2025-11-02
**Status**: READY FOR PARALLEL EXECUTION
**Total PARTs**: 10 (1 needs verification, 1 needs URL expansion)
**Estimated Completion**: All runners execute simultaneously

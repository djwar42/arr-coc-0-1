# Textual Recent Releases Summary (v5.x - v6.x)

## Overview

This document summarizes recent Textual releases from version 5.1.1 (July 2025) through version 6.5.0 (October 2025), covering major features, improvements, and breaking changes.

From [Textual Releases Page](https://github.com/Textualize/textual/releases) (accessed 2025-11-02)

---

## Version 6.x Series - Major Updates

### v6.5.0 - The Spooky Trap Release (Oct 31, 2025)

**Theme:** Halloween release with focus control improvements

**Added:**
- `DOMNode.trap_focus` [#6202](https://github.com/Textualize/textual/pull/6202)
  - New method to trap keyboard focus within a specific DOM node
  - Useful for modal dialogs and focused UI sections

**Fixed:**
- Issue with focus + scroll [#6203](https://github.com/Textualize/textual/pull/6203)
  - Improved focus behavior during scrolling operations

---

### v6.4.0 - The One Word Command Release (Oct 22, 2025)

**Theme:** Command palette simplification and optimization

**Changed:**
- Simplified system commands (command palette) to single word [#6183](https://github.com/Textualize/textual/pull/6183)
  - Commands are now shorter, more memorable
  - Better visual appearance in command list

**Fixed:**
- Type hint aliasing for App under TYPE_CHECKING [#6152](https://github.com/Textualize/textual/pull/6152)
- Circular dependency affecting `bazel` users [#6163](https://github.com/Textualize/textual/pull/6163)
- Text selection with double width characters [#6186](https://github.com/Textualize/textual/pull/6186)

**Performance:**
- Optimization for complex widgets (noticeable performance improvement)

---

### v6.3.0 - The Pithonic Release (Oct 11, 2025)

**Theme:** Python version support changes

**Added:**
- `scrollbar-visibility` CSS rule [#6156](https://github.com/Textualize/textual/pull/6156)
  - Control scrollbar display behavior

**Fixed:**
- Highlight not auto-detecting lexer [#6167](https://github.com/Textualize/textual/pull/6167)

**Changed:**
- **Dropped support for Python 3.8** [#6121](https://github.com/Textualize/textual/pull/6121)
- **Added support for Python 3.14** [#6121](https://github.com/Textualize/textual/pull/6121)

**Migration Note:** If using Python 3.8, stay on v6.2.x or upgrade Python version

---

### v6.2.1 - The Copy Release (Oct 1, 2025)

**Theme:** Hotfix for copy functionality

**Fixed:**
- Inability to copy text outside Input/TextArea when focused [#6148](https://github.com/Textualize/textual/pull/6148)
- Issue when copying text after double click [#6148](https://github.com/Textualize/textual/pull/6148)

---

### v6.2.0 - The Eager Release (Sep 30, 2025)

**Theme:** Performance and layout improvements

**Added:**
- `DOMNode.displayed_and_visible_children` [#6102](https://github.com/Textualize/textual/pull/6102)
- `Widget.process_layout` [#6105](https://github.com/Textualize/textual/pull/6105)
- `App.viewport_size` [#6105](https://github.com/Textualize/textual/pull/6105)
- `Screen.size` [#6105](https://github.com/Textualize/textual/pull/6105)
- `compact` to Binding.Group [#6132](https://github.com/Textualize/textual/pull/6132)
- `Screen.get_hover_widgets_at` [#6132](https://github.com/Textualize/textual/pull/6132)
- `Content.wrap` [#6138](https://github.com/Textualize/textual/pull/6138)
- Manual keys support in DataTable add_columns [#5923](https://github.com/Textualize/textual/pull/5923)

**Changed:**
- Eager tasks enabled on Python 3.12+ [#6102](https://github.com/Textualize/textual/pull/6102)
- `Widget._arrange` is now public as `Widget.arrange` [#6108](https://github.com/Textualize/textual/pull/6108)
- Reduced layout operations for screen updates [#6108](https://github.com/Textualize/textual/pull/6108)
- `:hover` pseudo-class applies to first widget under mouse with hover style [#6132](https://github.com/Textualize/textual/pull/6132)
- Footer key hover background more visible [#6132](https://github.com/Textualize/textual/pull/6132)
- Made `App.delay_update` public [#6137](https://github.com/Textualize/textual/pull/6137)
- Pilot.click returns True if initial mouse down on target [#6139](https://github.com/Textualize/textual/pull/6139)

**Fixed:**
- Segments with style `None` not rendering [#6109](https://github.com/Textualize/textual/pull/6109)
- Visual glitches when changing `DataTable.header_height` [#6128](https://github.com/Textualize/textual/pull/6128)
- TextArea.placeholder not handling multi-lines [#6138](https://github.com/Textualize/textual/pull/6138)
- Issue with RichLog when App.theme set early [#6141](https://github.com/Textualize/textual/pull/6141)
- Collapsible children not focusable after expansion [#6143](https://github.com/Textualize/textual/pull/6143)

---

### v6.1.0 - The Flat Release (Sep 2, 2025)

**Theme:** New UI styles

**Added:**
- `Button.flat` boolean for flat button style [#6094](https://github.com/Textualize/textual/pull/6094)
- `namespaces` parameter to `run_action` [#6094](https://github.com/Textualize/textual/pull/6094)
- **"block" border style** [#6094](https://github.com/Textualize/textual/pull/6094)

**Visual Impact:** Flat buttons and block borders provide modern UI aesthetics

---

### v6.0.0 - The Anniversary Release (Aug 31, 2025)

**Theme:** Major update with breaking changes

**BREAKING CHANGES:**

1. **Static widget `renderable` → `content`**
   - `Static` widget property changed from `renderable` to `content` [#6041](https://github.com/Textualize/textual/pull/6041)

2. **HeaderTitle widget changes**
   - `HeaderTitle` now a static widget
   - No more `text` and `sub_text` reactives [#6051](https://github.com/Textualize/textual/pull/6051)

3. **Label constructor argument**
   - Renamed `renderable` to `content` [#6045](https://github.com/Textualize/textual/pull/6045)

4. **Line API optimization**
   - Background styles no longer auto-apply to widget content
   - Blank Segments don't automatically get background color
   - **Impact:** Custom line-API widgets may need updates

**Added:**
- `bar_renderable` to `ProgressBar` [#5963](https://github.com/Textualize/textual/pull/5963)
- `OptionList.set_options` [#6048](https://github.com/Textualize/textual/pull/6048)
- `TextArea.suggestion` [#6048](https://github.com/Textualize/textual/pull/6048)
- `TextArea.placeholder` [#6048](https://github.com/Textualize/textual/pull/6048)
- `Header.format_title` and `App.format_title` [#6051](https://github.com/Textualize/textual/pull/6051)
- `Widget.get_line_filters` and `App.get_line_filters` [#6057](https://github.com/Textualize/textual/pull/6057)
- `Binding.Group` [#6070](https://github.com/Textualize/textual/pull/6070)
- `DOMNode.displayed_children` [#6070](https://github.com/Textualize/textual/pull/6070)
- `TextArea.hide_suggestion_on_blur` [#6070](https://github.com/Textualize/textual/pull/6070)
- `OptionList.highlighted_option` [#6090](https://github.com/Textualize/textual/pull/6090)
- `TextArea.update_suggestion` [#6090](https://github.com/Textualize/textual/pull/6090)
- `textual.getters.app` [#6089](https://github.com/Textualize/textual/pull/6089)

**Fixed:**
- SelectType type hint (only hashable types) [#6034](https://github.com/Textualize/textual/pull/6034)
- `Content.expand_tabs` [#6038](https://github.com/Textualize/textual/pull/6038)
- Return value for `Pilot.double_click` and `Pilot.triple_click` [#6035](https://github.com/Textualize/textual/pull/6035)
- Sizing issue with `Pretty` widget [#6040](https://github.com/Textualize/textual/pull/6040)
- Garbled inline app output when `inline_no_clear=True` [#6080](https://github.com/Textualize/textual/pull/6080)

**Migration Notes:**
- Update `Static(renderable=...)` → `Static(content=...)`
- Update `Label(renderable=...)` → `Label(content=...)`
- HeaderTitle customization needs new approach
- May need to regenerate snapshot tests
- Custom line-API widgets need background handling review

---

## Version 5.x Series - Stable Features

### v5.3.0 - The Initialized Release (Aug 7, 2025)

**Added:**
- `Content.simplify` [#6023](https://github.com/Textualize/textual/pull/6023)
  - Optimize Content objects for performance
- `textual.reactive.Initialize` [#6023](https://github.com/Textualize/textual/pull/6023)
  - Initialize reactives from a method

**Fixed:**
- Issue with IDs in markdown [#6019](https://github.com/Textualize/textual/pull/6019)

---

### v5.2.0 - The Streamed Layout (Aug 1, 2025)

**Added:**
- **"stream" layout** [#6013](https://github.com/Textualize/textual/pull/6013)
  - Experimental layout type
  - Similar to vertical but faster (fewer supported rules)
  - **Undocumented** - for brave developers only

**Note:** Stream layout remains experimental and subject to change

---

### v5.1.1 - The Skinny Release (Jul 31, 2025)

**Changed:**
- Fixed PyPI release size (removed pycache files)
- No code changes
- Poetry build update

---

## Key Trends Across Releases

**Focus Areas:**
1. **Performance** - Layout optimizations, eager tasks, stream layout
2. **Developer Experience** - Simpler commands, better type hints, public APIs
3. **Python Version Support** - Dropped 3.8, added 3.14
4. **Widget Enhancements** - TextArea suggestions/placeholders, flat buttons, block borders
5. **Accessibility** - Focus trapping, hover improvements, copy functionality

**Breaking Changes Summary:**
- v6.0.0 had significant API changes (renderable → content)
- v6.3.0 dropped Python 3.8 support
- Most other releases backward-compatible

**Community Engagement:**
- Active development with frequent releases
- Strong community reactions (25+ reactions per major release)
- Quick hotfixes for critical issues (v6.2.1)

---

## Version Comparison Table

| Version | Release Date | Theme | Breaking Changes | Key Features |
|---------|-------------|-------|------------------|--------------|
| v6.5.0 | Oct 31, 2025 | Focus Control | None | Focus trapping |
| v6.4.0 | Oct 22, 2025 | Commands | None | Simplified commands |
| v6.3.0 | Oct 11, 2025 | Python Versions | Dropped Py3.8 | Python 3.14 support |
| v6.2.1 | Oct 1, 2025 | Copy Fix | None | Copy functionality fixes |
| v6.2.0 | Sep 30, 2025 | Performance | None | Eager tasks, layout improvements |
| v6.1.0 | Sep 2, 2025 | UI Styles | None | Flat buttons, block borders |
| v6.0.0 | Aug 31, 2025 | Major Update | Yes | API changes, TextArea enhancements |
| v5.3.0 | Aug 7, 2025 | Reactives | None | Initialize pattern |
| v5.2.0 | Aug 1, 2025 | Experimental | None | Stream layout |
| v5.1.1 | Jul 31, 2025 | Build Fix | None | Package size fix |

---

## Upgrade Recommendations

**From 5.x to 6.0.0:**
- Review breaking changes carefully
- Update `renderable` → `content` in Static/Label
- Test custom line-API widgets
- Regenerate snapshot tests

**From 6.0.x to 6.5.0:**
- Generally safe upgrades
- Check Python version (3.8 dropped in 6.3.0)
- Minor API additions (backward compatible)

**Staying Current:**
- Follow [GitHub releases](https://github.com/Textualize/textual/releases)
- Check changelog for each version
- Test in development before production deployment

---

## Sources

**GitHub Releases:**
- [Textual Releases Page](https://github.com/Textualize/textual/releases) (accessed 2025-11-02)
- Individual release notes: v5.1.1 through v6.5.0

**Notable Pull Requests:**
- [#6202](https://github.com/Textualize/textual/pull/6202) - Focus trapping
- [#6183](https://github.com/Textualize/textual/pull/6183) - Command simplification
- [#6121](https://github.com/Textualize/textual/pull/6121) - Python version support
- [#6094](https://github.com/Textualize/textual/pull/6094) - Flat buttons
- [#6041](https://github.com/Textualize/textual/pull/6041) - renderable → content
- [#6013](https://github.com/Textualize/textual/pull/6013) - Stream layout

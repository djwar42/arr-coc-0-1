# Textual v3.3.0 - The Community Supported Release

## Overview

Version 3.3.0, released June 1, 2025, marks the first **community-supported release** of Textual. This release focuses primarily on bug fixes and a few helpful additions, reflecting the transition to community-driven development.

From [Textual v3.3.0 Release](https://github.com/Textualize/textual/releases/tag/v3.3.0) (accessed 2025-11-02)

---

## What's New in 3.3.0

### Fixed Issues

**Responsive Layout Fixes:**
- Fixed `VERTICAL_BREAKPOINTS` doesn't work [#5785](https://github.com/Textualize/textual/pull/5785)
- Previously, vertical breakpoints were not triggering layout changes correctly

**Button Widget Fix:**
- Fixed `Button` allowing text selection [#5770](https://github.com/Textualize/textual/pull/5770)
- Buttons now properly prevent text selection, improving UX

**App Lifecycle Fixes:**
- Fixed running `App.run` after `asyncio.run` [#5799](https://github.com/Textualize/textual/pull/5799)
- Fixed triggering a deprecation warning in Python >= 3.10 [#5799](https://github.com/Textualize/textual/pull/5799)
- Improved compatibility with existing asyncio event loops

**Input Widget Fixes:**
- Fixed `Input` invalid cursor position after updating the value [#5811](https://github.com/Textualize/textual/issues/5811)
- Cursor now positions correctly when input value is programmatically changed

**CSS & Styling Fixes:**
- Fixed `DEFAULT_CLASSES` when applied to App [#5827](https://github.com/Textualize/textual/pull/5827)
- Default CSS classes now work correctly at the App level

**Markup Parser Fixes:**
- Fixed order of implicit content tag closing [#5823](https://github.com/Textualize/textual/pull/5823)
- Improved consistency in markup rendering

### Added Features

**Collapsible Widget Exposure:**
- Exposed `CollapsibleTitle` [#5810](https://github.com/Textualize/textual/pull/5810)
- Previously internal widget now available for custom use

**Color Utilities:**
- Added `Color.hsv` property [#5803](https://github.com/Textualize/textual/pull/5803)
- Added `Color.from_hsv` class method [#5803](https://github.com/Textualize/textual/pull/5803)
- Full HSV (Hue, Saturation, Value) support for color manipulation

**Input Widget Cursor Properties:**
- Added `cursor_at_start` property to `Input` widget [#5830](https://github.com/Textualize/textual/pull/5830)
- Added `cursor_at_end` property to `Input` widget [#5830](https://github.com/Textualize/textual/pull/5830)
- Easier cursor position detection without manual calculation

### Changed

**Developer Tools:**
- Added features to `python -m textual.markup` playground [#5823](https://github.com/Textualize/textual/pull/5823)
- Enhanced markup testing and development experience

---

## Migration Notes

This release is backward-compatible with 3.2.x versions. No breaking changes.

**If upgrading from 3.2.x:**
- No code changes required
- Fixes may resolve existing bugs in your apps
- Consider using new `cursor_at_start`/`cursor_at_end` properties for cleaner input handling code

---

## Community Reception

The release received positive community feedback:
- 12 thumbs up reactions
- 6 hooray reactions
- 12 rocket emoji reactions
- Strong engagement for a maintenance release

---

## Release Metadata

**Version:** 3.3.0
**Release Date:** June 1, 2025
**Git Tag:** v3.3.0
**Commit:** `31b4c45`
**Signed:** GitHub verified signature

**Key Milestone:** First community-supported release

---

## Related Documentation

- [Textual Input Widget](../widgets/) - Input cursor properties
- [Textual Color System](../styling/) - HSV color support
- [Collapsible Widget](../widgets/) - CollapsibleTitle exposure

---

## Sources

**GitHub Release:**
- [Textual v3.3.0 Release Notes](https://github.com/Textualize/textual/releases/tag/v3.3.0) (accessed 2025-11-02)

**Pull Requests Referenced:**
- [#5785](https://github.com/Textualize/textual/pull/5785) - VERTICAL_BREAKPOINTS fix
- [#5770](https://github.com/Textualize/textual/pull/5770) - Button text selection fix
- [#5799](https://github.com/Textualize/textual/pull/5799) - App.run after asyncio.run fix
- [#5811](https://github.com/Textualize/textual/issues/5811) - Input cursor position fix
- [#5827](https://github.com/Textualize/textual/pull/5827) - DEFAULT_CLASSES fix
- [#5823](https://github.com/Textualize/textual/pull/5823) - Markup tag closing fix
- [#5810](https://github.com/Textualize/textual/pull/5810) - CollapsibleTitle exposure
- [#5803](https://github.com/Textualize/textual/pull/5803) - Color HSV support
- [#5830](https://github.com/Textualize/textual/pull/5830) - Input cursor properties

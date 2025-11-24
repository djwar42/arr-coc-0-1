# YouTube: Basic Screen Management in Textual

**Video**: Basic screen management in Textual: pushing, popping, and switching screens

## Overview

This video tutorial from the official Textualize channel demonstrates how to manipulate screens dynamically using Textual's screen management API. Learn how to push, pop, and switch between screens in your terminal user interface (TUI) applications.

**Channel**: Textualize (official)
**Duration**: 9:48
**Published**: December 12, 2023
**Views**: 3,400+

## Key Concepts Covered

From [Basic screen management in Textual: pushing, popping, and switching screens](https://www.youtube.com/watch?v=LJpR6u1ww7Q) (YouTube, accessed 2025-11-02):

### Screen Navigation Methods

The video covers three primary methods for dynamic screen management:

1. **push_screen()** - Adds a new screen to the screen stack
   - Displays the new screen on top of the current screen
   - Allows navigation to new screens while preserving the screen stack
   - Useful for modal dialogs, help screens, and nested workflows

2. **pop_screen()** - Removes the current screen from the stack
   - Returns to the previous screen
   - Undoes a push operation
   - Restores the previous application state

3. **switch_screen()** - Replaces the current screen with a new one
   - Navigates to a new screen without maintaining a stack
   - Clears the current screen instead of layering
   - Useful for multi-view applications with independent screens

### Screen Stack Behavior

The tutorial explains how Textual manages screens as a stack:

- Each application has a screen stack
- Pushing adds to the top (new screen displayed)
- Popping removes from the top (previous screen shown)
- Switching replaces the top screen
- The base screen is always at the bottom of the stack

## Practical Applications

**Modal Dialogs**: Use `push_screen()` to create modal dialog windows that appear on top of the main application interface.

**Nested Navigation**: Build applications with multiple levels of screens, where users can drill down and return to previous states using the screen stack.

**Help Screens**: Display contextual help using `push_screen()` for a temporary overlay that can be dismissed to return to the main screen.

**Multi-View Applications**: Use `switch_screen()` to build applications with completely different interfaces (settings, main menu, content areas) that don't need to maintain a stack.

## Related Video Content

The official Textualize YouTube channel has several complementary tutorials:

- **How to pass data to modal screens in Textual** (3:01) - Learn to customize modal screens by passing data when instantiating them
- **How to add a help screen to your Textual app** (5:56) - Use ModelScreen, bindings, and CSS for styling help screens
- **Debugging a Textual Application** (9:00) - Comprehensive debugging guide for TUI applications

## Sources

**Video Content**:
- [Basic screen management in Textual: pushing, popping, and switching screens](https://www.youtube.com/watch?v=LJpR6u1ww7Q) - Textualize Official YouTube Channel (accessed 2025-11-02)

**Official Documentation**:
- [Textual Screens Guide](https://textual.textualize.io/guide/screens/) - Official Textual documentation covering screen API

**Related Resources**:
- [Textualize Official YouTube Channel](https://www.youtube.com/@Textualize-official) - Official video tutorials and demonstrations
- [Textual GitHub Repository](https://github.com/Textualize/textual) - Source code and examples

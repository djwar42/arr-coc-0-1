# REPL Integration with Textual

## Overview

Integrating a REPL (Read-Eval-Print Loop) with Textual TUI applications presents unique challenges when you want to preserve REPL functionality (like readline shortcuts) while providing a visual interface. This document covers patterns and approaches for building TUI interfaces for existing REPL-based applications.

## The Challenge

When building a TUI for an existing REPL, the primary challenge is maintaining REPL functionality while gaining TUI benefits. Key considerations:

- **Preserving readline shortcuts**: Users expect readline features (history, shortcuts, tab completion)
- **Terminal embedding**: Running a terminal/REPL inside a TUI widget
- **Asynchronous communication**: Coordinating between REPL input and TUI updates
- **Layout integration**: Combining REPL input with other TUI components (source view, output display)

## Use Case: MPI Debugger with Textual

From [GitHub Discussion #4326](https://github.com/Textualize/textual/discussions/4326) (accessed 2025-11-02):

### Original Problem

User TomMelt wanted to build a TUI for an MPI debugger ([mdb](https://github.com/TomMelt/mdb/tree/add-client-server)) that:
- Has a REPL-like input based on Python's `cmd` module
- Uses `asyncio` with client-server architecture to send commands to debug processes
- Needs to display source code while stepping through the debugger

### Proposed Layout

```
┌─────────────────────────────────┐
│     Source Code View (Top)      │
│                                  │
├─────────────────────────────────┤
│  Command Output (Middle)         │
│                                  │
├─────────────────────────────────┤
│  REPL Input (Bottom)             │
└─────────────────────────────────┘
```

### Initial Approach Considered

Using Textual's [Input widget](https://textual.textualize.io/widgets/input/), but this loses REPL functionality:
- No readline shortcuts
- No command history navigation
- No tab completion from the original REPL

## Potential Solutions

### 1. Patching the `cmd` Module

From discussion comments (TomMelt, Apr 13, 2024):

> "I'm thinking that patching the `cmd` module in python might be an option too."

**Approach**: Modify or wrap Python's `cmd.Cmd` class to integrate with Textual's event system.

**Considerations**:
- Allows reuse of existing `cmd`-based REPL logic
- Need to bridge between `cmd.cmdloop()` and Textual's async event loop
- May require threading or asyncio integration

### 2. Forward Input to Existing REPL

**Approach**: Run the existing REPL shell loop and forward input from TUI to it.

**Pattern**:
```python
# Conceptual pattern
class DebuggerApp(App):
    def compose(self):
        yield SourceView()      # Top: source code
        yield OutputDisplay()   # Middle: command output
        yield Input()           # Bottom: command input

    async def on_input_submitted(self, event):
        # Forward to existing REPL/shell
        result = await self.shell_cmd_loop(event.value)
        self.query_one(OutputDisplay).append(result)
```

**Challenges**:
- Coordinating async execution between TUI and REPL
- Maintaining REPL state across TUI updates
- Handling REPL's blocking operations in async context

### 3. Custom Input Widget with Readline

**Approach**: Create a custom Textual widget that embeds readline functionality.

**Requirements**:
- Implement readline-like behavior in Textual widget
- Handle command history (up/down arrows)
- Support tab completion
- Integrate with existing `cmd` module commands

### 4. Terminal Emulator Widget (Advanced)

**Approach**: Embed a terminal emulator within Textual to run the REPL.

**Considerations**:
- Most complex but preserves 100% REPL functionality
- Would require terminal emulation library integration
- Textual doesn't currently provide a built-in terminal widget

## Community Status

From discussion (Sep 13, 2025):

> "Nothing? This has to be a pretty common use case" - brandon-fryslie

**Current State**: This remains an open question in the Textual community. No official solution or widget exists for REPL integration as of discussion date.

## Recommended Approaches

### For Simple REPLs

If your REPL is simple and doesn't heavily rely on readline:
1. Use Textual's `Input` widget
2. Implement command history manually
3. Add keyboard shortcuts for history navigation

### For Complex REPLs with readline

If you need full readline functionality:
1. Consider running REPL in separate thread/process
2. Use message passing between REPL and TUI
3. Display REPL output in Textual widgets while maintaining REPL in background

### For Debuggers and Development Tools

For debuggers like gdb, pdb, or custom MPI debuggers:
1. Split into command layer (reuse existing) and display layer (Textual)
2. Use asyncio to coordinate between layers
3. Display source code, variables, output in Textual widgets
4. Forward commands to backend debugger process

## Code Pattern: Async REPL Integration

```python
from textual.app import App, ComposeResult
from textual.widgets import Input, TextLog, Static
from textual.containers import Vertical
import asyncio

class REPLApp(App):
    """TUI wrapper for existing REPL."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Source View", id="source")
            yield TextLog(id="output")
            yield Input(placeholder="Enter command...")

    def on_mount(self):
        # Initialize your REPL/debugger backend
        self.repl_backend = MyREPLBackend()

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle command submission."""
        command = event.value
        event.input.clear()

        # Execute in REPL backend
        result = await self.repl_backend.execute(command)

        # Update TUI display
        output = self.query_one("#output", TextLog)
        output.write(result)

        # Update source view if needed
        if self.repl_backend.source_changed:
            source = self.query_one("#source", Static)
            source.update(self.repl_backend.get_source())
```

## Best Practices

1. **Separate Concerns**: Keep REPL logic separate from TUI display logic
2. **Async Coordination**: Use asyncio for non-blocking REPL execution
3. **State Management**: Maintain REPL state independently of TUI state
4. **Incremental Migration**: Start with simple TUI overlay, gradually add features
5. **Fallback Option**: Always allow users to run original REPL if TUI doesn't meet needs

## Related Resources

- [Python cmd module](https://docs.python.org/3/library/cmd.html) - Standard library REPL framework
- [Textual Input Widget](https://textual.textualize.io/widgets/input/) - Basic input handling
- [Textual Async Guide](https://textual.textualize.io/guide/workers/) - Async operations in Textual

## Open Questions

As of 2025-11-02, the following remain unresolved in the community:

1. **Built-in REPL Widget**: Will Textual provide a dedicated REPL/terminal widget?
2. **readline Integration**: Best practices for preserving readline functionality
3. **Terminal Embedding**: Recommended approach for embedding terminal emulators
4. **Common Patterns**: Established patterns for debugger TUIs

## Sources

**GitHub Discussions:**
- [Discussion #4326: "what is the best way to build a TUI for an existing REPL"](https://github.com/Textualize/textual/discussions/4326) - GitHub (accessed 2025-11-02)
  - Initial question by TomMelt (Mar 22, 2024)
  - Follow-up on cmd module patching (Apr 13, 2024)
  - Community interest note (Sep 13, 2025)

**Reference Projects:**
- [mdb - MPI debugger](https://github.com/TomMelt/mdb/tree/add-client-server) - Real-world example seeking REPL+TUI integration

**Official Documentation:**
- [Python cmd module](https://docs.python.org/3/library/cmd.html)
- [Textual Input Widget](https://textual.textualize.io/widgets/input/)

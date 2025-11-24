# Textual Testing Best Practices and Strategies

## Overview

Testing strategies specific to Textual TUI applications, combining unit testing principles with TUI-specific considerations. This guide synthesizes best practices from Textual's official documentation and general testing wisdom.

From [Official Testing Guide](https://textual.textualize.io/guide/testing/) and general testing best practices (accessed 2025-11-02)

## Do You Need to Write Tests?

**Short answer:** No, you don't *need* to write tests.

**Reality:** Almost always a good idea for:
- Confidence that app works as intended
- Finding bugs early in development
- Preventing regressions when refactoring
- Documenting expected behavior

Even experienced developers write bugs. Tests help catch them before users do.

From [Official Testing Guide - Do You Need Tests](https://textual.textualize.io/guide/testing/#do-you-need-to-write-tests)

## Testing Philosophy for TUIs

### Test User Behavior, Not Implementation

**Good:**
```python
async def test_user_can_submit_form():
    """Test what the user experiences."""
    async with app.run_test() as pilot:
        # User types name
        await pilot.click("#name-input")
        await pilot.press("J", "o", "h", "n")

        # User clicks submit
        await pilot.click("#submit-button")

        # User sees success message
        assert app.query_one("#status").renderable == "Success!"
```

**Bad:**
```python
async def test_internal_state():
    """Testing implementation details."""
    async with app.run_test() as pilot:
        # Directly manipulating internal state
        app._form_data["name"] = "John"
        app._submit_handler()
        assert app._success_flag == True
```

**Why:** Implementation can change, but user behavior should remain consistent.

## Test Organization Strategies

### Structure by Feature

```
tests/
├── test_login.py           # Login feature tests
├── test_dashboard.py       # Dashboard tests
├── test_settings.py        # Settings tests
└── test_navigation.py      # Navigation tests
```

### Structure by Test Type

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_validators.py
│   └── test_formatters.py
├── integration/            # Integration tests
│   ├── test_login_flow.py
│   └── test_data_sync.py
└── e2e/                    # End-to-end user flows
    └── test_complete_workflow.py
```

### Textual's Approach

Textual itself organizes tests by component/feature:
- See [Textual's tests directory](https://github.com/Textualize/textual/tree/main/tests/)
- Tests are named after the components they test
- Snapshot tests in separate directories

## Writing Effective Tests

### Test Names Should Be Descriptive

**Good:**
```python
async def test_pressing_enter_on_input_submits_form():
    """Clear what's being tested and expected outcome."""
    ...

async def test_clicking_cancel_button_closes_dialog():
    """Describes user action and result."""
    ...
```

**Bad:**
```python
async def test_input():  # What about input?
    ...

async def test_1():  # Meaningless name
    ...
```

### One Assertion Per Concept

**Good:**
```python
async def test_form_validation_rejects_empty_name():
    """Single concept: empty name is invalid."""
    async with app.run_test() as pilot:
        await pilot.click("#submit")
        assert app.query_one("#error").renderable == "Name required"

async def test_form_validation_rejects_short_password():
    """Single concept: short password is invalid."""
    async with app.run_test() as pilot:
        await pilot.click("#password")
        await pilot.press("1", "2", "3")
        await pilot.click("#submit")
        assert "at least 8 characters" in str(app.query_one("#error").renderable)
```

**Acceptable (related assertions):**
```python
async def test_rgb_button_clicks():
    """Multiple assertions testing same feature."""
    async with app.run_test() as pilot:
        await pilot.click("#red")
        assert app.screen.styles.background == Color.parse("red")

        await pilot.click("#green")
        assert app.screen.styles.background == Color.parse("green")

        await pilot.click("#blue")
        assert app.screen.styles.background == Color.parse("blue")
```

### Use Fixtures for Common Setup

```python
import pytest
from my_app import MyApp

@pytest.fixture
async def app():
    """Fixture providing app instance."""
    return MyApp()

@pytest.fixture
async def pilot(app):
    """Fixture providing running app with pilot."""
    async with app.run_test() as pilot:
        yield pilot

async def test_with_fixture(pilot):
    """Tests can reuse common setup."""
    await pilot.click("#button")
    assert pilot.app.button_clicked
```

## Handling Async and Timing

### Always Use pause() After State Changes

**Problem:**
```python
async def test_message_handling():
    async with app.run_test() as pilot:
        app.post_message(CustomMessage())
        # Message may not be processed yet!
        assert app.message_received  # May fail
```

**Solution:**
```python
async def test_message_handling():
    async with app.run_test() as pilot:
        app.post_message(CustomMessage())
        await pilot.pause()  # Wait for message processing
        assert app.message_received  # Now reliable
```

### Wait for Animations

```python
async def test_animated_transition():
    async with app.run_test() as pilot:
        await pilot.click("#animate-button")
        await pilot.wait_for_animation()  # Wait for animation
        assert app.animation_complete
```

### Delays When Needed

```python
async def test_debounced_input():
    """Test debounced search input."""
    async with app.run_test() as pilot:
        await pilot.press("a", "b", "c")
        await pilot.pause(delay=0.5)  # Wait for debounce
        assert app.search_executed
```

From [Official Testing Guide - Pausing](https://textual.textualize.io/guide/testing/#pausing-the-pilot)

## Testing Different Screen Sizes

### Test Responsive Layouts

```python
@pytest.mark.parametrize("size,expected_layout", [
    ((80, 24), "compact"),
    ((120, 40), "normal"),
    ((200, 60), "wide"),
])
async def test_responsive_layout(size, expected_layout):
    """Test layout adapts to screen size."""
    app = MyApp()
    async with app.run_test(size=size) as pilot:
        assert app.current_layout == expected_layout
```

### Test Minimum Dimensions

```python
async def test_minimum_screen_size():
    """Ensure app works at minimum size."""
    app = MyApp()
    async with app.run_test(size=(40, 10)) as pilot:
        # Verify critical UI elements are visible
        assert app.query_one("#main-menu").is_visible
```

## Testing Widget Interactions

### Test Widget Visibility

```python
async def test_modal_shows_and_hides():
    async with app.run_test() as pilot:
        # Modal hidden initially
        modal = app.query_one("#modal")
        assert not modal.is_visible

        # Show modal
        await pilot.click("#show-modal")
        await pilot.pause()
        assert modal.is_visible

        # Hide modal
        await pilot.press("escape")
        await pilot.pause()
        assert not modal.is_visible
```

### Test Focus Management

```python
async def test_tab_navigation():
    async with app.run_test() as pilot:
        # First input should have focus
        assert app.query_one("#first-input").has_focus

        # Tab to next input
        await pilot.press("tab")
        assert app.query_one("#second-input").has_focus

        # Tab to button
        await pilot.press("tab")
        assert app.query_one("#submit-button").has_focus
```

### Test Widget State Changes

```python
async def test_checkbox_toggle():
    async with app.run_test() as pilot:
        checkbox = app.query_one("#agree-checkbox")

        # Initially unchecked
        assert not checkbox.value

        # Click to check
        await pilot.click("#agree-checkbox")
        assert checkbox.value

        # Click to uncheck
        await pilot.click("#agree-checkbox")
        assert not checkbox.value
```

## Testing Error Conditions

### Test Validation

```python
async def test_email_validation():
    """Test invalid email shows error."""
    async with app.run_test() as pilot:
        await pilot.click("#email-input")
        await pilot.press("i", "n", "v", "a", "l", "i", "d")
        await pilot.click("#submit")
        await pilot.pause()

        error = app.query_one("#error-message")
        assert "valid email" in str(error.renderable).lower()
```

### Test Network Errors

```python
async def test_network_error_handling(monkeypatch):
    """Test app handles network failures gracefully."""
    # Mock network function to raise error
    async def mock_fetch():
        raise ConnectionError("Network unavailable")

    monkeypatch.setattr("my_app.fetch_data", mock_fetch)

    async with app.run_test() as pilot:
        await pilot.click("#load-data")
        await pilot.pause()

        error = app.query_one("#status")
        assert "connection error" in str(error.renderable).lower()
```

## Testing Data-Driven Widgets

### Testing DataTable

```python
async def test_data_table_population():
    """Test DataTable populates correctly."""
    async with app.run_test() as pilot:
        table = app.query_one(DataTable)

        # Check row count
        assert table.row_count == 10

        # Check cell content
        assert table.get_cell("row-1", "name") == "John Doe"

        # Test sorting
        await pilot.click(DataTable)  # Focus table
        await pilot.press("s")  # Sort command
        await pilot.pause()
        assert table.get_cell("row-1", "name") == "Alice Smith"
```

### Testing Tree Widget

```python
async def test_tree_expansion():
    """Test tree nodes expand/collapse."""
    async with app.run_test() as pilot:
        tree = app.query_one(Tree)
        root = tree.root

        # Expand root
        await pilot.click(Tree)
        await pilot.press("enter")
        await pilot.pause()
        assert root.is_expanded

        # Collapse root
        await pilot.press("enter")
        await pilot.pause()
        assert not root.is_expanded
```

## Snapshot Testing Integration

Combine Pilot tests with snapshot testing:

```python
async def test_initial_layout(snap_compare):
    """Snapshot test of initial layout."""
    assert snap_compare("my_app.py")

async def test_layout_after_interaction(snap_compare):
    """Snapshot after user interaction."""
    assert snap_compare("my_app.py", press=["tab", "enter"])
```

See `pytest-textual-snapshot` for full snapshot testing capabilities.

From [Official Testing Guide - Snapshot Testing](https://textual.textualize.io/guide/testing/#snapshot-testing)

## Mocking and Isolation

### Mock External Dependencies

```python
from unittest.mock import AsyncMock

async def test_api_call(monkeypatch):
    """Test with mocked API."""
    mock_api = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr("my_app.api_client.fetch", mock_api)

    async with app.run_test() as pilot:
        await pilot.click("#fetch-button")
        await pilot.pause()

        assert mock_api.called
        assert app.query_one("#status").renderable == "success"
```

### Isolate File System

```python
async def test_file_operations(tmp_path):
    """Test with temporary directory."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"theme": "dark"}')

    app = MyApp(config_path=str(config_file))
    async with app.run_test() as pilot:
        assert app.theme == "dark"
```

## Performance Testing

### Test Rendering Performance

```python
import time

async def test_large_table_performance():
    """Ensure large table renders in reasonable time."""
    app = MyApp()
    async with app.run_test() as pilot:
        start = time.time()
        await pilot.click("#load-1000-rows")
        await pilot.pause()
        duration = time.time() - start

        assert duration < 1.0  # Should render in under 1 second
```

### Test Memory with Large Datasets

```python
async def test_memory_efficient_scrolling():
    """Test app handles large scrollable content."""
    app = MyApp()
    async with app.run_test() as pilot:
        # Load large dataset
        await pilot.click("#load-10000-items")
        await pilot.pause()

        # Scroll through content
        for _ in range(100):
            await pilot.press("pagedown")

        # App should still be responsive
        await pilot.click("#top-button")
        await pilot.pause()
        assert app.at_top
```

## Test Coverage Goals

### Essential Coverage

**Must test:**
- Critical user flows (login, data submission, etc.)
- Error handling and validation
- Navigation and routing
- Key bindings and shortcuts

**Should test:**
- Edge cases (empty states, max lengths, etc.)
- Different screen sizes
- Theme switching
- Accessibility features

**Nice to test:**
- Animation behaviors
- Performance characteristics
- Internationalization

## Common Pitfalls

### Pitfall 1: Testing Implementation Instead of Behavior

**Bad:**
```python
# Testing internal method
assert app._internal_validate_email("test@test.com")
```

**Good:**
```python
# Testing user-visible behavior
await pilot.press("t", "e", "s", "t", "@", "t", "e", "s", "t", ".", "c", "o", "m")
await pilot.click("#submit")
assert not app.query_one("#error").is_visible
```

### Pitfall 2: Forgetting to Pause

```python
# Will likely fail
await pilot.press("enter")
assert app.submitted  # Race condition!

# Reliable
await pilot.press("enter")
await pilot.pause()  # Wait for message processing
assert app.submitted
```

### Pitfall 3: Not Testing Different Paths

```python
# Only testing happy path
async def test_submit():
    await pilot.click("#submit")
    assert app.success

# Test both paths
async def test_submit_with_valid_data():
    await pilot.click("#name")
    await pilot.press("J", "o", "h", "n")
    await pilot.click("#submit")
    await pilot.pause()
    assert app.success

async def test_submit_with_invalid_data():
    # Empty name
    await pilot.click("#submit")
    await pilot.pause()
    assert app.has_error
```

## Continuous Integration

### pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
]
```

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest pytest-asyncio
      - name: Run tests
        run: pytest
```

## Debugging Failed Tests

### Use Verbose Output

```bash
pytest -vv  # Very verbose
pytest -vv -s  # Show print statements
```

### Capture Screenshots on Failure

```python
async def test_with_screenshot_on_failure():
    async with app.run_test() as pilot:
        try:
            await pilot.click("#missing-button")
        except Exception:
            # Save SVG screenshot
            pilot.app.save_screenshot("failure.svg")
            raise
```

### Use pdb for Interactive Debugging

```python
async def test_with_debugger():
    async with app.run_test() as pilot:
        await pilot.click("#button")
        import pdb; pdb.set_trace()  # Debugger here
        assert app.some_state
```

## Sources

**Official Documentation:**
- [Testing Guide](https://textual.textualize.io/guide/testing/) - Official Textual docs (accessed 2025-11-02)
- [Textual Tests](https://github.com/Textualize/textual/tree/main/tests/) - Reference implementations

**General Testing Resources:**
- [Unit Testing Best Practices](https://betterprogramming.pub/unit-testing-best-practices-9bceeafe6edf) - Better Programming (Medium)
- [Python Testing with pytest](https://pythontest.com/pytest-book/) - Comprehensive pytest guide
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) - Async testing plugin

**Testing Philosophy:**
- [Test Behavior Not Implementation](https://testing.googleblog.com/2013/08/testing-on-toilet-test-behavior-not.html) - Google Testing Blog
- [The Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html) - Martin Fowler

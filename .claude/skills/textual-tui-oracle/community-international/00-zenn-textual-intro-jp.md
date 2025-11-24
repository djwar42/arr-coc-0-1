# Textual Introduction - Japanese Community Guide (Zenn.dev)

**Source**: [Textual紹介 (textual_intro)](https://zenn.dev/secondselection/articles/textual_intro)
**Author**: o.m (Second Selection Corporation)
**Published**: June 10, 2025
**Language**: Japanese (日本語)
**Platform**: Zenn.dev

## English Summary

This Japanese community article introduces Textual from a practical developer's perspective at Second Selection Corporation. The author transitioned to Linux-based development with Docker distributed processing and needed a way to display and input multiple things on a single screen. Rather than building a GUI or browser app, they chose TUI (Text User Interface) with Textual as a lightweight solution.

**Key insights from Japanese development perspective:**

### Why Textual? (Developer Context)
- Working primarily in Linux/Docker environments
- Need single-screen multi-input/display capability
- TUI preferred over "excessive" GUI or browser solutions
- SSH accessibility with low network overhead (character-based)

### Unique Japanese Developer Concerns

**Strengths emphasized:**
1. **Terminal communication** - SSH connectivity enables UI construction
2. **Low network load** - Character-based reduces communication overhead
3. **Simple pip installation** - Cross-platform environment setup
4. **CSS-like styling** - Familiar customization approach

**Challenges highlighted:**
1. **No VSCode extension** - Author couldn't find IDE support (as of article date)
2. **Learning curve** - Requires async/event-driven programming knowledge
3. **Limited components** - Fewer widgets than Tkinter/PyQt
4. **Best for simple tools** - Author recommends staying with "simple tools" due to framework quirks ("癖が強い" - strong personality/quirks)

### Cultural Development Insights

**Japanese pragmatic approach:**
- **吉 (kichi/yoshi)** - "It's wise to stick with simple tools" - conservative, practical advice
- Framework acknowledged as "quirky" but useful within constraints
- Active search for alternative TUI frameworks continues
- Focus on distributed systems (Docker, Linux, SSH) over desktop GUI

## Article Structure (Japanese Original)

### 1. What is Textual? (Textualとは？)

Textual is a Python framework for creating terminal applications with advanced text UIs, providing intuitive and simple APIs for complex interactions.

### 2. Merits of Textual (Textualのメリット)

**1. Easy creation of advanced text UIs**
- Build complex UI components (buttons, forms, lists) with concise code
- Python-based high-level text UI construction

**2. Async support (非同期対応)**
- Standard async processing support for high-performance apps
- Async user interactions and network requests for faster response
- **Note**: Japanese article emphasizes this as critical for their use case

**3. Simple and intuitive API**
- Beginner-friendly with minimal learning cost
- UI definitions and event handling with minimal code

**4. Intuitive layout creation**
- Combine widgets easily
- Responsive design - auto-adjusts to console screen size

**5. Cross-platform**
- Works on Linux, macOS, Windows (terminal-based)
- Platform-independent user experience
- **pip-only installation** (emphasized in Japanese context)

**6. Style customization**
- CSS-like stylesheets for appearance customization
- Easy color, font, size changes for visually appealing interfaces

**7. Terminal communication** (日本語特有の強調点)
- **SSH connectivity enables UI construction**
- **Character-based communication = low network load**
- Critical for distributed Docker/Linux environments

### 3. Demerits of Textual (Textualのデメリット)

**1. Not suitable for graphical interfaces**
- Terminal-based, not for GUI applications
- Not suitable for advanced graphics or animations

**2. Learning curve (学習曲線)**
- Requires async programming and event-driven programming knowledge
- Initial learning curve for beginners
- **No VSCode extension found** (Japanese-specific concern as of article date)

**3. Terminal dependency**
- Requires terminal environment familiarity
- Other GUI frameworks (Tkinter, PyQt) may be better for graphical apps
- Users unfamiliar with terminals need adjustment

**4. Limited components**
- Fewer widgets/components compared to Tkinter or PyQt
- Complex UI or custom widgets require additional work

**5. Performance limitations**
- Limited advanced graphical processing or real-time animations
- Performance constraints for richer interactions

### 4. Installation Method (インストール方法)

```bash
pip install textual
```

Simple pip-based installation (emphasized for cross-platform ease).

### 5. Basic Usage (基本的な使い方)

**Hello World example:**

```python
from textual.app import App
from textual.widgets import Text

class MyApp(App):
    def compose(self):
        yield Text("Hello, Textual!")

if __name__ == "__main__":
    MyApp.run()
```

Displays "Hello, Textual!" in terminal.

### 6. Layout Creation (レイアウトの作成)

**Multi-widget layout:**

```python
from textual.app import App
from textual.widgets import Button, Header, Footer

class LayoutApp(App):
    def compose(self):
        yield Header()
        yield Footer()
        yield Button("Click Me!")

if __name__ == "__main__":
    LayoutApp.run()
```

Uses `Header`, `Footer`, `Button` widgets to create full-screen layouts.

### 7. Adding Interaction (インタラクションの追加)

**Button click handler:**

```python
from textual.app import App
from textual.widgets import Button

class InteractiveApp(App):
    def compose(self):
        yield Button("Click Me!")

    async def on_button_pressed(self, event):
        print("Button was clicked!")

if __name__ == "__main__":
    InteractiveApp.run()
```

`on_button_pressed` method specifies button click behavior.

### 8. Advanced Features (高度な機能)

1. **Style customization**: CSS-like styling for UI
2. **Async processing**: Async programming support for improved performance
3. **Dialogs**: Modal windows, popups, complex UIs

### 10. Conclusion (さいごに)

**Author's pragmatic assessment:**

> "When searching for Python TUI frameworks, none had widgets as rich as Textual. However, it feels quite quirky (癖が強い), so it's wise to stick to creating simple tools."

**Japanese developer perspective:**
- **癖が強い** (kuse ga tsuyoi) - "has a strong personality/quirks" - acknowledges framework complexity
- **吉** (kichi) - "wise/auspicious" - pragmatic advice to limit scope
- Plans to continue with Textual but actively searching for alternatives
- Focus on distributed systems (Docker, Linux, SSH access)

## Japanese Cultural Programming Context

### Key Terms (Japanese)

**Technical:**
- **TUI** = テキストユーザーインタフェース (Text User Interface)
- **非同期** (hidobuki) = Async/asynchronous
- **ウィジェット** (wijetto) = Widget
- **レイアウト** (reiauto) = Layout

**Cultural/Idiomatic:**
- **癖が強い** (kuse ga tsuyoi) = "Has strong quirks/personality" - Japanese way of saying "complex/difficult" politely
- **吉** (kichi/yoshi) = "Auspicious/wise" - traditional term for "good decision"
- **大げさ** (oogesa) = "Excessive/exaggerated" - describing GUI/browser as overkill

### Development Environment Insights

**Japanese developer priorities:**
1. **Linux/Docker distributed processing** - Primary environment
2. **SSH accessibility** - Remote development critical
3. **Low network overhead** - Character-based protocols preferred
4. **Simple installation** - pip-only, no complex dependencies
5. **Cross-platform** - But Linux-focused in practice

**Tool selection criteria:**
- Practicality over features
- Conservative scope ("simple tools")
- Continuous evaluation of alternatives
- Acknowledgment of framework limitations

## Code Examples Analysis

All examples follow standard Textual patterns:

1. **Inherit from `App`** - Base application class
2. **Define `compose()` method** - Widget composition
3. **Use `yield`** - Declarative widget mounting
4. **Async event handlers** - `async def on_*` pattern
5. **`App.run()`** - Entry point execution

No Japanese-specific code patterns, but emphasis on:
- **Async first** - Every example highlights async capability
- **Minimal boilerplate** - Concise code valued
- **Widget composition** - `compose()` method central

## Cross-References to Oracle Knowledge

**Related Oracle Documentation:**

**Architecture:**
- [architecture/00-reactive-architecture.md](../architecture/00-reactive-architecture.md) - Async/reactive patterns
- [architecture/01-app-lifecycle.md](../architecture/01-app-lifecycle.md) - App class lifecycle
- [architecture/02-message-system.md](../architecture/02-message-system.md) - Event-driven architecture

**Core Concepts:**
- [core/00-app-basics.md](../core/00-app-basics.md) - App class fundamentals
- [core/01-widgets.md](../core/01-widgets.md) - Widget system
- [core/04-events.md](../core/04-events.md) - Event handling patterns

**Widgets:**
- [widgets/03-button.md](../widgets/03-button.md) - Button widget (used in examples)
- [widgets/07-header-footer.md](../widgets/07-header-footer.md) - Header/Footer widgets
- [widgets/14-static-text-label.md](../widgets/14-static-text-label.md) - Text/Static widgets

**Tutorials:**
- [tutorials/00-hello-world.md](../tutorials/00-hello-world.md) - Basic "Hello World" pattern
- [tutorials/01-widgets-and-layout.md](../tutorials/01-widgets-and-layout.md) - Layout creation

## Japanese Developer Recommendations

**Author's advice for Textual adoption:**

1. **Start small** - "Simple tools" scope recommended
2. **Accept quirks** - Framework has strong personality (癖が強い)
3. **Async knowledge required** - Not optional, must understand async/event-driven
4. **IDE support lacking** - No VSCode extension (as of June 2025)
5. **Compare alternatives** - Continue evaluating other TUI frameworks

**Best use cases (Japanese perspective):**
- Docker distributed systems
- SSH-accessible tooling
- Low-bandwidth network environments
- Linux/terminal-native workflows
- Quick prototyping for internal tools

**Avoid for:**
- Complex graphical applications (use GUI frameworks)
- Rich animations or graphics
- Non-terminal users
- Large-scale production apps (unless scoped conservatively)

## Community Context

**Second Selection Corporation** (株式会社セカンドセレクション):
- Japanese software company
- Focus on distributed systems (Docker, Linux)
- Actively recruiting developers
- Publishing technical articles on Zenn.dev

**Zenn.dev Platform:**
- Japanese developer community platform
- Technical article publishing (similar to Medium/Dev.to)
- Company "Publications" - organizations can publish articles
- Tagged: `Python`, `tui`, `textual`, `tech`

**Related Articles (by same author):**
- [Textual Tips](https://zenn.dev/secondselection/articles/textual_tips) - Follow-up tips article (see `01-zenn-textual-tips-jp.md`)

## Translation Notes

**Key Japanese phrases:**

1. **"どうも、セカンドセレクション前野です"**
   - "Hello, I'm Maeno from Second Selection"
   - Formal Japanese business introduction

2. **"大げさなと考えて"**
   - "Thinking it would be excessive"
   - Describing GUI/browser as overkill for needs

3. **"癖が強いように感じますので、簡単なツールを作るぐらいでとどめておくのが吉です"**
   - "It feels quite quirky, so it's wise to stick to creating simple tools"
   - Key pragmatic advice, uses traditional "吉" (auspicious/wise)

4. **"当分Textualと付き合っていく予定"**
   - "Plan to continue working with Textual for the time being"
   - But also "随時探し中" (continuously searching for alternatives)

## Sources

**Primary Source:**
- [Textual紹介 (Textual Introduction)](https://zenn.dev/secondselection/articles/textual_intro) - Zenn.dev article by o.m, Second Selection Corporation (June 10, 2025)

**Related Resources:**
- [Textual Tips](https://zenn.dev/secondselection/articles/textual_tips) - Follow-up article by same author
- [Second Selection Corporation](https://zenn.dev/p/secondselection) - Company profile on Zenn.dev

**Additional Context:**
- Zenn.dev - Japanese developer community platform
- Publication system - Company-authored technical articles
- Japanese software development culture and practices

---

**Access Date**: November 2, 2025
**Original Language**: Japanese (日本語)
**Translation**: English summary with preserved Japanese cultural context
**Platform**: Zenn.dev (Japanese developer community)

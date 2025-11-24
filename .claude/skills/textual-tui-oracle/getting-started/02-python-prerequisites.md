# Python Prerequisites for Textual Development

## Overview

Before diving into Textual TUI development, you should be comfortable with core Python fundamentals. This guide outlines the essential Python knowledge needed to effectively build terminal user interfaces.

## Prerequisites Checklist

### Level 1: Absolute Basics (Required)

- [ ] **Variables and Data Types**
  - Basic variable assignment and naming conventions
  - Integers, floats, strings, booleans
  - Type conversion (int(), str(), float())
  - Understanding mutability (mutable vs immutable types)

- [ ] **Basic Operators**
  - Arithmetic operators (+, -, *, /, //, %, **)
  - Comparison operators (==, !=, >, <, >=, <=)
  - Logical operators (and, or, not)
  - String concatenation and formatting (f-strings)

- [ ] **Control Flow**
  - if/elif/else statements
  - for loops and while loops
  - break and continue statements
  - Understanding indentation and blocks

- [ ] **Data Structures**
  - Lists (creation, indexing, slicing, methods)
  - Dictionaries (creation, access, iteration)
  - Tuples (immutable sequences)
  - Sets (unique elements)
  - List comprehensions

- [ ] **Functions**
  - Defining functions with def
  - Parameters and arguments
  - Return statements
  - Default arguments
  - *args and **kwargs for flexible arguments

**Resources for Level 1:**

From [trekhleb/learn-python](https://github.com/trekhleb/learn-python):
- Variables (src/getting_started/test_variables.py)
- Operators (src/operators/ directory)
- Data Types (src/data_types/ directory)
- Control Flow (src/control_flow/ directory)
- Functions (src/functions/ directory)

From [learnbyexample 100 Page Python Intro](https://learnbyexample.github.io/100_page_python_intro/):
- Numeric data types (Chapter 3)
- Strings and user input (Chapter 4)
- Defining functions (Chapter 5)
- Control structures (Chapter 6)

### Level 2: Intermediate Concepts (Highly Recommended)

- [ ] **Object-Oriented Programming (OOP)**
  - Class definition and instantiation
  - Instance variables and methods
  - self parameter
  - Inheritance (extending classes)
  - Understanding method resolution order (MRO)

- [ ] **Modules and Imports**
  - Creating and importing modules
  - Understanding packages
  - from X import Y syntax
  - Managing dependencies with pip and requirements.txt

- [ ] **Error Handling**
  - try/except/finally blocks
  - Understanding common exceptions
  - Raising custom exceptions
  - Context managers (with statement)

- [ ] **File I/O**
  - Reading and writing files
  - Using context managers (with statement)
  - File operations and paths

- [ ] **Async/Concurrency (Important for Textual)**
  - Basic understanding of async/await syntax
  - Coroutines and asyncio basics
  - Running async functions
  - Event loops (conceptual understanding)

**Resources for Level 2:**

From [trekhleb/learn-python](https://github.com/trekhleb/learn-python):
- Classes (src/classes/ directory)
- Modules and Packages (src/modules/ directory)
- Errors and Exceptions (src/exceptions/ directory)
- Files (src/files/ directory)

From [learnbyexample 100 Page Python Intro](https://learnbyexample.github.io/100_page_python_intro/):
- Exception handling (Chapter 9)
- Dealing with files (Chapter 19)
- Testing (Chapter 11)

From [learnbyexample Intermediate Resources](https://learnbyexample.github.io/py_resources/intermediate.html):
- Practical Python Programming - script writing and program organization
- Beyond the Basic Stuff with Python - OOP and practice projects
- Architecture Patterns with Python - design patterns for larger projects

### Level 3: Textual-Specific Python Skills (Essential)

- [ ] **Decorators**
  - Understanding @ decorator syntax
  - Function and method decorators
  - Built-in decorators (@property, @staticmethod, @classmethod)

- [ ] **String Manipulation**
  - String methods (split, join, strip, replace, format)
  - Regular expressions (re module) - basic patterns
  - String interpolation and formatting

- [ ] **Testing and Debugging**
  - Using pytest for testing
  - Python debugger (pdb) basics
  - print-based debugging vs proper debugging
  - Writing assertions
  - Test organization and fixtures

- [ ] **Code Style and Best Practices**
  - PEP 8 style guide adherence
  - Understanding linting (flake8, pylint)
  - Code organization and readability
  - Naming conventions

- [ ] **Type Hints (Modern Python)**
  - Basic type annotations
  - Type hints in function signatures
  - Understanding generic types (Optional, List, Dict, etc.)
  - Using type checkers like mypy (optional but helpful)

**Resources for Level 3:**

From [learnbyexample 100 Page Python Intro](https://learnbyexample.github.io/100_page_python_intro/):
- Debugging (Chapter 10) - using pdb for interactive debugging
- Testing (Chapter 11) - pytest and test organization
- Common beginner mistakes section

From [learnbyexample Testing Resources](https://learnbyexample.github.io/py_resources/intermediate.html#testing):
- Getting started with testing in Python (Real Python)
- TDD in Python with pytest
- Modern Test-Driven Development in Python

## Python Fundamentals by Topic

### Variables and Names

From [trekhleb/learn-python Variables](https://github.com/trekhleb/learn-python/blob/master/src/getting_started/test_variables.py):
- Variable naming conventions (snake_case)
- Reserved keywords
- Dynamic typing behavior
- Variable reassignment

### Collections and Iteration

Understanding these is crucial for working with:
- Widget lists and iteration
- Configuration dictionaries
- Data structures for state management

From [trekhleb/learn-python Data Types](https://github.com/trekhleb/learn-python/blob/master/src/data_types/):
- List methods: append(), extend(), insert(), remove(), pop()
- Dictionary operations: keys(), values(), items(), get()
- Set operations for unique values
- List and dict comprehensions

### Functions and Callbacks

Textual heavily uses callbacks for event handling, so understanding functions deeply is critical:
- Function composition
- Higher-order functions (functions that take/return functions)
- Closures and scope
- Default and keyword arguments

From [trekhleb/learn-python Functions](https://github.com/trekhleb/learn-python/blob/master/src/functions/):
- Function definition patterns
- Scoping with global/nonlocal
- Lambda expressions (for simple callbacks)
- Arbitrary arguments (*args, **kwargs)

### Asynchronous Programming (Textual-Specific)

Textual is built on asyncio, so understanding async/await is essential:
- Async functions and the await keyword
- Running async code with asyncio.run()
- Understanding event loops
- Concurrent task execution

From [learnbyexample Intermediate Resources](https://learnbyexample.github.io/py_resources/intermediate.html):
- Search for async and concurrency materials in intermediate section

## Testing and Debugging for TUI Applications

### Debugging TUI Applications

From [learnbyexample 100 Page Python Intro Debugging Chapter](https://learnbyexample.github.io/100_page_python_intro/debugging.html):
- Using pdb for interactive debugging
- Common debugging commands: n (next), s (step), c (continue), p (print), l (list)
- Setting breakpoints with breakpoint() function
- Using print() statements strategically

### Testing TUI Code

Key testing approaches for Textual:
- Unit testing individual functions and classes
- Integration testing widgets and components
- Using pytest fixtures for test setup
- Mocking and patching external dependencies

From [learnbyexample 100 Page Python Intro Testing](https://learnbyexample.github.io/100_page_python_intro/testing.html):
- pytest framework and test discovery
- Writing test functions (test_* prefix)
- Assertions and test organization

From [Textual Testing Guide](../testing/00-pilot-testing-guide.md):
- Textual's pilot testing framework for UI testing
- Testing user interactions
- Snapshot testing for UI changes

## Learning from Real Examples

### Python Exercises TUI App

From [learnbyexample TUI-apps: PythonExercises](https://github.com/learnbyexample/TUI-apps/tree/main/PythonExercises):
- Interactive exercises for beginner to intermediate Python
- Real Textual application demonstrating best practices
- Examples of:
  - Quiz and exercise presentation
  - User input handling in TUI
  - State management
  - Navigation between screens

### Python Regex Exercises TUI

From [learnbyexample TUI-apps: PyRegexExercises](https://github.com/learnbyexample/TUI-apps/tree/main/PyRegexExercises):
- 100+ exercises for Python regular expressions
- Demonstrates:
  - Interactive exercise delivery
  - Real-time validation
  - Feedback mechanisms in TUI
  - Complex state management

## Common Python Mistakes to Avoid

From [learnbyexample Debugging Guide - Common Beginner Mistakes](https://learnbyexample.github.io/100_page_python_intro/debugging.html#common-beginner-mistakes):

- [ ] Don't overwrite built-in names (str, list, dict, len)
- [ ] Avoid mutable default arguments in functions
- [ ] Understand the difference between = (assignment) and == (comparison)
- [ ] Be careful with variable scope (global/nonlocal)
- [ ] Don't confuse method calls with property access (widget.size vs widget.size())
- [ ] Understand None vs False vs empty collections

## Development Tools Setup

### Virtual Environments

From [learnbyexample 100 Page Python Intro](https://learnbyexample.github.io/100_page_python_intro/installing-modules-and-virtual-environments.html):
- Using venv to create isolated Python environments
- Activating virtual environments
- Installing packages with pip
- Creating requirements.txt files

### Code Quality Tools

- **pytest**: Testing framework
- **flake8**: Code style checker (PEP 8)
- **pylint**: Code analysis and linting
- **black**: Code formatter (optional but recommended)

From [learn-python repo .flake8 config](https://github.com/trekhleb/learn-python/blob/master/.flake8):
- Configuration for code style checking

## Progress Tracking

Use this checklist to track your Python preparation:

### Before Starting Textual

- [ ] All Level 1 items completed and tested
- [ ] Comfortable with at least one data structure (list or dict)
- [ ] Can write and call simple functions
- [ ] Understand if/else control flow
- [ ] Can run Python scripts from command line
- [ ] Have Python 3.8+ installed
- [ ] Have created and used a virtual environment

### Before Building Complex Apps

- [ ] All Level 2 items completed
- [ ] Comfortable with classes and OOP
- [ ] Can write basic tests with pytest
- [ ] Understand async/await basics
- [ ] Can debug code with pdb
- [ ] Familiar with common Python exceptions

### Advanced Development

- [ ] Level 3 items completed
- [ ] Comfortable with decorators
- [ ] Can write type hints
- [ ] Writing well-tested code as habit
- [ ] Understanding design patterns

## Next Steps

Once you've covered these prerequisites:

1. **Start with** [getting-started/00-installation.md](00-installation.md) - Set up Textual
2. **Follow** [getting-started/03-learning-path.md](03-learning-path.md) - Structured learning progression
3. **Review** [getting-started/01-official-tutorial.md](01-official-tutorial.md) - Official Textual tutorial
4. **Practice with** [examples](../examples/) - Real-world Textual applications

## Sources

**GitHub Learning Resources:**
- [trekhleb/learn-python](https://github.com/trekhleb/learn-python) - 17.4k stars, comprehensive Python playground
- [learnbyexample/TUI-apps](https://github.com/learnbyexample/TUI-apps) - Real Textual applications for learning
  - [PythonExercises](https://github.com/learnbyexample/TUI-apps/tree/main/PythonExercises) - Interactive Python exercises
  - [PyRegexExercises](https://github.com/learnbyexample/TUI-apps/tree/main/PyRegexExercises) - 100+ regex exercises

**Web Resources:**
- [learnbyexample 100 Page Python Intro](https://learnbyexample.github.io/100_page_python_intro/) - Comprehensive beginner guide
- [learnbyexample Python Intermediate Resources](https://learnbyexample.github.io/py_resources/intermediate.html) - Advanced topics (accessed 2025-11-02)
- [Real Python Testing Guide](https://realpython.com/python-testing/) - Complete testing reference
- [Real Python Textual Tutorial](https://realpython.com/python-textual/) - Textual-specific Python patterns

**Textual Oracle References:**
- [testing/00-pilot-testing-guide.md](../testing/00-pilot-testing-guide.md) - Textual testing framework
- [getting-started/00-installation.md](00-installation.md) - Textual setup
- [core-concepts/01-async-comfortable-tuis.md](../core-concepts/01-async-comfortable-tuis.md) - Async programming with Textual

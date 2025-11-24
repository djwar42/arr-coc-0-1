# Gradio Advanced Blocks & Layouts (2025)

**Category**: Practical Implementation
**Related**: [08-gradio-interface-and-blocks.md](../concepts/08-gradio-interface-and-blocks.md), [10-gradio-testing-best-practices.md](10-gradio-testing-best-practices.md)

## Overview

Gradio Blocks provides a powerful low-level API for building custom layouts and complex multi-component interfaces. This guide covers layout primitives (Row, Column, Group), advanced containers (Tabs, Accordions, Sidebar), dynamic UI with gr.render, custom components, and event listener patterns for Gradio 5.x (2025).

---

## Layout Primitives

### Row, Column, and Group

From [gradio.app/guides/controlling-layout](https://www.gradio.app/guides/controlling-layout):

**Basic Structure:**

```python
import gradio as gr

with gr.Blocks() as demo:
    # Default: vertical layout
    gr.Textbox(label="Input 1")
    gr.Textbox(label="Input 2")

    # Horizontal layout with Row
    with gr.Row():
        btn1 = gr.Button("Button 1")
        btn2 = gr.Button("Button 2")

    # Nested layouts
    with gr.Row():
        with gr.Column(scale=1):
            gr.Textbox(label="Left Column")
            gr.Textbox(label="Left Column 2")

        with gr.Column(scale=2):
            gr.Image(label="Right Column (2x wider)")
            gr.Button("Process")

demo.launch()
```

**Layout Hierarchy:**

- `gr.Blocks`: Top-level container
- `gr.Row`: Horizontal layout container
- `gr.Column`: Vertical layout container
- `gr.Group`: Visual grouping without layout changes

### Scale and Min Width

**Scale Property:**

```python
with gr.Row():
    btn0 = gr.Button("Button 0", scale=0)  # Fixed width, no expansion
    btn1 = gr.Button("Button 1", scale=1)  # Takes 1 unit of space
    btn2 = gr.Button("Button 2", scale=2)  # Takes 2 units (2x wider than btn1)

# Result: btn2 is twice as wide as btn1, btn0 stays fixed
```

**Min Width with Wrapping:**

```python
with gr.Row():
    # These will wrap to new line if window too narrow
    gr.Textbox(min_width=300, label="Field 1")
    gr.Textbox(min_width=300, label="Field 2")
    gr.Textbox(min_width=300, label="Field 3")
```

### Equal Height

```python
with gr.Row(equal_height=True):
    textbox = gr.Textbox(label="Input")  # Will match button height
    btn = gr.Button("Submit")             # Both same height
```

### Complex Layout Example

```python
with gr.Blocks() as demo:
    # Header row
    with gr.Row():
        gr.Markdown("# ML Model Interface")

    # Main content: 2-column layout
    with gr.Row():
        # Left sidebar (1/3 width)
        with gr.Column(scale=1, min_width=300):
            input_text = gr.Textbox(label="Input", lines=5)
            temperature = gr.Slider(0, 1, value=0.7, label="Temperature")
            max_tokens = gr.Number(label="Max Tokens", value=100)
            submit_btn = gr.Button("Generate", variant="primary")

        # Right content area (2/3 width)
        with gr.Column(scale=2, min_width=400):
            output_text = gr.Textbox(label="Output", lines=10)
            with gr.Row():
                copy_btn = gr.Button("Copy")
                clear_btn = gr.Button("Clear")
```

---

## Fill Height and Width

From [gradio.app/guides/controlling-layout](https://www.gradio.app/guides/controlling-layout):

### Fill Browser Width

```python
# Remove side padding, use full browser width
with gr.Blocks(fill_width=True) as demo:
    gr.Textbox(label="Full Width Input")
```

### Fill Browser Height

```python
# Make components expand to full viewport height
with gr.Blocks(fill_height=True) as demo:
    gr.Chatbot(scale=1)    # Expands to fill available space
    gr.Textbox(scale=0)    # Fixed height at bottom

# Useful for chat interfaces, dashboards, full-screen apps
```

**Chat Interface Example:**

```python
with gr.Blocks(fill_height=True, fill_width=True) as demo:
    with gr.Column(scale=1):
        # Chatbot fills remaining space
        chatbot = gr.Chatbot(scale=1, height=None)

        # Input bar fixed at bottom
        with gr.Row(scale=0):
            msg = gr.Textbox(
                scale=4,
                placeholder="Type a message...",
                show_label=False
            )
            send_btn = gr.Button("Send", scale=1)
```

---

## Tabs and Accordions

### Tabs for Organizing Content

From [gradio.app/guides/controlling-layout](https://www.gradio.app/guides/controlling-layout):

```python
import gradio as gr
import numpy as np

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Tool Interface")

    with gr.Tab("Text Tools"):
        text_input = gr.Textbox(label="Input Text")
        text_output = gr.Textbox(label="Output")
        text_button = gr.Button("Flip Text")
        text_button.click(flip_text, inputs=text_input, outputs=text_output)

    with gr.Tab("Image Tools"):
        with gr.Row():
            image_input = gr.Image(label="Input")
            image_output = gr.Image(label="Output")
        image_button = gr.Button("Flip Image")
        image_button.click(flip_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Settings"):
        gr.Markdown("## Configuration")
        api_key = gr.Textbox(label="API Key", type="password")
        save_btn = gr.Button("Save Settings")

demo.launch()
```

**Tab Features:**

- Consecutive tabs automatically grouped
- Only one tab visible at a time
- Tab switching handled automatically
- Can nest other layouts inside tabs

### Accordions for Collapsible Sections

```python
with gr.Blocks() as demo:
    gr.Markdown("# Model Dashboard")

    # Main controls always visible
    input_text = gr.Textbox(label="Input")
    output_text = gr.Textbox(label="Output")

    # Advanced settings in accordion (collapsed by default)
    with gr.Accordion("Advanced Settings", open=False):
        temperature = gr.Slider(0, 1, value=0.7, label="Temperature")
        top_p = gr.Slider(0, 1, value=0.9, label="Top P")
        frequency_penalty = gr.Slider(-2, 2, value=0, label="Frequency Penalty")

    # Optional debugging info
    with gr.Accordion("Debug Info", open=False):
        gr.Markdown("Token count, latency, etc.")
        debug_output = gr.JSON(label="Raw Response")

    process_btn = gr.Button("Process")
```

---

## Sidebar

From [gradio.app/guides/controlling-layout](https://www.gradio.app/guides/controlling-layout):

**Collapsible Sidebar Pattern:**

```python
import gradio as gr
import random

def generate_pet_name(animal_type, personality):
    prefixes = ["Fluffy", "Ziggy", "Bubbles", "Mochi"]
    suffixes = {
        "Cat": ["Whiskers", "Paws"],
        "Dog": ["Woofles", "Barkington"],
        "Bird": ["Feathers", "Chirpy"]
    }

    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes[animal_type])

    if personality == "Royal":
        suffix += " the Magnificent"

    return f"{prefix} {suffix}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Collapsible sidebar on left
    with gr.Sidebar(position="left"):
        gr.Markdown("# 🐾 Pet Name Generator")
        gr.Markdown("Configure your settings below")

        animal_type = gr.Dropdown(
            choices=["Cat", "Dog", "Bird"],
            label="Pet Type",
            value="Cat"
        )
        personality = gr.Radio(
            choices=["Normal", "Silly", "Royal"],
            label="Personality",
            value="Normal"
        )

    # Main content area
    name_output = gr.Textbox(label="Generated Name", lines=2)
    generate_btn = gr.Button("Generate Name! 🎲", variant="primary")

    generate_btn.click(
        fn=generate_pet_name,
        inputs=[animal_type, personality],
        outputs=name_output
    )

demo.launch()
```

**Sidebar Properties:**

- `position`: "left" or "right"
- Automatically collapsible
- Persists across page navigation
- Good for global settings/navigation

---

## Dynamic UI with gr.render

From [gradio.app/guides/dynamic-apps-with-render-decorator](https://www.gradio.app/guides/dynamic-apps-with-render-decorator) and [medium.com/data-science/gradio-beyond-the-interface](https://medium.com/data-science/gradio-beyond-the-interface-f37a4dae307d):

### The gr.render Decorator

**Basic Dynamic UI:**

```python
import gradio as gr

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Enter number of fields")

    @gr.render(inputs=input_text)
    def render_fields(num_str):
        """Dynamically create fields based on input"""
        try:
            num = int(num_str)
            for i in range(num):
                gr.Textbox(label=f"Field {i+1}")
        except:
            gr.Markdown("⚠️ Please enter a valid number")

demo.launch()
```

**Real-World Example: Model Selection:**

```python
with gr.Blocks() as demo:
    model_choice = gr.Radio(
        choices=["GPT-4", "Claude", "Llama"],
        label="Select Model",
        value="GPT-4"
    )

    @gr.render(inputs=model_choice)
    def show_model_settings(model):
        if model == "GPT-4":
            gr.Slider(0, 2, value=0.7, label="Temperature")
            gr.Number(label="Max Tokens", value=2000)
            gr.Checkbox(label="Use JSON Mode")

        elif model == "Claude":
            gr.Slider(0, 1, value=0.5, label="Temperature")
            gr.Textbox(label="System Prompt", lines=3)

        elif model == "Llama":
            gr.Slider(0, 1, value=0.7, label="Temperature")
            gr.Slider(0, 1, value=0.9, label="Top P")
            gr.Checkbox(label="Use 8-bit Quantization")

    input_text = gr.Textbox(label="Your prompt")
    output = gr.Textbox(label="Response")
    submit = gr.Button("Generate")
```

**Dynamic Form Builder:**

```python
with gr.Blocks() as demo:
    form_type = gr.Dropdown(
        choices=["Contact Form", "Survey", "Registration"],
        label="Form Type"
    )

    @gr.render(inputs=form_type)
    def build_form(form_type):
        if form_type == "Contact Form":
            gr.Textbox(label="Name")
            gr.Textbox(label="Email")
            gr.Textbox(label="Message", lines=5)

        elif form_type == "Survey":
            gr.Radio(
                choices=["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied"],
                label="How satisfied are you?"
            )
            gr.Slider(1, 10, label="Rate our service")
            gr.Textbox(label="Additional Comments", lines=3)

        elif form_type == "Registration":
            gr.Textbox(label="Username")
            gr.Textbox(label="Email")
            gr.Textbox(label="Password", type="password")
            gr.Checkbox(label="I agree to terms and conditions")

    submit_btn = gr.Button("Submit")
```

### State Management with gr.render

```python
import gradio as gr

with gr.Blocks() as demo:
    items = gr.State([])  # List of items
    item_input = gr.Textbox(label="Add item")
    add_btn = gr.Button("Add")

    @gr.render(inputs=items)
    def show_items(item_list):
        """Display current items with delete buttons"""
        gr.Markdown(f"### Items ({len(item_list)})")

        for idx, item in enumerate(item_list):
            with gr.Row():
                gr.Textbox(value=item, show_label=False, interactive=False)
                delete_btn = gr.Button("🗑️", scale=0)

                # Each delete button removes its item
                delete_btn.click(
                    lambda i=idx: items.value[:i] + items.value[i+1:],
                    outputs=items
                )

    def add_item(new_item, current_items):
        """Add new item to list"""
        return current_items + [new_item], ""

    add_btn.click(
        add_item,
        inputs=[item_input, items],
        outputs=[items, item_input]
    )

demo.launch()
```

---

## Custom Components

From [gradio.app/guides/custom-components-in-five-minutes](https://www.gradio.app/guides/custom-components-in-five-minutes):

### Creating Custom Components

**Workflow:**

```bash
# 1. Create component template
gradio cc create MyComponent --template SimpleTextbox

# 2. Directory structure
MyComponent/
├── backend/        # Python code
├── frontend/       # JavaScript/Svelte code
├── demo/          # Sample app
└── pyproject.toml  # Package metadata

# 3. Develop with hot reload
cd MyComponent
gradio cc dev

# 4. Build for distribution
gradio cc build

# 5. Publish to PyPI
gradio cc publish
```

**Simple Custom Component Example:**

```python
# backend/mycomponent.py
from gradio.components import Component

class ColorPicker(Component):
    """Custom color picker component"""

    def __init__(
        self,
        value="#000000",
        label="Color",
        **kwargs
    ):
        super().__init__(value=value, label=label, **kwargs)

    def preprocess(self, payload):
        """Convert frontend value to Python"""
        return payload  # Hex color string

    def postprocess(self, value):
        """Convert Python value to frontend"""
        return value

    def example_payload(self):
        return "#FF5733"

    def example_value(self):
        return "#FF5733"
```

**Using Custom Components:**

```python
import gradio as gr
from mycomponent import ColorPicker

def apply_color(color, text):
    """Apply color to text (HTML)"""
    return f'<span style="color:{color}">{text}</span>'

with gr.Blocks() as demo:
    color = ColorPicker(label="Choose Color")
    text = gr.Textbox(label="Text")
    output = gr.HTML(label="Preview")

    gr.Button("Apply").click(
        apply_color,
        inputs=[color, text],
        outputs=output
    )

demo.launch()
```

### Community Custom Components

From [gradio.app/custom-components/gallery](https://www.gradio.app/custom-components/gallery):

**Popular Custom Components:**

- `gradio_pdf`: PDF viewer and annotation
- `gradio_calendar`: Date/time picker
- `gradio_toggle`: Toggle switches
- `gradio_folium`: Interactive maps
- `gradio_molecule3d`: 3D molecule viewer

**Installation:**

```bash
pip install gradio_pdf
```

**Usage:**

```python
import gradio as gr
from gradio_pdf import PDF

with gr.Blocks() as demo:
    pdf_viewer = PDF(label="Upload PDF")
    text_output = gr.Textbox(label="Extracted Text")

    def extract_text(pdf_file):
        # Process PDF
        return "Extracted text..."

    pdf_viewer.change(extract_text, inputs=pdf_viewer, outputs=text_output)

demo.launch()
```

---

## Event Listeners & Data Flows

From [gradio.app/guides/blocks-and-event-listeners](https://www.gradio.app/guides/blocks-and-event-listeners):

### Multiple Data Flows

```python
import gradio as gr

def increase(num):
    return num + 1

with gr.Blocks() as demo:
    a = gr.Number(label="a")
    b = gr.Number(label="b")

    # Bidirectional data flow
    atob = gr.Button("a → b")
    btoa = gr.Button("b → a")

    atob.click(increase, inputs=a, outputs=b)
    btoa.click(increase, inputs=b, outputs=a)

demo.launch()
```

### Multi-Step Processing

```python
from transformers import pipeline
import gradio as gr

asr = pipeline("automatic-speech-recognition")
classifier = pipeline("text-classification")

def speech_to_text(audio):
    return asr(audio)["text"]

def text_to_sentiment(text):
    return classifier(text)[0]["label"]

with gr.Blocks() as demo:
    audio_input = gr.Audio(type="filepath")
    text_output = gr.Textbox(label="Transcription")
    sentiment_output = gr.Label(label="Sentiment")

    transcribe_btn = gr.Button("1. Transcribe")
    analyze_btn = gr.Button("2. Analyze Sentiment")

    # Chain: Audio → Text → Sentiment
    transcribe_btn.click(
        speech_to_text,
        inputs=audio_input,
        outputs=text_output
    )

    analyze_btn.click(
        text_to_sentiment,
        inputs=text_output,
        outputs=sentiment_output
    )

demo.launch()
```

### Event Chaining with .then()

```python
import gradio as gr
import time

def step1(text):
    time.sleep(1)
    return f"Step 1: {text}"

def step2(text):
    time.sleep(1)
    return f"Step 2: {text}"

def step3(text):
    time.sleep(1)
    return f"Step 3: {text}"

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input")
    output_text = gr.Textbox(label="Output")
    btn = gr.Button("Process Pipeline")

    # Chain events with .then()
    btn.click(step1, inputs=input_text, outputs=output_text) \
       .then(step2, inputs=output_text, outputs=output_text) \
       .then(step3, inputs=output_text, outputs=output_text)

demo.launch()
```

### Binding Multiple Triggers

From [gradio.app/guides/blocks-and-event-listeners](https://www.gradio.app/guides/blocks-and-event-listeners):

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

with gr.Blocks() as demo:
    name_input = gr.Textbox(label="Name", placeholder="Enter your name")
    output = gr.Textbox(label="Greeting")
    greet_btn = gr.Button("Greet")

    # Multiple triggers for same function
    gr.on(
        triggers=[name_input.submit, greet_btn.click],
        fn=greet,
        inputs=name_input,
        outputs=output
    )

demo.launch()
```

**Live Updates with gr.on:**

```python
with gr.Blocks() as demo:
    num1 = gr.Slider(1, 10, label="Number 1")
    num2 = gr.Slider(1, 10, label="Number 2")
    num3 = gr.Slider(1, 10, label="Number 3")
    total = gr.Number(label="Sum")

    # Auto-trigger on any input change
    @gr.on(inputs=[num1, num2, num3], outputs=total)
    def calculate_sum(a, b, c):
        return a + b + c

demo.launch()
```

---

## Visibility Control

From [gradio.app/guides/controlling-layout](https://www.gradio.app/guides/controlling-layout):

```python
import gradio as gr

def show_advanced(show):
    """Toggle visibility of advanced settings"""
    return gr.Column(visible=show)

with gr.Blocks() as demo:
    gr.Markdown("# Model Interface")

    # Basic controls (always visible)
    input_text = gr.Textbox(label="Input")
    output_text = gr.Textbox(label="Output")

    # Toggle for advanced settings
    show_advanced_checkbox = gr.Checkbox(label="Show Advanced Settings")

    # Advanced settings (hidden by default)
    with gr.Column(visible=False) as advanced_column:
        temperature = gr.Slider(0, 1, label="Temperature")
        top_p = gr.Slider(0, 1, label="Top P")
        max_tokens = gr.Number(label="Max Tokens")

    # Toggle visibility
    show_advanced_checkbox.change(
        show_advanced,
        inputs=show_advanced_checkbox,
        outputs=advanced_column
    )

    submit_btn = gr.Button("Generate")

demo.launch()
```

**Conditional UI Example:**

```python
with gr.Blocks() as demo:
    task_type = gr.Radio(
        choices=["Classification", "Generation", "Translation"],
        label="Task Type"
    )

    # Different inputs for each task
    with gr.Column(visible=True) as classification_ui:
        gr.Textbox(label="Text to classify")
        gr.Dropdown(choices=["Positive", "Negative", "Neutral"], label="Classes")

    with gr.Column(visible=False) as generation_ui:
        gr.Textbox(label="Prompt", lines=3)
        gr.Slider(0, 1, label="Creativity")

    with gr.Column(visible=False) as translation_ui:
        gr.Textbox(label="Source Text")
        gr.Dropdown(choices=["English", "Spanish", "French"], label="Target Language")

    def update_ui(task):
        """Show/hide UI based on task selection"""
        return {
            classification_ui: gr.Column(visible=task == "Classification"),
            generation_ui: gr.Column(visible=task == "Generation"),
            translation_ui: gr.Column(visible=task == "Translation")
        }

    task_type.change(
        update_ui,
        inputs=task_type,
        outputs=[classification_ui, generation_ui, translation_ui]
    )

demo.launch()
```

---

## Rendering Components Separately

From [gradio.app/guides/controlling-layout](https://www.gradio.app/guides/controlling-layout):

**Define Before Render:**

```python
import gradio as gr

# Define component outside Blocks context
input_textbox = gr.Textbox(label="Input")

with gr.Blocks() as demo:
    # Show examples first (requires input_textbox reference)
    gr.Examples(
        examples=["Hello", "Bonjour", "Hola"],
        inputs=input_textbox
    )

    # Render component in desired position
    input_textbox.render()

    output_textbox = gr.Textbox(label="Output")
    btn = gr.Button("Process")

demo.launch()
```

**Unrender and Re-render:**

```python
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("Column 1")
            textbox = gr.Textbox()

        with gr.Column():
            gr.Markdown("Column 2")
            textbox.unrender()  # Remove from Column 1

        with gr.Column():
            gr.Markdown("Column 3")
            textbox.render()    # Add to Column 3

demo.launch()
```

---

## Walkthrough Component

From [gradio.app/guides/controlling-layout](https://www.gradio.app/guides/controlling-layout):

**Multi-Step Guided Flow:**

```python
import gradio as gr

def next_step(current_step):
    """Progress to next step"""
    return f"step_{current_step + 1}"

with gr.Blocks() as demo:
    with gr.Walkthrough(elem_id="main_walkthrough") as walkthrough:

        with gr.Step(elem_id="step_1", label="Welcome"):
            gr.Markdown("# Welcome to the Setup Wizard")
            gr.Markdown("Click Next to begin")
            next_btn = gr.Button("Next")

        with gr.Step(elem_id="step_2", label="Configure"):
            gr.Markdown("## Configuration")
            api_key = gr.Textbox(label="API Key")
            next_btn2 = gr.Button("Next")

        with gr.Step(elem_id="step_3", label="Review"):
            gr.Markdown("## Review Settings")
            review_output = gr.Textbox(label="Summary")
            finish_btn = gr.Button("Finish")

    # Wire up navigation
    next_btn.click(lambda: "step_2", outputs=walkthrough)
    next_btn2.click(lambda: "step_3", outputs=walkthrough)
    finish_btn.click(lambda: "step_1", outputs=walkthrough)

demo.launch()
```

---

## Summary

Advanced Gradio layouts enable:

**Layout Control:**
- Row/Column/Group for structure
- Scale and min_width for sizing
- fill_height/fill_width for responsive design

**Organization:**
- Tabs for multiple views
- Accordions for collapsible sections
- Sidebar for persistent navigation

**Dynamic Interfaces:**
- gr.render for conditional UI
- State management with gr.State
- Component visibility toggling

**Extensibility:**
- Custom components with gradio cc
- Community component ecosystem
- Frontend/backend separation

**Event Patterns:**
- Multiple triggers with gr.on
- Event chaining with .then()
- Bidirectional data flows

**Best Practices:**
- Use gr.render for dynamic content
- Leverage scale for responsive layouts
- Combine Tabs + Accordions for complex UIs
- Implement Walkthrough for guided flows

These patterns transform Gradio from simple demos into sophisticated, production-ready applications with complex user interactions and workflows.

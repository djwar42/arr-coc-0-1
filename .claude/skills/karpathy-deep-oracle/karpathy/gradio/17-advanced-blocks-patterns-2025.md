# Advanced Gradio Blocks Patterns (2025)

Advanced patterns for building complex, interactive Gradio applications using Blocks API with dynamic UI, event chaining, and sophisticated state management.

## Overview

Gradio Blocks provides a low-level API for creating highly customizable applications beyond the capabilities of the Interface class. These advanced patterns enable dynamic component generation, complex event flows, and sophisticated user interactions essential for research applications like VLM testing and validation.

**Key capabilities covered:**
- Dynamic UI generation with `@gr.render`
- Complex nested layouts (Rows, Columns, Tabs, Accordions)
- Event chaining and dependencies (`.then()`, `.success()`, `.failure()`)
- State management (global, session, browser)
- Multi-trigger binding with `gr.on()`
- Component visibility control
- Key-based component preservation

---

## Dynamic UI with @gr.render

The `@gr.render` decorator enables components and event listeners to be created dynamically based on user input or state changes.

### Basic Dynamic Components

**Creating variable number of components:**

```python
import gradio as gr

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="input")

    @gr.render(inputs=input_text)
    def show_split(text):
        if len(text) == 0:
            gr.Markdown("## No Input Provided")
        else:
            for letter in text:
                gr.Textbox(letter)

demo.launch()
```

**Key points:**
- Function automatically re-runs when inputs change
- Components created in previous render are cleared
- Default triggers: `demo.load` and `.change` on inputs

### Custom Triggers

**Override default triggers:**

```python
with gr.Blocks() as demo:
    input_text = gr.Textbox(label="input")
    mode = gr.Radio(["textbox", "button"], value="textbox")

    @gr.render(inputs=[input_text, mode], triggers=[input_text.submit])
    def show_split(text, mode):
        if len(text) == 0:
            gr.Markdown("## No Input Provided")
        else:
            for letter in text:
                if mode == "textbox":
                    gr.Textbox(letter)
                else:
                    gr.Button(letter)
```

**Note:** If setting custom triggers and want automatic render at start, add `demo.load` to triggers list.

### Dynamic Event Listeners

**Event listeners must be defined inside render function:**

```python
import gradio as gr

with gr.Blocks() as demo:
    text_count = gr.State(1)
    add_btn = gr.Button("Add Box")
    add_btn.click(lambda x: x + 1, text_count, text_count)

    @gr.render(inputs=text_count)
    def render_count(count):
        boxes = []
        for i in range(count):
            box = gr.Textbox(key=i, label=f"Box {i}")
            boxes.append(box)

        def merge(*args):
            return " ".join(args)

        merge_btn.click(merge, boxes, output)

    merge_btn = gr.Button("Merge")
    output = gr.Textbox(label="Merged Output")

demo.launch()
```

**Critical rules:**
- Event listeners using render-created components must be inside render function
- Can reference components outside render function (like `merge_btn`, `output`)
- Previous render's event listeners are cleared on re-render

### Component Keys for Value Preservation

The `key=` parameter preserves component values across re-renders:

```python
import gradio as gr
import random

with gr.Blocks() as demo:
    number_of_boxes = gr.Slider(1, 5, step=1, value=3, label="Number of Boxes")

    @gr.render(inputs=[number_of_boxes])
    def create_boxes(number_of_boxes):
        for i in range(number_of_boxes):
            with gr.Row(key=f'row-{i}'):
                number_box = gr.Textbox(
                    label=f"Default Label",
                    info="Default Info",
                    key=f"box-{i}",
                    preserved_by_key=["label", "value"],
                    interactive=True
                )
                change_label_btn = gr.Button("Change Label", key=f"btn-{i}")

                change_label_btn.click(
                    lambda: gr.Textbox(
                        label=random.choice("ABCDE"),
                        info=random.choice("ABCDE")
                    ),
                    outputs=number_box
                )

demo.launch()
```

**Key benefits:**
1. **Browser performance** - Same element reused, no destroy/rebuild
2. **Value preservation** - Maintains user input across renders
3. **Property preservation** - Keep specified properties via `preserved_by_key=`

**Key requirements:**
- Parent layout components (Row, Column) must also have keys if nested
- Default preservation: "value" only
- Specify additional properties: `preserved_by_key=["label", "value", "info"]`

### Event Listener Keys

Key event listeners for performance and correctness:

```python
button.click(fn=my_function, inputs=inp, outputs=out, key="my-listener")
```

**Benefits:**
- Performance gains across re-renders
- Prevents errors when event triggered during re-render
- Gradio knows where to route data properly

---

## Complex Nested Layouts

### Rows and Columns

**Horizontal layout with gr.Row:**

```python
with gr.Blocks() as demo:
    with gr.Row():
        btn1 = gr.Button("Button 1")
        btn2 = gr.Button("Button 2")
```

**Equal height elements:**

```python
with gr.Row(equal_height=True):
    textbox = gr.Textbox()
    btn = gr.Button("Submit")
```

**Width control with scale:**

```python
with gr.Row():
    btn0 = gr.Button("Button 0", scale=0)  # No expansion
    btn1 = gr.Button("Button 1", scale=1)  # 1x expansion
    btn2 = gr.Button("Button 2", scale=2)  # 2x expansion
```

**Minimum width:**

```python
with gr.Row():
    btn = gr.Button("Button", min_width=300)
```

Row wraps if insufficient space for all `min_width` values.

### Nested Rows and Columns

**Complex research interface layout:**

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # Control panel
            model_dropdown = gr.Dropdown(["Model A", "Model B"], label="Model")
            query = gr.Textbox(label="Query")
            submit = gr.Button("Run")

        with gr.Column(scale=2, min_width=400):
            # Results area
            output_image = gr.Image(label="Output")
            metrics = gr.Textbox(label="Metrics")

demo.launch()
```

### Tabs and Accordions

**Tabbed interface:**

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# VLM Testing Interface")

    with gr.Tab("Single Image"):
        image = gr.Image()
        query = gr.Textbox(label="Query")
        output = gr.Textbox(label="Response")

    with gr.Tab("Batch Testing"):
        gallery = gr.Gallery()
        batch_query = gr.Textbox(label="Query")
        batch_output = gr.Dataframe()

    with gr.Accordion("Advanced Settings", open=False):
        temperature = gr.Slider(0, 1, value=0.7, label="Temperature")
        max_tokens = gr.Slider(1, 500, value=100, label="Max Tokens")

demo.launch()
```

### Sidebar

**Collapsible sidebar for controls:**

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Sidebar(position="left"):
        gr.Markdown("# Controls")
        checkpoint = gr.Dropdown(["ckpt-1", "ckpt-2"], label="Checkpoint")
        temperature = gr.Slider(0, 1, value=0.7, label="Temperature")

    # Main content area
    output = gr.Textbox(label="Model Output")

demo.launch()
```

### Fill Browser Height/Width

**Full width app:**

```python
with gr.Blocks(fill_width=True) as demo:
    # Removes side padding
    pass
```

**Full height app:**

```python
with gr.Blocks(fill_height=True) as demo:
    gr.Chatbot(scale=1)  # Expands to fill
    gr.Textbox(scale=0)  # Fixed height

demo.launch()
```

---

## Event Chaining and Dependencies

### Sequential Execution with .then()

Run events consecutively:

```python
import gradio as gr
import random
import time

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = random.choice(["How are you?", "Hello!", "Nice to meet you"])
        time.sleep(1)  # Simulate processing
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.launch()
```

### Conditional Chaining

**Execute only on success:**

```python
def process_data(data):
    if not data:
        raise ValueError("No data provided")
    return f"Processed: {data}"

def on_success(result):
    return f"Success: {result}"

def on_failure(error):
    return f"Error occurred: {error}"

btn.click(
    process_data,
    inputs=input_box,
    outputs=output_box
).success(
    on_success,
    inputs=output_box,
    outputs=status_box
).failure(
    on_failure,
    outputs=status_box
)
```

**Methods:**
- `.then()` - Always runs after previous event
- `.success()` - Runs only if previous succeeded
- `.failure()` - Runs only if previous raised error

### Multi-step Workflows

**VLM checkpoint comparison workflow:**

```python
import gradio as gr

with gr.Blocks() as demo:
    image = gr.Image(label="Input Image")
    query = gr.Textbox(label="Query")

    ckpt_a_output = gr.Textbox(label="Checkpoint A Output", visible=False)
    ckpt_b_output = gr.Textbox(label="Checkpoint B Output", visible=False)
    comparison = gr.Textbox(label="Comparison", visible=False)

    run_btn = gr.Button("Compare Checkpoints")

    def load_and_run_a(image, query):
        # Load checkpoint A and run inference
        return "Output from checkpoint A"

    def load_and_run_b(image, query):
        # Load checkpoint B and run inference
        return "Output from checkpoint B"

    def compare_outputs(out_a, out_b):
        return f"A: {out_a}\nB: {out_b}\nDifference: ..."

    run_btn.click(
        load_and_run_a,
        inputs=[image, query],
        outputs=ckpt_a_output
    ).then(
        load_and_run_b,
        inputs=[image, query],
        outputs=ckpt_b_output
    ).then(
        compare_outputs,
        inputs=[ckpt_a_output, ckpt_b_output],
        outputs=comparison
    ).then(
        lambda: [gr.Textbox(visible=True)] * 3,
        outputs=[ckpt_a_output, ckpt_b_output, comparison]
    )

demo.launch()
```

---

## State Management

### Global State

Variables outside functions are shared globally:

```python
import gradio as gr

# Shared between all users
visitor_count = 0

def increment_counter():
    global visitor_count
    visitor_count += 1
    return visitor_count

with gr.Blocks() as demo:
    number = gr.Textbox(label="Total Visitors")
    demo.load(increment_counter, inputs=None, outputs=number)

demo.launch()
```

**Use cases:** Visitor counters, shared caches, global configurations.

### Session State

Per-user state with `gr.State`:

```python
import gradio as gr

with gr.Blocks() as demo:
    cart = gr.State([])  # Each user has their own cart
    items = gr.CheckboxGroup(["Apple", "Banana", "Orange"])

    def add_items(new_items, previous_cart):
        cart = previous_cart + new_items
        return cart

    gr.Button("Add Items").click(add_items, [items, cart], cart)

    cart_size = gr.Number(label="Cart Size")
    cart.change(lambda cart: len(cart), cart, cart_size)

demo.launch()
```

**Key features:**
- State must be deepcopy-able
- Cleared on page refresh
- Stored for 60 minutes after tab close (configurable with `delete_cache`)
- `.change` listener triggers on state updates

### Browser State

Persistent state across sessions with `gr.BrowserState`:

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("Your settings will persist even after refresh")

    username = gr.Textbox(label="Username")
    api_key = gr.Textbox(label="API Key", type="password")

    local_storage = gr.BrowserState({"username": "", "api_key": ""})

    @demo.load(inputs=[local_storage], outputs=[username, api_key])
    def load_from_storage(saved):
        return saved.get("username", ""), saved.get("api_key", "")

    @gr.on([username.change, api_key.change], inputs=[username, api_key], outputs=[local_storage])
    def save_to_storage(user, key):
        return {"username": user, "api_key": key}

demo.launch()
```

**Features:**
- Data persists across page refreshes
- Stored in browser's localStorage
- Cleared on app restart (unless using hardcoded `storage_key`)

---

## Multi-Trigger Event Binding

### Using gr.on() for Multiple Triggers

Bind multiple triggers to one function:

```python
import gradio as gr

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    greet_btn = gr.Button("Greet")

    def greet(name, evt_data: gr.EventData):
        return f"Hello {name}!", evt_data.target.__class__.__name__

    gr.on(
        triggers=[name.submit, greet_btn.click],
        fn=greet,
        inputs=name,
        outputs=output
    )

demo.launch()
```

**Decorator syntax:**

```python
@gr.on(triggers=[name.submit, greet_btn.click], inputs=name, outputs=output)
def greet(name):
    return f"Hello {name}!"
```

### Automatic "Live" Events

Omit triggers to auto-bind to all input `.change` events:

```python
with gr.Blocks() as demo:
    with gr.Row():
        num1 = gr.Slider(1, 10)
        num2 = gr.Slider(1, 10)
        num3 = gr.Slider(1, 10)
    output = gr.Number(label="Sum")

    @gr.on(inputs=[num1, num2, num3], outputs=output)
    def sum(a, b, c):
        return a + b + c

demo.launch()
```

Automatically triggers on any input change.

---

## Advanced Input/Output Patterns

### Function Input: List vs Dict

**List notation (positional arguments):**

```python
def add(num1, num2):
    return num1 + num2

add_btn.click(add, inputs=[a, b], outputs=c)
```

**Dict notation (for many inputs):**

```python
def sub(data):
    return data[a] - data[b]

sub_btn.click(sub, inputs={a, b}, outputs=c)
```

Use dict notation when managing many components.

### Function Output: List vs Dict

**List notation:**

```python
def process(input):
    return output1_value, output2_value

btn.click(process, inputs=input, outputs=[output1, output2])
```

**Dict notation (selective updates):**

```python
def eat(food):
    if food > 0:
        return {food_box: food - 1, status_box: "full"}
    else:
        return {status_box: "hungry"}  # Only update status

btn.click(eat, inputs=food_box, outputs=[food_box, status_box])
```

**Note:** Must still specify all possible outputs in listener, even if conditionally updated.

### Skipping Component Updates

Use `gr.skip()` to leave values unchanged:

```python
import random
import gradio as gr

with gr.Blocks() as demo:
    numbers = [gr.Number(), gr.Number()]

    clear_btn = gr.Button("Clear")
    skip_btn = gr.Button("Skip")
    random_btn = gr.Button("Random")

    clear_btn.click(lambda: (None, None), outputs=numbers)
    skip_btn.click(lambda: [gr.skip(), gr.skip()], outputs=numbers)
    random_btn.click(lambda: (random.randint(0, 100), random.randint(0, 100)), outputs=numbers)

demo.launch()
```

**Difference:**
- `None` - Resets component to empty state
- `gr.skip()` - Leaves value unchanged

---

## Component Visibility Control

### Dynamic Visibility

Show/hide components based on logic:

```python
import gradio as gr

with gr.Blocks() as demo:
    name_box = gr.Textbox(label="Name")
    age_box = gr.Number(label="Age")
    symptoms_box = gr.CheckboxGroup(["Cough", "Fever", "Runny Nose"])
    submit_btn = gr.Button("Submit")

    with gr.Column(visible=False) as output_col:
        diagnosis_box = gr.Textbox(label="Diagnosis")
        summary_box = gr.Textbox(label="Summary")

    def submit(name, age, symptoms):
        return {
            submit_btn: gr.Button(visible=False),
            output_col: gr.Column(visible=True),
            diagnosis_box: "covid" if "Cough" in symptoms else "flu",
            summary_box: f"{name}, {age} y/o"
        }

    submit_btn.click(
        submit,
        [name_box, age_box, symptoms_box],
        [submit_btn, diagnosis_box, summary_box, output_col]
    )

demo.launch()
```

---

## Component Configuration Updates

### Updating Component Properties

Return new component to update configuration:

```python
import gradio as gr

def change_textbox(choice):
    if choice == "short":
        return gr.Textbox(lines=2, visible=True)
    elif choice == "long":
        return gr.Textbox(lines=8, visible=True, value="Lorem ipsum...")
    else:
        return gr.Textbox(visible=False)

with gr.Blocks() as demo:
    radio = gr.Radio(["short", "long", "none"], label="Essay length?")
    text = gr.Textbox(lines=2, interactive=True)
    radio.change(fn=change_textbox, inputs=radio, outputs=text)

demo.launch()
```

**Properties that can be updated:**
- `visible` - Show/hide component
- `value` - Component value
- `interactive` - Enable/disable interaction
- `lines`, `label`, `info` - Visual properties

Unspecified properties preserve previous values.

---

## Real-World Example: VLM Checkpoint Comparison

Combining multiple advanced patterns:

```python
import gradio as gr

with gr.Blocks(fill_height=True) as demo:
    # Global state for loaded models
    loaded_models = {}

    with gr.Sidebar():
        gr.Markdown("# VLM Checkpoint Comparison")
        checkpoint_count = gr.State(2)
        add_ckpt_btn = gr.Button("Add Checkpoint")
        add_ckpt_btn.click(lambda x: x + 1, checkpoint_count, checkpoint_count)

    # Dynamic checkpoint selectors
    @gr.render(inputs=checkpoint_count)
    def render_checkpoints(count):
        checkpoint_dropdowns = []
        for i in range(count):
            with gr.Row(key=f"ckpt-row-{i}"):
                dropdown = gr.Dropdown(
                    ["model-v1", "model-v2", "model-v3"],
                    label=f"Checkpoint {i+1}",
                    key=f"ckpt-{i}"
                )
                checkpoint_dropdowns.append(dropdown)

        # Run comparison
        def compare(*checkpoints):
            results = []
            for ckpt in checkpoints:
                # Load and run inference
                result = f"Output from {ckpt}"
                results.append(result)
            return "\n\n".join(results)

        compare_btn.click(compare, checkpoint_dropdowns, comparison_output)

    # Main interface
    with gr.Row():
        image_input = gr.Image(label="Input Image", scale=1)
        query_input = gr.Textbox(label="Query", scale=1)

    compare_btn = gr.Button("Compare Checkpoints", variant="primary")
    comparison_output = gr.Textbox(label="Comparison Results", lines=10)

demo.launch()
```

**Patterns used:**
- Dynamic UI with `@gr.render`
- Component keys for value preservation
- Nested layouts (Sidebar, Rows)
- State management
- Event chaining

---

## Sources

**Official Gradio Documentation:**
- [Dynamic Apps With Render Decorator](https://www.gradio.app/guides/dynamic-apps-with-render-decorator) (accessed 2025-10-31)
- [Blocks And Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners) (accessed 2025-10-31)
- [State In Blocks](https://www.gradio.app/guides/state-in-blocks) (accessed 2025-10-31)
- [Controlling Layout](https://www.gradio.app/guides/controlling-layout) (accessed 2025-10-31)

**Community Resources:**
- [Medium: Gradio Beyond the Interface](https://medium.com/data-science/gradio-beyond-the-interface-f37a4dae307d) - Advanced layout patterns
- [DataCamp: Building User Interfaces For AI Applications with Gradio](https://www.datacamp.com/tutorial/gradio-python-tutorial) - Comprehensive guide (2024)
- [GitHub: gradio-app/gradio](https://github.com/gradio-app/gradio) - Official repository with examples

**Related Documentation:**
- See `10-vlm-testing-patterns-2025.md` for VLM-specific interface patterns
- See `16-metrics-dashboards-logging-2025.md` for dashboard layout examples
- See `15-hf-spaces-research-deployment-2025.md` for deployment considerations

# Textual-Plotext: Terminal Plotting Widget

## Overview

**textual-plotext** is an official Textualize widget that wraps the [Plotext plotting library](https://github.com/piccolomo/plotext) for use in Textual applications. It enables rich data visualization directly in terminal UIs with support for line plots, scatter plots, bar charts, histograms, matrix plots, and more.

**Key Features:**
- Single `PlotextPlot` widget wrapping full Plotext API
- Support for basic plots (line, scatter, stem, log)
- Bar charts (vertical, horizontal, stacked, histogram)
- Special plots (error, event, matrix, confusion matrix, streaming)
- Decorator plots (shapes, text labels, lines)
- Automatic theme integration with Textual's dark/light modes
- Real-time data updates and streaming support
- Responsive resizing and replotting

## Installation

```bash
pip install textual-plotext
```

Test the library with the built-in demo:

```bash
python -m textual_plotext
```

## Basic Usage

### Simple Scatter Plot

The core difference from vanilla Plotext: use `PlotextPlot.plt` instead of importing `plotext` directly, and don't call `show()` (the widget handles rendering).

```python
from textual.app import App, ComposeResult
from textual_plotext import PlotextPlot

class ScatterApp(App[None]):

    def compose(self) -> ComposeResult:
        yield PlotextPlot()

    def on_mount(self) -> None:
        plt = self.query_one(PlotextPlot).plt
        y = plt.sin()  # sinusoidal test signal
        plt.scatter(y)
        plt.title("Scatter Plot")

if __name__ == "__main__":
    ScatterApp().run()
```

**Key Differences from Plotext:**
- Don't import `plotext` directly
- Access plotting functions via `PlotextPlot.plt`
- Don't call `plt.show()` - widget handles rendering automatically
- Widget refreshes on resize and reactive changes

## Widget API

### PlotextPlot Widget

The `PlotextPlot` widget exposes Plotext's full API through its `plt` property.

**Properties:**
- `plt` - Access to Plotext plotting functions
- `auto_theme` (bool) - Auto-switch themes with dark/light mode (default: True)

**Methods:**
- Standard Textual widget methods (refresh, mount, etc.)
- Call `refresh()` after updating plot data to trigger re-render

### Supported Plotext Functions

**Data Plotting:**
- `plt.plot()` - Line plot
- `plt.scatter()` - Scatter plot
- `plt.bar()` - Vertical/horizontal bar chart
- `plt.multiple_bar()` - Multiple bar series
- `plt.stacked_bar()` - Stacked bars
- `plt.hist()` - Histogram
- `plt.error()` - Error bars
- `plt.event_plot()` - Event timeline
- `plt.matrix_plot()` - Heatmap/matrix
- `plt.cmatrix()` - Confusion matrix

**Decorators:**
- `plt.vline()` / `plt.hline()` - Reference lines
- `plt.text()` - Text labels
- `plt.polygon()` / `plt.rectangle()` - Shapes

**Configuration:**
- `plt.title()` / `plt.xlabel()` / `plt.ylabel()` - Labels
- `plt.xscale()` / `plt.yscale()` - Axis scaling (linear/log)
- `plt.xlim()` / `plt.ylim()` - Axis limits
- `plt.grid()` - Grid lines
- `plt.date_form()` - Date formatting
- `plt.theme()` - Theme selection

**Utilities:**
- `plt.sin()` / `plt.square()` - Test signals
- `plt.colorize()` / `plt.uncolorize()` - Color helpers
- `plt.clear_data()` - Clear plot data (for updates)

### Unsupported Functions

Functions designed for REPL/interactive use are not exposed:

**File/System Operations:**
- `plt.save_text`, `plt.read_data`, `plt.write_data`
- `plt.download`, `plt.delete_file`
- `plt.script_folder`, `plt.parent_folder`, `plt.join_paths`

**Interactive/Testing:**
- `plt.interactive`, `plt.test`, `plt.time`
- Test data URLs (image, GIF, video, YouTube)

**Documentation Properties:**
- `plt.doc`, `plt.markers`, `plt.colors`, `plt.styles`, `plt.themes`

**No-op Functions:**
These can be called but do nothing in Textual context:
- `plt.clear_terminal`, `plt.show`, `plt.save_fig`

**Not Yet Supported:**
- GIF plots
- Video playback

## Plot Types and Examples

### Basic Plots

**Line Plot:**
```python
class LinePlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.plot(self.plt.sin())
        self.plt.title("Line Plot")
```

**Logarithmic Scale:**
```python
class LogPlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.plot(self.plt.sin(periods=2, length=10**4))
        self.plt.xscale("log")
        self.plt.yscale("linear")
        self.plt.grid(0, 1)
        self.plt.title("Logarithmic Plot")
        self.plt.xlabel("logarithmic scale")
        self.plt.ylabel("linear scale")

        # Workaround for Plotext log scale bug
        _ = self.plt.build()
        self.plt.xscale("linear")
```

**Stem Plot (filled):**
```python
class StemPlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.plot(self.plt.sin(), fillx=True)
        self.plt.title("Stem Plot")
```

**Multiple Data Sets:**
```python
class MultipleData(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.plot(self.plt.sin(), label="plot")
        self.plt.scatter(self.plt.sin(phase=-1), label="scatter")
        self.plt.title("Multiple Data Set")
```

**Multiple Axes:**
```python
class MultipleAxes(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.plot(
            self.plt.sin(),
            xside="lower",
            yside="left",
            label="lower left"
        )
        self.plt.plot(
            self.plt.sin(2, phase=-1),
            xside="upper",
            yside="right",
            label="upper right"
        )
        self.plt.title("Multiple Axes Plot")
```

### Bar Charts

**Vertical Bars:**
```python
class VerticalBar(PlotextPlot):
    def on_mount(self) -> None:
        pizzas = ["Sausage", "Pepperoni", "Mushrooms",
                  "Cheese", "Chicken", "Beef"]
        percentages = [14, 36, 11, 8, 7, 4]
        self.plt.bar(pizzas, percentages)
        self.plt.title("Most Favored Pizzas in the World")
```

**Horizontal Bars:**
```python
class HorizontalBar(PlotextPlot):
    def on_mount(self) -> None:
        pizzas = ["Sausage", "Pepperoni", "Mushrooms",
                  "Cheese", "Chicken", "Beef"]
        percentages = [14, 36, 11, 8, 7, 4]
        self.plt.bar(
            pizzas,
            percentages,
            orientation="horizontal",  # or "h"
            width=3/5
        )
        self.plt.title("Most Favoured Pizzas in the World")
```

**Multiple/Grouped Bars:**
```python
class MultipleBars(PlotextPlot):
    def on_mount(self) -> None:
        pizzas = ["Sausage", "Pepperoni", "Mushrooms",
                  "Cheese", "Chicken", "Beef"]
        male = [14, 36, 11, 8, 7, 4]
        female = [12, 20, 35, 15, 2, 1]
        self.plt.multiple_bar(pizzas, [male, female])
        self.plt.title("Most Favored Pizzas by Gender")
```

**Stacked Bars:**
```python
class StackedBars(PlotextPlot):
    def on_mount(self) -> None:
        pizzas = ["Sausage", "Pepperoni", "Mushrooms",
                  "Cheese", "Chicken", "Beef"]
        male = [14, 36, 11, 8, 7, 4]
        female = [12, 20, 35, 15, 2, 1]
        self.plt.stacked_bar(pizzas, [male, female])
        self.plt.title("Most Favored Pizzas by Gender")
```

**Histogram:**
```python
class Histogram(PlotextPlot):
    def on_mount(self) -> None:
        import random
        l = 7 * 10**4
        data1 = [random.gauss(0, 1) for _ in range(10 * l)]
        data2 = [random.gauss(3, 1) for _ in range(6 * l)]
        data3 = [random.gauss(6, 1) for _ in range(4 * l)]
        bins = 60
        self.plt.hist(data1, bins, label="mean 0")
        self.plt.hist(data2, bins, label="mean 3")
        self.plt.hist(data3, bins, label="mean 6")
        self.plt.title("Histogram Plot")
```

### Special Plots

**Matrix/Heatmap:**
```python
class MatrixPlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.title("Matrix Plot")

    def on_resize(self) -> None:
        # Regenerate matrix on resize for responsive heatmap
        p = 1
        matrix = [
            [
                (abs(r - self.size.height / 2) +
                 abs(c - self.size.width / 2)) ** p
                for c in range(self.size.width)
            ]
            for r in range(self.size.height)
        ]
        self.plt.clear_data()
        self.plt.matrix_plot(matrix)
```

**Confusion Matrix:**
```python
class ConfusionMatrix(PlotextPlot):
    def on_mount(self) -> None:
        import random
        l = 300
        actual = [random.randrange(0, 4) for _ in range(l)]
        predicted = [random.randrange(0, 4) for _ in range(l)]
        labels = ["Autumn", "Spring", "Summer", "Winter"]
        self.plt.cmatrix(actual, predicted, labels=labels)
```

**Error Plot:**
```python
class ErrorPlot(PlotextPlot):
    def on_mount(self) -> None:
        import random
        l = 20
        ye = [random.random() for _ in range(l)]
        xe = [random.random() for _ in range(l)]
        data = self.plt.sin(length=l)
        self.plt.error(data, xerr=xe, yerr=ye)
        self.plt.title("Error Plot")
```

**Event Timeline:**
```python
class EventPlot(PlotextPlot):
    def on_mount(self) -> None:
        from datetime import datetime
        import random

        self.plt.date_form("H:M")
        times = self.plt.datetimes_to_string([
            datetime(
                2022, 3, 27,
                random.randint(0, 23),
                random.randint(0, 59),
                random.randint(0, 59)
            )
            for _ in range(100)
        ])
        self.plt.event_plot(times)
```

### Decorator Plots

**Lines and Text:**
```python
class Decorators(PlotextPlot):
    def on_mount(self) -> None:
        pizzas = ["Sausage", "Pepperoni", "Mushrooms",
                  "Cheese", "Chicken", "Beef"]
        percentages = [14, 36, 11, 8, 7, 4]
        self.plt.bar(pizzas, percentages)
        self.plt.title("Labelled Bar Plot")

        # Add text labels above bars
        for i, pizza in enumerate(pizzas):
            self.plt.text(
                pizza,
                x=i + 1,
                y=percentages[i] + 1.5,
                alignment="center",
                color="red"
            )
        self.plt.ylim(0, 38)

        # Add reference lines
        self.plt.hline(20, "blue+")
        self.plt.vline(3, "magenta")
```

**Shapes:**
```python
class Shapes(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.title("Shapes")
        self.plt.polygon()  # Default triangle
        self.plt.rectangle()
        self.plt.polygon(sides=100)  # Circle approximation
```

## Real-Time Data and Updates

### Streaming Data Pattern

For live updating plots, use `set_interval()` with `clear_data()` and `refresh()`:

```python
class StreamingPlot(PlotextPlot):
    def on_mount(self) -> None:
        self.frame = 0
        self.plt.title("Streaming Data")
        # Update plot 4 times per second
        self.set_interval(0.25, self.plot)

    def plot(self) -> None:
        self.plt.clear_data()
        self.plt.scatter(
            self.plt.sin(
                periods=2,
                length=1_000,
                phase=(2 * self.frame) / 50
            )
        )
        self.refresh()  # Trigger re-render
        self.frame += 1
```

### Reactive Updates

Update plot when reactive variables change:

```python
class Weather(PlotextPlot):
    marker: var[str] = var("sd")

    def on_mount(self) -> None:
        self.plt.title("Temperature")
        self._data = []
        self._time = []

    def replot(self) -> None:
        """Redraw plot with current data."""
        self.plt.clear_data()
        self.plt.plot(self._time, self._data, marker=self.marker)
        self.refresh()

    def update(self, time: list, data: list) -> None:
        """Update plot data."""
        self._time = time
        self._data = data
        self.replot()

    def _watch_marker(self) -> None:
        """React to marker changes."""
        self.replot()
```

## Production Example: Weather Dashboard

From [textual_towers_weather.py](https://github.com/Textualize/textual-plotext/blob/main/examples/textual_towers_weather.py):

```python
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.message import Message
from textual.reactive import var
from textual.widgets import Footer, Header
from textual_plotext import PlotextPlot

class Weather(PlotextPlot):
    """Widget for plotting weather data."""

    marker: var[str] = var("sd")

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._unit = "Loading..."
        self._data: list[float] = []
        self._time: list[str] = []
        # Watch app theme changes
        self.watch(
            self.app, "theme",
            lambda: self.call_after_refresh(self.replot)
        )

    def on_mount(self) -> None:
        self.plt.date_form("Y-m-d H:M")
        self.plt.title(self._title)
        self.plt.xlabel("Time")

    def replot(self) -> None:
        """Redraw the plot."""
        self.plt.clear_data()
        self.plt.ylabel(self._unit)
        self.plt.plot(self._time, self._data, marker=self.marker)
        self.refresh()

    def update(self, data: dict, values: str) -> None:
        """Update from API response."""
        self._data = data["hourly"][values]
        self._time = [
            moment.replace("T", " ")
            for moment in data["hourly"]["time"]
        ]
        self._unit = data["hourly_units"][values]
        self.replot()

    def _watch_marker(self) -> None:
        self.replot()


class WeatherApp(App[None]):

    CSS = """
    Grid {
        grid-size: 2;
    }
    Weather {
        padding: 1 2;
    }
    """

    BINDINGS = [
        ("d", "app.toggle_dark", "Toggle light/dark mode"),
        ("m", "marker", "Cycle markers"),
        ("q", "app.quit", "Quit"),
    ]

    MARKERS = {
        "dot": "Dot",
        "hd": "High Definition",
        "fhd": "Higher Definition",
        "braille": "Braille",
        "sd": "Standard Definition",
    }

    marker: var[str] = var("sd")

    def compose(self) -> ComposeResult:
        yield Header()
        with Grid():
            yield Weather("Temperature", id="temperature")
            yield Weather("Wind Speed (10m)", id="windspeed")
            yield Weather("Precipitation", id="precipitation")
            yield Weather("Surface Pressure", id="pressure")
        yield Footer()

    def on_mount(self) -> None:
        self.gather_weather()

    @work(thread=True, exclusive=True)
    def gather_weather(self) -> None:
        """Worker thread for API calls."""
        from datetime import datetime, timedelta
        from json import loads
        from urllib.request import Request, urlopen

        end_date = datetime.now() - timedelta(days=365)
        start_date = end_date - timedelta(weeks=2)

        with urlopen(Request(
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=55.9533&longitude=-3.1883"
            f"&start_date={start_date.strftime('%Y-%m-%d')}"
            f"&end_date={end_date.strftime('%Y-%m-%d')}"
            "&hourly=temperature_2m,precipitation,"
            "surface_pressure,windspeed_10m"
        )) as result:
            data = loads(result.read().decode("utf-8"))
            self.post_message(self.WeatherData(data))

    class WeatherData(Message):
        def __init__(self, history: dict) -> None:
            self.history = history
            super().__init__()

    @on(WeatherData)
    def populate_plots(self, event: WeatherData) -> None:
        """Update plots with fetched data."""
        with self.batch_update():
            self.query_one("#temperature", Weather).update(
                event.history, "temperature_2m"
            )
            self.query_one("#windspeed", Weather).update(
                event.history, "windspeed_10m"
            )
            self.query_one("#precipitation", Weather).update(
                event.history, "precipitation"
            )
            self.query_one("#pressure", Weather).update(
                event.history, "surface_pressure"
            )

    def watch_marker(self) -> None:
        """Update all plots when marker changes."""
        self.sub_title = self.MARKERS[self.marker]
        for plot in self.query(Weather).results(Weather):
            plot.marker = self.marker

    def action_marker(self) -> None:
        """Cycle to next marker type."""
        from itertools import cycle
        markers = cycle(self.MARKERS.keys())
        self.marker = next(markers)


if __name__ == "__main__":
    WeatherApp().run()
```

**Key Patterns:**
- **Worker API**: Use `@work(thread=True)` for API calls to avoid blocking UI
- **Message passing**: Post custom messages to trigger plot updates
- **Batch updates**: Use `with self.batch_update()` when updating multiple widgets
- **Theme watching**: React to app theme changes for consistent styling
- **Marker configuration**: Use plot markers (dot, hd, fhd, braille, sd)

## Theming

### Built-in Themes

Plotext themes are supported, but some use ANSI colors that vary by terminal. For consistency, use the `textual-` prefixed themes which use full RGB colors.

**Textual-specific themes:**
- `textual-design-dark` - Dark mode theme (default in dark mode)
- `textual-design-light` - Light mode theme (default in light mode)
- All standard Plotext themes with `textual-` prefix for RGB consistency

### Auto-theming

By default, `PlotextPlot` automatically switches between dark/light themes when your app's theme changes:

```python
class MyApp(App):
    def compose(self) -> ComposeResult:
        # Will use textual-design-dark in dark mode
        # Will use textual-design-light in light mode
        yield PlotextPlot()
```

Disable auto-theming:

```python
class MyPlot(PlotextPlot):
    def on_mount(self) -> None:
        self.auto_theme = False
        self.plt.theme("textual-design-dark")  # Fixed theme
```

Watch theme changes manually:

```python
class MyPlot(PlotextPlot):
    def on_mount(self) -> None:
        # React to app theme changes
        self.watch(
            self.app, "theme",
            lambda: self.call_after_refresh(self.replot)
        )
```

## Known Issues and Workarounds

### Logarithmic Scale Bug

Plotext has a bug with repeated calls to `show()` or `build()` when using log scales. This causes `ValueError: math domain error` on second render.

**Workaround:**

After the first build, reset the scale to linear:

```python
class LogPlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.xscale("log")
        self.plt.plot(self.plt.sin(periods=2, length=10**4))

        # Force initial build
        _ = self.plt.build()

        # Reset to linear to avoid error on re-render
        self.plt.xscale("linear")
```

This pattern works for both `xscale` and `yscale` issues.

## Performance Considerations

**Plot Complexity:**
- Large datasets (10k+ points) render quickly in terminal
- Matrix plots scale with terminal size
- Use appropriate marker resolution (sd → hd → fhd → braille)

**Update Frequency:**
- Streaming plots: 4-10 Hz is smooth (`set_interval(0.1 to 0.25)`)
- Higher frequencies may cause flicker
- Use `clear_data()` before updating to avoid memory leaks

**Responsive Sizing:**
- Plots auto-resize with widget
- For matrix plots, regenerate data in `on_resize()` for best appearance
- Use `plotsize()` for explicit dimensions if needed

## Integration Patterns

### With DataFrames

```python
import pandas as pd

class DataFramePlot(PlotextPlot):
    def on_mount(self) -> None:
        df = pd.DataFrame({
            'x': range(100),
            'y': [i**2 for i in range(100)]
        })
        self.plt.plot(df['x'].tolist(), df['y'].tolist())
        self.plt.title("DataFrame Plot")
```

### With Textual Workers

```python
from textual import work

class AsyncDataPlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.title("Loading...")
        self.fetch_data()

    @work(thread=True)
    def fetch_data(self) -> None:
        # Expensive computation or API call
        import time
        time.sleep(2)
        data = [i**2 for i in range(100)]

        # Update on main thread
        self.call_from_thread(self.update_plot, data)

    def update_plot(self, data: list) -> None:
        self.plt.clear_data()
        self.plt.plot(data)
        self.plt.title("Data Loaded")
        self.refresh()
```

### Multiple Plots Layout

```python
from textual.containers import Grid

class Dashboard(App):
    CSS = """
    Grid {
        grid-size: 2 2;
        grid-gutter: 1;
    }
    PlotextPlot {
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Grid():
            yield LinePlot()
            yield BarPlot()
            yield ScatterPlot()
            yield MatrixPlot()
```

## Demo Application

Run the comprehensive demo to see all plot types:

```bash
python -m textual_plotext
```

The demo includes tabs for:
- **Basic Plots**: scatter, line, log, stem, multiple datasets
- **Bar Plots**: vertical, horizontal, multiple, stacked, histogram
- **Special Plots**: error, event, streaming, matrix, confusion matrix
- **Decorator Plots**: lines, text, shapes, animations

## Sources

**Official Repository:**
- [Textualize/textual-plotext](https://github.com/Textualize/textual-plotext) (accessed 2025-11-02)
  - [README.md](https://github.com/Textualize/textual-plotext/blob/main/README.md)
  - [textual_towers_weather.py](https://github.com/Textualize/textual-plotext/blob/main/examples/textual_towers_weather.py)
  - [__main__.py demo](https://github.com/Textualize/textual-plotext/blob/main/src/textual_plotext/__main__.py)

**Related:**
- [Plotext library](https://github.com/piccolomo/plotext) - Underlying plotting engine
- [Textual Widgets Guide](https://textual.textualize.io/widget_gallery/) - Official widget documentation

**Statistics:**
- 186 stars, 9 forks (as of 2025-11-02)
- Official Textualize project
- Last updated: 2024-11-21 (active maintenance)

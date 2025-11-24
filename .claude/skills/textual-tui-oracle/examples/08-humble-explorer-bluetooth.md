# HumBLE Explorer - Bluetooth Low Energy Scanner

**Project**: HumBLE Explorer
**Author**: Koen Vervloesem
**Repository**: https://github.com/koenvervloesem/humble-explorer
**Stars**: 74 | **Forks**: 3
**License**: MIT
**Documentation**: https://humble-explorer.readthedocs.io

## Overview

HumBLE Explorer is a cross-platform (Windows, Linux, macOS) human-friendly TUI program for scanning Bluetooth Low Energy (BLE) advertisements. Built with Textual and Bleak, it provides real-time BLE device discovery and advertisement decoding in an interactive terminal interface. Primarily designed for BLE developers and hardware debugging.

**Key Features**:
- Real-time BLE advertisement scanning with live updates
- Cross-platform Bluetooth hardware integration (Windows/Linux/macOS)
- Active and passive scanning modes
- Rich advertisement data decoding (manufacturer data, service UUIDs, RSSI)
- Device filtering by Bluetooth address
- Configurable data type visibility
- Color-coded timestamps and device addresses
- Auto-scrolling with manual navigation

## Hardware Integration

### Bluetooth Low Energy Stack

HumBLE Explorer uses **Bleak** (Bluetooth Low Energy platform Agnostic Klient) as the cross-platform BLE library, providing unified access to:

**Linux**: BlueZ D-Bus API
- Native BlueZ scanner with advertisement monitoring
- Support for passive scanning with OR patterns
- Configurable duplicate detection filters
- Direct access to Bluetooth adapters (hci0, hci1, etc.)

**macOS**: Core Bluetooth
- UUID-based device identification (default)
- Optional Bluetooth address access with `-m` flag
- Native macOS BLE stack integration

**Windows**: Windows Runtime (WinRT)
- Native Windows 10+ BLE support
- Automatic adapter detection

### Scanning Modes

```python
# Active scanning (default) - requests scan response data
scanner_kwargs = {"scanning_mode": "active"}

# Passive scanning - no scan requests sent
scanner_kwargs = {"scanning_mode": "passive"}
```

**Active Scanning**:
- Sends `SCAN_REQ` packets to discovered devices
- Devices respond with `SCAN_RSP` (scan response data)
- More information but increased radio traffic
- May include device name, additional manufacturer data

**Passive Scanning**:
- Listen-only mode, no packets transmitted
- Lower power consumption
- Stealthier device discovery
- Linux requires OR patterns for BlueZ compatibility

### Platform-Specific Configuration

From [app.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/app.py) lines 56-75:

```python
if system() == "Linux":
    if cli_args.scanning_mode == "passive":
        # Passive scanning with BlueZ needs at least one or_pattern
        # The following matches all devices
        self.scanner_kwargs["bluez"] = BlueZScannerArgs(
            or_patterns=[
                OrPattern(0, AdvertisementDataType.FLAGS, b"\x06"),
                OrPattern(0, AdvertisementDataType.FLAGS, b"\x1a"),
            ],
        )
    elif cli_args.scanning_mode == "active":
        # Disable duplicate detection for low-level view
        self.scanner_kwargs["bluez"] = BlueZScannerArgs(
            filters={"DuplicateData": True},
        )
elif system() == "Darwin":
    self.scanner_kwargs["cb"] = {"use_bdaddr": cli_args.macos_use_address}
```

## Architecture

### Component Structure

```
BLEScannerApp (Textual App)
├── BleakScanner (Bluetooth hardware interface)
├── DataTable (advertisement display)
├── SettingsWidget (data type toggles)
├── FilterWidget (address filtering)
├── Header/Footer (navigation)
└── RichRenderables (colored data display)
    ├── RichTime (color-coded timestamps)
    ├── RichDeviceAddress (color-coded addresses + OUI lookup)
    ├── RichAdvertisement (decoded BLE data)
    ├── RichRSSI (signal strength)
    ├── RichUUID (service UUID + names)
    ├── RichCompanyID (manufacturer lookup)
    └── RichHexData (hex/text display)
```

### Real-Time Scanning Pattern

From [app.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/app.py) lines 143-171:

```python
async def on_advertisement(
    self,
    device: BLEDevice,
    advertisement_data: AdvertisementData,
) -> None:
    """Show advertisement data on detection of a BLE advertisement."""
    # Store advertisement with timestamp
    now = datetime.now()
    self.advertisements.append((now, device.address, advertisement_data))

    # Create rich renderables
    table = self.query_one(DataTable)
    self.add_advertisement_to_table(
        table,
        RichTime(now),
        RichDeviceAddress(device.address),
        RichAdvertisement(advertisement_data, self.show_data_config()),
    )
```

**Async Callback Integration**:
- `detection_callback` registered with BleakScanner
- Called for every BLE advertisement received
- Non-blocking UI updates via Textual's async model
- Maintains full history for filtering and scrollback

### Data Table Management

From [app.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/app.py) lines 229-247:

```python
def add_advertisement_to_table(
    self,
    table: DataTable,
    now: RichTime,
    device_address: RichDeviceAddress,
    rich_advertisement: RichAdvertisement,
) -> None:
    """Add new row to table with time, address and advertisement."""
    if device_address.address.startswith(self.address_filter):
        table.add_row(
            now,
            device_address,
            rich_advertisement,
            height=max(device_address.height(), rich_advertisement.height()),
        )
        self.scroll_if_autoscroll()
```

**Dynamic Row Heights**:
- Calculated based on advertisement complexity
- Multi-line manufacturer data expands rows
- Service data trees adjust height automatically

## Textual Implementation Patterns

### Reactive UI Updates

From [app.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/app.py) line 46:

```python
class BLEScannerApp(App[None]):
    """A Textual app to scan for Bluetooth Low Energy advertisements."""

    address_filter = reactive("")  # Reactive attribute for filtering

    def watch_address_filter(self, old_filter: str, new_filter: str) -> None:
        """React when the reactive attribute address_filter changes."""
        self.recreate_table()
```

**Reactive Pattern**:
- `reactive("")` creates observable attribute
- `watch_address_filter()` automatically called on changes
- Triggers table rebuild with filtered data
- Preserves full advertisement history

### Custom Widgets

From [widgets.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/widgets.py) lines 16-30:

```python
class FilterWidget(Input):
    """A Textual widget to filter Bluetooth Low Energy advertisements."""

    def __init__(self, placeholder: str = "") -> None:
        super().__init__(placeholder=placeholder)
        self.display = False

    def on_blur(self) -> None:
        """Automatically hide widget on losing focus."""
        self.display = False
```

**Auto-Hide Pattern**:
- Widget visibility toggled with `display` attribute
- Automatically hides on focus loss (`on_blur`)
- Preserves filter state even when hidden
- User-friendly keyboard navigation

### Settings Sidebar

From [widgets.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/widgets.py) lines 44-80:

```python
def compose(self) -> ComposeResult:
    """Show switches."""
    yield Static("[b]Show data types[/b]\n")
    yield Horizontal(
        Static("Local name       ", classes="label"),
        Switch(value=True, id="local_name", classes="view"),
        classes="container",
    )
    yield Horizontal(
        Static("RSSI             ", classes="label"),
        Switch(value=True, id="rssi", classes="view"),
        classes="container",
    )
    # ... more switches for manufacturer_data, service_data, etc.
    yield Static("\n[b]Other settings[/b]\n")
    yield Horizontal(
        Static("Auto-scroll      ", classes="label"),
        Switch(value=True, id="autoscroll", classes="view"),
        classes="container",
    )
```

**Settings Management**:
- `Switch` widgets with unique IDs
- `view` class for data type toggles
- `on_switch_changed()` handler recreates table
- Dynamic content filtering without data loss

### Rich Renderables for Hardware Data

From [renderables.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/renderables.py) lines 61-96:

```python
class RichDeviceAddress:
    """Rich renderable that shows Bluetooth device address and OUI description."""

    def __init__(self, address: str) -> None:
        self.address = address
        self.style = Style(color=EIGHT_BIT_PALETTE[hash8(self.address)].hex)
        try:
            self.oui = oui[self.address[:8]]  # Lookup manufacturer from MAC prefix
        except (UnknownOUIError, WrongOUIFormatError):
            # macOS returns UUID instead of Bluetooth address
            self.oui = ""

    def height(self) -> int:
        """Return the number of lines this Rich renderable uses."""
        height = 1
        if self.oui:
            height += 1
        return height

    def __rich__(self) -> Text:
        """Render the RichDeviceAddress object."""
        if self.oui:
            return Text.assemble(Text(self.address, style=self.style), f"\n{self.oui}")
        return Text(self.address, style=self.style)
```

**OUI Lookup Integration**:
- First 3 bytes of MAC address identify manufacturer
- `bluetooth-numbers` library for OUI database
- Color-coding via hash for visual device tracking
- Graceful fallback for macOS UUIDs

### Advertisement Data Decoding

From [renderables.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/renderables.py) lines 261-300:

```python
class RichAdvertisement:
    """Rich renderable that shows advertisement data."""

    def __rich__(self) -> Table:
        """Render the RichAdvertisement object."""
        table = Table(show_header=False, show_edge=False, padding=0)

        # Show local name
        if self.data.local_name and self.show_data["local_name"]:
            table.add_row(
                Text.assemble("local name: ", (self.data.local_name, "green bold")),
            )

        # Show RSSI
        if self.data.rssi and self.show_data["rssi"]:
            table.add_row(Text.assemble("RSSI: ", RichRSSI(self.data.rssi).__rich__()))

        # Show manufacturer data with tree structure
        if self.data.manufacturer_data and self.show_data["manufacturer_data"]:
            tree = Tree("manufacturer data:")
            for cic, value in self.data.manufacturer_data.items():
                company_structure = Tree(
                    Text.assemble(
                        RichCompanyID(cic).__rich__(),  # Company name lookup
                        f" → {len(value)} bytes",
                    ),
                )
                company_structure.add(
                    Text.assemble("hex  → ", RichHexData(value).__rich__()),
                )
                company_structure.add(
                    Text.assemble("text → ", RichHexString(value).__rich__()),
                )
                tree.add(company_structure)
            table.add_row(tree)
```

**Rich Tree Display**:
- Manufacturer data decoded with company ID lookup
- Service UUIDs resolved to human-readable names
- Hex and ASCII representations side-by-side
- Collapsible tree structure for complex data

## Key Bindings and UI Flow

From [app.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/app.py) lines 38-44:

```python
BINDINGS = [
    ("q", "quit", "Quit"),
    ("f", "toggle_filter", "Filter"),
    ("s", "toggle_settings", "Settings"),
    ("t", "toggle_scan", "Toggle scan"),
    ("c", "clear_advertisements", "Clear"),
]
```

**Keyboard Navigation**:
- **Q**: Quit application
- **F**: Show/hide filter input (focus automatically set)
- **S**: Show/hide settings sidebar
- **T**: Start/stop BLE scanning
- **C**: Clear advertisement history
- **Tab**: Navigate between widgets
- **PgUp/PgDown/Arrows**: Scroll table when autoscroll disabled

### Start/Stop Scan Pattern

From [app.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/app.py) lines 249-262:

```python
async def start_scan(self) -> None:
    """Start BLE scan."""
    self.scanning = True
    self.set_title()
    await self.scanner.start()

async def stop_scan(self) -> None:
    """Stop BLE scan."""
    self.scanning = False
    self.set_title()
    await self.scanner.stop()
```

**Async Hardware Control**:
- Bleak scanner started/stopped asynchronously
- Title updates to show scanning status
- Non-blocking UI during hardware operations
- Clean resource management

## Installation and Usage

### Installation

From [README.rst](https://github.com/koenvervloesem/humble-explorer/blob/main/README.rst):

```bash
pip install humble-explorer
```

**Dependencies**:
- `textual` - TUI framework
- `bleak` - Cross-platform BLE library
- `bluetooth-numbers` - BLE specification databases (company IDs, service UUIDs, OUI)
- `rich` - Terminal formatting

### Command-Line Interface

```bash
# Default: active scanning on default adapter
humble-explorer

# Passive scanning mode
humble-explorer --scanning-mode passive
humble-explorer -s passive

# Specify Bluetooth adapter (Linux)
humble-explorer --adapter hci1
humble-explorer -a hci1

# macOS: Use Bluetooth address instead of UUID
humble-explorer --macos-use-address
humble-explorer -m

# Show version
humble-explorer --version

# Help
humble-explorer --help
```

### Device Filtering

**Address Filter** (press **F** in app):
```
address=DC
```
Shows only devices with addresses starting with "DC" (case-insensitive).

**Live Filtering**:
- Type filter in input widget
- Table updates in real-time
- Full history preserved (filtered items hidden, not deleted)
- Clear filter to show all devices again

## Hardware Integration Patterns

### Cross-Platform Bluetooth Adapter Access

**Linux Adapter Selection**:
```bash
# List available adapters
hciconfig

# Use specific adapter
humble-explorer -a hci1
```

**Platform Detection**:
```python
from platform import system

if system() == "Linux":
    # Use BlueZ-specific configuration
    scanner_kwargs["bluez"] = BlueZScannerArgs(...)
elif system() == "Darwin":
    # Use Core Bluetooth configuration
    scanner_kwargs["cb"] = {"use_bdaddr": True}
```

### Bluetooth Advertisement Types

HumBLE Explorer decodes:
- **Local Name**: Device name (complete or shortened)
- **RSSI**: Received Signal Strength Indicator (dBm)
- **TX Power**: Transmission power level
- **Manufacturer Data**: Company-specific data (with company ID lookup)
- **Service Data**: Data associated with service UUIDs
- **Service UUIDs**: Advertised GATT services (16-bit, 32-bit, 128-bit)

### Real-Time Performance

**Scanning Speed**:
- Active scanning: ~100ms per device (scan request + response)
- Passive scanning: Immediate (listen only)
- UI updates: Asynchronous, non-blocking
- Table rendering: Efficient with Textual's dirty-region updates

**Memory Management**:
- All advertisements stored in memory (`self.advertisements` list)
- Growing list with continuous scanning
- Press **C** to clear history and free memory
- No persistence (data lost on exit)

## Advanced Patterns

### Color-Coded Timestamps

From [renderables.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/renderables.py) lines 34-56:

```python
class RichTime:
    """Rich renderable that shows a time.

    All times within the same second are rendered in the same color.
    """

    def __init__(self, time: datetime) -> None:
        self.full_time = time.strftime("%H:%M:%S.%f")
        self.style = Style(
            color=EIGHT_BIT_PALETTE[hash8(time.strftime("%H:%M:%S"))].hex,
        )
```

**Visual Grouping**:
- Same-second timestamps share color
- Easy to identify burst of advertisements
- 256-color palette via `hash8()` function
- Microsecond precision displayed

### Hex and ASCII Display

From [renderables.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/renderables.py) lines 205-232:

```python
class RichHexString:
    """Rich renderable that shows hex data as a string.

    Non-printable characters are replaced by a dot.
    """

    def __rich__(self) -> str:
        result = []
        for byte in self.data:
            char = chr(byte)
            if char in PRINTABLE_CHARS:
                result.append(f" {char}")
            else:
                result.append(" .")
        return " ".join(result)
```

**Developer-Friendly Display**:
- Hex bytes and ASCII side-by-side
- Non-printable characters shown as dots
- Helps identify text vs binary data
- Standard hexdump-style visualization

## Use Cases

**BLE Development**:
- Debug advertisement data during development
- Verify service UUIDs and manufacturer data
- Monitor RSSI for range testing
- Compare active vs passive scanning behavior

**Hardware Debugging**:
- Discover BLE devices in area
- Analyze advertisement timing and frequency
- Identify manufacturer from MAC address (OUI)
- Troubleshoot connectivity issues

**Reverse Engineering**:
- Decode proprietary BLE protocols
- Analyze manufacturer-specific data formats
- Monitor device behavior over time
- Filter specific devices by address

## Testing and Quality

From [GitHub repository](https://github.com/koenvervloesem/humble-explorer):

**Testing Infrastructure**:
- Continuous Integration via GitHub Actions
- Codecov coverage tracking
- Cross-platform testing (Linux, macOS, Windows)
- Unit tests for renderables and widgets

**Code Quality**:
- Type hints throughout (`from __future__ import annotations`)
- Ruff for code linting and formatting
- Pre-commit hooks for consistency
- Comprehensive documentation (Sphinx + ReadTheDocs)

## Related Projects

From [README.rst](https://github.com/koenvervloesem/humble-explorer/blob/main/README.rst):

**Author's BLE Book**:
[Develop your own Bluetooth Low Energy Applications for Raspberry Pi, ESP32 and nRF52 with Python, Arduino and Zephyr](https://koen.vervloesem.eu/books/develop-your-own-bluetooth-low-energy-applications/)

**Accompanying Repository**:
https://github.com/koenvervloesem/bluetooth-low-energy-applications

## Key Takeaways

**Textual TUI Patterns**:
- Async hardware integration via callback pattern
- Reactive attributes for UI updates
- Custom widgets with auto-hide behavior
- Rich renderables for complex data visualization
- DataTable with dynamic row heights

**Hardware Integration**:
- Cross-platform BLE via Bleak library
- Platform-specific configuration handling
- Real-time hardware event processing
- Non-blocking async operations

**User Experience**:
- Color-coding for visual grouping
- Keyboard-driven navigation
- Live filtering without data loss
- Configurable data visibility
- Comprehensive help via CLI flags

## Sources

**GitHub Repository**:
- [koenvervloesem/humble-explorer](https://github.com/koenvervloesem/humble-explorer) (accessed 2025-11-02)
- [app.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/app.py) - Main Textual application
- [widgets.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/widgets.py) - Custom Textual widgets
- [renderables.py](https://github.com/koenvervloesem/humble-explorer/blob/main/src/humble_explorer/renderables.py) - Rich rendering classes
- [README.rst](https://github.com/koenvervloesem/humble-explorer/blob/main/README.rst) - Project overview

**Documentation**:
- [HumBLE Explorer Documentation](https://humble-explorer.readthedocs.io) (accessed 2025-11-02)
- [Usage Guide](https://humble-explorer.readthedocs.io/en/latest/usage.html) - Command-line arguments and UI guide

**Python Package**:
- [PyPI: humble-explorer](https://pypi.org/project/humble-explorer/)

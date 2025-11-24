# MeshTUI - LoRa Network Management TUI

## Status Note

**Package Status**: Limited Public Documentation

The `meshtui` package is listed on PyPI (https://pypi.org/project/meshtui/) as a "Meshcore TUI for LoRa networks" but has minimal public documentation, no discoverable GitHub repository, and limited search results as of 2025-11-02. This document covers the broader LoRa mesh network TUI ecosystem, focusing on related projects and patterns that would apply to building similar hardware-integrated TUIs.

## LoRa Mesh Network Context

### What is LoRa?

LoRa (Long Range) is a wireless modulation technique for IoT and M2M communication:
- Long-range: 2-15km depending on environment
- Low power: Battery life measured in years
- Low bandwidth: 0.3-50 kbps
- Operates in unlicensed spectrum (433/868/915 MHz)

### Mesh Network Projects

**Meshtastic** - The primary open-source LoRa mesh networking platform:
- Firmware for ESP32/nRF52 devices
- Android/iOS mobile apps
- Web client
- Python CLI tools
- GitHub: https://github.com/meshtastic

**MeshCore** - Alternative LoRa mesh firmware:
- Focuses on robust routing
- Hardware role definitions at flash time
- Companion radio architecture
- Python library: `meshcore` on PyPI

From [MeshCore PyPI](https://pypi.org/project/meshcore/) (accessed 2025-11-02):
- Base classes for communicating with MeshCore companion radios
- Hardware abstraction layer
- Network topology management

## Hardware Integration Patterns for LoRa TUIs

### Serial Communication

LoRa devices typically connect via:
- **USB Serial**: `/dev/ttyUSB0` or `/dev/ttyACM0` on Linux
- **Bluetooth Serial**: SPP profile for wireless connection
- **TCP/IP**: Some devices offer WiFi gateway mode

**Python Serial Libraries**:
```python
import serial
import pyserial

# Open LoRa device connection
ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)

# Read incoming mesh messages
while True:
    if ser.in_waiting:
        data = ser.readline()
        # Process mesh packet
```

### Hardware Communication Libraries

**pyLoRa** - Direct SX127x transceiver control:
```python
from SX127x.LoRa import *
from SX127x.board_config import BOARD

# Initialize board
BOARD.setup()

# Create LoRa instance
lora = LoRa()
lora.set_mode(MODE.STDBY)
lora.set_freq(915.0)  # MHz
```

**LoRaRF-Python** - Multi-chip LoRa library:
- Supports SX126x, SX127x, LLCC68
- Transmit and receive primitives
- Hardware abstraction
- GitHub: https://github.com/chandrawi/LoRaRF-Python

### Real-Time Network Monitoring

**Key Metrics for LoRa Mesh TUIs**:
- Signal strength (RSSI)
- Signal-to-noise ratio (SNR)
- Packet success rate
- Hop count / route quality
- Active nodes in mesh
- Channel utilization

**Textual Widgets for Hardware Data**:
```python
from textual.widgets import DataTable, Sparkline, Label

class NetworkStatsWidget(Static):
    def compose(self):
        yield Label("Signal Strength")
        yield Sparkline(data=self.rssi_history)
        yield DataTable()  # Node list with metrics

    def update_metrics(self, rssi: int, snr: float):
        """Update from hardware readings"""
        self.rssi_history.append(rssi)
        self.query_one(Sparkline).data = self.rssi_history
```

### Network Topology Display

**Mesh Network Visualization**:
- Node graph showing connections
- Hop paths between nodes
- Link quality indicators
- Real-time packet flow

**ASCII/Unicode Topology Rendering**:
```
Node Network Topology

  [Router-A]───strong───[Node-B]
       │                    │
    medium                weak
       │                    │
  [Node-C]───moderate───[Node-D]

Legend: strong: >-100dBm  moderate: -100 to -110dBm  weak: <-110dBm
```

**Textual DirectedGraph Widget** (if using rich-pixels or custom):
```python
from textual.widgets import Static
from rich.tree import Tree

class MeshTopology(Static):
    def render_topology(self, nodes: dict) -> Tree:
        tree = Tree("Mesh Network")
        for node_id, connections in nodes.items():
            branch = tree.add(f"[bold]{node_id}[/bold]")
            for conn in connections:
                quality = self._signal_quality(conn['rssi'])
                branch.add(f"{conn['target']} ({quality})")
        return tree
```

## TUI Design Patterns for Hardware Apps

### Asynchronous Hardware I/O

**Pattern**: Separate thread/worker for serial communication
```python
from textual.app import App
from textual.worker import Worker
import serial

class LoRaTUI(App):
    def on_mount(self):
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200)
        self.run_worker(self.read_hardware, exclusive=True)

    @work(thread=True)
    async def read_hardware(self):
        """Background thread for serial reads"""
        while True:
            if self.serial_port.in_waiting:
                data = self.serial_port.readline()
                self.call_from_thread(self.process_packet, data)

    def process_packet(self, data):
        """Main thread: Update UI with packet"""
        self.query_one(NetworkStatsWidget).add_packet(data)
```

### Event-Driven Architecture

**Hardware Events → Textual Messages**:
```python
from textual.message import Message

class PacketReceived(Message):
    def __init__(self, rssi: int, snr: float, payload: bytes):
        self.rssi = rssi
        self.snr = snr
        self.payload = payload
        super().__init__()

class NetworkMonitor(Widget):
    def on_packet_received(self, event: PacketReceived):
        """Handle incoming mesh packets"""
        self.update_signal_display(event.rssi, event.snr)
        self.log_packet(event.payload)
```

### Configuration Management

**Hardware Settings via TUI**:
```python
from textual.widgets import Input, Select, Button

class LoRaConfig(Static):
    def compose(self):
        yield Label("Frequency (MHz)")
        yield Input(value="915.0", id="freq")

        yield Label("Spreading Factor")
        yield Select(options=[
            ("SF7 - Fast, Short Range", 7),
            ("SF9 - Balanced", 9),
            ("SF12 - Slow, Long Range", 12)
        ], id="sf")

        yield Button("Apply to Radio", id="apply")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "apply":
            freq = float(self.query_one("#freq").value)
            sf = self.query_one("#sf").value
            self.set_radio_config(freq, sf)

    def set_radio_config(self, freq: float, sf: int):
        """Write config to hardware via serial"""
        cmd = f"AT+FREQ={freq}\r\n"
        self.app.serial_port.write(cmd.encode())
```

## Installation and Setup Patterns

### Hardware Detection

```python
import serial.tools.list_ports

def find_lora_devices():
    """Auto-detect connected LoRa radios"""
    devices = []
    for port in serial.tools.list_ports.comports():
        if 'CP2102' in port.description:  # Common USB-serial chip
            devices.append(port.device)
        elif 'CH340' in port.description:
            devices.append(port.device)
    return devices
```

### Permission Handling

**Linux**: Users need to be in `dialout` group:
```bash
sudo usermod -a -G dialout $USER
# Logout/login required
```

**udev Rules** for automatic permissions:
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666"
```

### Dependency Management

**Typical requirements.txt for LoRa TUI**:
```
textual>=0.40.0
pyserial>=3.5
rich>=13.0.0
# For specific LoRa chips
SX127x>=0.1.0  # or
LoRaRF-Python>=1.0.0
```

## Related Meshtastic TUI Projects

From search results (accessed 2025-11-02):

**Meshtastic Desktop Client** (Rust + TypeScript):
- GitHub: https://github.com/meshtastic (organization)
- Offline deployment and administration
- Ad-hoc mesh network management
- Built with modern TUI frameworks

**MeshCore Projects**:
- Active community on Facebook (LoRa, MESHTASTIC, MeshCore Projects group)
- Focus on robust routing compared to Meshtastic 2.6
- Hardware role definitions

From [Reddit r/meshtastic discussion](https://www.reddit.com/r/meshtastic/comments/1iz0qwq/mesh_26_routing_so_why_try_meshcore_now/) (accessed 2025-11-02):
- "Defined Hardware roles at flash"
- Competition drives innovation between projects
- MeshCore offers alternative routing algorithms

## Example: Minimal LoRa Monitor TUI

```python
"""
Minimal LoRa network monitor using Textual
Displays real-time packet stream and signal metrics
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static
from textual.containers import Vertical, Horizontal
from textual.worker import work
import serial

class SignalStrength(Static):
    """Display current RSSI/SNR"""
    rssi: int = -999
    snr: float = 0.0

    def render(self) -> str:
        quality = "Excellent" if self.rssi > -100 else "Good" if self.rssi > -110 else "Weak"
        return f"Signal: {self.rssi} dBm | SNR: {self.snr:.1f} dB | {quality}"

class PacketLog(DataTable):
    """Log of received packets"""
    def on_mount(self):
        self.add_columns("Time", "From", "RSSI", "Payload")

class LoRaMonitorApp(App):
    CSS = """
    SignalStrength {
        height: 3;
        border: solid green;
        padding: 1;
    }
    PacketLog {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "clear", "Clear Log"),
    ]

    def __init__(self, port: str = "/dev/ttyUSB0"):
        super().__init__()
        self.port = port
        self.serial_connection = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield SignalStrength()
        yield PacketLog()
        yield Footer()

    def on_mount(self):
        """Connect to hardware and start reading"""
        try:
            self.serial_connection = serial.Serial(self.port, 115200, timeout=1)
            self.run_worker(self.read_serial, exclusive=True)
        except serial.SerialException as e:
            self.notify(f"Failed to open {self.port}: {e}", severity="error")
            self.exit()

    @work(thread=True)
    async def read_serial(self):
        """Background worker: Read from serial port"""
        while True:
            if self.serial_connection.in_waiting:
                line = self.serial_connection.readline().decode('utf-8', errors='ignore')
                # Parse LoRa packet format (device-specific)
                if "RX:" in line:
                    self.call_from_thread(self.process_packet, line)

    def process_packet(self, line: str):
        """Main thread: Update UI with received packet"""
        # Parse example format: "RX: FROM:ABC123 RSSI:-95 SNR:8.5 DATA:Hello"
        import re
        match = re.search(r'FROM:(\w+)\s+RSSI:(-?\d+)\s+SNR:([\d.]+)\s+DATA:(.+)', line)
        if match:
            from_id, rssi, snr, data = match.groups()

            # Update signal strength display
            signal_widget = self.query_one(SignalStrength)
            signal_widget.rssi = int(rssi)
            signal_widget.snr = float(snr)
            signal_widget.refresh()

            # Add to packet log
            from datetime import datetime
            log = self.query_one(PacketLog)
            log.add_row(
                datetime.now().strftime("%H:%M:%S"),
                from_id,
                rssi,
                data[:30]  # Truncate long payloads
            )

    def action_clear(self):
        """Clear packet log"""
        self.query_one(PacketLog).clear()

if __name__ == "__main__":
    app = LoRaMonitorApp(port="/dev/ttyUSB0")
    app.run()
```

## Advanced Features

### Multi-Node Management

**Pattern**: Tab-based interface for multiple radios
```python
from textual.widgets import TabbedContent, TabPane

class MultiNodeManager(Static):
    def compose(self):
        with TabbedContent():
            with TabPane("Node A", id="node_a"):
                yield NodeMonitor(port="/dev/ttyUSB0")
            with TabPane("Node B", id="node_b"):
                yield NodeMonitor(port="/dev/ttyUSB1")
```

### Packet Analysis

**Hexdump Display**:
```python
from textual.widgets import TextArea

class PacketInspector(Static):
    def show_packet(self, raw_bytes: bytes):
        hex_view = ' '.join(f'{b:02x}' for b in raw_bytes)
        ascii_view = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw_bytes)

        text_area = self.query_one(TextArea)
        text_area.text = f"Hex:   {hex_view}\nASCII: {ascii_view}"
```

### Route Discovery

**Visualize packet paths through mesh**:
```python
class RouteTracer(Widget):
    def display_route(self, hops: list[str]):
        """Show how packet traversed mesh"""
        route_str = " → ".join(hops)
        self.update(f"Route: {route_str} ({len(hops)} hops)")
```

## Performance Considerations

### Serial I/O Optimization

- Use buffered reads to batch processing
- Parse in background thread
- Only update UI on significant changes (debounce rapid updates)

### Memory Management

- Limit packet log size (ring buffer)
- Archive old data to disk if needed
- Avoid storing raw bytes unnecessarily

### Responsiveness

- Don't block UI thread with serial waits
- Use workers for all hardware I/O
- Batch UI updates (collect packets, update once per 100ms)

## Testing Without Hardware

**Mock Serial Device**:
```python
from unittest.mock import Mock

class MockSerial:
    def __init__(self, *args, **kwargs):
        self.in_waiting = 0
        self._buffer = []

    def readline(self):
        if self._buffer:
            return self._buffer.pop(0)
        return b""

    def inject_packet(self, data: str):
        """Test helper: Inject simulated packet"""
        self._buffer.append(data.encode())
        self.in_waiting = len(self._buffer)

# Use in tests
app = LoRaMonitorApp()
app.serial_connection = MockSerial()
app.serial_connection.inject_packet("RX: FROM:TEST RSSI:-95 SNR:8.0 DATA:Hello")
```

## Summary

While `meshtui` itself has limited public documentation, the patterns for building LoRa network management TUIs are well-established:

**Core Requirements**:
1. Asynchronous serial communication (pyserial)
2. Hardware state monitoring (RSSI, SNR, node status)
3. Real-time data visualization (Textual widgets)
4. Network topology display
5. Configuration management

**Key Libraries**:
- `textual` - TUI framework
- `pyserial` - Serial communication
- `meshcore` - MeshCore device API (if using MeshCore hardware)
- `pyLoRa` or `LoRaRF-Python` - Direct LoRa chip control

**Related Projects**:
- Meshtastic (primary LoRa mesh ecosystem)
- MeshCore (alternative firmware with robust routing)

## Sources

**Web Research** (accessed 2025-11-02):
- [MeshCore PyPI Package](https://pypi.org/project/meshcore/) - Python library for MeshCore radios
- [Meshtastic GitHub Organization](https://github.com/meshtastic) - Open-source LoRa mesh networking
- [Meshtastic Documentation](https://meshtastic.org/docs/development/reference/github/) - Development reference
- [LoRaRF-Python GitHub](https://github.com/chandrawi/LoRaRF-Python) - Multi-chip LoRa library
- [Reddit r/meshtastic](https://www.reddit.com/r/meshtastic/) - Community discussions on LoRa mesh networking
- [pyLoRa PyPI](https://pypi.org/project/pyLoRa/) - SX127x transceiver library

**Search Notes**:
- Direct GitHub repository for "meshtui" package not found in public search results
- PyPI page https://pypi.org/project/meshtui/ reported but not accessible via web scraping (client challenge)
- Patterns documented based on related projects in LoRa TUI ecosystem

**Related Resources**:
- [Textual Framework](https://textual.textualize.io) - Official Textual documentation
- [PySerial Documentation](https://pyserial.readthedocs.io/) - Serial communication in Python

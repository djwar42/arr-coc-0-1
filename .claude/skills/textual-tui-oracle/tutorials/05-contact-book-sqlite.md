# Contact Book App with SQLite Integration

## Overview

Complete tutorial for building a contact book TUI application with Python, Textual, and SQLite. Demonstrates full CRUD operations, database integration patterns, form handling, DataTable usage, and dialog workflows.

**Level**: Intermediate
**Time**: 30-45 minutes
**Features**: Database CRUD, DataTable, Input forms, Confirmation dialogs, CSS styling

From [Build a Contact Book App With Python, Textual, and SQLite](https://realpython.com/contact-book-python-textual/) (accessed 2025-11-02)

---

## Project Structure

```
rpcontacts_project/
│
├── rpcontacts/
│   ├── __init__.py          # Package initialization
│   ├── __main__.py          # Entry point
│   ├── database.py          # SQLite operations
│   ├── rpcontacts.tcss      # CSS styling
│   └── tui.py              # TUI components
│
├── README.md
└── requirements.txt
```

---

## Installation

```bash
# Create project directory
mkdir rpcontacts_project/
cd rpcontacts_project/

# Set up virtual environment
python -m venv venv/
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install Textual
python -m pip install textual
```

---

## Database Design and Implementation

### Database Schema

The contacts database uses a single table with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Auto-generated unique identifier |
| `name` | TEXT | Contact's name |
| `phone` | TEXT | Contact's phone number |
| `email` | TEXT | Contact's email address |

### Database Class Implementation

```python
# rpcontacts/database.py
import pathlib
import sqlite3

DATABASE_PATH = pathlib.Path().home() / "contacts.db"

class Database:
    def __init__(self, db_path=DATABASE_PATH):
        self.db = sqlite3.connect(db_path)
        self.cursor = self.db.cursor()
        self._create_table()

    def _create_table(self):
        """Create contacts table if not exists"""
        query = """
            CREATE TABLE IF NOT EXISTS contacts(
                id INTEGER PRIMARY KEY,
                name TEXT,
                phone TEXT,
                email TEXT
            );
        """
        self._run_query(query)

    def _run_query(self, query, *query_args):
        """Execute SQL query with commit"""
        result = self.cursor.execute(query, [*query_args])
        self.db.commit()
        return result

    def get_all_contacts(self):
        """Retrieve all contacts from database"""
        result = self._run_query("SELECT * FROM contacts;")
        return result.fetchall()

    def get_last_contact(self):
        """Get most recently added contact"""
        result = self._run_query(
            "SELECT * FROM contacts ORDER BY id DESC LIMIT 1;"
        )
        return result.fetchone()

    def add_contact(self, contact):
        """Add new contact - contact is (name, phone, email) tuple"""
        self._run_query(
            "INSERT INTO contacts VALUES (NULL, ?, ?, ?);",
            *contact,
        )

    def delete_contact(self, id):
        """Delete contact by ID"""
        self._run_query(
            "DELETE FROM contacts WHERE id=(?);",
            id,
        )

    def clear_all_contacts(self):
        """Remove all contacts from database"""
        self._run_query("DELETE FROM contacts;")
```

**Key Patterns:**

1. **Connection Management**: Single connection established in `__init__`
2. **Query Helper**: `_run_query()` centralizes execute + commit pattern
3. **Auto-create Table**: `IF NOT EXISTS` ensures database setup on first run
4. **Parameterized Queries**: Uses `?` placeholders to prevent SQL injection
5. **Fetch Methods**: `fetchall()` vs `fetchone()` for different use cases

---

## Main Application Structure

### Package Initialization

```python
# rpcontacts/__init__.py
__version__ = "0.1.0"
```

### Entry Point

```python
# rpcontacts/__main__.py
from rpcontacts.database import Database
from rpcontacts.tui import ContactsApp

def main():
    app = ContactsApp(db=Database())
    app.run()

if __name__ == "__main__":
    main()
```

**Pattern**: Pass database instance to app at initialization for dependency injection.

---

## TUI Components

### Main Application Screen

```python
# rpcontacts/tui.py
from textual.app import App, on
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
)

class ContactsApp(App):
    CSS_PATH = "rpcontacts.tcss"
    BINDINGS = [
        ("m", "toggle_dark", "Toggle dark mode"),
        ("a", "add", "Add"),
        ("d", "delete", "Delete"),
        ("c", "clear_all", "Clear All"),
        ("q", "request_quit", "Quit"),
    ]

    def __init__(self, db):
        super().__init__()
        self.db = db

    def compose(self):
        yield Header()

        # Contacts list (DataTable)
        contacts_list = DataTable(classes="contacts-list")
        contacts_list.focus()
        contacts_list.add_columns("Name", "Phone", "Email")
        contacts_list.cursor_type = "row"
        contacts_list.zebra_stripes = True

        # Buttons panel
        add_button = Button("Add", variant="success", id="add")
        add_button.focus()
        buttons_panel = Vertical(
            add_button,
            Button("Delete", variant="warning", id="delete"),
            Static(classes="separator"),
            Button("Clear All", variant="error", id="clear"),
            classes="buttons-panel",
        )

        yield Horizontal(contacts_list, buttons_panel)
        yield Footer()

    def on_mount(self):
        self.title = "RP Contacts"
        self.sub_title = "A Contacts Book App With Textual & Python"
        self._load_contacts()

    def _load_contacts(self):
        """Load contacts from database into DataTable"""
        contacts_list = self.query_one(DataTable)
        for contact_data in self.db.get_all_contacts():
            id, *contact = contact_data
            contacts_list.add_row(*contact, key=id)

    def action_toggle_dark(self):
        self.dark = not self.dark

    def action_request_quit(self):
        def check_answer(accepted):
            if accepted:
                self.exit()

        self.push_screen(QuestionDialog("Do you want to quit?"), check_answer)
```

**Key Patterns:**

1. **DataTable Setup**: Configure columns, cursor, zebra stripes in `compose()`
2. **Database Integration**: Pass `Database` instance via constructor
3. **Load Pattern**: Unpack `(id, name, phone, email)`, use `id` as row key
4. **Bindings**: Map keyboard shortcuts to actions
5. **CSS Reference**: `CSS_PATH` links to external stylesheet

---

## Dialog Components

### Question Dialog (Confirmation)

```python
class QuestionDialog(Screen):
    def __init__(self, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message

    def compose(self):
        no_button = Button("No", variant="primary", id="no")
        no_button.focus()

        yield Grid(
            Label(self.message, id="question"),
            Button("Yes", variant="error", id="yes"),
            no_button,
            id="question-dialog",
        )

    def on_button_pressed(self, event):
        if event.button.id == "yes":
            self.dismiss(True)
        else:
            self.dismiss(False)
```

**Pattern**: Use `dismiss(value)` to return result from dialog screen.

### Input Dialog (Add Contact)

```python
class InputDialog(Screen):
    def compose(self):
        yield Grid(
            Label("Add Contact", id="title"),
            Label("Name:", classes="label"),
            Input(
                placeholder="Contact Name",
                classes="input",
                id="name",
            ),
            Label("Phone:", classes="label"),
            Input(
                placeholder="Contact Phone",
                classes="input",
                id="phone",
            ),
            Label("Email:", classes="label"),
            Input(
                placeholder="Contact Email",
                classes="input",
                id="email",
            ),
            Static(),
            Button("Cancel", variant="warning", id="cancel"),
            Button("Ok", variant="success", id="ok"),
            id="input-dialog",
        )

    def on_button_pressed(self, event):
        if event.button.id == "ok":
            name = self.query_one("#name", Input).value
            phone = self.query_one("#phone", Input).value
            email = self.query_one("#email", Input).value
            self.dismiss((name, phone, email))
        else:
            self.dismiss(())
```

**Key Patterns:**

1. **Input Retrieval**: Use `query_one("#id", Input).value` to get field values
2. **Return Data**: Return tuple `(name, phone, email)` or empty `()` on cancel
3. **Placeholders**: Guide users with placeholder text in Input widgets
4. **Grid Layout**: Organize labels and inputs in structured grid

---

## CRUD Operations

### Adding Contacts

```python
class ContactsApp(App):
    # ...

    @on(Button.Pressed, "#add")
    def action_add(self):
        def check_contact(contact_data):
            if contact_data:
                # Add to database
                self.db.add_contact(contact_data)
                # Get the newly added contact with ID
                id, *contact = self.db.get_last_contact()
                # Add to DataTable UI
                self.query_one(DataTable).add_row(*contact, key=id)

        self.push_screen(InputDialog(), check_contact)
```

**Flow**:
1. Launch InputDialog with `push_screen()`
2. Callback receives `(name, phone, email)` tuple or empty tuple
3. Insert into database if data provided
4. Retrieve contact with generated ID
5. Add to DataTable with matching key

**Critical**: Row `key` must match database `id` for deletion to work.

### Deleting Single Contact

```python
class ContactsApp(App):
    # ...

    @on(Button.Pressed, "#delete")
    def action_delete(self):
        contacts_list = self.query_one(DataTable)
        row_key, _ = contacts_list.coordinate_to_cell_key(
            contacts_list.cursor_coordinate
        )

        def check_answer(accepted):
            if accepted and row_key:
                # Delete from database
                self.db.delete_contact(id=row_key.value)
                # Remove from DataTable UI
                contacts_list.remove_row(row_key)

        name = contacts_list.get_row(row_key)[0]
        self.push_screen(
            QuestionDialog(f"Do you want to delete {name}'s contact?"),
            check_answer,
        )
```

**Key Techniques:**

1. **Get Selected Row**: Use `cursor_coordinate` + `coordinate_to_cell_key()`
2. **Extract ID**: Row key's `.value` property holds database ID
3. **Confirmation**: Show contact name in confirmation dialog
4. **Sync Delete**: Remove from database AND DataTable

### Clearing All Contacts

```python
class ContactsApp(App):
    # ...

    @on(Button.Pressed, "#clear")
    def action_clear_all(self):
        def check_answer(accepted):
            if accepted:
                # Clear database
                self.db.clear_all_contacts()
                # Clear DataTable UI
                self.query_one(DataTable).clear()

        self.push_screen(
            QuestionDialog("Are you sure you want to remove all contacts?"),
            check_answer,
        )
```

**Pattern**: Parallel operations on database and UI to maintain consistency.

---

## CSS Styling

```css
/* rpcontacts/rpcontacts.tcss */

/* Question Dialog */
QuestionDialog {
    align: center middle;
}

#question-dialog {
    grid-size: 2;
    grid-gutter: 1 2;
    grid-rows: 1fr 3;
    padding: 0 1;
    width: 60;
    height: 11;
    border: solid red;
    background: $surface;
}

#question {
    column-span: 2;
    height: 1fr;
    width: 1fr;
    content-align: center middle;
}

Button {
    width: 100%;
}

/* Contacts List */
.contacts-list {
    width: 3fr;
    padding: 0 1;
    border: solid green;
}

/* Buttons Panel */
.buttons-panel {
    align: center top;
    padding: 0 1;
    width: auto;
    border: solid red;
}

.separator {
    height: 1fr;
}

/* Input Dialog */
InputDialog {
    align: center middle;
}

#title {
    column-span: 3;
    height: 1fr;
    width: 1fr;
    content-align: center middle;
    color: green;
    text-style: bold;
}

#input-dialog {
    grid-size: 3 5;
    grid-gutter: 1 1;
    padding: 0 1;
    width: 50;
    height: 20;
    border: solid green;
    background: $surface;
}

.label {
    height: 1fr;
    width: 1fr;
    content-align: right middle;
}

.input {
    column-span: 2;
}
```

**Key CSS Patterns:**

1. **Center Dialogs**: `align: center middle` for Screen classes
2. **Grid Layouts**: `grid-size`, `grid-gutter`, `column-span` for structured forms
3. **Border Colors**: Visual distinction (green=list, red=danger actions)
4. **Responsive Sizing**: Use `fr` units for flexible layouts
5. **Variant Styling**: Button variants (success, warning, error) auto-styled

---

## Event Handling Patterns

### @on Decorator for Button Binding

```python
from textual.app import App, on

class ContactsApp(App):
    @on(Button.Pressed, "#add")
    def action_add(self):
        # Handle add button press
        pass

    @on(Button.Pressed, "#delete")
    def action_delete(self):
        # Handle delete button press
        pass
```

**Advantages**:
- Clean separation of event handlers
- Specific button targeting via CSS selector
- Auto-binds to action if method named `action_*`

### Callback Pattern for Dialogs

```python
def some_action(self):
    def process_result(result):
        if result:
            # Handle positive response
            pass

    self.push_screen(SomeDialog(), process_result)
```

**Pattern**: Inner function captures outer scope, processes dialog result.

---

## DataTable Integration

### Setup and Configuration

```python
contacts_list = DataTable(classes="contacts-list")
contacts_list.focus()
contacts_list.add_columns("Name", "Phone", "Email")
contacts_list.cursor_type = "row"  # Highlight entire rows
contacts_list.zebra_stripes = True  # Alternate row colors
```

### Adding Rows with Keys

```python
# Add single row
contacts_list.add_row("John Doe", "555-1234", "john@example.com", key=1)

# From database query
for contact_data in db.get_all_contacts():
    id, *contact = contact_data  # Unpack: id, name, phone, email
    contacts_list.add_row(*contact, key=id)
```

**Critical**: Use database ID as row `key` for later deletion/updates.

### Row Selection and Retrieval

```python
# Get selected row key
row_key, _ = contacts_list.coordinate_to_cell_key(
    contacts_list.cursor_coordinate
)

# Get row data
row_data = contacts_list.get_row(row_key)  # Returns tuple: (name, phone, email)
name = row_data[0]

# Remove row
contacts_list.remove_row(row_key)

# Clear all rows
contacts_list.clear()
```

---

## Form Validation Patterns

**Note**: This tutorial omits input validation for simplicity. Production apps should validate:

```python
class InputDialog(Screen):
    def on_button_pressed(self, event):
        if event.button.id == "ok":
            name = self.query_one("#name", Input).value
            phone = self.query_one("#phone", Input).value
            email = self.query_one("#email", Input).value

            # Add validation here:
            if not name.strip():
                # Show error message
                return

            if not self._validate_email(email):
                # Show error message
                return

            self.dismiss((name, phone, email))
        else:
            self.dismiss(())

    def _validate_email(self, email):
        # Email validation logic
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None
```

**Recommended Validations**:
- Name: Non-empty, reasonable length
- Phone: Format validation (regex or library)
- Email: Format validation (regex or library)
- All fields: SQL injection prevention (parameterized queries handle this)

---

## Testing the Database

```python
# Interactive testing
from rpcontacts.database import Database

db = Database()

# Add test data
data = [
    ("Linda", "111-2222-3333", "linda@example.com"),
    ("Joe", "111-2222-3333", "joe@example.com"),
    ("Lara", "111-2222-3333", "lara@example.com"),
]

for contact in data:
    db.add_contact(contact)

# Retrieve all
contacts = db.get_all_contacts()
print(contacts)
# Output: [(1, 'Linda', '111-2222-3333', 'linda@example.com'), ...]

# Get last added
last = db.get_last_contact()
print(last)
# Output: (3, 'Lara', '111-2222-3333', 'lara@example.com')

# Delete contact
db.delete_contact(1)

# Clear all
db.clear_all_contacts()
```

---

## Running the Application

```bash
# From project root
python -m rpcontacts
```

**Keyboard Shortcuts:**
- `M` - Toggle dark/light mode
- `A` - Add new contact
- `D` - Delete selected contact
- `C` - Clear all contacts
- `Q` - Quit application

---

## Complete Application Flow

### Add Contact Flow
1. User presses `A` or clicks "Add" button
2. `action_add()` launches `InputDialog`
3. User fills form fields (name, phone, email)
4. User clicks "Ok" button
5. Dialog returns `(name, phone, email)` tuple
6. Callback adds contact to database
7. Callback retrieves contact with generated ID
8. Callback adds contact to DataTable UI

### Delete Contact Flow
1. User selects row in DataTable
2. User presses `D` or clicks "Delete" button
3. `action_delete()` gets selected row key (database ID)
4. Confirmation dialog shows contact name
5. User clicks "Yes"
6. Callback deletes from database by ID
7. Callback removes row from DataTable UI

### Clear All Flow
1. User presses `C` or clicks "Clear All" button
2. `action_clear_all()` shows confirmation dialog
3. User clicks "Yes"
4. Callback executes `DELETE FROM contacts`
5. Callback clears DataTable UI

---

## Key Architectural Patterns

### 1. Separation of Concerns
- `database.py`: Pure SQLite operations
- `tui.py`: UI components and event handling
- `__main__.py`: Application entry point

### 2. Dependency Injection
```python
app = ContactsApp(db=Database())
```
Allows for easier testing and database swapping.

### 3. Screen Stack Management
```python
self.push_screen(DialogScreen(), callback)
```
Dialogs overlay main screen, dismissed with result.

### 4. Row Key Synchronization
```python
# When adding:
id, *contact = db.get_last_contact()
table.add_row(*contact, key=id)

# When deleting:
db.delete_contact(id=row_key.value)
table.remove_row(row_key)
```
Critical pattern for maintaining consistency between database and UI.

### 5. Inner Function Callbacks
```python
def action_add(self):
    def check_contact(contact_data):
        if contact_data:
            # Process contact
            pass

    self.push_screen(InputDialog(), check_contact)
```
Captures outer scope for clean dialog result processing.

---

## Extension Ideas

From the tutorial's "Next Steps" section:

### Add New Data Fields
```python
# Database schema expansion
CREATE TABLE contacts(
    id INTEGER PRIMARY KEY,
    name TEXT,
    phone TEXT,
    email TEXT,
    website TEXT,      -- New field
    birthday TEXT,     -- New field
    notes TEXT,        -- New field
    photo BLOB         -- New field (binary)
);

# UI expansion
contacts_list.add_columns("Name", "Phone", "Email", "Birthday")
```

### Search Capability
```python
class Database:
    def search_contacts(self, query):
        result = self._run_query(
            """SELECT * FROM contacts
               WHERE name LIKE ? OR email LIKE ? OR phone LIKE ?;""",
            f"%{query}%", f"%{query}%", f"%{query}%"
        )
        return result.fetchall()

# UI: Add Input widget for search
search_input = Input(placeholder="Search contacts...")
```

### Backup/Export Feature
```python
import shutil
import json

class Database:
    def backup_database(self, backup_path):
        """Create database file backup"""
        shutil.copy2(self.db_path, backup_path)

    def export_to_json(self, json_path):
        """Export contacts to JSON"""
        contacts = self.get_all_contacts()
        data = [
            {"id": c[0], "name": c[1], "phone": c[2], "email": c[3]}
            for c in contacts
        ]
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
```

### Edit Existing Contact
```python
class Database:
    def update_contact(self, id, contact):
        """Update existing contact"""
        self._run_query(
            "UPDATE contacts SET name=?, phone=?, email=? WHERE id=?;",
            *contact, id
        )

class EditDialog(Screen):
    def __init__(self, contact_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact_data = contact_data

    def compose(self):
        # Pre-populate Input widgets with existing data
        yield Input(value=self.contact_data[0], id="name")
        # ... etc
```

---

## Common Pitfalls

### 1. Row Key Mismatch
**Problem**: Adding rows without database ID as key breaks deletion.
```python
# Wrong:
table.add_row(name, phone, email)  # No key!

# Correct:
table.add_row(name, phone, email, key=database_id)
```

### 2. Missing Validation
**Problem**: Invalid data stored in database.
**Solution**: Validate all inputs before database insertion.

### 3. UI/Database Desync
**Problem**: Database updated but UI not refreshed (or vice versa).
**Solution**: Always update both in same code block:
```python
self.db.add_contact(contact_data)
id, *contact = self.db.get_last_contact()
self.query_one(DataTable).add_row(*contact, key=id)
```

### 4. Forgetting to Commit
**Problem**: Changes not persisted to database.
**Solution**: Always `commit()` after `execute()` (or use helper method).

---

## Performance Considerations

### Database Optimization
```python
# For large datasets, add indexes
CREATE INDEX idx_name ON contacts(name);
CREATE INDEX idx_email ON contacts(email);

# Use transactions for bulk operations
db.cursor.execute("BEGIN TRANSACTION")
for contact in large_contact_list:
    db.add_contact(contact)
db.db.commit()
```

### UI Optimization
```python
# For large contact lists, use pagination
contacts_list.loading = True  # Show loading indicator
# Load contacts in worker thread
contacts_list.loading = False
```

---

## Related Textual Concepts

**From this tutorial**:
- DataTable widget configuration and row management
- Dialog screens with `push_screen()` and `dismiss()`
- `@on` decorator for event binding
- CSS grid layouts for forms
- Input widgets with placeholders

**See also**:
- [Widgets Reference](../widgets/00-widgets-overview.md) - All Textual widgets
- [Screens Guide](../concepts/02-screens-navigation.md) - Screen management patterns
- [Events Guide](../concepts/04-events-messages.md) - Event system
- [CSS Guide](../styling/00-css-overview.md) - Textual CSS reference

---

## Sources

**Source Tutorial**:
- [Build a Contact Book App With Python, Textual, and SQLite](https://realpython.com/contact-book-python-textual/) - Real Python tutorial by Leodanis Pozo Ramos (accessed 2025-11-02)

**Related Documentation**:
- [Textual DataTable Widget](https://textual.textualize.io/widgets/data_table/)
- [Textual Screens Guide](https://textual.textualize.io/guide/screens/)
- [Python sqlite3 Documentation](https://docs.python.org/3/library/sqlite3.html)

**Additional References**:
- [SQLite CREATE TABLE](https://www.sqlite.org/lang_createtable.html)
- [SQLite Parameterized Queries](https://www.sqlite.org/lang_expr.html#varparam)

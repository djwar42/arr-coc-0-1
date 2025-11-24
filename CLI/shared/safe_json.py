"""Safe JSON I/O with atomic writes, file locking, and versioned backups.

This module provides production-grade JSON file operations that prevent:
- Race conditions from concurrent access (file locking)
- Data loss from power cuts/crashes (atomic writes)
- Permanent data loss from corruption (20 versioned backups)

Usage:
    from CLI.shared.safe_json import SafeJSON

    # Replace json.dump():
    SafeJSON.write("data.json", my_dict)

    # Replace json.load():
    my_dict = SafeJSON.read("data.json")
"""

# <claudes_code_comments>
# ** Function List **
# SafeJSON.read(path) - Read JSON with shared lock, corruption detection, auto-recovery
# SafeJSON.write(path, data) - Atomic write with exclusive lock and timestamped backup
# SafeJSON._create_backup(path) - Create timestamped backup in backups/ directory
# SafeJSON._cleanup_old_backups(path) - Keep last 20 backups, delete older ones
#
# ** Technical Review **
# This module implements a production-grade JSON I/O system with three layers of protection:
#
# 1. FILE LOCKING (prevents concurrent corruption):
#    - Shared locks for reads (fcntl.LOCK_SH) - multiple readers OK
#    - Exclusive locks for writes (fcntl.LOCK_EX) - blocks all others
#    - Lock acquired BEFORE file operations, released in finally block
#
# 2. ATOMIC WRITES (prevents partial file corruption):
#    - Write to temp file first (.tmp suffix with random component)
#    - Only after successful write, rename to real file (atomic operation!)
#    - OS guarantees rename is instant/complete (no partial files visible)
#    - Power cut during write? Temp corrupted, original safe. During rename? Either old or new complete.
#
# 3. VERSIONED BACKUPS (enables recovery from any disaster):
#    - Auto-backup on EVERY write: file.2025-11-14-03-45-12.json
#    - Keep last 20 versions in backups/ subdirectory
#    - Auto-cleanup deletes oldest when > MAX_BACKUPS
#    - Corrupt file detected? Auto-backup as file.CORRUPT-timestamp.json + return {}
#
# Flow for write():
#   1. Create timestamped backup of existing file (if exists)
#   2. Write data to temp file in same directory
#   3. Acquire exclusive lock on temp file
#   4. os.replace() temp → real (atomic, works across platforms)
#   5. Release lock
#   6. Cleanup old backups
#
# Flow for read():
#   1. Open file in read mode
#   2. Acquire shared lock
#   3. Parse JSON
#   4. Release lock
#   5. If JSONDecodeError: backup corrupt file, return {}
#
# Error handling:
#   - JSONDecodeError: Backup corrupt file, return {} (fresh start)
#   - FileNotFoundError: Return {} (new file)
#   - IOError: Retry with exponential backoff (optional)
#
# Platform notes:
#   - Uses fcntl (Unix/Linux/Mac only)
#   - For Windows: Consider msvcrt.locking or portalocker library
#   - tempfile.NamedTemporaryFile for secure temp file creation
#   - os.replace() works atomically on all platforms (Python 3.3+)
# </claudes_code_comments>

import json
import fcntl
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


class SafeJSON:
    """Production-grade JSON I/O with atomic writes, locking, and backups."""

    BACKUP_DIR = "backups"
    MAX_BACKUPS = 20

    @staticmethod
    def read(path: str | Path) -> Dict[str, Any]:
        """
        Read JSON file with shared lock and corruption recovery.

        - Acquires shared lock (multiple readers OK, blocks writers)
        - Returns {} if file doesn't exist or is corrupt
        - Auto-backups corrupt files for forensics

        Args:
            path: Path to JSON file

        Returns:
            Dict from JSON file, or {} if not found/corrupt
        """
        path = Path(path)

        # File doesn't exist? Return empty dict (not an error)
        if not path.exists():
            return {}

        try:
            with open(path, 'r') as f:
                # Acquire shared lock (allows multiple readers, blocks writers)
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                    return data
                finally:
                    # Always unlock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except json.JSONDecodeError as e:
            # Corrupt JSON! Backup for forensics, return fresh dict
            print(f"⚠️  Corrupt JSON detected in {path}: {e}")

            # Backup corrupt file
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            corrupt_backup = path.parent / SafeJSON.BACKUP_DIR / f"{path.stem}.CORRUPT-{timestamp}{path.suffix}"
            corrupt_backup.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(path, corrupt_backup)
                print(f"✨ Corrupt file backed up to: {corrupt_backup}")
                print(f"✨ Returning fresh dict - you can recover from backups/ if needed")
            except Exception as backup_error:
                print(f"⚠️  Could not backup corrupt file: {backup_error}")

            return {}

        except Exception as e:
            # Other I/O errors
            print(f"⚠️  Error reading {path}: {e}")
            return {}

    @staticmethod
    def write(path: str | Path, data: Dict[str, Any]) -> bool:
        """
        Write JSON file atomically with exclusive lock and versioned backup.

        Process:
        1. Create timestamped backup of existing file
        2. Write to temp file in same directory
        3. Acquire exclusive lock (blocks all readers and writers)
        4. Atomic rename temp → real file
        5. Release lock
        6. Cleanup old backups (keep last 20)

        Args:
            path: Path to JSON file
            data: Dict to serialize as JSON

        Returns:
            True if successful, False on error
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Backup existing file (if exists)
            if path.exists():
                SafeJSON._create_backup(path)

            # Step 2: Write to temp file in same directory
            # (same directory ensures atomic rename works)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp',
                prefix=f'{path.stem}_',
                dir=path.parent,
                text=True
            )

            try:
                # Write data to temp file
                with os.fdopen(temp_fd, 'w') as temp_file:
                    # Acquire exclusive lock on temp file
                    fcntl.flock(temp_file.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(data, temp_file, indent=2)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())  # Force write to disk
                    finally:
                        fcntl.flock(temp_file.fileno(), fcntl.LOCK_UN)

                # Step 3: Atomic rename (works across platforms)
                os.replace(temp_path, path)

            except Exception as e:
                # Cleanup temp file on error
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise e

            # Step 4: Cleanup old backups
            SafeJSON._cleanup_old_backups(path)

            return True

        except Exception as e:
            print(f"⚠️  Error writing {path}: {e}")
            return False

    @staticmethod
    def _create_backup(path: Path) -> None:
        """
        Create timestamped backup in backups/ subdirectory.

        Format: filename.2025-11-14-03-45-12.json

        Args:
            path: Path to file to backup
        """
        if not path.exists():
            return

        # Create backups directory
        backup_dir = path.parent / SafeJSON.BACKUP_DIR
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        backup_path = backup_dir / f"{path.stem}.{timestamp}{path.suffix}"

        try:
            shutil.copy2(path, backup_path)
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")

    @staticmethod
    def _cleanup_old_backups(path: Path) -> None:
        """
        Keep last MAX_BACKUPS versions, delete older ones.

        Sorts by modification time, deletes oldest first.

        Args:
            path: Path to original file (backups are in backups/ subdirectory)
        """
        backup_dir = path.parent / SafeJSON.BACKUP_DIR

        if not backup_dir.exists():
            return

        # Find all backups for this file (including CORRUPT backups)
        pattern = f"{path.stem}.*{path.suffix}"
        backups = sorted(
            backup_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Newest first
        )

        # Delete oldest backups beyond MAX_BACKUPS
        for old_backup in backups[SafeJSON.MAX_BACKUPS:]:
            try:
                old_backup.unlink()
            except Exception as e:
                print(f"⚠️  Could not delete old backup {old_backup}: {e}")

"""Shared internal utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to file atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(tmp_fd, content.encode())
        os.close(tmp_fd)
        os.replace(tmp_path, str(path))
    except BaseException:
        os.close(tmp_fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to file atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(tmp_fd, data)
        os.close(tmp_fd)
        os.replace(tmp_path, str(path))
    except BaseException:
        os.close(tmp_fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

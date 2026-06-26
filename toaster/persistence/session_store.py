"""Persisting the *session* (small, recomputable state) as JSON.

This is intentionally separate from the label sidecar. Labels are the
deliverable; the session is convenience state — which file is open, which schema,
the camera pose, the active class, recent files. Transient things (groupings, the
undo history) are deliberately *not* serialized: they are cheap to recompute.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["SessionState", "SessionStore"]


@dataclass
class SessionState:
    """Lightweight, JSON-serializable session preferences."""

    cloud_path: str | None = None
    schema_path: str | None = None
    active_class: int = 0
    camera: list[float] | None = None  # opaque viewer camera pose
    recent_files: list[str] = field(default_factory=list)

    def remember(self, path: str, limit: int = 10) -> None:
        """Move ``path`` to the front of the recent-files list."""
        self.recent_files = [path] + [p for p in self.recent_files if p != path]
        del self.recent_files[limit:]


class SessionStore:
    """Loads and saves a :class:`SessionState` at a fixed JSON path."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _default_path()

    def load(self) -> SessionState:
        """Load the session, returning defaults if the file is missing/corrupt."""
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return SessionState()
        known = {f for f in SessionState.__dataclass_fields__}
        return SessionState(**{k: v for k, v in data.items() if k in known})

    def save(self, state: SessionState) -> Path:
        """Write the session to disk and return its path."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(state), indent=2))
        return self.path


def _default_path() -> Path:
    """``~/.config/toaster/session.json`` (XDG-friendly)."""
    base = Path.home() / ".config" / "toaster"
    return base / "session.json"

"""Allow ``python -m toaster`` to launch the desktop app."""

from __future__ import annotations

import sys

from .desktop import main

if __name__ == "__main__":
    sys.exit(main())

"""Normals viewer package entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from .cli import app as cli_app

__all__ = ["main"]


def main() -> None:
    """Launch the graphical user interface."""

    from .gui import run_gui

    run_gui()


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    main()

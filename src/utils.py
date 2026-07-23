#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Shared utility functions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_TABLES = PROJECT_ROOT / "outputs" / "tables"


def get_project_tree(max_depth: int = 3) -> str:
    """Generate a text representation of the project directory tree.

    Excludes .git, .venv, __pycache__, and node_modules.
    """
    exclude = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache"}
    lines: list[str] = []

    def _walk(directory: Path, prefix: str = "", depth: int = 0) -> None:
        if depth > max_depth:
            return
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        entries = [e for e in entries if e.name not in exclude]
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)

    lines.append(directory.name if (directory := PROJECT_ROOT) else ".")
    _walk(PROJECT_ROOT)
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_project_tree())

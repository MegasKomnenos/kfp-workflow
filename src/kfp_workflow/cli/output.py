"""Structured output helpers for CLI commands."""

from __future__ import annotations

import json
from typing import Any, Sequence, Tuple

from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

# ---- Status color mapping ----

_RUN_STATE_STYLES = {
    "SUCCEEDED": "bold green",
    "RUNNING": "bold blue",
    "PENDING": "yellow",
    "FAILED": "bold red",
    "CANCELED": "dim red",
    "CANCELING": "dim red",
    "PAUSED": "yellow",
    "SKIPPED": "dim",
}

_ISVC_STATUS_STYLES = {
    "True": "bold green",
    "False": "bold red",
    "Unknown": "yellow",
}


def style_run_state(state: str) -> str:
    """Return a Rich-markup styled run state string."""
    style = _RUN_STATE_STYLES.get(state, "")
    return f"[{style}]{state}[/{style}]" if style else state


def style_isvc_ready(ready: str) -> str:
    """Return a Rich-markup styled InferenceService readiness string."""
    style = _ISVC_STATUS_STYLES.get(ready, "")
    return f"[{style}]{ready}[/{style}]" if style else ready


def print_json(data: Any) -> None:
    """Print data as syntax-highlighted JSON to stdout."""
    console.print_json(json.dumps(data, default=str))


def print_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    """Print a Rich table to stdout."""
    table = Table(title=title, show_lines=False)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def print_kv(pairs: Sequence[Tuple[str, str]]) -> None:
    """Print key-value pairs as a two-column detail view."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in pairs:
        table.add_row(key, value)
    console.print(table)

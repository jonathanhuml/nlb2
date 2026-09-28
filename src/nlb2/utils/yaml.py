"""Small YAML loading helper with a minimal fallback parser.

PyYAML is declared as a project dependency. The fallback is only for lightweight
local smoke tests in environments where the dependency has not been installed.
It supports the simple nested mapping style used by this repository's configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML recipes or JSON snapshots, preserving JSON numeric types."""

    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        data = yaml.safe_load(text)
        return {} if data is None else data
    except ModuleNotFoundError:
        return _parse_simple_mapping_yaml(text)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_mapping_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, value = stripped.split(":", 1)

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root

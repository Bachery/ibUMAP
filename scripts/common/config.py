from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _strip_comment(raw_line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(raw_line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return raw_line[:index].rstrip()
    return raw_line.rstrip()


def simple_yaml_load(text: str) -> Any:
    """Parse the small YAML subset used by the experiment configs (fallback when PyYAML is missing)."""
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        cleaned = _strip_comment(raw_line)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return None, index
        current_indent, current_text = lines[index]
        if current_indent < indent:
            return None, index
        if current_text.startswith("- "):
            return parse_list(index, current_indent)
        return parse_map(index, current_indent)

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while index < len(lines):
            line_indent, text = lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or not text.startswith("- "):
                break
            payload = text[2:].strip()
            index += 1
            if payload == "":
                value, index = parse_block(index, indent + 2)
                items.append(value)
            elif ":" in payload:
                key, raw_value = payload.split(":", 1)
                item: dict[str, Any] = {key.strip(): parse_scalar(raw_value.strip())}
                if index < len(lines) and lines[index][0] > indent:
                    nested, index = parse_map(index, lines[index][0])
                    if isinstance(nested, dict):
                        item.update(nested)
                items.append(item)
            else:
                items.append(parse_scalar(payload))
        return items, index

    def parse_map(index: int, indent: int) -> tuple[dict[str, Any], int]:
        mapping: dict[str, Any] = {}
        while index < len(lines):
            line_indent, text = lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or text.startswith("- "):
                break
            if ":" not in text:
                raise ValueError(f"Invalid YAML line: {text}")
            key, raw_value = text.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            index += 1
            if raw_value:
                mapping[key] = parse_scalar(raw_value)
            elif index < len(lines) and lines[index][0] > indent:
                mapping[key], index = parse_block(index, lines[index][0])
            else:
                mapping[key] = None
        return mapping, index

    if not lines:
        return {}
    data, next_index = parse_block(0, lines[0][0])
    if next_index != len(lines):
        raise ValueError("Could not parse entire YAML document")
    return data


def load_yaml(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return simple_yaml_load(text)
    return yaml.safe_load(text)


def resolve_path(value: str | Path | None, *, base_dir: str | Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


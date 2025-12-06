"""
Utilities for splitting JSON templates by measurements for batch inference.

Aligned with inference_with_template_split.ipynb notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
PathT = tuple[str, ...]


def is_measurement_node(node: Any) -> bool:
    """
    Check if a node is a measurement (leaf dict with 'valeur' key).
    """
    return isinstance(node, dict) and "valeur" in node


def normalize_consecutive_duplicates(path: PathT) -> PathT:
    """Remove consecutive duplicates in path ('A','A','B' -> 'A','B')."""
    if not path:
        return path
    norm = [path[0]]
    for k in path[1:]:
        if k != norm[-1]:
            norm.append(k)
    return tuple(norm)


def collect_measurements(node: Any, base_path: PathT = ()) -> list[tuple[PathT, JsonDict]]:
    """
    Recursively traverse JSON and return list of [(path, leaf_dict), ...]
    where leaf_dict is a measurement object (dict with 'valeur').
    """
    out: list[tuple[PathT, JsonDict]] = []

    if is_measurement_node(node):
        out.append((base_path, node))
        return out

    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(collect_measurements(v, base_path + (k,)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(collect_measurements(v, base_path + (f"[{i}]",)))
    
    return out


def insert_path(root: JsonDict, path: PathT, value: JsonDict, dedup: bool = True) -> None:
    """Insert value into root at path (creating sub-dicts as needed)."""
    if dedup:
        path = normalize_consecutive_duplicates(path)
    if not path:
        root.update(value)
        return

    cur = root
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def split_dict_by_measurements(
    data: JsonDict,
    max_objects_per_part: int,
    dedup_consecutive: bool = True,
) -> list[JsonDict]:
    """
    Split a JSON dict into parts of <= max_objects_per_part measurement leaves.
    
    Each part contains only the branches necessary to reach its leaves.
    
    Args:
        data: Input JSON dictionary.
        max_objects_per_part: Maximum number of measurement nodes per part.
        dedup_consecutive: Whether to remove consecutive duplicate keys in paths.
    
    Returns:
        List of partial JSON dictionaries.
    """
    if max_objects_per_part <= 0:
        raise ValueError("max_objects_per_part must be > 0")

    leaves = collect_measurements(data)
    if not leaves:
        return [{}]

    parts: list[JsonDict] = []
    for i in range(0, len(leaves), max_objects_per_part):
        chunk = leaves[i:i + max_objects_per_part]
        partial: JsonDict = {}
        for path, leaf in chunk:
            insert_path(partial, path, leaf, dedup=dedup_consecutive)
        parts.append(partial)

    return parts


def split_template_by_measurements(
    template: JsonDict | str | Path,
    max_objects_per_part: int,
    dedup_consecutive: bool = True,
) -> list[str]:
    """
    Split a JSON template into parts by measurements, returning JSON strings.
    
    Args:
        template: JSON dict, JSON string, or file path to JSON template.
        max_objects_per_part: Maximum number of measurement nodes per part.
        dedup_consecutive: Whether to remove consecutive duplicate keys.
    
    Returns:
        List of JSON strings, one per part.
    """
    # Load template if needed
    if isinstance(template, (str, Path)):
        path = Path(template)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            # Try parsing as JSON string
            data = json.loads(template)
    elif isinstance(template, str):
        data = json.loads(template)
    else:
        data = template

    parts = split_dict_by_measurements(data, max_objects_per_part, dedup_consecutive)
    
    return [json.dumps(part, ensure_ascii=False, indent=2) for part in parts]


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two JSON dictionaries.
    
    b takes precedence over a for conflicting keys.
    """
    result = dict(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


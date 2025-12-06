from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from amalytics_ml.config import EvalConfig


def _normalize_scalar(value: Any, cfg: EvalConfig) -> Any:
    """Normalize scalars for comparison."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if cfg.numeric_precision is None:
            return float(value)
        return round(float(value), cfg.numeric_precision)

    text = str(value)
    if cfg.strip_whitespace:
        text = " ".join(text.split())
    if not cfg.case_sensitive:
        text = text.lower()
    return text


def _should_ignore(path: Sequence[str], ignored: set[str], separator: str) -> bool:
    joined = separator.join(path)
    return joined in ignored


def flatten_structure(
    data: Any,
    cfg: EvalConfig,
    path: tuple[str, ...] = (),
    ignored: set[str] | None = None,
) -> dict[str, Any]:
    """
    Flatten nested dict/list structures into {path: value} pairs where leaves are scalars.
    """
    ignored = ignored or set()
    if _should_ignore(path, ignored, cfg.path_separator):
        return {}

    if isinstance(data, Mapping):
        flattened: dict[str, Any] = {}
        for key, value in data.items():
            flattened.update(flatten_structure(value, cfg, path + (str(key),), ignored))
        return flattened

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        flattened: dict[str, Any] = {}
        for idx, value in enumerate(data):
            flattened.update(flatten_structure(value, cfg, path + (f"[{idx}]",), ignored))
        return flattened

    joined = cfg.path_separator.join(path)
    return {joined: data}


def confusion_counts_for_record(
    prediction: Mapping[str, Any],
    reference: Mapping[str, Any],
    cfg: EvalConfig,
) -> dict[str, int]:
    """
    Compute TP/FP/FN counts for a single prediction vs reference pair.
    """
    ignored = set(cfg.ignore_keys)
    pred_flat = flatten_structure(prediction, cfg, ignored=ignored)
    ref_flat = flatten_structure(reference, cfg, ignored=ignored)

    keys = set(pred_flat) | set(ref_flat)
    counts = {"tp": 0, "fp": 0, "fn": 0}

    for key in keys:
        truth_val = _normalize_scalar(ref_flat.get(key), cfg)
        pred_val = _normalize_scalar(pred_flat.get(key), cfg)

        if truth_val is None and cfg.ignore_null_ground_truth:
            if pred_val is not None:
                counts["fp"] += 1
            continue

        if truth_val is None:
            if pred_val is not None:
                counts["fp"] += 1
            continue

        if pred_val is None:
            counts["fn"] += 1
        elif pred_val == truth_val:
            counts["tp"] += 1
        else:
            counts["fp"] += 1
            counts["fn"] += 1

    return counts


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


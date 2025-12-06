from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from amalytics_ml.config import SyntheticDataConfig


def _default_schema() -> Mapping[str, Any]:
    return {}


def generate_synthetic_blood_tests(
    n: int,
    config: SyntheticDataConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Generate n synthetic blood test reports as structured dictionaries.

    Args:
        n: Number of synthetic reports to create.
        config: Optional SyntheticDataConfig. If None, a minimal default is used.

    Returns:
        List of length n containing structured blood test dictionaries.
    """

    cfg = config or SyntheticDataConfig(schema=_default_schema())
    reports: list[dict[str, Any]] = []

    for _ in range(n):
        report: dict[str, Any] = {}
        for category, subcats in cfg.schema.items():
            report[category] = {}
            for subcat, params in subcats.items():
                report[category][subcat] = {}
                for name, rng in params.items():
                    min_val = rng.get("min", 0)
                    max_val = rng.get("max", 1)
                    value = random.uniform(min_val, max_val)
                    report[category][subcat][name] = round(value, 2)

        reports.append(report)

    return reports


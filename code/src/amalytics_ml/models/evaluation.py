from __future__ import annotations

from typing import Any

from amalytics_ml.config import EvalConfig
from amalytics_ml.utils.metrics import confusion_counts_for_record, safe_divide


def _counts_to_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_model(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    config: EvalConfig | None = None,
) -> dict[str, float]:
    """
    Compute precision/recall/F1 for structured JSON predictions vs ground truth.

    Args:
        predictions: List of model outputs (already parsed JSON dictionaries).
        ground_truth: List of ground-truth dictionaries aligned with predictions.
        config: Optional evaluation configuration; defaults applied if None.

    Returns:
        Dictionary containing micro-average precision/recall/F1. When config.average == "macro",
        additional macro metrics are included (precision_macro, recall_macro, f1_macro).
    """

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground_truth must have the same length.")

    cfg = config or EvalConfig()

    total_tp = total_fp = total_fn = 0
    macro_precisions: list[float] = []
    macro_recalls: list[float] = []
    macro_f1s: list[float] = []

    for pred, truth in zip(predictions, ground_truth):
        counts = confusion_counts_for_record(pred, truth, cfg)
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if cfg.average == "macro":
            prec, rec, f1 = _counts_to_metrics(tp, fp, fn)
            macro_precisions.append(prec)
            macro_recalls.append(rec)
            macro_f1s.append(f1)

    precision_micro, recall_micro, f1_micro = _counts_to_metrics(total_tp, total_fp, total_fn)
    metrics: dict[str, float] = {
        "precision": precision_micro,
        "recall": recall_micro,
        "f1": f1_micro,
    }

    if cfg.average == "macro" and macro_precisions:
        metrics.update(
            {
                "precision_macro": sum(macro_precisions) / len(macro_precisions),
                "recall_macro": sum(macro_recalls) / len(macro_recalls),
                "f1_macro": sum(macro_f1s) / len(macro_f1s),
            }
        )

    return metrics


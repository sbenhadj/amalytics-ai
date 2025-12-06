import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amalytics_ml.config import EvalConfig
from amalytics_ml.models.evaluation import evaluate_model


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a list of JSON objects from a .json or .jsonl file.

    - For .json: expects either a list[dict] or a dict (wrapped as [dict]).
    - For .jsonl: one JSON object per line.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # If it's a single dict, wrap it into a list for consistency
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError(
            f"Unsupported JSON structure in {path}: "
            f"expected dict or list[dict], got {type(data)}"
        )

    raise ValueError(
        f"Unsupported file extension for {path}. "
        f"Use .json or .jsonl for predictions and ground truth."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Amalytics LLM on structured medical outputs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to EvalConfig JSON file (paths & options for evaluation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config JSON
    with config_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # Instantiate EvalConfig
    cfg = EvalConfig(**cfg_dict)

    # Load predictions and ground truth
    preds_path = Path(cfg.predictions_path)
    gt_path = Path(cfg.ground_truth_path)

    predictions = load_json_or_jsonl(preds_path)
    ground_truth = load_json_or_jsonl(gt_path)

    # Sanity check: same length
    if len(predictions) != len(ground_truth):
        print(
            f"WARNING: Number of predictions ({len(predictions)}) "
            f"differs from ground truth ({len(ground_truth)}). "
            f"Evaluation may be misaligned."
        )

    # Run evaluation
    metrics = evaluate_model(predictions, ground_truth, cfg)

    # Pretty-print metrics
    print("=== Evaluation metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Optionally save metrics to file
    metrics_output_path = getattr(cfg, "metrics_output_path", None)
    if metrics_output_path:
        out_path = Path(metrics_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()

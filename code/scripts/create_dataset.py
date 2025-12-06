"""
CLI script to create training dataset from PDF reports and JSON templates.

Usage:
    python scripts/create_dataset.py \
        --reports ./data/reports \
        --filled ./data/templates/filled \
        --empty ./data/templates/empty \
        --output ./data/dataset.jsonl
"""

import argparse
import sys
from pathlib import Path

# --- Resolve project root and add src/ to sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # points to `code/`
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amalytics_ml.data.preprocessing import create_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create training dataset (JSONL) from PDF reports and JSON templates."
    )
    parser.add_argument(
        "--reports",
        required=True,
        help="Directory containing PDF report files.",
    )
    parser.add_argument(
        "--filled",
        required=True,
        help="Directory containing filled JSON templates (ground truth annotations).",
    )
    parser.add_argument(
        "--empty",
        required=True,
        help="Directory containing empty JSON templates (input templates).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the JSONL dataset file.",
    )
    args = parser.parse_args()

    # --- Resolve paths: try CWD first, then ROOT_DIR ---
    def resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        # Try from current working directory first
        if path.exists():
            return path.resolve()
        # Try from project root (code/)
        candidate = ROOT_DIR / path_str
        if candidate.exists():
            return candidate.resolve()
        # If neither exists, return resolved relative to CWD (will be created)
        return path.resolve()

    report_dir = resolve_path(args.reports)
    filled_dir = resolve_path(args.filled)
    empty_dir = resolve_path(args.empty)
    output_path = resolve_path(args.output)

    # --- Validate input directories exist ---
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")
    if not filled_dir.exists():
        raise FileNotFoundError(f"Filled templates directory not found: {filled_dir}")
    if not empty_dir.exists():
        raise FileNotFoundError(f"Empty templates directory not found: {empty_dir}")

    print(f"Report directory: {report_dir}")
    print(f"Filled templates: {filled_dir}")
    print(f"Empty templates: {empty_dir}")
    print(f"Output dataset: {output_path}")
    print()

    # --- Create dataset ---
    create_training_dataset(
        report_dir=report_dir,
        filled_template_dir=filled_dir,
        empty_template_dir=empty_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()


"""
CLI script for complete model evaluation pipeline.

This script automates the entire evaluation process:
1. Loads a test dataset (PDFs + ground truth + templates)
2. Generates predictions for all test samples using inference
3. Calculates evaluation metrics (precision, recall, F1)
4. Saves predictions and metrics

Usage:
    python scripts/eval_full.py --config configs/eval_full_dummy.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from natsort import natsorted

# --- Resolve project root and add src/ to sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # points to `code/`
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amalytics_ml.config import FullEvalConfig, InferenceConfig, EvalConfig
from amalytics_ml.models.inference import run_inference
from amalytics_ml.models.evaluation import evaluate_model


def extract_base(filename: str) -> str:
    """Return filename without extension."""
    return os.path.splitext(filename)[0]


def normalize_name(filename: str) -> str:
    """
    Normalize filename by removing extension and common suffixes.
    
    Removes .json/.pdf extension and _template, _filled, _empty suffixes.
    """
    base = os.path.splitext(filename)[0]
    return base.replace("_template", "").replace("_filled", "").replace("_empty", "")


def resolve_path(path_str: str, root_dir: Path) -> Path:
    """Resolve a path string, trying CWD first, then root_dir."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    # Try from current working directory first
    if path.exists():
        return path.resolve()
    # Try from project root
    candidate = root_dir / path_str
    if candidate.exists():
        return candidate.resolve()
    # Return resolved path (may not exist yet, will be created)
    return path.resolve()


def is_hf_repo_id(path_str: str) -> bool:
    """Check if a path string is a HuggingFace model repository ID."""
    return "/" in path_str and not path_str.startswith(("./", "../", "/"))


def find_matching_files(
    base_name: str,
    directory: Path,
    extension: str,
    normalize: bool = True,
) -> Path | None:
    """
    Find a file matching a base name in a directory.
    
    Args:
        base_name: Base name to match (without extension).
        directory: Directory to search in.
        extension: File extension (e.g., '.pdf', '.json').
        normalize: Whether to normalize filenames before matching.
    
    Returns:
        Path to matching file or None if not found.
    """
    if not directory.exists():
        return None
    
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
        
        if not file_path.suffix.lower() == extension.lower():
            continue
        
        file_base = extract_base(file_path.name)
        if normalize:
            file_base = normalize_name(file_base)
            compare_base = normalize_name(base_name)
        else:
            compare_base = base_name
        
        if file_base == compare_base:
            return file_path
    
    return None


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete evaluation pipeline: inference + metrics for a test dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to FullEvalConfig JSON file (relative to project root or absolute).",
    )
    args = parser.parse_args()
    
    # --- Resolve config path ---
    config_path = resolve_path(args.config, ROOT_DIR)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # --- Load FullEvalConfig from JSON ---
    with config_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    
    cfg = FullEvalConfig(**cfg_dict)
    
    print("=" * 60)
    print("Full Evaluation Pipeline")
    print("=" * 60)
    
    # --- Resolve dataset directories ---
    test_reports_dir = resolve_path(cfg.test_reports_dir, ROOT_DIR)
    test_gt_dir = resolve_path(cfg.test_ground_truth_dir, ROOT_DIR)
    test_empty_dir = None
    if cfg.test_empty_templates_dir:
        test_empty_dir = resolve_path(cfg.test_empty_templates_dir, ROOT_DIR)
    
    if not test_reports_dir.exists():
        raise FileNotFoundError(f"Test reports directory not found: {test_reports_dir}")
    if not test_gt_dir.exists():
        raise FileNotFoundError(f"Test ground truth directory not found: {test_gt_dir}")
    
    # --- Find all PDF files in test reports directory ---
    pdf_files = natsorted([f for f in test_reports_dir.iterdir() if f.suffix.lower() == ".pdf"])
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {test_reports_dir}")
    
    print(f"\n📁 Found {len(pdf_files)} test PDFs in {test_reports_dir}")
    
    # --- Resolve model and LoRA paths ---
    if is_hf_repo_id(cfg.model_path):
        model_path_str = cfg.model_path
    else:
        model_path = resolve_path(cfg.model_path, ROOT_DIR)
        model_path_str = str(model_path)
    
    lora_path_str = ""
    if cfg.lora_path:
        lora_path = resolve_path(cfg.lora_path, ROOT_DIR)
        lora_path_str = str(lora_path)
    
    # --- Load or resolve template ---
    template_obj = cfg.template
    if template_obj is None and cfg.template_path:
        template_path = resolve_path(cfg.template_path, ROOT_DIR)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        template_obj = load_json_file(template_path)
    
    if template_obj is None:
        raise ValueError(
            "Template must be provided via 'template' or 'template_path' in config."
        )
    
    # --- Convert template to dict if it's a string ---
    if isinstance(template_obj, str):
        try:
            template_obj = json.loads(template_obj)
        except json.JSONDecodeError:
            raise ValueError(f"Template string is not valid JSON: {template_obj[:100]}...")
    
    if not isinstance(template_obj, dict):
        raise TypeError(f"Template must be a dict, got {type(template_obj)}")
    
    # --- Create output directory for predictions ---
    predictions_output_dir = resolve_path(cfg.predictions_output_dir, ROOT_DIR)
    predictions_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Predictions will be saved to: {predictions_output_dir}")
    
    # --- Build InferenceConfig from FullEvalConfig ---
    inference_cfg = InferenceConfig(
        model_path=cfg.model_path,
        lora_path=cfg.lora_path,
        template=template_obj,
        bos_token=cfg.bos_token,
        system_tag=cfg.system_tag,
        user_tag=cfg.user_tag,
        eot_token=cfg.eot_token,
        system_prompt=cfg.system_prompt,
        user_prompt_template=cfg.user_prompt_template,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        return_scores=cfg.return_scores,
        eos_token_id=cfg.eos_token_id,
        pad_token_id=cfg.pad_token_id,
        load_in_4bit=cfg.load_in_4bit,
        bnb_compute_dtype=cfg.bnb_compute_dtype,
        device_map=cfg.device_map,
        extra_generation_kwargs=cfg.extra_generation_kwargs,
        use_batch_inference=cfg.use_batch_inference,
        max_measurements_per_batch=cfg.max_measurements_per_batch,
        dedup_consecutive_keys=cfg.dedup_consecutive_keys,
        apply_anonymization=cfg.apply_anonymization,
        anonymization_secret_key=cfg.anonymization_secret_key,
        anonymization_use_ner=cfg.anonymization_use_ner,
        anonymization_ner_model_path=cfg.anonymization_ner_model_path,
        anonymization_output_dir=cfg.anonymization_output_dir,
    )
    
    # --- Generate predictions for each PDF ---
    print(f"\n🚀 Starting inference on {len(pdf_files)} test samples...")
    print("-" * 60)
    
    predictions: List[Dict[str, Any]] = []
    ground_truths: List[Dict[str, Any]] = []
    prediction_files: List[Path] = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_base = extract_base(pdf_file.name)
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        # Find matching ground truth
        gt_file = find_matching_files(pdf_base, test_gt_dir, ".json", normalize=True)
        if gt_file is None:
            print(f"  ⚠️  Warning: No ground truth found for {pdf_file.name}, skipping...")
            skipped_count += 1
            continue
        
        # Load ground truth
        try:
            gt_dict = load_json_file(gt_file)
        except Exception as e:
            print(f"  ❌ Error loading ground truth {gt_file.name}: {e}, skipping...")
            skipped_count += 1
            continue
        
        # Find matching empty template (if per-file templates are used)
        empty_template = template_obj  # Default: use shared template
        if test_empty_dir:
            empty_template_file = find_matching_files(
                pdf_base, test_empty_dir, ".json", normalize=True
            )
            if empty_template_file:
                try:
                    empty_template = load_json_file(empty_template_file)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not load empty template {empty_template_file.name}: {e}, using shared template")
        
        # Update inference config with this template
        inference_cfg.template = empty_template
        
        # Run inference
        try:
            result = run_inference(
                model_path=model_path_str,
                lora_path=lora_path_str,
                input_text=pdf_file,
                config=inference_cfg,
            )
            
            predictions.append(result.parsed_json)
            ground_truths.append(gt_dict)  # Only add ground truth if inference succeeded
            
            # Save prediction to file
            pred_file = predictions_output_dir / f"{pdf_base}_prediction.json"
            with pred_file.open("w", encoding="utf-8") as f:
                json.dump(result.parsed_json, f, ensure_ascii=False, indent=2)
            prediction_files.append(pred_file)
            
            processed_count += 1
            print(f"  ✅ Prediction saved to {pred_file.name}")
            
        except Exception as e:
            error_count += 1
            print(f"  ❌ Error during inference: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            # Skip this sample - don't add prediction or ground truth to maintain alignment
            continue
    
    print("-" * 60)
    print(f"\n📊 Inference Summary:")
    print(f"   ✅ Successfully processed: {processed_count}")
    print(f"   ⚠️  Skipped (no ground truth): {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📝 Total predictions: {len(predictions)}")
    
    # --- Verify alignment ---
    if len(predictions) != len(ground_truths):
        print(
            f"\n❌ ERROR: Mismatch between predictions ({len(predictions)}) "
            f"and ground truth ({len(ground_truths)}). This should not happen!"
        )
        raise ValueError(
            f"Predictions and ground truth are not aligned: "
            f"{len(predictions)} predictions vs {len(ground_truths)} ground truth"
        )
    
    if not predictions:
        raise ValueError("No predictions were generated. Cannot compute metrics.")
    
    # --- Build EvalConfig from FullEvalConfig ---
    eval_cfg = EvalConfig(
        predictions_path="",  # Not used, we pass predictions directly
        ground_truth_path="",  # Not used, we pass ground truth directly
        ignore_keys=cfg.ignore_keys,
        case_sensitive=cfg.case_sensitive,
        strip_whitespace=cfg.strip_whitespace,
        numeric_precision=cfg.numeric_precision,
        ignore_null_ground_truth=cfg.ignore_null_ground_truth,
        path_separator=cfg.path_separator,
        average=cfg.average,
        metrics_output_path=cfg.metrics_output_path,
    )
    
    # --- Calculate metrics ---
    print(f"\n📊 Calculating evaluation metrics...")
    print("-" * 60)
    
    metrics = evaluate_model(predictions, ground_truths, eval_cfg)
    
    # --- Display results ---
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    
    # --- Save metrics to file ---
    if cfg.metrics_output_path:
        metrics_path = resolve_path(cfg.metrics_output_path, ROOT_DIR)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Metrics saved to: {metrics_path}")
    
    # --- Save predictions list (JSONL format) for easy loading later ---
    predictions_jsonl_path = predictions_output_dir / "all_predictions.jsonl"
    with predictions_jsonl_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"💾 All predictions saved as JSONL to: {predictions_jsonl_path}")
    
    # --- Save ground truth list (JSONL format) ---
    gt_jsonl_path = predictions_output_dir / "all_ground_truth.jsonl"
    with gt_jsonl_path.open("w", encoding="utf-8") as f:
        for gt in ground_truths:
            f.write(json.dumps(gt, ensure_ascii=False) + "\n")
    print(f"💾 All ground truth saved as JSONL to: {gt_jsonl_path}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


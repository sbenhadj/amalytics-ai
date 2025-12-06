import argparse
import json
import sys
from pathlib import Path

import pdfplumber
from huggingface_hub import login

# Authenticate with HuggingFace
login("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")

# --- Resolve project root and add src/ to sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # points to `code/`
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amalytics_ml.config import InferenceConfig
from amalytics_ml.models.inference import run_inference


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pdfplumber."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Amalytics LLaMA-based inference on a medical report."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to InferenceConfig JSON file (relative to project root or absolute).",
    )
    parser.add_argument(
        "--output",
        help="Base path for output files (without extension). "
             "Will create <output>_result.json and <output>_confidence.json",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        help="Raw extracted text from PDF (passed directly as a string).",
    )
    group.add_argument(
        "--report-pdf",
        dest="report_pdf",
        help="Path to a PDF report; text will be extracted automatically.",
    )
    args = parser.parse_args()

    # --- Resolve config path: try CWD first, then ROOT_DIR ---
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Check if it exists relative to CWD
        if not config_path.exists():
            # Fall back to ROOT_DIR
            config_path = ROOT_DIR / args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # --- Load InferenceConfig from JSON ---
    with config_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = InferenceConfig(**cfg_dict)

    # --- Resolve model path ---
    # HuggingFace repo IDs look like "org/model-name" - don't resolve them as file paths
    def is_hf_repo_id(path_str: str) -> bool:
        return "/" in path_str and not path_str.startswith(("./", "../", "/"))

    if is_hf_repo_id(cfg.model_path):
        # It's a HuggingFace model ID, keep as-is
        model_path_str = cfg.model_path
    else:
        model_path = Path(cfg.model_path)
        if not model_path.is_absolute():
            if not model_path.exists():
                model_path = ROOT_DIR / cfg.model_path
        # If model_path is a run directory, try to locate a checkpoint with config.json
        if model_path.exists() and not (model_path / "config.json").exists():
            candidate_root = model_path / "llama-3-lora-finetuned"
            if candidate_root.exists():
                checkpoints = sorted(candidate_root.glob("checkpoint-*"))
                for ckpt in reversed(checkpoints):
                    if (ckpt / "config.json").exists():
                        model_path = ckpt
                        break
        model_path_str = str(model_path)

    # --- Resolve LoRA path: try CWD first, then ROOT_DIR (skip if empty) ---
    lora_path_str = ""
    if cfg.lora_path:
        lora_path = Path(cfg.lora_path)
        if not lora_path.is_absolute():
            if not lora_path.exists():
                lora_path = ROOT_DIR / cfg.lora_path
        lora_path_str = str(lora_path)

    # --- Ensure we actually have a template loaded ---
    if cfg.template is None and cfg.template_path:
        template_path = Path(cfg.template_path)
        if not template_path.is_absolute():
            if not template_path.exists():
                template_path = ROOT_DIR / cfg.template_path

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with template_path.open("r", encoding="utf-8") as f:
            cfg.template = json.load(f)

    if cfg.template is None:
        raise ValueError(
            "InferenceConfig must provide 'template' or 'template_path'. "
            "Template is still None after loading."
        )

    # --- Enable return_scores if output is requested ---
    if args.output:
        cfg.return_scores = True

    # --- Determine input text ---
    if args.report_pdf:
        pdf_path = Path(args.report_pdf)
        if not pdf_path.is_absolute():
            if not pdf_path.exists():
                pdf_path = ROOT_DIR / args.report_pdf
        input_text = extract_text_from_pdf(pdf_path)
    else:
        input_text = args.text or ""

    # --- Run inference ---
    result = run_inference(
        model_path=model_path_str,
        lora_path=lora_path_str,
        input_text=input_text,
        config=cfg,
    )

    # --- Output results ---
    if args.output:
        output_base = Path(args.output)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filled template JSON
        result_path = output_base.parent / f"{output_base.stem}_result.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result.parsed_json, f, ensure_ascii=False, indent=2)
        print(f"Filled template saved to: {result_path}")
        
        # Save confidence scores JSON
        if result.confidence_scores:
            confidence_path = output_base.parent / f"{output_base.stem}_confidence.json"
            with confidence_path.open("w", encoding="utf-8") as f:
                json.dump(result.confidence_scores, f, ensure_ascii=False, indent=2)
            print(f"Confidence scores saved to: {confidence_path}")
        else:
            print("Warning: No confidence scores available (return_scores may be disabled)")
    else:
        # Print JSON result to stdout
        print(json.dumps(result.parsed_json, ensure_ascii=False, indent=2))
        if result.confidence_scores:
            print("\n--- Confidence Scores ---")
            print(json.dumps(result.confidence_scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
Dataset preprocessing for training.

Creates training dataset (JSONL format) from PDF reports and JSON templates.
Aligned with the original finetuning.ipynb notebook (Cell 17).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from natsort import natsorted


# LLaMA 3.1 chat format tokens
BOS = "<|begin_of_text|>"
SYS = "<|start_header_id|>system<|end_header_id|>"
USER = "<|start_header_id|>user<|end_header_id|>"
ASSIST = "<|start_header_id|>assistant<|end_header_id|>"
EOT = "<|eot_id|>"

SYSTEM_PROMPT = (
    "You are a precise and reliable medical AI assistant. "
    "Your task is to extract relevant entities from a medical report and return a filled JSON object based on the provided template. "
    "Ensure the output is valid JSON and matches the structure of the template. "
    "Only use values found in the report, respect the required fields and units and keep fields unchanged if no value is found."
)

USER_PROMPT_TEMPLATE = (
    "Extract relevant information from the following medical report to fill fields in the JSON template.\n\n"
    "Medical Report:\n{}\n\n"
    "JSON Template:\n{}\n\n"
    "Your response must only contain the completed JSON object, nothing else."
)


@dataclass
class DatasetCreationConfig:
    """Configuration for creating training dataset."""
    report_dir: str | Path
    filled_template_dir: str | Path
    empty_template_dir: str | Path
    output_path: str | Path
    bos_token: str = BOS
    system_tag: str = SYS
    user_tag: str = USER
    assistant_tag: str = ASSIST
    eot_token: str = EOT
    system_prompt: str = SYSTEM_PROMPT
    user_prompt_template: str = USER_PROMPT_TEMPLATE


def extract_base(filename: str) -> str:
    """Return filename without extension."""
    return os.path.splitext(filename)[0]


def normalize_annotation_name(filename: str) -> str:
    """
    Normalize annotation filename.
    
    Removes .json extension and _template, _filled, _empty suffixes.
    """
    base = os.path.splitext(filename)[0]
    return base.replace("_template", "").replace("_filled", "").replace("_empty", "")


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract plain text from a PDF using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        Extracted text content.
    """
    import pdfplumber
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def build_training_prompt(
    report_text: str,
    empty_template: dict[str, Any],
    filled_template: dict[str, Any],
    cfg: DatasetCreationConfig | None = None,
) -> str:
    """
    Build a complete training prompt from report and templates.
    
    Args:
        report_text: Extracted text from PDF report.
        empty_template: Empty JSON template structure.
        filled_template: Filled JSON template (ground truth).
        cfg: Optional configuration for prompt tokens.
    
    Returns:
        Complete prompt string in LLaMA 3.1 chat format.
    """
    cfg = cfg or DatasetCreationConfig(
        report_dir="",
        filled_template_dir="",
        empty_template_dir="",
        output_path="",
    )
    
    report_text = report_text.strip()
    empty_template_str = json.dumps(empty_template, indent=2, ensure_ascii=False)
    filled_json_str = json.dumps(filled_template, indent=2, ensure_ascii=False)
    
    user_prompt = cfg.user_prompt_template.format(report_text, empty_template_str)
    
    full_prompt = (
        f"{cfg.bos_token}"
        f"{cfg.system_tag}\n{cfg.system_prompt}{cfg.eot_token}"
        f"{cfg.user_tag}\n{user_prompt}{cfg.eot_token}\n"
        f"{cfg.assistant_tag}\n{filled_json_str}{cfg.eot_token}"
    )
    
    return full_prompt


def create_training_dataset(
    report_dir: str | Path,
    filled_template_dir: str | Path,
    empty_template_dir: str | Path,
    output_path: str | Path,
    config: DatasetCreationConfig | None = None,
) -> None:
    """
    Create a training dataset (JSONL) from PDF reports and JSON templates.
    
    This function replicates the logic from finetuning.ipynb Cell 17.
    
    Args:
        report_dir: Directory containing PDF report files.
        filled_template_dir: Directory containing filled JSON templates.
        empty_template_dir: Directory containing empty JSON templates.
        output_path: Path to output JSONL file.
        config: Optional configuration for prompt construction.
    """
    cfg = config or DatasetCreationConfig(
        report_dir=report_dir,
        filled_template_dir=filled_template_dir,
        empty_template_dir=empty_template_dir,
        output_path=output_path,
    )
    
    report_dir = Path(report_dir)
    filled_template_dir = Path(filled_template_dir)
    empty_template_dir = Path(empty_template_dir)
    output_path = Path(output_path)
    
    # Collect and sort files
    report_files = natsorted([f for f in os.listdir(report_dir) if f.endswith(".pdf")])
    annotation_files = natsorted([f for f in os.listdir(filled_template_dir) if f.endswith(".json")])
    empty_template_files = natsorted([f for f in os.listdir(empty_template_dir) if f.endswith(".json")])
    
    # Extract basenames (without extension) and normalize
    report_basenames = natsorted([extract_base(f) for f in report_files])
    annotation_basenames = natsorted([
        normalize_annotation_name(f).replace("_template", "") for f in annotation_files
    ])
    empty_basenames = natsorted([
        normalize_annotation_name(f).replace("_template", "") for f in empty_template_files
    ])
    
    # Validate that all three directories have matching files
    if report_basenames != annotation_basenames or report_basenames != empty_basenames:
        raise ValueError(
            "Mismatch between reports, filled, or empty templates:\n"
            f"Reports: {report_basenames}\n"
            f"Annotations: {annotation_basenames}\n"
            f"Empty: {empty_basenames}"
        )
    
    # Load all data
    reports = []
    filled_templates = []
    empty_templates = []
    
    for rfile, afile, etfile in zip(report_files, annotation_files, empty_template_files):
        rpath = report_dir / rfile
        apath = filled_template_dir / afile
        epath = empty_template_dir / etfile
        
        # Extract text from PDF
        reports.append(extract_text_from_pdf(rpath))
        
        # Load filled template
        with apath.open("r", encoding="utf-8") as f:
            filled_templates.append(json.load(f))
        
        # Load empty template
        with epath.open("r", encoding="utf-8") as f:
            empty_templates.append(json.load(f))
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build and save prompts
    with output_path.open("w", encoding="utf-8") as out_file:
        for report, filled, empty in zip(reports, filled_templates, empty_templates):
            full_prompt = build_training_prompt(report, empty, filled, cfg)
            out_file.write(json.dumps({"text": full_prompt}, ensure_ascii=False) + "\n")
    
    print(f"✅ Dataset created successfully at: {output_path}")
    print(f"   Total examples: {len(reports)}")


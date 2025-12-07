from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import dtype as torch_dtype


@dataclass
class TrainConfig:
    """Configuration bundle for training LLaMA-based models with LoRA adapters."""

    base_model_path: str
    data_path: str
    output_dir: str
    lora_output_dir: str
    logging_dir: str = "./logs"
    tokenizer_max_length: int = 3000
    test_size: float = 0.2
    seed: int = 42
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    num_train_epochs: int = 5
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 10
    report_to: str = "tensorboard"
    fp16: bool = True
    early_stopping_patience: int = 1
    load_in_4bit: bool = True
    bnb_compute_dtype: torch_dtype = torch.float16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    lora_target_modules: Sequence[str] = field(default_factory=lambda: ("q_proj", "v_proj"))

    def ensure_output_dirs(self) -> None:
        """Create directories that need to exist before training starts."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.lora_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Configuration bundle controlling prompt construction and generation for inference."""

    model_path: str
    lora_path: str
    template: Mapping[str, Any] | str | None = None
    template_path: str | None = None
    bos_token: str = "<|begin_of_text|>"
    system_tag: str = "<|start_header_id|>system<|end_header_id|>"
    user_tag: str = "<|start_header_id|>user<|end_header_id|>"
    eot_token: str = "<|eot_id|>"
    system_prompt: str = (
        "You are a precise and reliable medical AI assistant. "
        "Your task is to extract relevant entities from a medical report and return a filled JSON object based on the provided template. "
        "Ensure the output is valid JSON and matches the structure of the template. "
        "Only use values found or logically inferred from the report. Leave fields unchanged if no value is found."
    )
    user_prompt_template: str = (
        "Extract relevant information from the following medical report to fill fields in the JSON template.\n\n"
        "Medical Report:\n{report_text}\n\n"
        "JSON Template:\n{template_text}\n\n"
        "Your response must only contain the completed JSON object, nothing else."
    )
    max_new_tokens: int = 3000
    do_sample: bool = False
    return_scores: bool = False
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    load_in_4bit: bool = True
    bnb_compute_dtype: torch_dtype = torch.float16
    device_map: str | dict[str, Any] | None = "auto"
    extra_generation_kwargs: dict[str, Any] = field(default_factory=dict)
    # Batch inference with template splitting
    use_batch_inference: bool = False
    max_measurements_per_batch: int = 2
    dedup_consecutive_keys: bool = True
    # Anonymization options
    apply_anonymization: bool = False
    anonymization_secret_key: str = "sbh86"
    anonymization_use_ner: bool = True
    anonymization_ner_model_path: str = ""
    anonymization_output_dir: str | None = None


@dataclass
class EvalConfig:
    """
    Configuration for model evaluation.

    Attributes allow callers to control how dictionaries are flattened and how values are compared.
    """

    # Required file paths (must come before fields with defaults)
    predictions_path: str
    ground_truth_path: str

    # Optional behaviour/configuration flags
    ignore_keys: Iterable[str] = ()
    case_sensitive: bool = False
    strip_whitespace: bool = True
    numeric_precision: int | None = 4
    ignore_null_ground_truth: bool = True
    path_separator: str = "."
    average: str = "micro"
    metrics_output_path: str | None = None


@dataclass
class FullEvalConfig:
    """
    Configuration for complete evaluation pipeline: inference + metrics.
    
    Combines InferenceConfig parameters with EvalConfig parameters,
    plus dataset paths for automated evaluation.
    """
    
    # Dataset paths
    test_reports_dir: str  # Directory containing PDF test reports
    test_ground_truth_dir: str  # Directory containing filled JSON templates (ground truth)
    
    # Inference configuration (from InferenceConfig)
    model_path: str
    lora_path: str

    test_empty_templates_dir: str | None = None  # Directory containing empty templates (if None, uses single template)
    
    template: Mapping[str, Any] | str | None = None
    template_path: str | None = None
    bos_token: str = "<|begin_of_text|>"
    system_tag: str = "<|start_header_id|>system<|end_header_id|>"
    user_tag: str = "<|start_header_id|>user<|end_header_id|>"
    eot_token: str = "<|eot_id|>"
    system_prompt: str = (
        "You are a precise and reliable medical AI assistant. "
        "Your task is to extract relevant entities from a medical report and return a filled JSON object based on the provided template. "
        "Ensure the output is valid JSON and matches the structure of the template. "
        "Only use values found or logically inferred from the report. Leave fields unchanged if no value is found."
    )
    user_prompt_template: str = (
        "Extract relevant information from the following medical report to fill fields in the JSON template.\n\n"
        "Medical Report:\n{report_text}\n\n"
        "JSON Template:\n{template_text}\n\n"
        "Your response must only contain the completed JSON object, nothing else."
    )
    max_new_tokens: int = 3000
    do_sample: bool = False
    return_scores: bool = False
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    load_in_4bit: bool = True
    bnb_compute_dtype: torch_dtype = torch.float16
    device_map: str | dict[str, Any] | None = "auto"
    extra_generation_kwargs: dict[str, Any] = field(default_factory=dict)
    use_batch_inference: bool = False
    max_measurements_per_batch: int = 2
    dedup_consecutive_keys: bool = True
    apply_anonymization: bool = False
    anonymization_secret_key: str = "sbh86"
    anonymization_use_ner: bool = True
    anonymization_ner_model_path: str = ""
    anonymization_output_dir: str | None = None
    
    # Evaluation configuration (from EvalConfig)
    ignore_keys: Iterable[str] = ()
    case_sensitive: bool = False
    strip_whitespace: bool = True
    numeric_precision: int | None = 4
    ignore_null_ground_truth: bool = True
    path_separator: str = "."
    average: str = "micro"
    
    # Output paths
    predictions_output_dir: str = "./evaluation_predictions"  # Where to save predictions
    metrics_output_path: str | None = None  # Where to save metrics JSON


@dataclass
class SyntheticDataConfig:
    """Parameters controlling synthetic blood-test generation."""

    schema: Mapping[str, Any]
    selection_probs: Sequence[float] = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0)
    selection_weights: Sequence[float] = (0.2, 0.1, 0.2, 0.1, 0.15, 0.15, 0.1)
    value_noise: float = 0.2






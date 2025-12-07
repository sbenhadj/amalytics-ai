from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from amalytics_ml.config import InferenceConfig
from amalytics_ml.utils.template_split import (
    split_template_by_measurements,
    deep_merge,
)
from amalytics_ml.data.anonymization import (
    extract_text_from_pdf,
    anonymize_text,
    AnonymizationConfig,
)


@dataclass
class InferenceResult:
    """Container for inference results including optional confidence scores."""
    parsed_json: dict[str, Any]
    confidence_scores: dict[str, float] | None = None
    raw_text: str | None = None


def _template_to_string(template: str | Mapping[str, Any] | None) -> str:
    if template is None:
        raise ValueError("InferenceConfig.template must be provided to build the prompt.")
    if isinstance(template, str):
        return template
    return json.dumps(template, ensure_ascii=False, indent=2)


def _build_prompt(report_text: str, template_str: str, cfg: InferenceConfig) -> str:
    return (
        f"{cfg.bos_token}"
        f"{cfg.system_tag}\n{cfg.system_prompt}{cfg.eot_token}"
        f"{cfg.user_tag}\n"
        f"{cfg.user_prompt_template.format(report_text=report_text, template_text=template_str)}"
        f"{cfg.eot_token}"
    )


def _load_model_and_tokenizer(model_path: str, lora_path: str, cfg: InferenceConfig):
    # CRITICAL: Match original notebook behavior exactly
    # If LoRA path is provided, load model and tokenizer from LoRA config's base_model_name_or_path
    # This ensures both model and tokenizer match the training setup
    if lora_path:
        from peft import PeftConfig
        try:
            peft_config = PeftConfig.from_pretrained(lora_path)
            base_model_name = peft_config.base_model_name_or_path
            # Use the base model name from LoRA config for both model and tokenizer
            # This matches: model_name = config.base_model_name_or_path in original notebook
            actual_model_path = base_model_name
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        except Exception as e:
            # Fallback to model_path if LoRA config can't be loaded
            print(f"[WARNING] Could not load PeftConfig from {lora_path}: {e}")
            print(f"[WARNING] Falling back to model_path: {model_path}")
            actual_model_path = model_path
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    else:
        # No LoRA, use model_path directly
        actual_model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model kwargs - match original notebook exactly
    # Original: load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, device_map="auto"
    model_kwargs: dict[str, Any] = {
        "device_map": cfg.device_map,
        "low_cpu_mem_usage": True,
    }
    
    if cfg.load_in_4bit:
        # Use BitsAndBytesConfig for 4-bit quantization (recommended approach)
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=cfg.bnb_compute_dtype,
            )
            model_kwargs["quantization_config"] = quantization_config
        except ImportError:
            # Fallback to deprecated method if BitsAndBytesConfig not available
            # This matches the original notebook which uses deprecated method
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = cfg.bnb_compute_dtype
    else:
        # Use float16 to reduce memory when not using 4-bit quantization
        model_kwargs["dtype"] = torch.float16

    # Load model from actual_model_path (base_model_name if LoRA, else model_path)
    # This matches: model = AutoModelForCausalLM.from_pretrained(model_name, ...) in original
    model = AutoModelForCausalLM.from_pretrained(actual_model_path, **model_kwargs)
    
    # Only load LoRA if path is provided
    # This matches: model = PeftModel.from_pretrained(model, peft_model_id) in original
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    
    # Set to eval mode - matches: model = model.eval() in original
    model = model.eval()
    return model, tokenizer


def _calculate_confidence(
    parsed: dict[str, Any],
    generated_ids: torch.Tensor,
    scores: tuple,
    tokens: list[str],
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    """
    Calculate per-field confidence scores based on token probabilities.
    
    Aligned with the original calculate_confidence function from LLAMA(script).ipynb.
    Expects a 3-level nested JSON structure:
        Category -> SubCategory -> Field (with valeur/unité)
    
    Args:
        parsed: The parsed JSON output from the model.
        generated_ids: Tensor of generated token IDs.
        scores: Tuple of score tensors from model.generate(output_scores=True).
        tokens: List of token strings corresponding to generated_ids.
        tokenizer: The tokenizer used for decoding.
    
    Returns:
        Dictionary mapping field names to their average confidence scores.
    """
    # Calculate confidence for each token
    confidences = []
    for i, score in enumerate(scores):
        # Handle different score tensor shapes
        # Scores can be: [batch_size, vocab_size] or [vocab_size]
        if score.dim() == 2:
            # Batch case: take first element if batch_size=1, otherwise error
            if score.shape[0] == 1:
                probs = F.softmax(score[0], dim=-1)
            else:
                raise ValueError(f"Expected single sample scores, got batch size {score.shape[0]}")
        elif score.dim() == 1:
            # Single sample case
            probs = F.softmax(score, dim=-1)
        else:
            raise ValueError(f"Unexpected score tensor shape: {score.shape}")
        
        # Extract token ID (handle both tensor and int)
        if isinstance(generated_ids, torch.Tensor):
            token_id = generated_ids[i].item()
        else:
            token_id = generated_ids[i] if isinstance(generated_ids[i], int) else int(generated_ids[i])
        
        # Get confidence for this token
        if token_id >= probs.shape[0]:
            # Token ID out of range - skip or use fallback
            confidence = 0.0
        else:
            confidence = probs[token_id].item()
        confidences.append(confidence)

    # Build token-confidence pairs
    token_conf_pairs = list(zip(tokens, confidences))

    # Build character-level confidence map
    decoded_tokens = []
    for token, conf in token_conf_pairs:
        decoded_piece = tokenizer.convert_tokens_to_string([token])
        decoded_tokens.append((decoded_piece, conf))

    reconstructed_text = ""
    char_conf_map: list[float] = []
    for piece, conf in decoded_tokens:
        reconstructed_text += piece
        char_conf_map.extend([conf] * len(piece))

    # Collect all leaf keys using the original 3-level structure
    # Structure: Category (k1) -> SubCategory (k2) -> Field (key with valeur/unité)
    total_keys: list[str] = []
    for k1, v1 in parsed.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                if isinstance(v2, dict):
                    keys = list(v2.keys())
                    total_keys = total_keys + keys

    # Calculate confidence for each field (original algorithm)
    entity_confidences: dict[str, float] = {}
    i = 0
    
    for k1, v1 in parsed.items():
        if not isinstance(v1, dict):
            continue
        for k2, v2 in v1.items():
            if not isinstance(v2, dict):
                continue
            for key, value in v2.items():
                key_str = json.dumps(key, ensure_ascii=False)
                pos = reconstructed_text.find(key_str)
                
                if pos == -1:
                    # Key not found in text - skip
                    i += 1
                    continue
                
                if i + 1 < len(total_keys):
                    next_key = json.dumps(total_keys[i + 1], ensure_ascii=False)
                    next_pos = reconstructed_text.find(next_key, pos + len(key_str))
                    if next_pos != -1 and next_pos > pos:
                        # Found next key - take confidence between pos and next_pos
                        confs = char_conf_map[pos:next_pos]
                        if confs:
                            avg_conf = sum(confs) / len(confs)
                            entity_confidences[key] = round(avg_conf, 4)
                    else:
                        # Next key not found - take from pos to end
                        if pos < len(char_conf_map):
                            confs = char_conf_map[pos:]
                            if confs:
                                avg_conf = sum(confs) / len(confs)
                                entity_confidences[key] = round(avg_conf, 4)
                else:
                    # Last key - take from pos to end
                    if pos < len(char_conf_map):
                        confs = char_conf_map[pos:]
                        if confs:
                            avg_conf = sum(confs) / len(confs)
                            entity_confidences[key] = round(avg_conf, 4)
                i += 1

    return entity_confidences


def _parse_llm_output(s: str) -> dict[str, Any] | None:
    """
    Parse LLM output that may be strict JSON or Python-style dict.
    
    Returns dict or None if parsing fails.
    """
    s = s.strip()
    # Try JSON first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try Python literal
    try:
        import ast
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _extract_json_from_response(response_text: str) -> dict[str, Any] | None:
    """
    Extract JSON object from model response.
    
    Matches the original notebook behavior: simple regex search for JSON block.
    """
    # Original notebook approach: simple regex search
    match = re.search(r"\{[\s\S]+\}", response_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find JSON with balanced braces
    brace_start = response_text.find("{")
    if brace_start != -1:
        json_candidate = response_text[brace_start:]
        # Count braces to find potential end (handle strings properly)
        depth = 0
        end_pos = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_candidate):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
        
        if end_pos > 0:
            try:
                return json.loads(json_candidate[:end_pos])
            except json.JSONDecodeError:
                pass
    
    return None


def _run_batch_inference(
    model: Any,
    tokenizer: AutoTokenizer,
    input_text: str,
    template: dict[str, Any] | str,
    cfg: InferenceConfig,
) -> InferenceResult:
    """
    Run batch inference with template splitting for optimized processing.
    
    This function:
    1. Splits the template by measurements
    2. Creates batch prompts
    3. Runs batch inference
    4. Merges results
    
    Args:
        model: Loaded model instance.
        tokenizer: Tokenizer instance.
        input_text: Medical report text.
        template: Template to split (dict or JSON string).
        cfg: Inference configuration.
    
    Returns:
        InferenceResult with merged JSON and confidence scores.
    """
    # Split template into parts
    if isinstance(template, str):
        template_dict = json.loads(template)
    else:
        template_dict = template
    
    template_parts = split_template_by_measurements(
        template_dict,
        max_objects_per_part=cfg.max_measurements_per_batch,
        dedup_consecutive=cfg.dedup_consecutive_keys,
    )
    
    if not template_parts:
        raise ValueError("Template splitting produced no parts.")
    
    # Build prompts for each part
    prompts = []
    for part_json_str in template_parts:
        prompt = _build_prompt(input_text, part_json_str, cfg)
        prompts.append(prompt)
    
    # Set padding side for batch generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # Tokenize batch
        model_max_length = getattr(tokenizer, "model_max_length", 2048)
        if model_max_length > 100000:
            model_max_length = 2048
        
        max_input_length = model_max_length - cfg.max_new_tokens
        if max_input_length < 512:
            max_input_length = 512
        
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)
        
        eos_token_id = cfg.eos_token_id if cfg.eos_token_id is not None else tokenizer.eos_token_id
        pad_token_id = cfg.pad_token_id if cfg.pad_token_id is not None else tokenizer.eos_token_id
        
        generation_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": cfg.return_scores,
        }
        # Update with extra kwargs, but remove sampling params if do_sample=False
        extra_kwargs = cfg.extra_generation_kwargs.copy()
        if not cfg.do_sample:
            # Remove sampling parameters when do_sample=False
            extra_kwargs.pop("temperature", None)
            extra_kwargs.pop("top_p", None)
            extra_kwargs.pop("top_k", None)
        generation_kwargs.update(extra_kwargs)
        
        # Batch generation
        with torch.inference_mode():
            outputs = model.generate(**tokenized, **generation_kwargs)
        
        # Process each result in the batch
        sequences = outputs.sequences
        batch_size = sequences.size(0)
        input_padded_len = tokenized["input_ids"].shape[1]
        
        merged_result: dict[str, Any] = {}
        merged_confidences: dict[str, Any] = {}
        all_raw_texts: list[str] = []
        
        for i in range(batch_size):
            seq_i = sequences[i]
            gen_region = seq_i[input_padded_len:]
            
            # Skip if generation region is empty
            if gen_region.size(0) == 0:
                print(f"[WARNING] Batch item {i+1} produced empty generation, skipping.")
                continue
            
            # Cut at first EOS (but don't include EOS token itself)
            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                eos_pos = (gen_region == eos_id).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    gen_region = gen_region[:eos_pos[0]]
            
            # Skip if nothing left after EOS removal
            if gen_region.size(0) == 0:
                print(f"[WARNING] Batch item {i+1} had only EOS token, skipping.")
                continue
            
            # Decode generated text
            gen_text = tokenizer.decode(gen_region, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            all_raw_texts.append(gen_text)
            
            # Extract and parse JSON
            extracted_json = _extract_json_from_response(gen_text)
            if not extracted_json:
                # Try parsing with fallback
                extracted_json = _parse_llm_output(gen_text)
            
            if extracted_json and isinstance(extracted_json, dict):
                merged_result = deep_merge(merged_result, extracted_json)
                
                # Calculate confidence if requested
                if cfg.return_scores and hasattr(outputs, "scores") and outputs.scores:
                    try:
                        # Get the actual generated length (before EOS)
                        gen_length = gen_region.size(0)
                        # Ensure we don't exceed available scores
                        T_i = min(gen_length, len(outputs.scores))
                        
                        if T_i == 0:
                            continue
                        
                        # Extract generated token IDs for this batch item
                        gen_ids_i = gen_region[:T_i]
                        
                        # Extract scores for this batch item
                        # outputs.scores[t] has shape [batch_size, vocab_size]
                        # We need the scores for batch item i
                        scores_i = []
                        for t in range(T_i):
                            score_t = outputs.scores[t]  # Shape: [batch_size, vocab_size]
                            if i < score_t.shape[0]:
                                # Extract score for this batch item
                                score_i = score_t[i]  # Shape: [vocab_size]
                                scores_i.append(score_i)
                            else:
                                break
                        
                        if len(scores_i) != T_i:
                            # Mismatch - skip confidence for this item
                            print(f"[WARNING] Score length mismatch for batch item {i+1}: {len(scores_i)} vs {T_i}")
                            continue
                        
                        # Convert token IDs to list for tokenizer
                        gen_ids_list = gen_ids_i.tolist()
                        tokens_i = tokenizer.convert_ids_to_tokens(gen_ids_list)
                        
                        # Calculate confidence
                        entity_conf_i = _calculate_confidence(
                            parsed=extracted_json,
                            generated_ids=gen_ids_i,
                            scores=tuple(scores_i),
                            tokens=tokens_i,
                            tokenizer=tokenizer,
                        )
                        
                        if entity_conf_i:
                            # Merge confidences (flat dict structure - keys are field names)
                            merged_confidences.update(entity_conf_i)
                    except Exception as e:
                        import traceback
                        print(f"[WARNING] Could not calculate confidence for batch item {i+1}: {e}")
                        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        
        # If no results were merged, return empty dict with warning
        if not merged_result:
            print("[WARNING] Batch inference produced no valid JSON results.")
            # Return empty template structure if possible
            if isinstance(template_dict, dict):
                merged_result = template_dict.copy()
            else:
                merged_result = {}
        
        return InferenceResult(
            parsed_json=merged_result,
            confidence_scores=merged_confidences if merged_confidences else None,
            raw_text="\n---\n".join(all_raw_texts) if all_raw_texts else None,
        )
    
    except Exception as e:
        tokenizer.padding_side = original_padding_side
        raise RuntimeError(f"Batch inference failed: {e}") from e


def run_inference(
    model_path: str,
    lora_path: str,
    input_text: str | Path,
    config: InferenceConfig | None = None,
) -> InferenceResult:
    """
    Load a fine-tuned (base + LoRA) model, run inference on the given text or PDF, and return JSON output.

    Args:
        model_path: Path to the base causal language model.
        lora_path: Directory containing the LoRA adapter weights.
        input_text: The medical report text to process, or a Path to a PDF file.
        config: Optional inference configuration overriding defaults.

    Returns:
        InferenceResult containing parsed JSON and optional confidence scores.
    """

    if config is None:
        raise ValueError("InferenceConfig instance must be provided to run_inference.")
    cfg = config
    
    # Handle PDF input or text input
    text_content = input_text
    
    # Check if input is a PDF file path
    is_pdf_path = False
    pdf_path = None
    
    if isinstance(input_text, Path):
        # Direct Path object - check extension
        if input_text.suffix.lower() == '.pdf':
            is_pdf_path = True
            pdf_path = input_text
    elif isinstance(input_text, str):
        # String input - check if it's a file path (not plain text)
        # Only consider it a PDF path if:
        # 1. Ends with .pdf AND
        # 2. Looks like a file path (contains / or \ or is a valid path) AND
        # 3. Either exists or is a reasonable path (not too long, no special chars that suggest it's text)
        input_lower = input_text.lower().strip()
        
        # Check if it ends with .pdf and looks like a path
        if input_lower.endswith('.pdf'):
            path_obj = Path(input_text)
            # Consider it a path if it has path separators or is a reasonable file path
            has_path_separator = '/' in input_text or '\\' in input_text
            is_reasonable_path = len(input_text) < 260 and not '\n' in input_text
            
            if has_path_separator or is_reasonable_path:
                is_pdf_path = True
                pdf_path = path_obj
    
    if is_pdf_path:
        # Input is a PDF path
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text from PDF
        text_content = extract_text_from_pdf(pdf_path)
    
    # Ensure we have a string
    if not isinstance(text_content, str):
        raise TypeError(f"Expected str or PDF path, got {type(input_text)}")
    
    # Apply anonymization if enabled
    if cfg.apply_anonymization:
        anon_config = AnonymizationConfig(
            secret_key=cfg.anonymization_secret_key.encode() if isinstance(cfg.anonymization_secret_key, str) else cfg.anonymization_secret_key,
            use_ner=cfg.anonymization_use_ner,
            ner_model_path=cfg.anonymization_ner_model_path,
        )
        text_content = anonymize_text(text_content, anon_config)
    
    # Use the processed text content
    input_text = text_content
    
    model, tokenizer = _load_model_and_tokenizer(model_path, lora_path, cfg)
    
    # Use batch inference if enabled
    if cfg.use_batch_inference:
        template_obj = cfg.template
        
        # Template should already be loaded by the script, but handle edge cases
        if template_obj is None:
            raise ValueError("Template must be provided via template or template_path for batch inference.")
        
        # Convert string template to dict if needed
        if isinstance(template_obj, str):
            try:
                template_obj = json.loads(template_obj)
            except json.JSONDecodeError:
                # Maybe it's a file path
                template_path = Path(template_obj)
                if template_path.exists():
                    with template_path.open("r", encoding="utf-8") as f:
                        template_obj = json.load(f)
                else:
                    raise ValueError(f"Could not parse template as JSON or find as file: {template_obj[:100]}...")
        
        if not isinstance(template_obj, dict):
            raise TypeError(f"Template must be a dict, got {type(template_obj)}")
        
        return _run_batch_inference(model, tokenizer, input_text, template_obj, cfg)
    
    # Standard single inference
    template_str = _template_to_string(cfg.template)
    prompt = _build_prompt(input_text, template_str, cfg)
    
    # Tokenize - match original notebook exactly: NO truncation
    # Original: inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    eos_token_id = cfg.eos_token_id if cfg.eos_token_id is not None else tokenizer.eos_token_id
    pad_token_id = cfg.pad_token_id if cfg.pad_token_id is not None else tokenizer.eos_token_id

    # Always output scores if we want to calculate confidence
    generation_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": cfg.return_scores,
    }
    # Update with extra kwargs, but remove sampling params if do_sample=False
    extra_kwargs = cfg.extra_generation_kwargs.copy()
    if not cfg.do_sample:
        # Remove sampling parameters when do_sample=False
        extra_kwargs.pop("temperature", None)
        extra_kwargs.pop("top_p", None)
        extra_kwargs.pop("top_k", None)
    generation_kwargs.update(extra_kwargs)

    outputs = model.generate(**inputs, **generation_kwargs)
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[-1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Try to extract JSON with multiple strategies
    parsed = _extract_json_from_response(response_text)
    
    if parsed is None:
        # If we still can't parse JSON, return the template as-is with raw response
        print(f"[WARNING] Could not extract valid JSON from model response.")
        print(f"[DEBUG] Raw response (first 500 chars): {response_text[:500]}")
        
        # Return the original template as fallback
        if isinstance(cfg.template, dict):
            parsed = cfg.template.copy()
        else:
            parsed = {"error": "Could not parse model response", "raw_response": response_text[:1000]}
    
    # Calculate confidence scores if requested and we have valid scores
    confidence_scores = None
    if cfg.return_scores and hasattr(outputs, "scores") and outputs.scores and parsed:
        try:
            # Ensure we have matching lengths
            gen_length = generated_ids.size(0)
            scores_length = len(outputs.scores)
            
            if gen_length > scores_length:
                # Generated more tokens than we have scores for - truncate
                generated_ids = generated_ids[:scores_length]
            
            tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
            
            # Extract scores - for single inference, scores[t] has shape [1, vocab_size]
            scores_list = []
            for t, score in enumerate(outputs.scores):
                if score.dim() == 2:
                    # Batch case: take first (and only) element
                    if score.shape[0] == 1:
                        scores_list.append(score[0])
                    else:
                        raise ValueError(f"Expected single sample, got batch size {score.shape[0]}")
                elif score.dim() == 1:
                    scores_list.append(score)
                else:
                    raise ValueError(f"Unexpected score shape: {score.shape}")
            
            confidence_scores = _calculate_confidence(
                parsed=parsed,
                generated_ids=generated_ids,
                scores=tuple(scores_list),
                tokens=tokens,
                tokenizer=tokenizer,
            )
        except Exception as e:
            import traceback
            print(f"[WARNING] Could not calculate confidence scores: {e}")
            if cfg.return_scores:  # Only print traceback if scores were explicitly requested
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")

    return InferenceResult(
        parsed_json=parsed,
        confidence_scores=confidence_scores,
        raw_text=response_text,
    )


from __future__ import annotations

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer

from amalytics_ml.config import TrainConfig


def train_model(config: TrainConfig) -> str:
    """
    Fine-tune a LLaMA-based causal LM with LoRA adapters using the provided configuration.

    Args:
        config: Training hyperparameters, dataset location, and output directories.

    Returns:
        Path to the directory that stores the trained LoRA adapter weights.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=config.bnb_compute_dtype,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=config.lora_task_type,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenized = tokenizer(
            examples["text"],
            truncation=False,
            max_length=config.tokenizer_max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = load_dataset("json", data_files=config.data_path, split="train")
    split_dataset = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    train_dataset = split_dataset["train"].map(tokenize_fn, batched=True)
    eval_dataset = split_dataset["test"].map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        fp16=config.fp16,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    trainer.train()

    config.ensure_output_dirs()
    trainer.save_model(config.lora_output_dir)
    tokenizer.save_pretrained(config.lora_output_dir)

    return config.lora_output_dir


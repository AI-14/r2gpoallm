import argparse
import logging
import os

import pandas as pd
import torch
from jury import Jury, load_metric
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from radgraph import F1RadGraph
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ORPOConfig, SFTConfig

from datasets import Dataset


def get_dataset(train_filepath: str, val_filepath: str) -> list[Dataset]:
    """Prepares sft datasets.

    Args:
        train_filepath (str): Path to train csv file.
        val_filepath (str): Path to val csv file.

    Returns:
        list[Dataset]: Train and val datasets.
    """

    train_ds = Dataset.from_pandas(pd.read_csv(train_filepath, encoding="utf-8"))
    val_ds = Dataset.from_pandas(pd.read_csv(val_filepath, encoding="utf-8"))
    return [train_ds, val_ds]


def get_model_with_lora(args: argparse.Namespace) -> any:
    """Prepares the model.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        any: Model.
    """

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_repo,
        token=args.hf_token,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias=args.lora_bias,
        task_type=args.lora_task_type,
    )

    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    return base_model


def get_tokenizer(args: argparse.Namespace) -> any:
    """Prepares tokenizer.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        any: Tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_repo,
        token=args.hf_token,
        add_bos_token=True,
        add_eos_token=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def get_sft_training_args(args: argparse.Namespace) -> SFTConfig:
    """Prepares training arguments for sft.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        SFTConfig: Configuration.
    """

    return SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.eval_strategy,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        max_seq_length=args.max_seq_len,
        dataset_text_field=args.dataset_text_field,
    )


def get_orpo_training_args(args: argparse.Namespace) -> ORPOConfig:
    """Prepares training arguments for orpo.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        ORPOConfig: Configuration.
    """

    return ORPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.eval_strategy,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        max_length=args.max_seq_len,
    )


def gen_prompt_sft(row: pd.DataFrame) -> str:
    """Generates the sft prompt.

    Args:
        row (pd.DataFrame): Single row containing all the fields.

    Returns:
        str: Prompt.
    """

    prompt: str = (
        "###INSTRUCTION:\n"
        + "You are an expert radiologist. Generate FINDINGS section not exceeding 100 words by understanding details based on the given corresponding sections below.\n"
        + "PAST-REPORTS:\n"
        + f"{row['similar_past_reports']}\n"
        + "CONTEXTUAL-TAGS:\n"
        + f"{row['tags']}\n"
        + "IMPRESSIONS:\n"
        + f"{row['impression']}\n"
        + "###RESPONSE:\n"
    )
    return prompt


def gen_prompt_po(row: pd.DataFrame) -> str:
    """Generates the po prompt.

    Args:
        row (pd.DataFrame): Single row containing all the fields.

    Returns:
        str: Prompt.
    """

    prompt: str = (
        "###INSTRUCTION:\n"
        + "You are an expert radiologist. Enhance and improve PREDICTED-FINDINGS section not exceeding 100 words by understanding details based on the given corresponding sections below.\n"
        + "PAST-REPORTS:\n"
        + f"{row['similar_past_reports']}\n"
        + "CONTEXTUAL-TAGS:\n"
        + f"{row['tags']}\n"
        + "IMPRESSIONS:\n"
        + f"{row['impression']}\n"
        + "PREDICTED-FINDINGS:\n"
        + f"{row['prediction_fg']}\n"
        + "###RESPONSE:\n"
    )
    return prompt


def get_logger(logging_dir: str) -> logging.Logger:
    """Builds logger.

    Args:
        logging_dir (str): Path to logging directory.

    Returns:
        logging.Logger: Logger.
    """

    if not os.path.isdir(logging_dir):
        os.makedirs(logging_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(f"{logging_dir}/exp.log", mode="w")
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def calculate_metrics(
    predictions: list[list[str]], references: list[list[str]], logger: logging.Logger
) -> None:
    """Computes NLG metrics and F1-RadGraph metrics.

    Args:
        predictions (list[list[str]]): Containing all the predicted sentences.
        references (list[list[str]]): Containing all the ground truth sentences.
        logger (logging.Logger): Logger.
    """

    # NLG ----------------------------------------
    metrics = [
        load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
        load_metric("bleu", resulting_name="bleu_2", compute_kwargs={"max_order": 2}),
        load_metric("bleu", resulting_name="bleu_3", compute_kwargs={"max_order": 3}),
        load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
        load_metric("meteor", resulting_name="meteor"),
        load_metric("rouge", resulting_name="rouge"),
    ]

    scorer = Jury(metrics=metrics, run_concurrent=False)
    scores = scorer(predictions=predictions, references=references)

    logger.info(f"BLEU-1: {scores['bleu_1']['score']:.5f}")
    logger.info(f"BLEU-2: {scores['bleu_2']['score']:.5f}")
    logger.info(f"BLEU-3: {scores['bleu_3']['score']:.5f}")
    logger.info(f"BLEU-4: {scores['bleu_4']['score']:.5f}")
    logger.info(f"METEOR: {scores['meteor']['score']:.5f}")
    logger.info(f"ROUGE-L: {scores['rouge']['rougeL']:.5f}")

    # F1-RadGraph ----------------------------------------
    predictions = sum(predictions, [])
    references = sum(references, [])
    f1radgraph = F1RadGraph(reward_level="all")
    mean_reward, _, _, _ = f1radgraph(hyps=predictions, refs=references)
    logger.info(f"Simple, partial, complete scores: {mean_reward}")

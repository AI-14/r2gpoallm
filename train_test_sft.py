import argparse
import gc
import logging
import os
import warnings

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import SFTTrainer

from configs import seeds, sft
from utils import (
    calculate_metrics,
    gen_prompt_sft,
    get_dataset,
    get_logger,
    get_model_with_lora,
    get_sft_training_args,
    get_tokenizer,
)

warnings.filterwarnings("ignore")
tqdm.pandas()


def train(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Train the llm.

    Args:
        args (argparse.Namespace): Arguments.
        logger (logging.Logger): Logger.
    """

    # Prepare components for training ---------------------------
    base_model = get_model_with_lora(args)
    tokenizer = get_tokenizer(args)
    training_args = get_sft_training_args(args)
    train_dataset, val_dataset = get_dataset(
        args.sft_train_filepath, args.sft_val_filepath
    )

    # Train ---------------------------
    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    trainer.train()
    logger.info("Trainer history:")
    for hist in trainer.state.log_history:
        logger.info(hist)
    pd.DataFrame(trainer.state.log_history).to_csv(
        f"{args.results_dir}/trainer_history.csv"
    )

    # Save adapters ---------------------------
    trainer.model.save_pretrained(os.path.join(args.output_dir, args.adapter_model_dir))
    tokenizer.save_pretrained(os.path.join(args.output_dir, args.adapter_model_dir))
    logger.info("Saved new sft adapter model and its tokenizer")

    # Free memory ---------------------------
    del base_model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Merge adapters with model and save ---------------------------
    adapter_model = AutoPeftModelForCausalLM.from_pretrained(
        f"{args.output_dir}/{args.adapter_model_dir}",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    full_model = adapter_model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.output_dir, args.adapter_model_dir),
    )
    full_model.save_pretrained(
        os.path.join(args.output_dir, args.merged_model_dir), safe_serialization=True
    )
    tokenizer.save_pretrained(os.path.join(args.output_dir, args.merged_model_dir))
    logger.info("Saved full merged model and its tokenizer")


def test(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Tests the llm.

    Args:
        args (argparse.Namespace): Arguments.
        logger (logging.Logger): Logger.
    """

    # Prepare components for inference ---------------------------
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.output_dir, args.merged_model_dir),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.output_dir, args.merged_model_dir),
        add_bos_token=True,
        add_eos_token=False,
    )

    # Prepare prompts ---------------------------
    df = pd.read_csv(args.sft_test_filepath, encoding="utf-8")
    df["prompt"] = df.progress_apply(gen_prompt_sft, axis=1)

    # Inference ---------------------------
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    predictions_sft: list[str] = []
    for prompt in tqdm(df["prompt"].values.tolist(), leave=True):
        sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            num_beams=3,
            num_return_sequences=1,
        )
        gen_text = sequences[0]["generated_text"]
        gen_text = gen_text.split("###RESPONSE:\n")[1].lower().strip()
        predictions_sft.append(gen_text)

    # Save results ---------------------------
    df["prediction_sft"] = predictions_sft
    df.to_csv(f"{args.results_dir}/sft_predictions.csv", index=False)

    # Get results ---------------------------
    logger.info("Results of llm-sft:")
    df = pd.read_csv(f"{args.results_dir}/sft_predictions.csv", encoding="utf-8")
    preds: list[list[str]] = []
    refs: list[list[str]] = []
    for pred, ref in zip(
        df["prediction_sft"].values.tolist(), df["findings"].values.tolist()
    ):
        preds.append([pred])
        refs.append([ref])

    calculate_metrics(preds, refs, logger)


def main() -> None:
    """The main flow of this file."""

    args = sft.get_sft_args()

    # Configure necessities ---------------------------
    logger = get_logger(args.logging_dir)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    logger.info(f"EXPERIMENT SETTING:\n{args}")
    seeds.seed_everything(args.seed)

    # Train and test ---------------------------
    train(args, logger)
    test(args, logger)


if __name__ == "__main__":
    main()

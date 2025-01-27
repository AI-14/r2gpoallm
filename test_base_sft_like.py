import argparse
import logging
import os
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from configs import base, seeds
from utils import (
    calculate_metrics,
    gen_prompt_sft,
    get_logger,
)

warnings.filterwarnings("ignore")
tqdm.pandas()


def test_base_sft_like(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Tests the base llm in sft like form.

    Args:
        args (argparse.Namespace): Arguments.
        logger (logging.Logger): Logger.
    """

    # Prepare components for inference ---------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_repo,
        token=args.hf_token,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_repo,
        token=args.hf_token,
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
    predictions_base: list[str] = []
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
        predictions_base.append(gen_text)

    # Save results ---------------------------
    df["prediction_base"] = predictions_base
    df.to_csv(f"{args.results_dir}/base_sft_like_predictions.csv", index=False)

    # Get results ---------------------------
    logger.info("Results of llm-base-sft-like:")
    df = pd.read_csv(
        f"{args.results_dir}/base_sft_like_predictions.csv", encoding="utf-8"
    )
    preds: list[list[str]] = []
    refs: list[list[str]] = []
    for pred, ref in zip(
        df["prediction_base"].values.tolist(), df["findings"].values.tolist()
    ):
        preds.append([pred])
        refs.append([ref])

    calculate_metrics(preds, refs, logger)


def main() -> None:
    """The main flow of this file."""

    args = base.get_base_args()

    # Configure necessities ---------------------------
    logger = get_logger(args.logging_dir)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    logger.info(f"EXPERIMENT SETTING:\n{args}")
    seeds.seed_everything(args.seed)

    # Test ---------------------------
    test_base_sft_like(args, logger)


if __name__ == "__main__":
    main()

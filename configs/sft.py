import argparse


def get_sft_args() -> argparse.Namespace:
    """Configures arguments for sft.

    Returns:
        argparse.Namespace: Arguments.
    """

    parser = argparse.ArgumentParser()

    # Files and directories ---------------------------
    parser.add_argument(
        "--sft_train_filepath", type=str, default="datasets/iuxray/sft_train.csv"
    )
    parser.add_argument(
        "--sft_val_filepath", type=str, default="datasets/iuxray/sft_val.csv"
    )
    parser.add_argument(
        "--sft_test_filepath", type=str, default="datasets/iuxray/sft_test.csv"
    )
    parser.add_argument("--base_model_repo", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output_dir", type=str, default="sft-ckpts")
    parser.add_argument("--logging_dir", type=str, default="sft-logs")
    parser.add_argument("--adapter_model_dir", type=str, default="sft-adapter")
    parser.add_argument("--merged_model_dir", type=str, default="sft-merged")
    parser.add_argument("--results_dir", type=str, default="sft-res")

    # Lora setting ---------------------------
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")

    # Trainer setting ---------------------------
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=3e-2)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--max_seq_len", type=int, default=600)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--dataset_text_field", type=str, default="prompt")

    # Other setting ---------------------------
    parser.add_argument("--hf_token", type=str, default="<add_your_hf_token>")
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()

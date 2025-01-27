import argparse


def get_base_args() -> argparse.Namespace:
    """Configures arguments for base llm.

    Returns:
        argparse.Namespace: Arguments.
    """

    parser = argparse.ArgumentParser()

    # Files and directories ---------------------------
    parser.add_argument(
        "--sft_test_filepath", type=str, default="datasets/iuxray/sft_test.csv"
    )
    parser.add_argument(
        "--po_test_filepath", type=str, default="datasets/iuxray/po_test.csv"
    )
    parser.add_argument("--base_model_repo", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--logging_dir", type=str, default="base-sft-like-logs")
    parser.add_argument("--results_dir", type=str, default="base-sft-like-res")

    # Other setting ---------------------------
    parser.add_argument("--hf_token", type=str, default="<add_your_hf_token>")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()

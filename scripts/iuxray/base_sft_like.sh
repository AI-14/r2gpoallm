python test_base_sft_like.py \
--sft_test_filepath datasets/iuxray/sft_test.csv \
--po_test_filepath datasets/iuxray/po_test.csv \
--base_model_repo microsoft/Phi-3-mini-4k-instruct \
--logging_dir base-sft-like-logs \
--results_dir base-sft-like-res \
--max_new_tokens 80 \
--hf_token \
--seed 1234
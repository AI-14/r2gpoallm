python test_base_po_like.py \
--sft_test_filepath datasets/covctr/sft_test.csv \
--po_test_filepath datasets/covctr/po_test.csv \
--base_model_repo microsoft/Phi-3-mini-4k-instruct \
--logging_dir base-po-like-logs \
--results_dir base-po-like-res \
--max_new_tokens 100 \
--hf_token \
--seed 1234
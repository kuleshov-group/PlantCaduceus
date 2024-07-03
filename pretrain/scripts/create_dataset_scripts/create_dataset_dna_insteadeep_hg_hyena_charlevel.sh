#!/bin/bash
# run from project roots
source scripts/setup_env.sh
conda activate llmlib
python scripts/convert_bert_dataset_to_mds_streaming.py --dataset InstaDeepAI/human_reference_genome  --tokenizer llmlib/llmlib/tokenization/hg38_char_tokenizer-maxlen-2048 --out_root=./my-dna-dataset-instadeep-hg-charlevel-no-concat --splits train validation --compression zstd

source scripts/setup_env.sh
conda activate llmlib
python scripts/convert_bert_dataset_to_mds_streaming.py --dataset InstaDeepAI/human_reference_genome  --tokenizer   "InstaDeepAI/nucleotide-transformer-500m-human-ref" --out_root=./my-dna-dataset-instadeep-hg-no-concat --splits train validation

# accelerate launch src/zero-shot-eval.py motif_acc \
#   --model model/PlantCAD2-Small-l24-d0768 \
#   --input_tsv data/GCF_002870075.5_Lsat_Salinas_v15_genomic_rand_chunk_1024_samples_100k.tsv \
#   --mask_idx 510,511,512 \
#   --motif_len 3 \
#   --batch_size 10
accelerate launch src/zero-shot-eval.py motif_acc \
  --model model/plantcad2_small_lettuce_20260511_230631 \
  --input_tsv data/GCF_002870075.5_Lsat_Salinas_v15_genomic_rand_chunk_1024_samples_100k.tsv \
  --mask_idx 510,511,512 \
  --motif_len 3 \
  --batch_size 10


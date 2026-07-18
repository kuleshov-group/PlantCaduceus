python src/visualize_confidence.py \
  --models model/PlantCAD2-Small-l24-d0768/ model/plantcad2_small_lettuce_20260511_230631 \
  --fna data/Salinas_v15.ltr.ssr.hardmasked.fa \
  --bed data/genes.bed \
  --gff data/GCF_002870075.5_Lsat_Salinas_v15_genomic.gff \
  --regulator data/regulator.tab \
  --output-dir ./confidence_plots_masked/
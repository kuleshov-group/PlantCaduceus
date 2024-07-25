# [PlantCaduceus: Cross-species Modeling of Plant Genomes at Single Nucleotide Resolution](https://plantcaduceus.github.io/)

## Requirements
```
pytorch >= 2.0
transformers >= 4.0
mamba-ssm<=2.0.0
```

## Using PlantCaduceus with Hugging Face

The example notebook to use PlantCaduceus to get embeddings and logits score is available in the `notebooks/examples.ipynb` directory. 

Pre-trained PlantCaduceus models have been uploaded to Hugging Face. The available models are:
- PlantCaduceus_l20: [kuleshov-group/PlantCaduceus_l20](https://huggingface.co/kuleshov-group/PlantCaduceus_l20)
    - Trained on sequences of length 512bp, with a model size of 256 and 20 layers.
- PlantCaduceus_l24: [kuleshov-group/PlantCaduceus_l24](https://huggingface.co/kuleshov-group/PlantCaduceus_l24)
    - Trained on sequences of length 512bp, with a model size of 256 and 24 layers.
- PlantCaduceus_l28: [kuleshov-group/PlantCaduceus_l28](https://huggingface.co/kuleshov-group/PlantCaduceus_l28)
    - Trained on sequences of length 512bp, with a model size of 256 and 28 layers.
- PlantCaduceus_l32: [kuleshov-group/PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32)
    - Trained on sequences of length 512bp, with a model size of 256 and 32 layers.


## Fine-tune PlantCaduceus
We fine-tuned the PlantCaduceus by training an XGBoost model on top of the embedding for each task. The fine-tuning script is available in the `src` directory. The script takes the following arguments:

```
python src/fine_tune.py \
    -train train.tsv \ # training data, data format: https://huggingface.co/datasets/kuleshov-group/cross-species-single-nucleotide-annotation/tree/main/TIS
    -valid valid.tsv \ # validation data, the same format as the training data
    -test test_rice.tsv \ # test data (optional), the same format as the training data
    -model 'kuleshov-group/PlantCaduceus_l20' \ # pre-trained model name
    -output ./output \ # output directory
    -device 'cuda:1' # GPU device to dump embeddings
```

## Zero-shot score to estimate mutation effect
We used the log-likelihood difference between the reference and the alternative alleles to estimate the mutation effect. The script is available in the `src` directory. The script takes the following arguments:
```
python src/zero_shot_score.py \
    -input examples/example_snp.tsv \ 
    -output output.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \ # pre-trained model name
    -device 'cuda:1' # GPU device to dump embeddings
```

The data structure of `example_snp.tsv` is as follows:
```
chr	start	end	pos	ref	alt	sequences
chr1	315858	316370	316114	C	T	CTCTCCCGATGTCCTCTCGTCGTCTTATCCAGATTCCGGAGCCGATCGAACGCAGGGGAGATACACTCCAGTGGGCATGAAGTGGTTCCAAACCCCAATACCGAAGCGTTGAGTCGATTCGCTCGCTGCTGAAGTGGTTCCTTGCATGGCCGGAGCCAGTGCGTCCTGCTCCATGGCCGCCGGAGCCTGCTGCGCCTGCCCATCGTCACTTTTCCCCCACCGCCGTCCTCGGCGCCACTCCCGCACCATCATGTGCAGCTCCCTTCCGTCCTTATCCCTGCCATTCCCAATCCCAGCGCCTACCAGCGGCGGCGGACGCCTGCGCATCTTCTCCGGCAGCGCCAACCCGGTGCTGGCGCAGGAGATCGCGTGCTACCTTGGGATGGAGCTGGGCCAGATCAAGATCAAGCGGTTCGCGGATGGCGAGATCTACGTGCAGCTGCAAGAGAGCGTGCGTGGCTGCGACGTGTTCCTGGTGCAGCCCACCTGCCCTCCCGCCAACGAGAACCTCA
```

## Pre-train PlantCaduceus with huggingface
```
WANDB_PROJECT=PlantCaduceus python src/HF_pre_train.py --do_train 
    --report_to wandb --prediction_loss_only True --remove_unused_columns False --dataset_name 'kuleshov-group/Angiosperm_16_genomes' --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 --optim adamw_torch \
    --dataloader_num_workers 16 --preprocessing_num_workers 16 --seed 32 \
    --save_strategy steps --save_steps 1000 --evaluation_strategy steps --eval_steps 1000 --logging_steps 10 \
    --max_steps 120000 --warmup_steps 1000 \
    --save_total_limit 20 --learning_rate 2E-4 --lr_scheduler_type constant_with_warmup \
    --run_name test --overwrite_output_dir \
    --output_dir "PlantCaduceus_train_1" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 4 --tokenizer_name 'kuleshov-group/PlantCaduceus_l20' --config_name 'kuleshov-group/PlantCaduceus_l20'
```


## Citation
```bibtex
@article {Zhai2024.06.04.596709,
    author = {Zhai, Jingjing and Gokaslan, Aaron and Schiff, Yair and Berthel, Ana and Liu, Zong-Yan and Miller, Zachary R and Scheben, Armin and Stitzer, Michelle C and Romay, Cinta and Buckler, Edward S. and Kuleshov, Volodymyr},
    title = {Cross-species plant genomes modeling at single nucleotide resolution using a pre-trained DNA language model},
    elocation-id = {2024.06.04.596709},
    year = {2024},
    doi = {10.1101/2024.06.04.596709},
    URL = {https://www.biorxiv.org/content/early/2024/06/05/2024.06.04.596709},
    eprint = {https://www.biorxiv.org/content/early/2024/06/05/2024.06.04.596709.full.pdf},
    journal = {bioRxiv}
}
```
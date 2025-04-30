![Static Badge](https://img.shields.io/badge/Linux-blue?logo=Linux&logoColor=white)
![GitHub Repo stars](https://img.shields.io/github/stars/kuleshov-group/PlantCaduceus)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/kuleshov-group/PlantCaduceus)
[![DOI](https://zenodo.org/badge/DOI/10.1101/2024.06.04.596709.svg)](https://doi.org/10.1101/2024.06.04.596709)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg?style=flat)](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a)
<a href="https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a">
  <img alt="Hugging Face Downloads" src="https://img.shields.io/badge/dynamic/json?color=blue&label=downloads&query=downloads&url=https://huggingface.co/api/models/kuleshov-group/PlantCaduceus_l32">
</a>
<p align="center">
  <img src="img/logo.jpg" alt="logo" width="20%">
</p>

## Table of Contents
- [PlantCAD overview](#plantcad-overview)
- [Model summary](#plantcad-model-summary)
- [Run PlantCaduceus demo on Google Colab](#run-plantcaduceus-demo-on-google-colab)
- [Run PlantCaduceus locally (creating conda environment)](#run-plantcaduceus-locally-creating-conda-environment)
  - [Test if mamba_ssm is installed correctly](#testing-if-mamba_ssm-is-installed-correctly)
- [Train an XGBoost classifier using PlantCAD embeddings](#train-an-xgboost-classifier-using-plantcad-embeddings)
- [Use the trained XGBoost classifiers](#use-the-trained-xgboost-classifiers)
- [Estimate mutation effect with zero-shot strategy](#zero-shot-score-to-estimate-mutation-effect)
- [Inference speed test](#inference-speed-test)
- [Pre-train PlantCAD with huggingface](#pre-train-plantcad-with-huggingface)
- [Citation](#citation)

## [PlantCAD overview](https://plantcaduceus.github.io/)

PlantCaduceus, with its short name of **PlantCAD**, is a plant DNA LM based on the [Caduceus](https://arxiv.org/abs/2403.03234) architecture, which extends the efficient [Mamba](https://arxiv.org/abs/2312.00752) linear-time sequence modeling framework to incorporate bi-directionality and reverse complement equivariance, specifically designed for DNA sequences. PlantCAD is pre-trained on a curated dataset of 16 Angiosperm genomes. PlantCAD showed state-of-the-art cross species performance in predicting TIS, TTS, Splice Donor and Splice Acceptor. The zero-shot of PlantCAD enables identifying genome-wide deleterious mutations and known causal variants in Arabidopsis, Sorghum and Maize.

## PlantCAD model summary
Pre-trained PlantCAD models have been uploaded to [HuggingFace 🤗](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a). Here's the summary of four PlantCAD models with different parameter sizes.
| Model | Sequence Length | Model Size | Embedding Size |
|-------|----------------|------------|----------------|
| [PlantCaduceus_l20](https://huggingface.co/kuleshov-group/PlantCaduceus_l20) | 512bp | 20M | 384 |
| [PlantCaduceus_l24](https://huggingface.co/kuleshov-group/PlantCaduceus_l24) | 512bp | 40M | 512 |
| [PlantCaduceus_l28](https://huggingface.co/kuleshov-group/PlantCaduceus_l28) | 512bp | 128M | 768 |
| [PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32) | 512bp | 225M | 1024 |

## Run PlantCaduceus demo on Google Colab
Here's an example notebook to show how to run PlantCAD on google colab: [PlantCAD google colab](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=drive_link)

**Note**: please make sure to set runtime to GPU/TPU when running google colab notebooks

## Run PlantCaduceus locally (creating conda environment)
```
conda env create -f env/environment.yml
conda activate PlantCAD
pip install -r env/requirements.txt --no-build-isolation
```

#### Test if mamba_ssm is installed correctly
```python
import torch
from mamba_ssm import Mamba
```

- If not, please re-install mamba_ssm by running the following command:
```bash
pip uninstall mamba-ssm
pip install mamba-ssm==2.2.0 --no-build-isolation
```
The example notebook to use PlantCAD to get embeddings and logits score is available in the `notebooks/examples.ipynb` directory. 


## Train an XGBoost classifier using PlantCAD embeddings
We trained an XGBoost model on top of the PlantCAD embedding for each task to evaluate its performance. The script is available in the `src` directory. The script takes the following arguments:

```
python src/train_XGBoost.py \
    -train train.tsv \ # training data, data format: https://huggingface.co/datasets/kuleshov-group/cross-species-single-nucleotide-annotation/tree/main/TIS
    -valid valid.tsv \ # validation data, the same format as the training data
    -test test_rice.tsv \ # test data (optional), the same format as the training data
    -model 'kuleshov-group/PlantCaduceus_l20' \ # pre-trained model name
    -output ./output \ # output directory
    -device 'cuda:0' # GPU device to dump embeddings
```

## Use the trained XGBoost classifiers
The trained XGBoost classifiers in the paper are available [here](classifiers), the following script is used for prediction with XGBoost model
```
python src/predict_XGBoost.py \
    -test test_rice.tsv \           
    -model 'kuleshov-group/PlantCaduceus_l20' \ # pre-trained model
    -classifier classifiers/PlantCaduceus_l20/TIS_XGBoost.json \  # the trained XGBoost classifier
    -device 'cuda:0' \ # GPU device to dump embeddings
    -output ./output # output directory
```


## Zero-shot score to estimate mutation effect
We used the log-likelihood difference between the reference and the alternative alleles to estimate the mutation effect. The script is available in the `src` directory. 

#### Using vcf files as input
```
python src/zero_shot_score.py \
    -input-vcf examples/example_maize_snp.vcf \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output example_output.vcf \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -device 'cuda:0'
```

The maize reference genome can be downloaded from [here](https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0.fa.gz)

#### Prepare input files from VCF

Here's the pipeline to prepare input files from VCF
```bash
# prepare bed
inputVCF="input.vcf"
genomeFA="genome.fa"
output="snp_info.tsv" # this could be the input file of the zero_shot_score.py code
grep -v '#' ${inputVCF} | awk -v OFS="\t" '{print $1,$2-256,$2+256}' > ${inputVCF}.bed
bedtools getfasta -tab -fi ${genomeFA} -bed ${inputVCF}.bed -fo ${inputVCF}.seq.tsv
awk -v OFS="\t" '{print $1,$2-256, $2+256,$2,$4,$5}' ${inputVCF} | paste - <(cut -f2 ${inputVCF}.seq.tsv) > ${output}_tmp
# add header
echo -e "chr\tstart\tend\tpos\tref\talt\tsequences" > ${output}
cat ${output}_tmp >> ${output}
rm ${inputVCF}.bed ${inputVCF}.seq.tsv ${output}_tmp
```

Then getting zero-shot scores with this code.

```
python src/zero_shot_score.py \
    -input examples/example_snp.tsv \ 
    -output output.tsv \
    -model 'kuleshov-group/PlantCaduceus_l32' \ # pre-trained model name
    -device 'cuda:0' # GPU device to dump embeddings
```

**Note**: we would highly recommend using the largest model ([PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32)) for the zero-shot score estimation.


## Inference speed test
The inference speed is highly dependent on the model size and GPU type, we tested on some commonly used GPUs. With 5,000 SNPs, the inference speed is as follows:

<table>
    <tr>
        <th>Model</th>
        <th>H100</th>
        <th>A100</th>
        <th>A6000</th>
        <th>3090</th>
        <th>A5000</th>
        <th>A40</th>
        <th>2080</th>
    </tr>
    <tr>
        <td>PlantCaduceus_l20</td>
        <td>16s</td>
        <td>19s</td>
        <td>24s</td>
        <td>25s</td>
        <td>25s</td>
        <td>26s</td>
        <td>44s</td>
    </tr>
    <tr>
        <td>PlantCaduceus_l24</td>
        <td>21s</td>
        <td>27s</td>
        <td>35s</td>
        <td>37s</td>
        <td>42s</td>
        <td>38s</td>
        <td>71s</td>
    </tr>
    <tr>
        <td>PlantCaduceus_l28</td>
        <td>31s</td>
        <td>43s</td>
        <td>62s</td>
        <td>69s</td>
        <td>77s</td>
        <td>67s</td>
        <td>137s</td>
    </tr>
    <tr>
        <td>PlantCaduceus_l32</td>
        <td>47s</td>
        <td>66s</td>
        <td>94s</td>
        <td>116s</td>
        <td>130s</td>
        <td>107s</td>
        <td>232s</td>
    </tr>
</table>

## Pre-train PlantCAD with huggingface
```
WANDB_PROJECT=PlantCAD python src/HF_pre_train.py --do_train 
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
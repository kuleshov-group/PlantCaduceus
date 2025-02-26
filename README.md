# [PlantCaduceus: Cross-species Modeling of Plant Genomes at Single Nucleotide Resolution](https://plantcaduceus.github.io/)

## Creating the PlantCAD environment
```
conda env create -f env/environment.yml
conda activate PlantCAD
pip install -r env/requirements.txt --no-build-isolation
```

#### Testing if mamba_ssm is installed correctly
```python
import torch
from mamba_ssm import Mamba
```

- If not, please re-install mamba_ssm by running the following command:
```bash
pip uninstall mamba-ssm
pip install mamba-ssm==2.2.0 --no-build-isolation
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


## Training an XGBoost classifier using PlantCaduceus embeddings
We trained an XGBoost model on top of the PlantCaduceus embedding for each task to evaluate its performance. The script is available in the `src` directory. The script takes the following arguments:

```
python src/train_XGBoost.py \
    -train train.tsv \ # training data, data format: https://huggingface.co/datasets/kuleshov-group/cross-species-single-nucleotide-annotation/tree/main/TIS
    -valid valid.tsv \ # validation data, the same format as the training data
    -test test_rice.tsv \ # test data (optional), the same format as the training data
    -model 'kuleshov-group/PlantCaduceus_l20' \ # pre-trained model name
    -output ./output \ # output directory
    -device 'cuda:0' # GPU device to dump embeddings
```

## Using the trained XGBoost classifiers
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
We used the log-likelihood difference between the reference and the alternative alleles to estimate the mutation effect. The script is available in the `src` directory. The script takes the following arguments:
```
python src/zero_shot_score.py \
    -input examples/example_snp.tsv \ 
    -output output.tsv \
    -model 'kuleshov-group/PlantCaduceus_l32' \ # pre-trained model name
    -device 'cuda:1' # GPU device to dump embeddings
```

**Note**: we would highly recommend using the largest model ([PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32)) for the zero-shot score estimation.

- We also provide a pipeline to generate input files from VCF and genome FASTA files
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


### Inference speed test
The inference speed is highly dependent on the model size and GPU type, we tested on some commonly used GPUs. With 5,000 SNPs, the inference speed is as follows:

<table>
    <tr>
        <th>Model</th>
        <th>GPU</th>
        <th>Time</th>
    </tr>
    <tr>
        <td rowspan="7" style="text-align:center; vertical-align:middle;">PlantCaduceus_l20</td>
        <td>H100</td>
        <td>16s</td>
    </tr>
    <tr>
        <td>A100</td>
        <td>19s</td>
    </tr>
    <tr>
        <td>A6000</td>
        <td>24s</td>
    </tr>
    <tr>
        <td>3090</td>
        <td>25s</td>
    </tr>
    <tr>
        <td>A5000</td>
        <td>25s</td>
    </tr>
    <tr>
        <td>A40</td>
        <td>26s</td>
    </tr>
    <tr>
        <td>2080</td>
        <td>44s</td>
    </tr>
    <tr>
        <td rowspan="7" style="text-align:center; vertical-align:middle;">PlantCaduceus_l24</td>
        <td>H100</td>
        <td>21s</td>
    </tr>
    <tr>
        <td>A100</td>
        <td>27s</td>
    </tr>
    <tr>
        <td>A6000</td>
        <td>35s</td>
    </tr>
    <tr>
        <td>3090</td>
        <td>37s</td>
    </tr>
    <tr>
        <td>A40</td>
        <td>38s</td>
    </tr>
    <tr>
        <td>A5000</td>
        <td>42s</td>
    </tr>
    <tr>
        <td>2080</td>
        <td>71s</td>
    </tr>
    <tr>
        <td rowspan="7" style="text-align:center; vertical-align:middle;">PlantCaduceus_l28</td>
        <td>H100</td>
        <td>31s</td>
    </tr>
    <tr>
        <td>A100</td>
        <td>43s</td>
    </tr>
    <tr>
        <td>A6000</td>
        <td>62s</td>
    </tr>
    <tr>
        <td>A40</td>
        <td>67s</td>
    </tr>
    <tr>
        <td>3090</td>
        <td>69s</td>
    </tr>
    <tr>
        <td>A5000</td>
        <td>77s</td>
    </tr>
    <tr>
        <td>2080</td>
        <td>137s</td>
    </tr>
    <tr>
        <td rowspan="7" style="text-align:center; vertical-align:middle;">PlantCaduceus_l32</td>
        <td>H100</td>
        <td>47s</td>
    </tr>
    <tr>
        <td>A100</td>
        <td>66s</td>
    </tr>
    <tr>
        <td>A6000</td>
        <td>94s</td>
    </tr>
    <tr>
        <td>A40</td>
        <td>107s</td>
    </tr>
    <tr>
        <td>3090</td>
        <td>116s</td>
    </tr>
    <tr>
        <td>A5000</td>
        <td>130s</td>
    </tr>
    <tr>
        <td>2080</td>
        <td>232s</td>
    </tr>
</table>

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
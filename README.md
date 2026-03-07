![Static Badge](https://img.shields.io/badge/Linux-blue?logo=Linux&logoColor=white)
![GitHub Repo stars](https://img.shields.io/github/stars/kuleshov-group/PlantCaduceus)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/kuleshov-group/PlantCaduceus)
[![PlantCAD](https://zenodo.org/badge/DOI/10.1073/pnas.2421738122.svg)](https://doi.org/10.1073/pnas.2421738122)
![PlantCAD Downloads](https://img.shields.io/badge/dynamic/json?color=blue&label=PlantCAD+downloads&query=downloads&url=https://huggingface.co/api/models/kuleshov-group/PlantCaduceus_l32)
[![PlantCAD2](https://zenodo.org/badge/DOI/10.1101/2025.08.27.672609.svg)](https://doi.org/10.1101/2025.08.27.672609)
![PlantCAD2 Downloads](https://img.shields.io/badge/dynamic/json?color=blue&label=PlantCAD2+downloads&query=downloads&url=https://huggingface.co/api/models/kuleshov-group/PlantCAD2-Small-l24-d0768)
[![PlantCAD 🤗](https://img.shields.io/badge/🤗-PlantCAD-yellow.svg?style=flat)](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a)
[![PlantCAD2 🤗](https://img.shields.io/badge/🤗-PlantCAD2-yellow.svg?style=flat)](https://huggingface.co/collections/kuleshov-group/plantcad2-67e437e241a382671371a572)

<p align="center">
  <img src="img/logo.jpg" alt="logo" width="20%">
</p>

## 🚀 [PlantCAD2](https://huggingface.co/collections/kuleshov-group/plantcad2-67e437e241a382671371a572) is here! ([paper](https://www.biorxiv.org/content/10.1101/2025.08.27.672609v3))

A new DNA foundation model for angiosperms, with [LoRA fine-tuned models](https://huggingface.co/collections/plantcad/fine-tuned-plantcad2-models-68b316a57616134fa7a1b6b6) for accessible chromatin, gene expression, and protein translation.

## Table of Contents

- [PlantCAD overview](#plantcad-overview)
- [Quick Start](#quick-start)
- [Model summary](#model-summary)
- [Installation](#installation)
- [Quick example](#quick-example)
- [Usage guides](#usage-guides)
- [Citations](#citations)

## [PlantCAD overview](https://plantcaduceus.github.io/)

PlantCaduceus, with its short name of **PlantCAD**, is a plant DNA LM based on the [Caduceus](https://arxiv.org/abs/2403.03234) architecture, which extends the efficient [Mamba](https://arxiv.org/abs/2312.00752) linear-time sequence modeling framework to incorporate bi-directionality and reverse complement equivariance, specifically designed for DNA sequences. PlantCAD is pre-trained on a curated dataset of 16 Angiosperm genomes. PlantCAD showed state-of-the-art cross species performance in predicting TIS, TTS, Splice Donor and Splice Acceptor. The zero-shot of PlantCAD enables identifying genome-wide deleterious mutations and known causal variants in Arabidopsis, Sorghum and Maize.

## Quick Start

**New to PlantCAD?** Try our [Google Colab demo](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=sharing) - no installation required!

**For local usage:** See installation instructions [here](docs/local-install.md), then use `notebooks/examples.ipynb` to get started.

## Model summary

Pre-trained models have been uploaded to **HuggingFace 🤗**: [PlantCAD](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a) and [PlantCAD2](https://huggingface.co/collections/plantcad/fine-tuned-plantcad2-models-68b316a57616134fa7a1b6b6).

| Model | Max Input Length | Model Size | Embedding Size |
| :--- | :--- | :--- | :--- |
| **PlantCAD** | | | |
| [PlantCaduceus_l20](https://huggingface.co/kuleshov-group/PlantCaduceus_l20) | 512bp | 20M | 384 |
| [PlantCaduceus_l24](https://huggingface.co/kuleshov-group/PlantCaduceus_l24) | 512bp | 40M | 512 |
| [PlantCaduceus_l28](https://huggingface.co/kuleshov-group/PlantCaduceus_l28) | 512bp | 128M | 768 |
| [PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32) | 512bp | 225M | 1024 |
| **PlantCAD2** | | | |
| [PlantCAD2-Small](https://huggingface.co/kuleshov-group/PlantCAD2-Small-l24-d0768) | 8192bp | 88M | 768 |
| [PlantCAD2-Medium](https://huggingface.co/kuleshov-group/PlantCAD2-Medium-l48-d1024) | 8192bp | 311M | 1024 |
| [PlantCAD2-Large](https://huggingface.co/kuleshov-group/PlantCAD2-Large-l48-d1536) | 8192bp | 694M | 1536 |

> **⚠️ Important:** The "Max Input Length" is a hard limit — your input sequences **cannot** exceed this length. Use `-contextSize 512` for PlantCAD models and up to `-contextSize 8192` for PlantCAD2 models. See [Model Recommendations](docs/model-recommendations.md) for guidance on which model to use.

## Installation

| Option | Best for |
| :--- | :--- |
| [Google Colab](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=sharing) | Beginners — no installation required |
| [Local installation](docs/local-install.md) | Regular use — requires NVIDIA GPU |
| [Docker](docker/README.md) | Reproducible environments |

## Quick example

Get sequence embeddings with PlantCAD:

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
model = AutoModelForMaskedLM.from_pretrained(
    'kuleshov-group/PlantCaduceus_l32', trust_remote_code=True
).to(device)

sequence = "CTTAATTAATATTGCCTTTGTAATAACGCGCGAAACACAAATCTTCTCTGCCTAATGCAGTAGTCATGTGTTGACTCCTTCAAAATTTCCAAGAAGTTAGTGGCTGGTGTGTCATTGTCTTCATCTTTTTTTTTTTTTTTTTAAAAATTGAATGCGACATGTACTCCTCAACGTATAAGCTCAATGCTTGTTACTGAAACATCTCTTGTCTGATTTTTTCAGGCTAAGTCTTACAGAAAGTGATTGGGCACTTCAATGGCTTTCACAAATGAAAAAGATGGATCTAAGGGATTTGTGAAGAGAGTGGCTTCATCTTTCTCCATGAGGAAGAAGAAGAATGCAACAAGTGAACCCAAGTTGCTTCCAAGATCGAAATCAACAGGTTCTGCTAACTTTGAATCCATGAGGCTACCTGCAACGAAGAAGATTTCAGATGTCACAAACAAAACAAGGATCAAACCATTAGGTGGTGTAGCACCAGCACAACCAAGAAGGGAAAAGATCGATGATCG"

input_ids = tokenizer.encode_plus(
    sequence, return_tensors="pt", return_attention_mask=False,
    return_token_type_ids=False
)["input_ids"].to(device)

with torch.inference_mode():
    outputs = model(input_ids=input_ids, output_hidden_states=True)

embeddings = outputs.hidden_states[-1].to(torch.float32).cpu().numpy()

# Average forward and reverse complement embeddings
hidden_size = embeddings.shape[-1] // 2
forward = embeddings[..., 0:hidden_size]
reverse = embeddings[..., hidden_size:][:, ::-1, :]
averaged_embeddings = (forward + reverse) / 2
print(averaged_embeddings.shape)
```

See `notebooks/examples.ipynb` for more detailed examples.

## Usage guides

| Guide | Description |
| :--- | :--- |
| **[Zero-shot SNP & Region Scoring](docs/zero-shot-scoring.md)** | Score variants (VCF) or genomic regions (BED) using log-likelihood ratios |
| **[Zero-shot SV Scoring](docs/zero-shot-scoring-sv.md)** | Score structural variants (deletions & insertions) |
| **[XGBoost Classifiers](docs/xgboost-classifiers.md)** | Train or use pre-trained classifiers for TIS, TTS, splice sites |
| **[In-silico Mutagenesis](pipelines/in-silico-mutagenesis/README.md)** | Large-scale simulation and analysis of genetic variants |
| **[Fine-tuned PlantCAD2 Models](docs/PlantCAD2-overview.md)** | LoRA models for chromatin, expression, translation |
| **[Zero-shot Evaluation](docs/zero-shot-eval.md)** | PlantCAD2 zero-shot benchmark results |
| **[Pre-training](docs/pre-training.md)** | Pre-train or fine-tune PlantCAD models from scratch |
| **[Model Recommendations](docs/model-recommendations.md)** | Which model to use, inference speed benchmarks, GPU memory guide |

## Citations

If you find PlantCAD useful for your research, please consider citing our paper:

- Zhai, J., Gokaslan, A., Schiff, Y., Berthel, A., Liu, Z. Y., Lai, W. L., Miller, Z. R., Scheben, A., Stitzer, M. C., Romay, M. C., Buckler, E. S., & Kuleshov, V. (2025). Cross-species modeling of plant genomes at single nucleotide resolution using a pretrained DNA language model. Proceedings of the National Academy of Sciences, 122(24), e2421738122. https://doi.org/10.1073/pnas.2421738122
- Zhai J., Gokaslan A., Hsu SK., Chen SP., Liu ZY., Marroquin E., Czech E., Cannon B., Berthel A., Romay MC., Pennell M., Kuleshov V.* Buckler ES*. PlantCAD2: A Long-Context DNA Language Model for Cross-Species Functional Annotation in Angiosperms. bioRxiv. 2025. Nov 19. doi: https://doi.org/10.1101/2025.08.27.672609

## Contact

Maintained by **Jingjing Zhai**.

- For collaboration inquiries: [jz963@cornell.edu](mailto:jz963@cornell.edu) or [zhaijingjing603@gmail.com](mailto:zhaijingjing603@gmail.com)
- General questions, bug reports, and feature requests: please [open an issue](https://github.com/plantcad/plantcad/issues)

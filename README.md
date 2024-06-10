# PlantCaduceus
Cross-species plant genomes modeling at single nucleotide resolution using a pre-trained dna language model

## Using PlantCaduceus with Hugging Face

Pre-trained PlantCaduceus models have been uploaded to Hugging Face. The available models are:
- PlantCaduceus_l20: [kuleshov-group/PlantCaduceus_l20](https://huggingface.co/kuleshov-group/PlantCaduceus_l20)
    - Trained on sequences of length 512bp, with a model size of 256 and 20 layers.
- PlantCaduceus_l24: [kuleshov-group/PlantCaduceus_l24](https://huggingface.co/kuleshov-group/PlantCaduceus_l24)
    - Trained on sequences of length 512bp, with a model size of 256 and 24 layers.
- PlantCaduceus_l28: [kuleshov-group/PlantCaduceus_l28](https://huggingface.co/kuleshov-group/PlantCaduceus_l28)
    - Trained on sequences of length 512bp, with a model size of 256 and 28 layers.
- PlantCaduceus_l32: [kuleshov-group/PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32)
    - Trained on sequences of length 512bp, with a model size of 256 and 32 layers.

To use PlantCaduceus with Hugging Face, you can use the following code snippet:

```python
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import torch
model_path = 'kuleshov-group/PlantCaduceus_l32'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

sequence = "ATGCGTACGATCGTAG"
encoding = tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
input_ids = encoding["input_ids"].to(device)
with torch.inference_mode():
    outputs = model(input_ids=input_ids, output_hidden_states=True)
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
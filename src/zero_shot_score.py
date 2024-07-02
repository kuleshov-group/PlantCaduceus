import pandas as pd
import torch,sys
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse, sys, os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", dest="inputDF", type=str, default=None, help="The directory of input tab-separated file")
    parser.add_argument("-output", dest = "output", default=None, help = "The directory of output")
    parser.add_argument("-model", dest = "model", default=None, help = "The directory of pre-trained model")
    parser.add_argument("-device", dest = "device", default="cuda:0", help = "The device to run the model")
    parser.add_argument("-batchSize", dest = "batchSize", default=128, type=int, help = "The batch size for the model")
    parser.add_argument("-numWorkers", dest = "numWorkers", default=4, type=int, help = "The number of workers for the model")
    parser.add_argument("-tokenIdx", dest = "tokenIdx", default=255, type=int, help = "The index of the nucleotide")
    args = parser.parse_args()
    return args

class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']
        input_ids[0, 255] = self.tokenizer.mask_token_id # mask the 255th token
        return {
            'sequence': sequence,
            'input_ids': input_ids
        }

def load_model_and_tokenizer(model_dir, device):
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

def create_dataloader(sequences, tokenizer, batch_size):
    dataset = SequenceDataset(sequences, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def extract_logits(model, dataloader, device, tokenIdx, tokenizer):
    nucleotides = list('acgt')
    results = []
    for batch in tqdm(dataloader, desc="Inference..."):
        curIDs = batch['input_ids'].to(device)
        curIDs = curIDs.squeeze(1)
        with torch.inference_mode():
            outputs = model(input_ids=curIDs)
        all_logits = outputs.logits
        logits = all_logits[:, tokenIdx, [tokenizer.get_vocab()[nc] for nc in nucleotides]] # get the logits for the mask>
        probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
        results.append(probs)
    return np.concatenate(results, axis=0)

def zero_shot_score(snpDF, logits):
    nucleotides = ['A', 'C', 'G', 'T']
    res = []
    for idx, (snp, probs) in enumerate(zip(snpDF.itertuples(index=False), logits)):
        refAllele = getattr(snp, 'ref')
        altAllele = getattr(snp, 'alt')
        refProb = probs[nucleotides.index(refAllele)]
        altProb = probs[nucleotides.index(altAllele)]
        res.append(np.log(altProb / refProb))
    return res

def main():
    args = parse_args()
    snpDF = pd.read_csv(args.inputDF, delimiter='\t')
    sequences = snpDF['sequences'].tolist()
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    loader = create_dataloader(sequences, tokenizer, args.batchSize)
    logits = extract_logits(model, loader, args.device, args.tokenIdx, tokenizer)
    scores = zero_shot_score(snpDF, logits)
    snpDF['zeroShotScore'] = scores
    snpDF.to_csv(args.output, sep='\t', index=False)

if __name__ == "__main__":
    main()

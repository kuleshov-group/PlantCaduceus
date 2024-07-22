import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse, os
from tqdm import tqdm
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", dest="inputDF", type=str, default=None, help="The directory of input tab-separated file")
    parser.add_argument("-output", dest="output", default=None, help="The directory of output")
    parser.add_argument("-model", dest="model", default=None, help="The directory of pre-trained model")
    parser.add_argument("-device", dest="device", default="cuda:0", help="The device to run the model")
    parser.add_argument("-batchSize", dest="batchSize", default=128, type=int, help="The batch size for the model")
    parser.add_argument("-numWorkers", dest="numWorkers", default=4, type=int, help="The number of workers for the model")
    parser.add_argument("-tokenIdx", dest="tokenIdx", default=255, type=int, help="The index of the nucleotide to mask")
    args = parser.parse_args()
    return args

class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, tokenIdx):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.tokenIdx = tokenIdx

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
        input_ids[0, self.tokenIdx] = self.tokenizer.mask_token_id # mask the specified token index
        return {
            'sequence': sequence,
            'input_ids': input_ids
        }

def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

def create_dataloader(sequences, tokenizer, batch_size, tokenIdx):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, tokenizer, tokenIdx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def extract_logits(model, dataloader, device, tokenIdx, tokenizer):
    logging.info("Extracting logits")
    nucleotides = list('acgt')
    results = []
    for batch in tqdm(dataloader, desc="Inference..."):
        curIDs = batch['input_ids'].to(device)
        curIDs = curIDs.squeeze(1)
        with torch.inference_mode():
            outputs = model(input_ids=curIDs)
        all_logits = outputs.logits
        logits = all_logits[:, tokenIdx, [tokenizer.get_vocab()[nc] for nc in nucleotides]] # get the logits for the masked token
        probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
        results.append(probs)
    return np.concatenate(results, axis=0)

def zero_shot_score(snpDF, logits):
    logging.info("Calculating zero-shot scores")
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
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    
    logging.info(f"Reading input data from {args.inputDF}")
    snpDF = pd.read_csv(args.inputDF, delimiter='\t')
    sequences = snpDF['sequences'].tolist()

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    logging.info("Creating data loader")
    loader = create_dataloader(sequences, tokenizer, args.batchSize, args.tokenIdx)

    logging.info("Extracting logits")
    logits = extract_logits(model, loader, args.device, args.tokenIdx, tokenizer)

    logging.info("Calculating zero-shot scores")
    scores = zero_shot_score(snpDF, logits)

    snpDF['zeroShotScore'] = scores
    logging.info(f"Writing output data to {args.output}")
    snpDF.to_csv(args.output, sep='\t', index=False)

if __name__ == "__main__":
    main()
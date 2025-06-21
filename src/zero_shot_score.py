import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse, sys
from tqdm import tqdm
import logging
import vcf
from Bio import SeqIO
import gzip


def parse_args():
    parser = argparse.ArgumentParser()
    inputGroup = parser.add_mutually_exclusive_group(required=True)
    inputGroup.add_argument("-input-table", dest="inputDF", type=str, default=None,
                            help="The directory of input tab-separated file. Required columns: ref, alt, sequences")
    inputGroup.add_argument("-input-vcf", dest="inputVCF", type=str, default=None,
                            help="The directory of input vcf")
    parser.add_argument("-input-fasta", dest="inputFasta", type=str, default=None,
                        help="The directory of input fasta. Required if using VCF")
    parser.add_argument("-output", dest="output", default=None, help="The directory of output")
    parser.add_argument("-outBED", action="store_true", dest="outBED", default=False,
                        help="Output in BED format instead of tab-separated file, only works with -input-table")
    parser.add_argument("-model", dest="model", default=None, help="The directory of pre-trained model")
    parser.add_argument("-device", dest="device", default="cuda:0", help="The device to run the model")
    parser.add_argument("-batchSize", dest="batchSize", default=128, type=int, help="The batch size for the model")
    parser.add_argument("-numWorkers", dest="numWorkers", default=4, type=int,
                        help="The number of workers for the model")
    parser.add_argument("-tokenIdx", dest="tokenIdx", default=255, type=int, help="The index of the nucleotide to mask")
    args = parser.parse_args()

    if args.inputVCF is not None and args.inputFasta is None:
        sys.exit("-input-fasta is required with -input-vcf")

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
        input_ids[0, self.tokenIdx] = self.tokenizer.mask_token_id  # mask the specified token index
        return {
            'sequence': sequence,
            'input_ids': input_ids
        }


def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")

    # Determine the appropriate dtype based on the GPU capabilities
    def get_optimal_dtype():
        if not torch.cuda.is_available():
            logging.info("Using float32 as no GPU is available.")
            return torch.float32

        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)

        if capability[0] >= 8:  # sm_80 or higher
            logging.info("Using bfloat16 as the GPU supports sm_80 or higher.")
            return torch.bfloat16
        elif capability[0] >= 6:  # sm_60 or higher
            logging.info("Using float16 as the GPU supports sm_60 or higher.")
            return torch.float16
        else:
            logging.info("Using float32 as the GPU does not support float16 or bfloat16.")
            return torch.float32

    optimal_dtype = get_optimal_dtype()

    # Load the model with the selected dtype
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)

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
    for batch in tqdm(dataloader):
        curIDs = batch['input_ids'].to(device)
        curIDs = curIDs.squeeze(1)
        with torch.inference_mode():
            outputs = model(input_ids=curIDs)
        all_logits = outputs.logits
        logits = all_logits[:, tokenIdx,
                 [tokenizer.get_vocab()[nc] for nc in nucleotides]]  # get the logits for the masked token
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


def zero_shot_score_vcf(args, recordIndices, logits):
    logging.info("Calculating zero-shot scores")
    nucleotides = ['A', 'C', 'G', 'T']

    reader = vcf.Reader(filename=args.inputVCF)
    writer = vcf.Writer(open(args.output, "w"), reader)

    currentIdx = 0

    rIdx = 0
    for record in reader:
        if currentIdx == recordIndices[rIdx]:
            scores = []

            refAllele = record.REF
            refProb = logits[rIdx][nucleotides.index(refAllele)]

            for alt in record.ALT:
                if alt.type == "SNV":
                    altAllele = alt.sequence
                    altProb = logits[rIdx][nucleotides.index(altAllele)]
                    scores.append(np.log(altProb / refProb))
                else:
                    scores.append(".")

            record.INFO["plantCAD_zero_shot"] = ",".join(str(zs) for zs in scores)

            writer.write_record(record)

            rIdx += 1
        currentIdx += 1

    writer.close()


def seq_from_vcf(args):
    logging.info(f"Reading input data from {args.inputVCF}")

    # load fasta
    if args.inputFasta.endswith(".gz"):
        with gzip.open(args.inputFasta, "rt") as file:
            fastaDict = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    else:
        fastaDict = SeqIO.to_dict(SeqIO.parse(args.inputFasta, "fasta"))

    sequences = []
    recordIndices = []  # keep track of which record a sequence belongs to
    vcfReader = vcf.Reader(filename=args.inputVCF)
    recordIdx = 0
    prevChrom = ""
    addIdx = 512 - args.tokenIdx

    for record in vcfReader:
        if any([alt.type == "SNV" for alt in record.ALT]):
            chrom = record.CHROM
            pos = record.POS - 1  # convert to zero-based indexing

            try:
                if pos - args.tokenIdx < 0:
                    seq = str(fastaDict[chrom][0:pos + addIdx].seq).upper().rjust(512, "N")
                else:
                    seq = str(fastaDict[chrom][pos - args.tokenIdx:pos + addIdx].seq).upper().ljust(512, "N")

                sequences.append(seq)
                recordIndices.append(recordIdx)

                # RAM saving: remove chromosomes when the corresponding vcf records have been processed
                if prevChrom != chrom:
                    if prevChrom != "":
                        fastaDict.pop(prevChrom)
                    prevChrom = chrom
            except:
                print("Error processing VCF at record " + str(recordIdx))
                print("Check that VCF file is sorted and chromosome names match FASTA file.")
                print(record)
                sys.exit()
        recordIdx += 1
    return sequences, recordIndices


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()

    if args.inputDF is not None:

        logging.info(f"Reading input data from {args.inputDF}")
        snpDF = pd.read_csv(args.inputDF, delimiter='\t')
        logging.info("Filtering out invalid SNPs")
        logging.info(
            f"Filtered out {len(snpDF) - len(snpDF[snpDF['ref'].isin(['A', 'C', 'G', 'T']) & snpDF['alt'].isin(['A', 'C', 'G', 'T'])])} invalid SNPs")
        snpDF = snpDF[snpDF['ref'].isin(['A', 'C', 'G', 'T']) & snpDF['alt'].isin(['A', 'C', 'G', 'T'])]
        sequences = snpDF['sequences'].tolist()
    else:  # get sequence from fasta
        sequences, recordIndices = seq_from_vcf(args)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    logging.info("Creating data loader")
    loader = create_dataloader(sequences, tokenizer, args.batchSize, args.tokenIdx)

    logits = extract_logits(model, loader, args.device, args.tokenIdx, tokenizer)

    if args.inputDF is not None:
        scores = zero_shot_score(snpDF, logits)

        snpDF['zeroShotScore'] = scores
        if args.outBED:
            logging.info("Outputting results in BED format")
            snpDF['start'] = snpDF['pos'] - 1
            snpDF['end'] = snpDF['pos']
            snpDF = snpDF[['chr', 'start', 'end', 'ref', 'alt', 'zeroShotScore']]
            snpDF.to_csv(args.output, sep='\t', index=False, header=False)
        else:
            logging.info("Outputting results in tab-separated format")
            snpDF.to_csv(args.output, sep='\t', index=False)
    else:
        zero_shot_score_vcf(args, recordIndices, logits)
    logging.info(f"Zero-shot scores saved to {args.output}")


if __name__ == "__main__":
    main()

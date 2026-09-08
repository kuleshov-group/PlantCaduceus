import pandas as pd
import torch
from accelerate import Accelerator
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
    parser = argparse.ArgumentParser(
        description="Score Structural Variants (SVs) using a pre-trained language model."
    )
    parser.add_argument("-input-vcf", dest="inputVCF", type=str, required=True,
                        help="The input VCF file containing SVs (Deletions).")
    parser.add_argument("-input-fasta", dest="inputFasta", type=str, required=True,
                        help="The directory of reference genome fasta file.")
    parser.add_argument("-output", dest="output", required=True, help="The output VCF file.")
    parser.add_argument("-model", dest="model", required=True, help="The directory of pre-trained model.")
    parser.add_argument("-device", dest="device", default="cuda:0", help="The device to run the model (default: cuda:0).")
    parser.add_argument("-batchSize", dest="batchSize", default=32, type=int, help="The batch size for the model (default: 32).")
    parser.add_argument("-contextSize", dest="contextSize", default=8192, type=int,
                        help="The context window size (default: 8192).")
    parser.add_argument("-flank-size", dest="flankSize", default=5, type=int,
                        help="The size of the flank to score around the SV junction (default: 5).")
    
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
            return_token_type_ids=False,
            truncation=True, # Should not happen if contextSize is managed correctly, but good safety
            max_length=len(sequence) # Allow some buffer
        )
        input_ids = encoding['input_ids']
        return {
            'input_ids': input_ids
        }


def load_model_and_tokenizer(model_dir, device):
    logging.info(f"Loading model and tokenizer from {model_dir}")
    n_gpus = torch.cuda.device_count()
    logging.info(f"Using {n_gpus} GPU(s) for inference.")

    def get_optimal_dtype():
        if not torch.cuda.is_available():
            return torch.float32
        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)
        if capability[0] >= 8:
            return torch.bfloat16
        elif capability[0] >= 6:
            return torch.float16
        else:
            return torch.float32

    optimal_dtype = get_optimal_dtype()

    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=optimal_dtype, device_map="auto"
        )
        model.to(optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer


def create_dataloader(sequences, tokenizer, batch_size):
    logging.info(f"Creating DataLoader with batch size {batch_size}")
    dataset = SequenceDataset(sequences, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def extract_logits_at_indices(model, dataloader, device, tokenizer, indices_list):
    """
    Extract logits at specific indices for each sequence in the dataloader.
    indices_list: list of lists, where indices_list[i] contains the positions to extract for sequence i.
    """
    logging.info("Extracting logits")
    # Use lowercase for vocab lookup as per model requirement
    nucleotides = list('acgt')
    vocab_indices = [tokenizer.get_vocab()[nc] for nc in nucleotides]
    
    all_extracted_probs = []
    
    seq_idx = 0
    for batch in tqdm(dataloader):
        curIDs = batch['input_ids'].to(device)
        curIDs = curIDs.squeeze(1)
        batch_size = curIDs.shape[0]

        with torch.inference_mode():
            outputs = model(input_ids=curIDs)
        
        # outputs.logits: (batch, seq_len, vocab)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=2) # (batch, seq_len, vocab)
        
        for b in range(batch_size):
            # Get the indices we care about for this sequence
            target_indices = indices_list[seq_idx]
            
            # Extract probs for acgt at those indices
            # target_indices is a list of positions
            # We want probs[b, pos, vocab_indices]
            
            # Check bounds
            seq_len = probs.shape[1]
            valid_indices = [idx for idx in target_indices if 0 <= idx < seq_len]
            if len(valid_indices) != len(target_indices):
                logging.warning(f"Sequence {seq_idx}: Some indices out of bounds. Max len {seq_len}, requested {target_indices}")
            
            extracted = probs[b, valid_indices, :][:, vocab_indices].cpu().numpy()
            all_extracted_probs.append(extracted)
            
            seq_idx += 1
            
    return all_extracted_probs


def load_fasta(fasta_path):
    logging.info(f"Loading reference genome from {fasta_path}")
    if fasta_path.endswith(".gz"):
        with gzip.open(fasta_path, "rt") as file:
            return SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    else:
        return SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))


def seq_from_vcf_sv(args):
    logging.info(f"Reading input VCF from {args.inputVCF}")
    fastaDict = load_fasta(args.inputFasta)
    
    vcfReader = vcf.Reader(filename=args.inputVCF)
    
    ref_sequences = []
    mut_sequences = []
    
    # We need to track which positions to extract for each sequence
    ref_indices_list = []
    mut_indices_list = []
    
    records = []
    
    half = args.contextSize // 2
    flank = args.flankSize
    
    count = 0
    
    for record in vcfReader:
        # Determine SV Type
        is_deletion = False
        is_insertion = False
        
        if "SVTYPE" in record.INFO:
            if record.INFO["SVTYPE"] == "DEL":
                is_deletion = True
            elif record.INFO["SVTYPE"] == "INS":
                is_insertion = True
        else:
            # Heuristic based on length
            ref_len = len(record.REF)
            if record.ALT and record.ALT[0]:
                alt_len = len(record.ALT[0].sequence)
                if ref_len > alt_len:
                    is_deletion = True
                elif alt_len > ref_len:
                    is_insertion = True

        if is_deletion:
            chrom = record.CHROM
            # VCF POS is 1-based. It is the base BEFORE the deletion usually (if standard).
            # e.g. REF=ATCG, ALT=A. POS=100. A is at 100. TCG (101-103) is deleted.
            # We want to score the junction formed by A (100) and the base after G (104).
            
            pos = record.POS - 1 # 0-based index of the base 'A'
            
            # Calculate deleted region
            ref_seq_str = str(record.REF)
            alt_seq_str = str(record.ALT[0].sequence)
            
            # Verify that ALT is a prefix of REF (standard VCF) or just handle generic replacement
            # For pure deletion, ALT should be the first char of REF.
            # If complex indel, we just take REF and ALT as given.
            
            del_start_idx = pos + len(alt_seq_str) # First deleted base index
            del_end_idx = pos + len(ref_seq_str)   # First kept base index after deletion
            
            if 'END' in record.INFO:
                end_1indexed = record.INFO['END']
            else:
                end_1indexed = record.POS + len(ref_seq_str) - 1
            
            center_1indexed = (record.POS + end_1indexed) // 2
            
            # For pysam/BioPython fetch, we use 0-indexed coordinates
            ref_seq_start = center_1indexed - half
            ref_seq_end = center_1indexed + half
            
            # Bounds check
            chrom_len = len(fastaDict[chrom])
            if ref_seq_start < 0:
                logging.warning(f"Sequence {chrom}:{pos} out of bounds. Skipping.")
                continue

            if ref_seq_end > chrom_len:
                logging.warning(f"Sequence {chrom}:{pos} out of bounds. Skipping.")
                continue

            if len(record.REF) >= half - flank:
                logging.warning(f"Skipping Deletion at {chrom}:{pos} because length {len(record.REF)} >= half {half} - flank {flank}")
                continue
            
            ref_seq_content = str(fastaDict[chrom].seq[ref_seq_start:ref_seq_end]).upper()
            
            # Pad if necessary (though usually context is large enough)
            if len(ref_seq_content) != args.contextSize:
                logging.error(f"Sequence {chrom}:{pos} has length {len(ref_seq_content)}, expected {args.contextSize}. Skipping.")
                exit(1)
            
            # Indices in RefSeq corresponding to the flanks
            del_start_for_flanks = record.POS - 1  # 0-indexed position of kept base
            del_end_for_flanks = end_1indexed  # 0-indexed start of right flank
            
            ref_left_indices = [p - ref_seq_start for p in range(del_start_for_flanks - flank, del_start_for_flanks)]
            ref_right_indices = [p - ref_seq_start for p in range(del_end_for_flanks, del_end_for_flanks + flank)]
            
            ref_indices = ref_left_indices + ref_right_indices
            
            # MutSeq Construction

            del_start_0indexed = record.POS - 1
            mut_left_start = del_start_0indexed - half
            mut_left_end = del_start_0indexed
            
            if 'END' in record.INFO:
                del_end_0indexed = record.INFO['END']
            else:
                del_end_0indexed = record.POS + len(ref_seq_str) - 1
            mut_right_start = del_end_0indexed
            mut_right_end = del_end_0indexed + half
            
            # Handle bounds
            left_pad = ""
            if mut_left_start < 0:
                left_pad = "N" * (-mut_left_start)
                mut_left_start = 0
            
            mut_left_seq = str(fastaDict[chrom].seq[mut_left_start:mut_left_end]).upper()
            mut_left_seq = left_pad + mut_left_seq
            
            mut_right_seq = str(fastaDict[chrom].seq[mut_right_start:mut_right_end]).upper()
            if len(mut_right_seq) < half:
                mut_right_seq = mut_right_seq.ljust(half, 'N')
                
            mut_seq_content = mut_left_seq + mut_right_seq
            
            # Indices in MutSeq
            # Junction is at 'half'.
            # Left flank: [half - flank, half]
            mut_left_indices = list(range(half - flank, half))
            # Right flank: [half, half + flank]
            mut_right_indices = list(range(half, half + flank))
            
            mut_indices = mut_left_indices + mut_right_indices
            
            ref_sequences.append(ref_seq_content)
            mut_sequences.append(mut_seq_content)
            ref_indices_list.append(ref_indices)
            mut_indices_list.append(mut_indices)
            records.append(record)
            
            count += 1
            
        elif is_insertion:
            chrom = record.CHROM
            pos = record.POS # 1-based, anchor position
            
            # RefSeq: Centered at POS
            # We want context around the insertion point (after anchor).
            # Anchor is at pos-1 (0-indexed).
            # Insertion happens between pos-1 and pos.
            
            center_1indexed = pos
            ref_seq_start = center_1indexed - half
            ref_seq_end = center_1indexed + half
            
            # Bounds check
            chrom_len = len(fastaDict[chrom])
            if ref_seq_start < 0:
                logging.warning(f"Sequence {chrom}:{pos} out of bounds. Skipping.")
                continue
            if ref_seq_end > chrom_len:
                logging.warning(f"Sequence {chrom}:{pos} out of bounds. Skipping.")
                continue
            
            ins_seq = str(record.ALT[0].sequence)[1:]
            ins_len = len(ins_seq)
            
            if ins_len >= half - flank:
                logging.warning(f"Skipping Insertion at {chrom}:{pos} because length {ins_len} >= half {half} - flank {flank}")
                continue
            
            ref_seq_content = str(fastaDict[chrom].seq[ref_seq_start:ref_seq_end]).upper()
            
            if len(ref_seq_content) != args.contextSize:
                logging.error(f"Sequence {chrom}:{pos} has length {len(ref_seq_content)}, expected {args.contextSize}. Skipping.")
                exit(1)
            
            # Ref Indices (Flanks around insertion point)
            # Anchor index relative to ref_seq_start
            anchor_idx_ref = (pos - 1) - ref_seq_start
            ref_left_indices = list(range(anchor_idx_ref - flank, anchor_idx_ref))
            ref_right_indices = list(range(anchor_idx_ref + len(record.REF), anchor_idx_ref + flank + len(record.REF)))
            ref_indices = ref_left_indices + ref_right_indices
            
            # MutSeq Construction
            # Insert ALT[1:] after Anchor.
            

            mut_seq_left = str(fastaDict[chrom].seq[ref_seq_start:(center_1indexed-1)]).upper() + ins_seq
            right_start = center_1indexed+len(record.REF)-1
            right_end = right_start + (args.contextSize - len(mut_seq_left))
            mut_seq_right = str(fastaDict[chrom].seq[right_start:right_end]).upper()
            mut_seq_content = mut_seq_left + mut_seq_right

            mut_left_indices = list(range(anchor_idx_ref - flank, anchor_idx_ref))
            mut_right_indices = list(range(anchor_idx_ref + len(record.ALT[0]) - 1, anchor_idx_ref + flank + len(record.ALT[0]) - 1))
            mut_indices = mut_left_indices + mut_right_indices
            
            
            ref_sequences.append(ref_seq_content)
            mut_sequences.append(mut_seq_content)
            ref_indices_list.append(ref_indices)
            mut_indices_list.append(mut_indices)
            records.append(record)
            
            count += 1

    logging.info(f"Processed {count} SV records.")
    return ref_sequences, mut_sequences, ref_indices_list, mut_indices_list, records


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    args = parse_args()
    
    # Load Data
    ref_seqs, mut_seqs, ref_indices, mut_indices, records = seq_from_vcf_sv(args)
    
    if not ref_seqs:
        logging.error("No SVs found or processed.")
        sys.exit(0)

    # Load Model
    accelerator = Accelerator()
    device = accelerator.device
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Inference on Ref
    logging.info("Running inference on Reference Sequences...")
    ref_loader = create_dataloader(ref_seqs, tokenizer, args.batchSize)
    ref_probs = extract_logits_at_indices(model, ref_loader, device, tokenizer, ref_indices)

    # Inference on Mut
    logging.info("Running inference on Mutated Sequences...")
    mut_loader = create_dataloader(mut_seqs, tokenizer, args.batchSize)
    mut_probs = extract_logits_at_indices(model, mut_loader, device, tokenizer, mut_indices)
    
    # Calculate Scores and Write Output
    logging.info(f"Calculating scores and writing to {args.output}")
    
    # Prepare VCF writer
    vcf_reader = vcf.Reader(filename=args.inputVCF)
    # Add header line for the new score
    vcf_reader.infos['PlantCAD_SV_Score'] = vcf.parser._Info(
        'PlantCAD_SV_Score', 1, 'Float', 'Averaged Delta Log Probability for SV flanks', None, None, None
    )
    vcf_writer = vcf.Writer(open(args.output, 'w'), vcf_reader)
    
    # Map records to scores
    # Note: We only processed a subset of records (the ones identified as SVs/Deletions)
    # We need to write ALL records, adding scores to the processed ones.
    # But vcf_reader iterator is consumed. We should re-open or match by ID/Pos.
    # Simpler: Iterate the original file again, and pop from our results if it matches.
    # Or just write the records we have if the user only cares about those?
    # Usually we want to output the full VCF annotated.
    
    # Let's create a map of (CHROM, POS, REF, ALT) -> Score
    score_map = {}
    
    for i, record in enumerate(records):
        r_probs = ref_probs[i] # (2*flank, 4)
        m_probs = mut_probs[i] # (2*flank, 4)
        
        # Calculate score
        # The old pipeline gets the nucleotide from MutSeq at the center positions (4091-4100)
        # and then scores log(P_mut(nt) / P_ref(nt)) where:
        # - P_mut is extracted from MutSeq at those positions
        # - P_ref is extracted from RefSeq at the FLANK positions (which are different!)
        # 
        # So we get the sequence from MutSeq at mut_indices
        current_mut_seq = mut_seqs[i]
        current_mut_indices = mut_indices[i]
        
        scores = []
        # Use lowercase for matching sequence to nucleotides
        nucleotides = list('acgt')
        
        for j, idx in enumerate(current_mut_indices):
            nt = current_mut_seq[idx].lower() # Get nucleotide from MUTSEQ
            if nt not in nucleotides:
                scores.append(0)
                continue
            
            if j >= len(r_probs) or j >= len(m_probs):
                scores.append(0)
                continue

            nt_idx = nucleotides.index(nt)
            
            p_ref = r_probs[j][nt_idx]
            p_mut = m_probs[j][nt_idx]
            
            if p_ref > 0 and p_mut > 0:
                s = np.log(p_mut / p_ref)
            else:
                s = 0
            scores.append(s)
            
        avg_score = np.mean(scores)
        
        # Key for map: (CHROM, POS, REF, ALT_str)
        key = (record.CHROM, record.POS, str(record.REF), str(record.ALT[0]))
        score_map[key] = avg_score

    # Re-read and write
    vcf_reader_2 = vcf.Reader(filename=args.inputVCF)
    for record in vcf_reader_2:
        key = (record.CHROM, record.POS, str(record.REF), str(record.ALT[0]))
        if key in score_map:
            record.INFO['PlantCAD_SV_Score'] = float(f"{score_map[key]:.4f}")
        vcf_writer.write_record(record)
        
    vcf_writer.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()

import os
import argparse
import random

from tqdm import trange, tqdm
from Bio import SeqIO

CHUNK_SIZE = 8192
NUM_SAMPLES = 100_000
MOTIF_LEN = 3


def parse_genes(gff_path: str) -> dict[str, list[tuple[int, int]]]:
    """Return {seq_id: [(start, end), ...]} with 0-based half-open intervals."""
    genes: dict[str, list[tuple[int, int]]] = {}
    with open(gff_path) as f:
        for line in tqdm(f, desc="Processing GFF lines"):
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[2] != "gene":
                continue
            seq_id = parts[0]
            start = int(parts[3]) - 1  # GFF is 1-indexed
            end = int(parts[4])        # end is inclusive in GFF → exclusive here
            genes.setdefault(seq_id, []).append((start, end))
    return genes


def write_sequence_chunks(
    input_fasta: str,
    input_gff: str,
    output_path: str | None,
    num_samples: int = NUM_SAMPLES,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    if output_path is None:
        input_filename = os.path.splitext(input_fasta)[0]
        output_path = f"{input_filename}_rand_chunk_{chunk_size}_samples_{num_samples // 1000}k.tsv"

    half = chunk_size // 2
    motif_center_offset = half - MOTIF_LEN // 2  # 0-based position of motif start within chunk

    seqs: dict[str, str] = {r.id: str(r.seq) for r in SeqIO.parse(input_fasta, "fasta")}
    genes = parse_genes(input_gff)

    # Keep only genes that fit a full chunk around any motif within them
    eligible: list[tuple[str, int, int]] = []
    for seq_id, gene_list in genes.items():
        if seq_id not in seqs:
            continue
        seq_len = len(seqs[seq_id])
        for gstart, gend in gene_list:
            gene_len = gend - gstart
            if gene_len < MOTIF_LEN:
                continue
            # Motif start (0-based) within full sequence: gstart..gend-MOTIF_LEN
            # Chunk requires: motif_start - (half - MOTIF_LEN//2) >= 0
            #             and motif_start - (half - MOTIF_LEN//2) + chunk_size <= seq_len
            motif_min = gstart
            motif_max = gend - MOTIF_LEN  # inclusive upper bound for motif start
            chunk_start_min = motif_min - motif_center_offset
            chunk_start_max = motif_max - motif_center_offset
            # Clamp to sequence bounds
            chunk_start_lo = max(0, chunk_start_min)
            chunk_start_hi = min(seq_len - chunk_size, chunk_start_max)
            if chunk_start_lo > chunk_start_hi:
                continue
            eligible.append((seq_id, gstart, gend))

    total_genes = sum(len(v) for v in genes.values())
    print(f"Total genes: {total_genes}")
    print(f"Eligible genes: {len(eligible)}")

    if not eligible:
        raise ValueError("No eligible genes found — try a smaller chunk size.")

    with open(output_path, "w", newline="") as out_file:
        out_file.write("sequence\n")
        for _ in trange(num_samples):
            while True:
                seq_id, gstart, gend = random.choice(eligible)
                seq = seqs[seq_id]
                seq_len = len(seq)

                motif_start = random.randint(gstart, gend - MOTIF_LEN)
                chunk_start = motif_start - motif_center_offset
                chunk_start = max(0, min(seq_len - chunk_size, chunk_start))

                chunk = seq[chunk_start:chunk_start + chunk_size]
                motif = seq[motif_start:motif_start + MOTIF_LEN]
                if all(b in "ATCGatcg" for b in motif):
                    break

            out_file.write(f"{chunk}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TSV with gene-centered sequence chunks from a FASTA + GFF file."
    )
    parser.add_argument(
        "input_fasta",
        nargs="?",
        default="data/GCF_002870075.5_Lsat_Salinas_v15_genomic.fna",
        help="Input FASTA file path.",
    )
    parser.add_argument(
        "input_gff",
        nargs="?",
        default="data/GCF_002870075.5_Lsat_Salinas_v15_genomic.gff",
        help="Input GFF file path.",
    )
    parser.add_argument(
        "output_tsv",
        nargs="?",
        default=None,
        help="Output TSV file path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of random sequence chunks to sample.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Length of each output sequence chunk (default: 8192).",
    )
    args = parser.parse_args()

    write_sequence_chunks(
        args.input_fasta,
        args.input_gff,
        args.output_tsv,
        args.num_samples,
        args.chunk_size,
    )

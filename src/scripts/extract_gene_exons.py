#!/usr/bin/env python3
"""
Extract genes from a query GFF and their corresponding exons from a reference GFF.

Usage:
    python extract_gene_exons.py genes.gff reference.gff -o output.gff
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class GffRecord:
    seqname: str
    source: str
    feature: str
    start: int
    end: int
    score: str
    strand: str
    frame: str
    attributes: str

    def __str__(self):
        return "\t".join([
            self.seqname, self.source, self.feature,
            str(self.start), str(self.end),
            self.score, self.strand, self.frame,
            self.attributes,
        ])


def parse_gff(path: str, feature_filter: Optional[Set[str]] = None) -> List[GffRecord]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Limit to 9 fields so embedded tabs in the attributes column are preserved
            parts = line.split("\t", 8)
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature_filter and feature not in feature_filter:
                continue
            records.append(GffRecord(
                seqname=parts[0],
                source=parts[1],
                feature=feature,
                start=int(parts[3]),
                end=int(parts[4]),
                score=parts[5],
                strand=parts[6],
                frame=parts[7],
                attributes=parts[8],
            ))
    return records


def build_exon_index(exons: List[GffRecord]) -> Dict[str, List[GffRecord]]:
    """Index exons by seqname for fast lookup."""
    index = defaultdict(list)
    for exon in exons:
        index[exon.seqname].append(exon)
    return index


def find_exons_in_gene(gene: GffRecord, exon_index: Dict[str, List[GffRecord]]) -> List[GffRecord]:
    """Return exons fully contained within the gene's coordinates on the same seqname."""
    candidates = exon_index.get(gene.seqname, [])
    return [
        e for e in candidates
        if e.start >= gene.start and e.end <= gene.end
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Output genes from a query GFF with their exons from a reference GFF."
    )
    parser.add_argument("genes_gff", help="GFF file with gene records of interest")
    parser.add_argument("reference_gff", help="Reference GFF file containing exon records")
    parser.add_argument("-o", "--output", default="-", help="Output GFF file (default: stdout)")
    parser.add_argument(
        "--gene-feature", default="gene",
        help="Feature type to treat as gene in the query GFF (default: gene)"
    )
    parser.add_argument(
        "--exon-feature", default="exon",
        help="Feature type to treat as exon in the reference GFF (default: exon)"
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None,
        help="Maximum number of genes to extract (default: all)"
    )
    args = parser.parse_args()

    genes = parse_gff(args.genes_gff, feature_filter={args.gene_feature})
    if args.limit is not None:
        genes = genes[: args.limit]
    if not genes:
        sys.exit(f"No '{args.gene_feature}' records found in {args.genes_gff}")

    all_exons = parse_gff(args.reference_gff, feature_filter={args.exon_feature})
    exon_index = build_exon_index(all_exons)

    out = open(args.output, "w") if args.output != "-" else sys.stdout
    try:
        out.write("##gff-version 3\n")
        total_exons = 0
        for gene in genes:
            out.write(str(gene) + "\n")
            exons = find_exons_in_gene(gene, exon_index)
            exons.sort(key=lambda e: e.start)
            for exon in exons:
                out.write(str(exon) + "\n")
            total_exons += len(exons)
    finally:
        if args.output != "-":
            out.close()

    print(
        f"Wrote {len(genes)} gene(s) and {total_exons} exon(s) to "
        f"{'stdout' if args.output == '-' else args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

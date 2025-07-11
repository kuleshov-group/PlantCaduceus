import random, sys
from collections import defaultdict

input_vcf = sys.argv[1]
output_vcf = sys.argv[2]
random.seed(42)

# Storage
header = []
variant_dict = defaultdict(list)

# Read VCF
with open(input_vcf, "r") as f:
    for line in f:
        if line.startswith("#"):
            header.append(line)
            continue
        fields = line.strip().split("\t")
        info_field = fields[7]

        consequence_field = None
        for field in info_field.split(";"):
            if field.startswith("CSQ="):  # VEP standard tag
                consequence_field = field[4:].split(",")[0].split("|")[1]  # only first consequence for first transcript
                break
            elif field.startswith("Consequence="):  # alternate tag
                consequence_field = field.split("=")[1]
                break

        if not consequence_field or "&" in consequence_field:
            continue

        # Group into categories
        variant_dict[consequence_field].append(line)

# Downsample
output_lines = []

# Handle intergenic separately
intergenic_lines = variant_dict.get("intergenic_variant", [])
intergenic_sampled = random.sample(intergenic_lines, min(len(intergenic_lines), 200_000))
output_lines.extend(intergenic_sampled)

# Handle all other types
for cons, lines in variant_dict.items():
    if cons == "intergenic_variant":
        continue
    if len(lines) > 100_000:
        lines = random.sample(lines, 100_000)
    output_lines.extend(lines)

# Write output
with open(output_vcf, "w") as out:
    out.writelines(header)
    out.writelines(output_lines)

print(f"Saved: {output_vcf}")

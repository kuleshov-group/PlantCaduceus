import random, sys

input_vcf = sys.argv[1]
output_vcf = sys.argv[2]

# Buckets
intergenic = []
other = []
header = []

# Read and classify
with open(input_vcf, "r") as f:
    for line in f:
        if line.startswith("#"):
            header.append(line)
            continue
        info_field = line.strip().split("\t")[7]
        consequence_field = None
        for field in info_field.split(";"):
            if field.startswith("CSQ="):  # standard VEP tag
                consequence_field = field[4:].split(",")[0].split("|")[1]  # first transcript consequence
                break
            elif field.startswith("Consequence="):  # alternate tag
                consequence_field = field.split("=")[1]
                break
        if not consequence_field:
            continue

        # Skip multi-consequence
        if ";" in consequence_field:
            continue

        if "intergenic_variant" in consequence_field:
            intergenic.append(line)
        else:
            other.append(line)

# Downsample
random.seed(42)
intergenic_sampled = random.sample(intergenic, min(len(intergenic), 200_000))
other_sampled = random.sample(other, min(len(other), 100_000))

# Write output
with open(output_vcf, "w") as out:
    out.writelines(header)
    out.writelines(other_sampled)
    out.writelines(intergenic_sampled)

print(f"Saved: {output_vcf}")
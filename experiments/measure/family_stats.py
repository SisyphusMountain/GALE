"""Statistics of the HOGENOM families used in the paper's paragraph "Where the time goes" and in the section on numbers:
the size of each family from gpurec's preprocessing (clades, splits), how far below the floating-point range their
likelihoods fall at the fitted rates, and the median minus log-likelihood of the family sets of the experiments.
Prints:
  1. clades and splits of the family MEDIAN_FAMILY (the family profiled with callgrind);
  2. from PRECISION_TSV (written by precision_all.py at the fitted rates; column loss_bits_fp64 = -log2 of the likelihood
     of a family): the median and largest -log2 likelihood, the families below the smallest normal single-precision
     number (-log2 likelihood above 126) and below the smallest normal double-precision number (above 1022), the share
     of all splits and clades in the latter, and the one of them with the fewest splits (the family of the
     scaled-against-double profile);
  3. from PER_FAM_LIKELIHOODS (per_fam_likelihoods.txt of a full fit: "family log-likelihood"), the median of minus
     the log-likelihood over all families and over each family list given after it.
usage: family_stats.py FAMILIES_FILE SPECIES_TREE MEDIAN_FAMILY PRECISION_TSV PER_FAM_LIKELIHOODS LIST_FILE..."""
import csv
import statistics as st
import sys
from pathlib import Path

import gpurec

FAM, SP, MEDIAN_FAMILY, TSV, LL = sys.argv[1:6]
LISTS = sys.argv[6:]
# -log2 of the smallest normal single- and double-precision numbers (2^-126 and 2^-1022): facts of the formats.
SINGLE_MIN_NORMAL_BITS = 126
DOUBLE_MIN_NORMAL_BITS = 1022

data = gpurec.load_families(families=Path(FAM), species_tree=Path(SP), min_covered_species=4, trim_ratio=0.0,
                            max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False)
clades = {name: int(f["C"]) for name, f in zip(data.names, data.families)}
splits = {name: int(f["N_splits"]) for name, f in zip(data.names, data.families)}
print(f"{len(clades)} families, {sum(clades.values()):,} clades, {sum(splits.values()):,} splits")
print(f"1. {MEDIAN_FAMILY}: {clades[MEDIAN_FAMILY]} clades, {splits[MEDIAN_FAMILY]} splits")

bits = {row["family"]: float(row["loss_bits_fp64"]) for row in csv.DictReader(open(TSV), delimiter="\t")}
assert set(bits) == set(clades), "the precision table and the families file name different families"
values = sorted(bits.values())
under_single = [k for k in bits if bits[k] > SINGLE_MIN_NORMAL_BITS]
under_double = [k for k in bits if bits[k] > DOUBLE_MIN_NORMAL_BITS]
print(f"2. -log2 likelihood at the fitted rates: median {st.median(values):.1f}, largest {values[-1]:.2f} bits")
print(f"   below 2^-{SINGLE_MIN_NORMAL_BITS}: {len(under_single)} families; below 2^-{DOUBLE_MIN_NORMAL_BITS}: {len(under_double)} families, "
      f"holding {100 * sum(splits[k] for k in under_double) / sum(splits.values()):.1f} % of all splits and "
      f"{100 * sum(clades[k] for k in under_double) / sum(clades.values()):.1f} % of all clades")
fewest = min(under_double, key=lambda k: splits[k])
print(f"   below 2^-{DOUBLE_MIN_NORMAL_BITS} with the fewest splits: {fewest} ({clades[fewest]} clades, {splits[fewest]} splits)")

ll = {line.split()[0]: float(line.split()[1]) for line in open(LL) if line.strip()}
print(f"3. median -ln likelihood per family: all {len(ll)} families {st.median(-v for v in ll.values()):.1f}")
for list_file in LISTS:
    names = [line.strip() for line in open(list_file) if line.strip()]
    print(f"   {Path(list_file).stem} ({len(names)} families): {st.median(-ll[k] for k in names):.1f}")

"""The same run as run_cli.sh, from Python (run from this folder)."""
from pathlib import Path

import gpurec

data = gpurec.load_families(
    families=Path("families.txt"),
    species_tree=Path("species.nwk"),
    min_covered_species=4,
    trim_ratio=0.0,
    max_clade_split_ratio=None,
    fraction_missing=None,
    skip_family_filtering=False,
)
result = gpurec.fit(data, parametrization="global", start_rates=(0.1, 0.1, 0.1), config=gpurec.load_fit_config(None))
print("rates:", result.rates)
print("log-likelihoods:", result.log_likelihoods)
samples = result.sample_reconciliations(n=10, seed=42)
gpurec.write_alerax(result, samples, Path("output_python"))

"""Families covering fewer than four species (--min-covered-species 1, as Huang et al. ran AleRax):
one gene, two genes and three genes with a duplication. AleRax 1.4 and 1.5 crash on the first two
(a one-leaf or two-leaf Newick gene tree), so the reference is the CPU oracle, which agrees with
AleRax 1.5.0 on the third to 1e-12 nats once AleRax's fixed points converge."""
import math

import pytest
import torch

from gpurec import evaluate, load_families, load_fit_config
from tests._fraction_missing_oracle import oracle_nll

pytestmark = pytest.mark.gpu

SPECIES = "((A:1,B:1)AB:1,(C:1,D:1)CD:1)R;"
GENES = {"one": "A_1;", "two": "(A_1:1,B_1:1);", "three": "((A_1:1,A_2:1):1,C_1:1);"}
RATES = (0.1, 0.2, 0.05)


def test_small_families_match_the_oracle(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    species = tmp_path / "species.nwk"
    species.write_text(SPECIES + "\n")
    lines = ["[FAMILIES]"]
    for name, newick in GENES.items():
        (tmp_path / f"{name}.nwk").write_text(newick + "\n")
        lines += [f"- {name}", f"gene_tree = {tmp_path / f'{name}.nwk'}"]
    families = tmp_path / "families.txt"
    families.write_text("\n".join(lines) + "\n")
    data = load_families(
        families=families, species_tree=species, min_covered_species=1, trim_ratio=0.0,
        max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    assert sorted(data.names) == sorted(GENES)
    config = load_fit_config(None)
    config.precision.model_dtype = "float64"
    config.validate()
    result = evaluate(data, "global", RATES, config)
    for name in GENES:
        expected = -oracle_nll(str(species), str(tmp_path / f"{name}.nwk"), *RATES, 0.0) * math.log(2.0)
        assert abs(result.log_likelihoods[name] - expected) < 1e-8, (name, result.log_likelihoods[name], expected)

import math
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu
RELEASE = Path(__file__).parent / "data" / "release"


@pytest.fixture
def data(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec import load_families

    monkeypatch.chdir(RELEASE)
    return load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )


def test_fit_config_keeps_the_fitting_rate_box(tmp_path):
    from gpurec import load_fit_config

    override = tmp_path / "c.toml"
    override.write_text("[solver]\ne_max_iter = 200\n")
    assert load_fit_config(None).rates.max_rate == 2.0
    config = load_fit_config(override)
    assert config.solver.e_max_iter == 200 and config.rates.max_rate == 2.0


def test_evaluate_matches_the_pytorch_model(data):
    from gpurec import GeneReconModel, evaluate, load_fit_config

    config = load_fit_config(None)
    result = evaluate(data, parametrization="global", rates=(0.1, 0.2, 0.05), config=config)
    model = GeneReconModel("species.nwk", ["fam1.nwk"], mode="genewise", config=config)
    theta = torch.log2(torch.tensor([[0.1, 0.2, 0.05]], device="cuda", dtype=model.theta.dtype))
    expected = -float(model.genewise_loss_vector(theta=theta)[0]) * math.log(2.0)
    assert result.log_likelihoods["fam1"] == pytest.approx(expected, rel=1e-6)
    # fam3 is fam1 with renamed genes and a mapping file
    assert result.log_likelihoods["fam3"] == pytest.approx(result.log_likelihoods["fam1"], rel=1e-5)
    assert result.rates == pytest.approx({"D": 0.1, "L": 0.2, "T": 0.05})


def test_global_fit_improves_the_likelihood(data):
    from gpurec import evaluate, fit, load_fit_config

    config = load_fit_config(None)
    start = evaluate(data, parametrization="global", rates=(0.1, 0.1, 0.1), config=config)
    fitted = fit(data, parametrization="global", start_rates=(0.1, 0.1, 0.1), config=config)
    assert sum(fitted.log_likelihoods.values()) >= sum(start.log_likelihoods.values()) - 1e-4
    assert set(fitted.rates) == {"D", "L", "T"}
    assert all(value > 0 for value in fitted.rates.values())


def test_per_family_fit_improves_each_family(data):
    from gpurec import evaluate, fit, load_fit_config

    config = load_fit_config(None)
    start = evaluate(data, parametrization="per-family", rates=(0.1, 0.1, 0.1), config=config)
    fitted = fit(data, parametrization="per-family", start_rates=(0.1, 0.1, 0.1), config=config)
    assert set(fitted.rates) == set(data.names)
    for name in data.names:
        assert fitted.log_likelihoods[name] >= start.log_likelihoods[name] - 1e-4


def test_bad_rates_are_rejected(data):
    from gpurec import evaluate, load_fit_config

    with pytest.raises(ValueError, match="positive"):
        evaluate(data, parametrization="global", rates=(0.1, 0.0, 0.1), config=load_fit_config(None))


def test_single_gene_family(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec import evaluate, load_families, load_fit_config

    gene = tmp_path / "single.nwk"
    gene.write_text("5_1;\n")
    data = load_families(
        families=[gene], species_tree=RELEASE / "species.nwk", min_covered_species=1, trim_ratio=0.0,
        max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    result = evaluate(data, parametrization="global", rates=(0.1, 0.1, 0.1), config=load_fit_config(None))
    assert math.isfinite(result.log_likelihoods["single"])


@pytest.mark.parametrize("parametrization", ["global", "per-family"])
def test_fit_over_several_batches_matches_one_batch_evaluation(tmp_path, monkeypatch, parametrization):
    """12 families forced into several batches (small clade budget) fit to finite log-likelihoods
    equal to a one-batch evaluation at the fitted rates."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    import shutil

    from gpurec import fit, interface, load_families, load_fit_config

    lines = ["[FAMILIES]"]
    for name in ("fam1.nwk", "fam2.nwk", "fam3.nwk", "fam3.map", "species.nwk"):
        shutil.copy(RELEASE / name, tmp_path / name)
    for copy in range(4):
        for family in ("fam1", "fam2", "fam3"):
            lines += [f"- copy{copy}_{family}", f"gene_tree = {family}.nwk"]
            if family == "fam3":
                lines.append("mapping = fam3.map")
    (tmp_path / "families.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.chdir(tmp_path)
    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    assert data.n_families == 12
    config = load_fit_config(None)

    from gpurec.data import FamilyData

    budgets = []
    build_model = FamilyData.build_model

    def recording_build_model(self, indices, mode, solver_options, config, clade_budget):
        budgets.append(clade_budget)
        return build_model(self, indices, mode, solver_options, config, clade_budget)

    with monkeypatch.context() as patch:
        # Each family has 27 to 39 clades; a budget of 80 clades per batch gives several batches.
        config.batching.clade_budget = 80
        patch.setattr(FamilyData, "build_model", recording_build_model)
        fitted = fit(data, parametrization=parametrization, start_rates=(0.1, 0.1, 0.1), config=config)
        assert len(fitted.model.batch_statics) > 1
    # Every model built while fitting (not only the final one) used the small budget.
    assert len(budgets) >= 2 and set(budgets) == {80}

    one_batch = interface._result(data, parametrization, fitted.log2_rates, config, None, 0.0)
    assert len(one_batch.model.batch_statics) == 1
    assert all(math.isfinite(value) for value in fitted.log_likelihoods.values())
    for name in data.names:
        assert fitted.log_likelihoods[name] == pytest.approx(one_batch.log_likelihoods[name], rel=1e-6)

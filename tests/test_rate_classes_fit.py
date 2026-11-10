"""Rate classes on the GPU: equal class rates reproduce the global model, the class gradient matches
central differences, a class fit is at least as good as the global fit (the models are nested), and
the command line writes one row of rates per node."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gpurec import evaluate, fit, load_families, load_fit_config
from gpurec.fit.cladewise_fit import fit_cladewise, make_cladewise_objective
from gpurec.io.rate_classes import parse_rate_classes

pytestmark = pytest.mark.gpu

RELEASE = Path(__file__).parent / "data" / "release"


@pytest.fixture(autouse=True)
def _in_release(monkeypatch):
    # families.txt names its gene trees relative to the working directory, as AleRax does
    monkeypatch.chdir(RELEASE)


def _setup():
    data = load_families(
        families=RELEASE / "families.txt", species_tree=RELEASE / "species.nwk", min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    config = load_fit_config(None)
    config.precision.model_dtype = "float64"
    config.validate()
    species = data.species
    internal = [node for node in range(len(species.labels)) if not species.is_leaf(node) and node != species.root]
    # two classes per rate: the subtree of the root's first child, and the rest
    first = species.child1[species.root]
    text = f"{species.labels[first]} DLT\n"
    # and a third duplication class deeper down, if the tree has one
    deeper = [node for node in internal if node != first and species.parent[node] == first]
    if deeper:
        text += f"{species.labels[deeper[0]]} D\n"
    return data, config, parse_rate_classes(text, species)


def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def test_equal_class_rates_reproduce_the_global_model():
    _skip_without_cuda()
    data, config, classes = _setup()
    rates = (0.1, 0.2, 0.05)
    glob = evaluate(data, "global", rates, config)
    cls = evaluate(data, "cladewise", rates, config, classes)
    for name in data.names:
        assert abs(glob.log_likelihoods[name] - cls.log_likelihoods[name]) < 1e-9, name


def test_class_gradient_matches_central_differences():
    _skip_without_cuda()
    data, config, classes = _setup()
    model = data.build_model(list(range(data.n_families)), "specieswise", config.solver, config,
                             config.batching.clade_budget)
    objective = make_cladewise_objective(model, classes)
    rng = np.random.default_rng(1)
    x = np.log2(rng.uniform(0.05, 0.4, size=sum(classes.counts)))
    _, grad = objective(x)
    eps = 1e-5
    for i in range(len(x)):
        up, down = x.copy(), x.copy()
        up[i] += eps
        down[i] -= eps
        fd = (objective(up)[0] - objective(down)[0]) / (2 * eps)
        assert abs(fd - grad[i]) / max(1.0, abs(fd)) < 1e-4, (i, grad[i], fd)


def test_cladewise_fit_is_at_least_as_good_as_the_global_fit():
    _skip_without_cuda()
    data, config, classes = _setup()
    glob = fit(data, "global", (0.1, 0.1, 0.1), config)
    cls = fit(data, "cladewise", (0.1, 0.1, 0.1), config, classes)
    assert sum(cls.log_likelihoods.values()) >= sum(glob.log_likelihoods.values()) - 1e-6
    assert cls.log2_rates.shape == (len(data.species.labels), 3)


def test_cli_writes_the_rates_of_every_node(tmp_path):
    _skip_without_cuda()
    from gpurec.cli.main import main

    data, _config, classes = _setup()
    path = tmp_path / "classes.txt"
    path.write_text("".join(f"{classes.starts[0][i]} D\n" for i in range(1, classes.counts[0])))
    out = tmp_path / "out"
    assert main(["-f", str(RELEASE / "families.txt"), "-s", str(RELEASE / "species.nwk"), "-p", str(out),
                 "--model-parametrization", str(path), "-g", "0", "--precision", "float64"]) == 0
    rows = (out / "model_parameters" / "model_parameters.txt").read_text().splitlines()
    labels = data.species.labels
    assert rows[0] == "node D L T" and len(rows) == len(labels) + 1
    rates = {row.split()[0]: tuple(float(v) for v in row.split()[1:]) for row in rows[1:]}
    assert rates.keys() == set(labels)
    assert all(math.isfinite(v) and v > 0 for node_rates in rates.values() for v in node_rates)
    # Nodes of one class share its rate. Whether a class moves away from the global rate is the data's
    # business: here the global D optimum sits at the lower bound and the class fit stops at once.
    written = parse_rate_classes(path.read_text(), data.species)
    for kind in range(3):
        per_class = {}
        for node, label in enumerate(labels):
            per_class.setdefault(written.class_of[kind][node], set()).add(rates[label][kind])
        assert all(len(values) == 1 for values in per_class.values()), kind


def test_a_cladewise_fit_that_does_not_converge_raises():
    _skip_without_cuda()
    data, config, classes = _setup()
    config.cladewise.max_iter, config.cladewise.gtol = 1, 1e-12
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_cladewise(data, classes, start_log2_rates=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64),
                      config=config, verbose=False)


"""The forward simulator (gpurec.simulate_families, crates/gpurec-backtrack/src/simulate.rs) draws
families from exactly the model GALE's likelihood computes.

1. Exactness (GPU, float64): on species trees of 2 and 3 leaves, how often each outcome comes out
   against GALE's own probability of it. GALE reads every gene tree as unrooted, so an outcome is
   an unrooted gene tree with the species of each gene. GALE's likelihood of one named unrooted
   tree is the mean, over its rootings, of the probability of the rooted named tree, conditioned
   on at least one gene (tests/_history_enumeration.py, ``log_likelihood_nats``). The outcome's
   frequency is that likelihood times the number of rootings, divided by the number of renamings
   of the genes within each species that give the same named unrooted tree back (the named trees
   the outcome stands for are equally likely; there are as many as renamings, divided by that
   number). Pearson's chi-square over the outcomes expected at least _LEAST_EXPECTED times, plus
   one bin for all the others. The cases with global rates, equal receiver weights and the
   uniform start go through ``gpurec.evaluate``; per-species rates, receiver weights and the root
   start are options of neither it nor GALE's fit, so they are set on GALE's model
   (``GeneReconModel``) and the losses taken from its solver.
2. Recovery (GPU): GALE's global fit on simulated families finds the simulated rates, within the
   standard errors given by the curvature of its likelihood.
3. The old Python simulator (GALE-experiments, tools_from_gale_6c05a21/tools/simulate), run once
   (GALE-experiments/simulator_crosscheck): the mean visible events per branch and the number of
   genes per family agree, in distribution, with the frozen numbers.
4. The share of attempts that leave no gene equals the extinction probability of the model.
5. The same seed gives the same families for any number of threads; the folder written reads
   back in GALE; the command line refuses a missing option; a minimum gene count draws small
   families again.
6. The per-species counts made in Rust equal the Python rules, and the flat arrays describe the
   same families as the objects.

The random draws use fixed seeds, so every run gives the same numbers; the bounds are such that a
correct simulator would also pass with other seeds, except with the small probabilities given.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

import gpurec

FROZEN = Path(__file__).parent / "data" / "frozen" / "old_simulator_five_leaves.json"
# The smallest p-value a chi-square may give, and the largest |z| of one entry: a correct
# simulator fails one of the about 15 tests below with probability about 15 x 1e-4.
_LEAST_P = 1e-4
_LARGEST_Z = 4.5
# An outcome gets its own bin when expected at least this often (the normal approximation of a
# count); the outcomes looked at have at most this many genes.
_LEAST_EXPECTED = 25
_MOST_GENES = 4
_MAX_COPIES = 1_000_000
_THREE = "((X,Y)XY,Z)R;\n"
_TWO = "(A,B)R;\n"
_FIVE = "(((A,B)AB,C)ABC,(D,E)DE)R;\n"
# (species tree, rates, receiver weights, families): global rates; high transfer and loss;
# per-species rates with unequal receiver weights, on three and on two leaves.
_CASES = {
    "global": (_THREE, (0.30, 0.38, 0.22), "uniform", 200_000),
    "high transfer and loss": (_THREE, (0.10, 1.50, 1.20), "uniform", 200_000),
    "per species, three leaves": (
        _THREE, {"X": (0.45, 0.30, 0.20), "Y": (0.10, 0.60, 0.05), "XY": (0.30, 0.25, 0.80),
                 "Z": (0.20, 0.90, 0.40), "R": (0.50, 0.35, 0.60)},
        {"X": 1.2, "Y": -0.7, "XY": 0.4, "Z": 0.0, "R": 0.0}, 200_000),
    "per species, two leaves": (
        _TWO, {"A": (0.60, 0.30, 0.90), "B": (0.15, 1.20, 0.30), "R": (0.40, 0.50, 0.70)},
        {"A": 1.5, "B": -0.5, "R": 0.0}, 200_000),
}


def _species(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "species.nwk"
    path.write_text(text)
    return path


def _unrooted_edges(newick: str) -> tuple[list[str], list[tuple[int, int]]]:
    """The leaf names of a Newick tree (no branch lengths, no internal names) and the edges of the
    tree read as unrooted: the root, a node with two neighbours, is removed and its two children
    joined. Leaves are the nodes 0 to len(names) - 1."""
    names, internal, edges, stack = [], [], [], []
    token = ""
    for ch in newick.strip().rstrip(";"):
        if ch in "(),":
            if token:
                names.append(token)
                stack[-1].append(("leaf", len(names) - 1))
                token = ""
            if ch == "(":
                stack.append([])
            elif ch == ")":
                children = stack.pop()
                internal.append(children)
                node = ("internal", len(internal) - 1)
                if stack:
                    stack[-1].append(node)
                else:
                    root = node
        else:
            token += ch
    if token:  # a tree of one gene
        return [token], []
    number = {("leaf", i): i for i in range(len(names))}
    for i in range(len(internal)):
        number[("internal", i)] = len(names) + i
    for i, children in enumerate(internal):
        if ("internal", i) != root:
            edges += [(number[("internal", i)], number[c]) for c in children]
    a, b = internal[root[1]]
    edges.append((number[a], number[b]))
    return names, edges


def _canonical(edges, label) -> str:
    """The text of the unrooted tree with leaf v shown as label[v], the same for every drawing of
    it: the smallest over the rootings (one per edge) of the rooted text, children sorted."""
    if not edges:
        return label[0]
    around = defaultdict(list)
    for u, v in edges:
        around[u].append(v)
        around[v].append(u)

    def text(v, parent):
        if v < len(label):
            return label[v]
        return "(" + ",".join(sorted(text(c, v) for c in around[v] if c != parent)) + ")"

    return min("(" + ",".join(sorted((text(u, v), text(v, u)))) + ")" for u, v in edges)


def _unrooted_outcome(family):
    """(key of the family's gene tree read as unrooted, with each gene's species; the number of
    its rootings; the number of renamings of its genes within each species that give the same
    named unrooted tree back)."""
    names, edges = _unrooted_edges(family.newick)
    species = [family.gene_species[name] for name in names]
    key = _canonical(edges, species)
    named = _canonical(edges, names)
    groups = [[i for i in range(len(names)) if species[i] == s] for s in sorted(set(species))]
    renamings = 0
    for orders in itertools.product(*(itertools.permutations(g) for g in groups)):
        label = list(names)
        for group, order in zip(groups, orders):
            for i, j in zip(group, order):
                label[i] = names[j]
        renamings += _canonical(edges, label) == named
    return key, max(1, len(edges)), renamings


def _outcomes(simulation):
    """{outcome: [count, representative family, rootings, renamings]} of the families with at most
    _MOST_GENES genes, their gene trees read as unrooted."""
    table = {}
    for family in simulation.families:
        if len(family.gene_species) > _MOST_GENES:
            continue
        key, rootings, renamings = _unrooted_outcome(family)
        table.setdefault(key, [0, family, rootings, renamings])[0] += 1
    return table


def _family_data(folder, species, families):
    """The families written as files and read by GALE, and GALE's float64 configuration."""
    folder.mkdir()
    config_path = folder / "config.toml"
    config_path.write_text('[precision]\nmodel_dtype = "float64"\n')
    config = gpurec.load_fit_config(config_path)
    entries = ["[FAMILIES]"]
    for i, family in enumerate(families):
        (folder / f"{i}.nwk").write_text(family.newick + "\n")
        (folder / f"{i}.link").write_text("".join(f"{g} {s}\n" for g, s in family.gene_species.items()))
        entries += [f"- o{i}", f"gene_tree = {folder / f'{i}.nwk'}", f"mapping = {folder / f'{i}.link'}"]
    (folder / "families.txt").write_text("\n".join(entries) + "\n")
    data = gpurec.load_families(families=folder / "families.txt", species_tree=species, min_covered_species=1,
                                trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None,
                                skip_family_filtering=True)
    return data, config


def _gale_log2_public(folder, species, families, rates):
    """GALE's log2 likelihood of each family's gene tree (read as unrooted, as GALE reads every gene
    tree), from the public ``gpurec.evaluate`` at global rates (float64): receiver weights equal,
    uniform start."""
    data, config = _family_data(folder, species, families)
    nats = gpurec.evaluate(data, "global", rates, config).log_likelihoods
    return np.array([nats[f"o{i}"] / math.log(2.0) for i in range(len(families))])


def _gale_log2(folder, species, families, rates, weights, origination):
    """GALE's log2 likelihood (float64) of each family's gene tree, read as unrooted, at rates per
    branch, receiver weights and start set on its model: GALE's fit and ``gpurec.evaluate`` have no
    per-species rates, receiver weights or root start, so the losses are taken from its solver
    directly, family by family."""
    from gpurec.api._execution import _origination_log_probs
    from gpurec.core.inference.solver import nll_vector_from_root_rows, solve_resident_e_pi

    data, config = _family_data(folder, species, families)
    labels = list(data.species.labels)
    model = data.build_model(list(range(data.n_families)), "specieswise", config.solver, config,
                             config.batching.clade_budget)
    rows = [rates[label] if isinstance(rates, dict) else rates for label in labels]
    w = [0.0 if weights == "uniform" else weights[label] for label in labels]
    start = [0.0 if origination == "uniform" or s == data.species.root else -math.inf for s in range(len(labels))]
    loss = torch.empty(data.n_families, dtype=torch.float64)
    with torch.no_grad():
        model.theta.copy_(torch.log2(torch.tensor(rows, dtype=torch.float64)))
        model.receiver_weights.copy_(torch.tensor(w, dtype=torch.float64))
        model.origination_weights.copy_(torch.tensor(start, dtype=torch.float64))
        o_lp, o_p = _origination_log_probs(model.origination_weights, like=model.theta,
                                           accumulator_dtype=torch.float64)
        for static in model.batch_statics:
            E, _, _, _, root_rows, *_ = solve_resident_e_pi(static, model.theta, model.receiver_weights,
                                                             warm_start_E=None)
            loss[static.family_index_tensor.cpu()] = nll_vector_from_root_rows(
                root_rows, E, origination_log_probs=o_lp, origination_probs=o_p,
                accumulator_dtype=torch.float64).cpu()
    position = {name: k for k, name in enumerate(data.names)}
    return np.array([-float(loss[position[f"o{i}"]]) for i in range(len(families))])


def _chi_square(table, log2_likelihood, n):
    """Pearson's chi-square of the outcomes expected at least _LEAST_EXPECTED times, plus one bin
    of all others: (p-value, number of bins, largest |z|, the outcome of the largest |z| with its
    observed and expected counts). ``table``: {outcome: [count, family, rootings, renamings]}."""
    keys = list(table) + ["all others"]
    counts = np.array([count for count, _, _, _ in table.values()])
    rootings = np.array([r for _, _, r, _ in table.values()])
    renamings = np.array([k for _, _, _, k in table.values()])
    expected = n * np.exp2(log2_likelihood) * rootings / renamings
    tested = expected >= _LEAST_EXPECTED
    keys = [k for k, t in zip(keys, list(tested) + [True]) if t]
    observed = np.append(counts[tested], n - counts[tested].sum())
    expected = np.append(expected[tested], n - expected[tested].sum())
    assert expected[-1] > _LEAST_EXPECTED
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    z = (observed - expected) / np.sqrt(expected)
    worst = int(np.argmax(np.abs(z)))
    return (stats.chi2.sf(chi2, len(observed) - 1), len(observed), float(np.abs(z[worst])),
            (keys[worst], int(observed[worst]), float(expected[worst])))


@pytest.mark.gpu
@pytest.mark.parametrize("case, origination", [(case, "uniform") for case in _CASES]
                         + [("global", "root"), ("per species, three leaves", "root")])
def test_outcome_frequencies_match_gale_probabilities(tmp_path, case, origination):
    text, rates, weights, n = _CASES[case]
    species = _species(tmp_path, text)
    simulation = gpurec.simulate_families(species, rates, weights, origination, n, 20261106, _MAX_COPIES, 1, 8)
    assert simulation.attempts_over_cap == 0
    table = _outcomes(simulation)
    families = [f for _, f, _, _ in table.values()]
    if isinstance(rates, tuple) and weights == "uniform" and origination == "uniform":
        tree = _gale_log2_public(tmp_path / "gale", species, families, rates)
    else:
        tree = _gale_log2(tmp_path / "gale", species, families, rates, weights, origination)
    p, bins, largest, (key, observed, expected) = _chi_square(table, tree, n)
    print(f"{case}, {origination} start: unrooted gene trees p {p:.3g} over {bins} bins, largest |z| {largest:.2f} "
          f"(outcome {key}: {observed} seen, {expected:.1f} expected)")
    assert bins >= 10 and p > _LEAST_P and largest < _LARGEST_Z


@pytest.mark.gpu
def test_gale_recovers_the_simulated_rates(tmp_path):
    from gpurec.api._execution import stream_batches

    true = (0.12, 0.25, 0.08)
    species = _species(tmp_path, "((((A,B),(C,D)),((E,F),G)),((H,I),(J,(K,L))));\n")
    gpurec.simulate_families(species, true, "uniform", "uniform", 1000, 7, _MAX_COPIES, 1, 8).write(tmp_path / "data")
    config_path = tmp_path / "float64.toml"
    config_path.write_text('[precision]\nmodel_dtype = "float64"\n')
    config = gpurec.load_fit_config(config_path)
    data = gpurec.load_families(families=tmp_path / "data" / "families.txt", species_tree=species,
                                min_covered_species=1, trim_ratio=0.0, max_clade_split_ratio=None,
                                fraction_missing=None, skip_family_filtering=True)
    assert data.n_families == 1000
    result = gpurec.fit(data, "global", (0.1, 0.1, 0.1), config)
    fitted = result.log2_rates[0].double().cpu()
    model = data.build_model(list(range(data.n_families)), "global", config.solver, config,
                             config.batching.clade_budget)

    def slope(x):  # of the loss summed over the families, in bits per log2 unit
        theta = x.to(model.theta).contiguous()
        return stream_batches(model.batch_statics, theta, model.receiver_weights, model.origination_weights,
                              genewise=False, need_grad=True)[1].double().cpu()

    # curvature of the loss (bits, per log2 unit) by central differences of the slopes
    step = 1e-3
    hessian = torch.stack([(slope(fitted + step * e) - slope(fitted - step * e)) / (2 * step)
                           for e in torch.eye(3, dtype=torch.float64)])
    covariance = torch.linalg.inv((hessian + hessian.T) / 2 * math.log(2.0))  # nats: Fisher information
    z = (fitted - torch.log2(torch.tensor(true, dtype=torch.float64))) / covariance.diagonal().sqrt()
    print("fitted", torch.exp2(fitted).tolist(), "standard errors (log2)", covariance.diagonal().sqrt().tolist(),
          "z", z.tolist())
    assert z.abs().max() < _LARGEST_Z


def test_agrees_with_the_old_python_simulator(tmp_path):
    old = json.loads(FROZEN.read_text())
    n = 200_000
    simulation = gpurec.simulate_families(_species(tmp_path, old["species_tree"] + "\n"), old["rates_D_L_T"],
                                          old["receiver_log_weights"], "uniform", n, 5, _MAX_COPIES, 1, 8)
    labels = list(simulation.species.labels)
    columns = [0, 1, 2, 3, 8]  # speciations, duplications, losses, transfers, transfers_to
    counts = np.array([np.array(family.event_counts)[:, columns] for family in simulation.families])
    old_mean = np.array([old["mean"][label] for label in labels])
    old_variance = np.array([old["variance"][label] for label in labels])
    se = np.sqrt(counts.var(axis=0, ddof=1) / n + old_variance / old["families"])
    z = (counts.mean(axis=0) - old_mean)[se > 0] / se[se > 0]
    # each z on its own: the 44 means move together (one large family raises every branch's
    # counts), so their squares do not add up to a chi-square
    assert len(z) >= 40 and np.abs(z).max() < _LARGEST_Z, z
    # number of genes: two-sample chi-square, the sizes from the first one the old simulator gave
    # fewer than 50 times on in one bin
    new = Counter(len(f.gene_species) for f in simulation.families)
    old_sizes = Counter({int(k): v for k, v in old["genes_histogram"].items()})
    cut = min(k for k in range(1, max(old_sizes) + 2) if old_sizes[k] < 50)
    table = [[c[k] for k in range(1, cut)] + [sum(v for k, v in c.items() if k >= cut)] for c in (new, old_sizes)]
    assert stats.chi2_contingency(np.array(table))[1] > _LEAST_P


def test_attempts_without_gene_match_the_extinction_probability(tmp_path):
    _, rates, weights, _ = _CASES["per species, three leaves"]
    n = 100_000
    simulation = gpurec.simulate_families(_species(tmp_path, _THREE), rates, weights, "uniform", n, 3,
                                          _MAX_COPIES, 1, 8)
    species, labels = simulation.species, simulation.species.labels
    p = np.array([np.array([1.0, *rates[label]]) / (1.0 + sum(rates[label])) for label in labels])  # S D L T
    receive = np.zeros((len(labels), len(labels)))
    for e in range(len(labels)):
        banned, a = {e}, species.parent[e]
        while a >= 0:
            banned.add(a)
            a = species.parent[a]
        valid = [h for h in range(len(labels)) if h not in banned]
        receive[e, valid] = np.exp([weights[labels[h]] for h in valid])
        receive[e] /= receive[e].sum()
    E = np.zeros(len(labels))
    for _ in range(10_000):  # the smallest fixed point, reached from 0
        below = np.array([0.0 if species.is_leaf(e) else E[species.child1[e]] * E[species.child2[e]]
                          for e in range(len(labels))])
        E = p[:, 0] * below + p[:, 1] * E * E + p[:, 2] + p[:, 3] * E * (receive @ E)
    extinct = float(E.mean())
    attempts = n + simulation.attempts_without_gene
    z = (simulation.attempts_without_gene / attempts - extinct) / math.sqrt(extinct * (1 - extinct) / attempts)
    assert abs(z) < _LARGEST_Z, (simulation.attempts_without_gene / attempts, extinct)


def _folder_texts(folder: Path) -> dict:
    return {str(p.relative_to(folder)): p.read_text() for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name not in ("settings.json", "families.txt")}


def test_min_genes_draws_small_families_again(tmp_path):
    """A minimum of 3 genes: no kept family is smaller, the attempts drawn again are counted, the
    output does not depend on the threads, and a family that already left 3 genes with a minimum of 1
    is the same family (each family draws from its own generator, a small attempt is just drawn again)."""
    species = _species(tmp_path, _FIVE)
    loose = gpurec.simulate_families(species, (0.2, 0.4, 0.2), "uniform", "uniform", 500, 3, _MAX_COPIES, 1, 4)
    assert loose.attempts_too_few_genes == 0
    strict = {threads: gpurec.simulate_families(species, (0.2, 0.4, 0.2), "uniform", "uniform", 500, 3,
                                                _MAX_COPIES, 3, threads) for threads in (1, 4)}
    for threads, run in strict.items():
        run.write(tmp_path / f"strict{threads}")
    assert _folder_texts(tmp_path / "strict1") == _folder_texts(tmp_path / "strict4")
    small = sum(len(f.gene_species) < 3 for f in loose.families)
    assert small > 50 and strict[4].attempts_too_few_genes >= small
    assert min(len(f.gene_species) for f in strict[4].families) >= 3
    for a, b in zip(loose.families, strict[4].families):
        if len(a.gene_species) >= 3:
            assert a.newick == b.newick and a.history == b.history
    settings = json.loads((tmp_path / "strict4" / "settings.json").read_text())
    assert settings["min_genes"] == 3 and settings["attempts_too_few_genes"] == strict[4].attempts_too_few_genes
    with pytest.raises(ValueError, match="min_genes"):
        gpurec.simulate_families(species, (0.2, 0.4, 0.2), "uniform", "uniform", 5, 3, _MAX_COPIES, 0, 1)


def test_same_seed_same_families_for_any_threads(tmp_path):
    species = _species(tmp_path, _FIVE)
    runs = {threads: gpurec.simulate_families(species, (0.4, 0.3, 0.3), "uniform", "uniform", 2000, 11,
                                              _MAX_COPIES, 1, threads) for threads in (1, 3, 8)}
    for threads, run in runs.items():
        run.write(tmp_path / str(threads))
    assert _folder_texts(tmp_path / "1") == _folder_texts(tmp_path / "3") == _folder_texts(tmp_path / "8")
    other = gpurec.simulate_families(species, (0.4, 0.3, 0.3), "uniform", "uniform", 2000, 12, _MAX_COPIES, 1, 8)
    assert [f.newick for f in other.families] != [f.newick for f in runs[1].families]
    capped = gpurec.simulate_families(species, (2.0, 0.1, 0.5), "uniform", "uniform", 200, 11, 50, 1, 8)
    copies = [1 + 2 * (f.history[0] + f.history[1] + f.history[3]) for f in capped.families]
    assert capped.attempts_over_cap > 0 and max(copies) <= 50
    # per-family rates: the odd families have no duplication
    rates = [(0.0, 0.2, 0.3) if i % 2 else (1.0, 0.2, 0.3) for i in range(200)]
    by_family = gpurec.simulate_families(species, rates, "uniform", "uniform", 200, 11, _MAX_COPIES, 1, 8)
    duplications = [f.history[1] for f in by_family.families]
    assert sum(duplications[1::2]) == 0 and sum(duplications[0::2]) > 50


def test_written_folder_reads_in_gale_and_command_line(tmp_path):
    from gpurec.cli.main import main

    species = _species(tmp_path, _FIVE)
    rates = tmp_path / "model_parameters.txt"
    rates.write_text("node D L T\n" + "".join(f"{x} 0.2 0.3 0.1\n" for x in ("A", "B", "AB", "C", "ABC", "D", "E", "DE", "R")))
    receivers = tmp_path / "receiver_weights.txt"
    receivers.write_text("node weight probability status\n" + "".join(
        f"{x} {w} 0.1 fitted\n" for x, w in zip(("A", "B", "AB", "C", "ABC", "D", "E", "DE", "R"), range(9))))
    common = ["simulate", "--species-tree", str(species), "--origination", "UNIFORM", "--families", "300",
              "--seed", "4", "--max-copies", "100000", "--min-genes", "1", "--threads", "2"]
    assert main(common + ["--rates-file", str(rates), "--receiver-weights", str(receivers), "--out", str(tmp_path / "a")]) == 0
    weights = {x: float(w) for x, w in zip(("A", "B", "AB", "C", "ABC", "D", "E", "DE", "R"), range(9))}
    gpurec.simulate_families(species, (0.2, 0.3, 0.1), weights, "uniform", 300, 4, 100000, 1, 5).write(tmp_path / "b")
    assert _folder_texts(tmp_path / "a") == _folder_texts(tmp_path / "b")
    settings = json.loads((tmp_path / "a" / "settings.json").read_text())
    assert settings["seed"] == 4 and settings["attempts_without_gene"] >= 0
    with pytest.raises(SystemExit):  # no option has a default
        main(common + ["--rates", "0.2", "0.3", "0.1", "--out", str(tmp_path / "c")])
    with pytest.raises(ValueError, match="origination"):
        gpurec.simulate_families(species, (0.2, 0.3, 0.1), "uniform", "lca", 3, 4, 100, 1, 1)
    # families of one or two genes too: min_covered_species 1 (gale's default, 4, would drop them)
    data = gpurec.load_families(families=tmp_path / "a" / "families.txt",
                                species_tree=tmp_path / "a" / "species_tree.newick", min_covered_species=1,
                                trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None,
                                skip_family_filtering=True)
    assert data.n_families == 300 and tuple(data.species.labels) == ("A", "B", "AB", "C", "ABC", "D", "E", "DE", "R")
    assert all((tmp_path / "a" / "reconciliations" / f"{name}.xml").exists() for name in data.names)


# 6. The flat arrays and the Rust counts (CPU): the per-species counts Rust makes equal the Python
#    rules of gpurec.reconciliation.files.species_event_counts family by family, on every case of 1,
#    and the arrays describe the same trees as the objects.
_WIDE = "((((A,B),(C,D)),((E,F),G)),((H,I),(J,(K,L))));\n"


@pytest.mark.parametrize("case", list(_CASES))
def test_rust_counts_equal_the_python_rules_family_by_family(tmp_path, case):
    from gpurec.reconciliation.files import species_event_counts

    text, rates, weights, _ = _CASES[case]
    simulation = gpurec.simulate_families(_species(tmp_path, text), rates, weights, "uniform", 2000, 9,
                                          _MAX_COPIES, 1, 4)
    assert simulation.families
    for family in simulation.families:
        assert family.event_counts == species_event_counts(family.reconciliation, simulation.species), family.name


def test_flat_arrays_describe_the_same_families_as_the_objects(tmp_path):
    from gpurec.simulate import simulate_family_arrays

    species = _species(tmp_path, _WIDE)
    arrays = simulate_family_arrays(species, (0.3, 0.5, 0.4), "uniform", "root", 500, 5, _MAX_COPIES, 1, 2, False)
    assert arrays.event_counts is None
    assert len(arrays.node_offsets) == len(arrays.gene_offsets) == 501
    root = arrays.species_tree.root
    for f in range(500):
        lo, hi = int(arrays.node_offsets[f]), int(arrays.node_offsets[f + 1])
        parent, children, event = arrays.parent[lo:hi], arrays.children[lo:hi], arrays.event[lo:hi]
        assert parent[0] == -1 and (parent[1:] >= 0).all()
        assert arrays.branch[lo] == root          # the root start: every family begins on the root branch
        for k in range(hi - lo):
            for c in children[k]:
                assert c == -1 or parent[c] == k
        genes = arrays.gene[lo:hi][event == 0]
        g0, g1 = int(arrays.gene_offsets[f]), int(arrays.gene_offsets[f + 1])
        assert sorted(genes.tolist()) == list(range(g1 - g0))
    sim = gpurec.simulate_families(species, (0.3, 0.5, 0.4), "uniform", "root", 500, 5, _MAX_COPIES, 1, 2)
    for f, family in enumerate(sim.families):
        assert family.newick == arrays.newick[f]
        assert list(family.gene_species) == arrays.gene_names[int(arrays.gene_offsets[f]):int(arrays.gene_offsets[f + 1])]

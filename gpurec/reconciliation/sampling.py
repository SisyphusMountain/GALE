"""Draw reconciliation samples for every family, one forward solve per batch."""
from __future__ import annotations

import hashlib

import torch

from gpurec.core.backtracking.input import _load_native_module, batch_forward, family_sampler_arguments
from gpurec.reconciliation.tree import ReconciledGeneTree, tree_from_sampler_nodes


def sample_seed(seed: int, family: str, index: int) -> int:
    """A 64-bit seed for sample ``index`` of ``family``: independent of batching and family order."""
    digest = hashlib.sha256(f"{seed}\0{family}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def iter_family_reconciliations(model, data, n: int, seed: int):
    """Yield ``(family name, n sampled reconciliations)`` family by family, one batch in memory at a time.

    Families come in the model's batch order, not in ``data.names`` order. Each family's samples
    use the same seeds as :func:`sample_family_reconciliations`, so the trees are the same.
    """
    if n < 0:
        raise ValueError(f"the number of samples must be non-negative, got {n}")
    native = _load_native_module()
    for static in model.batch_statics:
        # Gradient tracking is switched off only while this batch is computed, not across the
        # ``yield`` below, where the caller's own code runs.
        with torch.no_grad():
            forward = batch_forward(model, static)
            arguments = None
            batch = []
            for local, family_index in enumerate(static.family_indices):
                name = data.names[family_index]
                family = model.families[family_index]
                clade_to_gene = {
                    int(clade): gene for clade, gene in zip(family["leaf_row_index"], family["leaf_gene_names"])
                }
                seeds = [sample_seed(seed, name, i) for i in range(n)]
                arguments = family_sampler_arguments(model, static, forward, local)
                batch.append((name, [
                    tree_from_sampler_nodes(nodes, clade_to_gene)
                    for nodes in native.sample_reconciliations_many(*arguments, seeds)
                ]))
        # ``arguments`` holds views of ``forward``: both go before the caller's code runs.
        del forward, arguments
        yield from batch


def sample_family_reconciliations(model, data, n: int, seed: int) -> dict[str, list[ReconciledGeneTree]]:
    samples = dict(iter_family_reconciliations(model, data, n, seed))
    return {name: samples[name] for name in data.names}

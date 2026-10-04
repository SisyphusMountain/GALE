# gpurec.core.backtracking

Backtracking support for sampling concrete reconciliations from a fitted model after forward
probabilities have been computed. A reconciliation sample answers, for every node of a sampled
gene tree, which event (speciation, duplication, transfer, loss, ...) put it there and on which
species-tree branch.

## Files

- `input.py`: loads the Rust native backtracking extension (`gpurec_backtrack`) and provides the
  building blocks used to draw samples:
  - `batch_forward(model, static)`: runs the forward solve (`E`, `Ebar`, `Pi`, `Pibar` and the
    log event probabilities) once for a whole batch of families. Every family in the batch shares
    this one forward pass, so sampling many families, or many samples per family, does not redo
    it.
  - `family_sampler_arguments(model, static, forward, local_family_index)`: slices the batch-wide
    tensors from `batch_forward` down to one family and converts them to the 16 NumPy arrays the
    native sampler takes, in the fixed order it expects (root clade id, leaf-to-species map, split
    tables, `Pi`/`Pibar`, `E`/`Ebar`, per-species event log-probabilities, transfer-receiver
    log-probabilities, and the species-tree topology helpers).
  - `sample_reconciliations(model, *, family_index, seed)`: the single-family, single-sample
    entry point. It finds the batch that contains `family_index` (raising `IndexError` if no
    batch does), builds that batch's forward pass and per-family arguments, and calls the native
    `sample_reconciliations_torch` for one sample.
- `gpurec/reconciliation/sampling.py::sample_family_reconciliations` is the higher-level,
  many-families/many-samples-per-family entry point (used by `FitResult.sample_reconciliations`);
  for each batch and each family in it, it calls the native `sample_reconciliations_many` once
  with a list of per-sample seeds (derived from `(seed, family_name, sample_index)`, so results do
  not depend on batch layout or family order), instead of calling the single-sample sampler in a
  Python loop.

## Sampled node format

Both native functions return one sampled reconciliation as a list of nodes, node 0 being the
root. Each node is a 4-tuple `(event, species, (left, right), clade)`:

- `event`: one of `"leaf"`, `"speciation"`, `"duplication"`, `"transfer"`, `"loss"`.
- `species`: the index of the species-tree node the gene-tree node reconciles to.
- `(left, right)`: the indices, in the same list, of the node's two children, or `(-1, -1)` when
  it has none yet (a leaf, or a node awaiting `set_children`).
- `clade`: the id of the gene-tree clade (from the family's clade table) the node reconciles.

`gpurec/reconciliation/tree.py::tree_from_sampler_nodes` walks this node list (using the
consecutive-loss encoding: a `"speciation"`/`"transfer"` node whose recorded child is itself a
`"loss"` node collapses to a single `SL`/`TL` event) and builds a `ReconciledGeneTree`, the object
the RecPhyloXML/Newick/event-count writers consume.

Generated `__pycache__` files are interpreter artifacts and are not part of the source implementation.

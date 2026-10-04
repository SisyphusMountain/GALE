# gpurec

`gpurec` computes gene-tree/species-tree reconciliation under the "undated
DTL" model (duplication, transfer, loss — the standard model used by tools
such as AleRax, with no assumption on the timing of species divergences). On
a fixed species tree and a set of gene families, it computes the
reconciliation likelihood, fits duplication/loss/transfer rates by maximum
likelihood, and draws sample reconciliations (which gene-tree node maps to
which event, on which species-tree branch). The heavy computation runs on
Triton/CUDA kernels, so a CUDA GPU is required; there is no CPU fallback.

## Installation

Requirements:
- Linux.
- A CUDA-capable GPU.
- Python 3.11 or 3.12.
- A Rust toolchain, installed with [rustup](https://rustup.rs/) (building
  `gpurec` compiles two small Rust extension modules).

Install the latest tagged release directly from GitHub:

```bash
pip install git+https://github.com/SisyphusMountain/gpurec@v0.1.0
```

or, from a clone:

```bash
git clone https://github.com/SisyphusMountain/gpurec
cd gpurec
pip install .
```

Building downloads the `rustree` crate from GitHub as a Cargo dependency
(needed by the Rust extensions), so a network connection is required at
build time.

## Quick start, command line

```bash
cd examples/quickstart
gpurec -f families.txt -s species.nwk -p output -g 10 --seed 42
```

This fits one shared set of duplication/loss/transfer rates over the example
families and draws 10 sample reconciliations per family into `output/`. See
`examples/quickstart/run_cli.sh`.

Because `gpurec`'s command line accepts
[AleRax](https://github.com/BenoitMorel/AleRax) 1.4.0's arguments (the
subset described below) and writes the same output files, an AleRax command
line works after replacing the program name `alerax` by `gpurec`.

## Quick start, Python

```python
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
```

See `examples/quickstart/run_python.py`.

`result.model` is the underlying `torch.nn.Module` (a `GeneReconModel`); it
can be used directly in a custom PyTorch optimization loop instead of
`gpurec.fit`. Its `theta` parameter holds log-base-2 rates, one row `(D, L,
T)` per family, and calling the model returns the negative log-likelihood
**in bits** (not the natural-log likelihood that `result.log_likelihoods`
reports — the two differ by a factor of `log(2)`):

```python
import math
import torch

# evaluate (not fit) starting rates, to get a model no gpurec optimizer has touched yet:
result = gpurec.evaluate(data, parametrization="global", rates=(0.1, 0.1, 0.1), config=gpurec.load_fit_config(None))
model = result.model  # a torch.nn.Module; model.theta has shape (n_families, 3)
optimizer = torch.optim.Adam([model.theta], lr=0.05)
for step in range(50):
    optimizer.zero_grad()
    loss_bits = model()  # negative log-likelihood, summed over families, in bits
    loss_bits.backward()
    optimizer.step()
    # keep the rates inside the same box gpurec's own fitting uses ([1e-6, 2.0]):
    gpurec.clamp_log_rate_(model.theta, min_rate=1e-6, max_rate=2.0)
with torch.no_grad():
    log_likelihood_nats = -float(model()) * math.log(2.0)
```

## Supported AleRax options

`gpurec`'s command line accepts a subset of
[AleRax](https://github.com/BenoitMorel/AleRax) 1.4.0's command-line
arguments. Unless stated, argument names, meanings and default values match
AleRax's.

### Implemented

| Argument | AleRax default | gpurec behaviour |
|---|---|---|
| `-f/--families FILE` | required | families file (see "Families file" below) |
| `-s/--species-tree FILE` | required | Newick file; AleRax's named-tree options (`Random`, `MiniNJ`, ...) are rejected |
| `-p/--prefix DIR` | `alerax_output` | output folder; refused, before any work, if it already holds results of an earlier run (see below) |
| `--model-parametrization` | `GLOBAL` | `GLOBAL` or `PER-FAMILY` |
| `--d`, `--l`, `--t` | 0.1 each, must be > 0 | starting rates, or the rates used with `--fix-rates` |
| `--fix-rates` | off | evaluate likelihoods only |
| `-g/--gene-tree-samples N` | 100 | samples per family; 0 = no sampling |
| `--seed N` | 123 | sampling seed |
| `--fraction-missing-file FILE` | none | lines `species fraction`; must cover every leaf species |
| `--min-covered-species N` | 4 | AleRax filtering (see "Family filtering" below) |
| `--trim-ratio r` | 0 | AleRax filtering; r must be at least 0 and less than 1 |
| `--max-clade-split-ratio x` | -1 (off) | AleRax filtering |
| `--skip-family-filtering` | off | skip the file-readability checks, as in AleRax |
| `--verbose-opt` | off | log each optimizer step |

### Accepted only with the value that describes gpurec's model

`-r/--rec-model UndatedDTL`, `--transfer-constraint PARENTS`,
`--gene-tree-rooting UNIFORM`, `--origination UNIFORM`,
`--species-tree-search SKIP`. Any other value is an error.
(`PARENTS` is AleRax's name for "a transfer may not go to the donor or to
one of its ancestors", which is gpurec's rule.)

### Rejected

Every other AleRax option (for example `--rate-optimizer`,
`--memory-savings`, `--cleanup-ccp`, `--highways`, `--no-tl`,
`--prune-species-tree`, `--speciation-gamma-categories`,
`--ccp-sample-frequency`, `--infer-speciation-order`) stops the program
before any work, with one message listing every unsupported option given.
An option that would silently do nothing is not accepted. The same message
lists any word that is neither an option nor an option's value (for
example a stray file name). Old AleRax spellings are rejected with a
pointer: `--per-family-rates` (use `--model-parametrization PER-FAMILY`),
`--verbose-opt-rates` (use `--verbose-opt`), `--per-species-rates`
(per-species rates are not supported) and `--gene-tree-sample-frequency`
(now `--ccp-sample-frequency`, not supported).

### Output folder

gpurec has no checkpoints, so it never writes into a folder that already
holds the results of an earlier run: if the `-p/--prefix` folder contains
`per_fam_likelihoods.txt`, `model_parameters/`, `reconciliations/`,
`species_trees/` or `perSpeciesCoverage.txt`, the program stops before any
work (whatever `-g` is) and names what it found. An empty folder, or one
holding only `gpurec.log` or unrelated files, is used as is (`gpurec.log`
is appended to).

### gpurec-only arguments

`--precision float32|float64`, `--config FILE` (TOML; solver, precision and
rate settings — see "Configuration" below). They have no AleRax
counterpart.

### Families file

Lines before `[FAMILIES]` are ignored; `#` starts a comment; whitespace is
removed from every line (as AleRax does, so paths cannot contain spaces);
`- name` starts a family; keys `gene_tree` or `starting_gene_tree` (a `.ale`
file or a file of Newick trees) and `mapping`; keys `alignment`,
`likelihoods` and `subst_model` are accepted and ignored because they only
concern sequence likelihoods, which gpurec does not use; any other key is
an error. Relative paths resolve against the current directory, as in
AleRax.

From Python, `load_families` additionally accepts a list of tree files or a
directory (`*.ale`, `*.newick`, `*.nwk`); family names are then the file
names without extension, and duplicate names are an error.

### Gene-to-species mapping

- `mapping=` file whose first line contains `:`: `species:gene1;gene2;...`.
- Otherwise: `gene species` per line.
- No mapping file: the species is the part of the gene name before the first
  `_`; a name without `_` maps to the species of the same name.
- Every gene must map to a leaf of the species tree; otherwise the program
  stops, naming the family and the gene.

### Family filtering, as in AleRax 1.4.0

- Without `--skip-family-filtering`, a family whose gene-tree or mapping
  file cannot be opened, or is empty, is dropped with a logged message.
- `--min-covered-species N`: keep a family when it covers at least N species
  (counted from its mapping, or from the first tree's labels).
- `--trim-ratio r`: sort families by increasing clade count and keep the
  first `floor(N*(1-r))`.
- `--max-clade-split-ratio x` (when x > 0): keep a family when
  `x*(2*leaves-3) >= clades`.
- Dropped families appear in no output file; they are listed in the log.

## Output files

Written under the folder given by `-p/--prefix`. Unless stated, names,
columns and row order match AleRax 1.4.0.

### Species tree and node names

Internal nodes of the species tree that have no usable label (empty, equal
to another label, or a number such as a bootstrap value) are named as
AleRax does: `Node_<leftmost leaf of left child>_<leftmost leaf of right
child>_<k>`, with `k` counting from 0 until the name is unique, assigned in
post-order. These names are used in every file.

- `species_trees/starting_species_tree.newick`
- `species_trees/inferred_species_tree.newick` (identical, all branch
  lengths 1.0, as AleRax writes without species-tree search)

### Likelihoods and rates

- `per_fam_likelihoods.txt`: `family logL` per line, natural log, sorted by
  increasing value then by name.
- `GLOBAL`: `model_parameters/model_parameters.txt`, header `node D L T`,
  one row per species node (2n-1 rows) in AleRax node order.
- `PER-FAMILY`: `model_parameters/<family>_rates.txt`, line 1 `D L T`,
  line 2 the three rates.

### Other files

- `perSpeciesCoverage.txt`: `species fraction`, leaf species, sorted by
  increasing fraction then name.
- `gpurec.log`: the full command line, gpurec version, GPU name, filtering
  messages, optimizer progress, the rates (GLOBAL: one line; PER-FAMILY:
  the smallest and largest value of each rate, and one line per family
  with `--verbose-opt`), Python warnings, timings.

### Reconciliation samples (`-g N`, N > 0)

`<i>` counts from 0, without padding.

The command line samples and writes one batch of families at a time and
keeps only running totals for the whole-dataset files, so its memory use
does not grow with the number of families. From Python,
`FitResult.sample_reconciliations` returns every family's samples at once
(a dictionary), which does.

- `reconciliations/all/<family>_sample_<i>.xml`: RecPhyloXML with the same
  header, elements and nesting as AleRax's writer: `<spTree>`, then
  `<recGeneTree>`; gene leaves named by gene, internal gene nodes `NULL`;
  events `leaf`, `speciation` (also for speciation-with-loss),
  `duplication`, `branchingOut` (transfer and transfer-with-loss), `loss`,
  and `transferBack` on the recipient branch.
- `reconciliations/all/<family>_samples.newick`: one sampled gene tree per
  line, internal nodes named by their event (`S`, `D`, `T`). These trees
  have no branch lengths — gpurec's parsed clades do not keep the input
  trees' branch lengths.
- `reconciliations/all/<family>_eventCounts_<i>.txt`: 8 lines `NAME:count`
  for `S SL D DL T TL L Leaf` (DL and L are 0, as in AleRax).
- `reconciliations/all/<family>_speciesEventCounts_<i>.txt`: header
  `species_label, speciations, duplications, losses, transfers, presence,
  origination, copies, singletons, transfers_to`; all-zero rows omitted.
- `reconciliations/all/<family>_transfers_<i>.txt`: `source destination 1`
  per transfer, in event order.
- `reconciliations/summaries/<family>_meanSpeciesEventCounts.txt` and
  `<family>_meanTransfers.txt`.
- `reconciliations/totalSpeciesEventCounts.txt`,
  `reconciliations/totalTransfers.txt`,
  `reconciliations/origins/<species>.txt` (one per species node).
  `totalSpeciesEventCounts.txt` and `totalTransfers.txt` are sums of each
  family's mean, added family by family; the command line adds families in
  batch order and `write_alerax` in `data.names` order, so the two can
  differ in a value's last digit.

### Deliberate differences from AleRax

- Numbers are written at full precision, not AleRax's 6 significant digits;
  any parser of AleRax's files reads them.
- Rows of `speciesEventCounts_<i>.txt` are sorted by species name (AleRax's
  order there is unspecified; its summary files are already sorted).
- `<family>_samples.newick` has no branch lengths: gpurec's parsed clades do
  not keep the input trees' branch lengths.
- Not written: `ccps/`, `checkpoint/`, `alerax.log` (replaced by
  `gpurec.log`), `perSpeciesMissing.txt`, `.alerec`, `consensus_50.newick`.

## Configuration

`--config file.toml` (command line) or `gpurec.load_fit_config(path)`
(Python) overrides solver, precision and rate settings with a TOML file.
A file need only list the fields it wants to change; everything else keeps
its default. The sections that take effect when fitting or evaluating are:

- `[precision]`: `model_dtype` and `accumulator_dtype` (`"float32"` or
  `"float64"`).
- `[rates]`: the rate box used when fitting: `min_rate`, `max_rate`, and
  `bound_active_eps` (how close to an edge of the box a rate must be to
  count as sitting on that edge). `init_rate` is rejected with an error:
  fitting starts from `--d`/`--l`/`--t` (`start_rates` in Python), never
  from `init_rate`.
- `[solver]`: settings of the fixed-point and gradient solvers.
  `e_tangent_tol` and `use_hvp_warm_start` are rejected with an error: no
  fit, evaluation or sampling step in gpurec 0.1 reads either one from the
  config file. Which of the remaining keys have an effect depends on the
  step:
  - The final likelihoods (`per_fam_likelihoods.txt`,
    `result.log_likelihoods`) and the reconciliation samples use the
    `[solver]` values as written, except `warm_adjoint`, which is off
    there. These steps compute no gradient, so the keys that change their
    result are `e_max_iter` and `e_tol` (the extinction-probability
    iteration) and `pi_iters` (the reconciliation-probability iteration).
  - While fitting, the fitting routines choose their own iteration counts:
    `pi_iters` and `neumann_terms` from the file are replaced by the
    routines' fixed values (16, and 64 where more accuracy is needed).
  - The GLOBAL fit uses every other `[solver]` key as written
    (`warm_adjoint = true` takes effect only when the GPU has room for the
    extra memory it needs).
  - The PER-FAMILY fit uses only `e_max_iter`, `e_tol`,
    `e_adjoint_max_iter`, `e_adjoint_tol`, `adjoint_pruning_threshold`,
    `use_adjoint_pruning` and `pibar_side_threshold`. It turns
    `warm_adjoint` on by itself (off only for its final accuracy checks).

The GPU memory budget is fixed in gpurec 0.1 (85% of total GPU memory, minus
1 GiB of headroom) and cannot be set from the config file; a `[memory]`,
`[newton]` or `[regularizer]` section is rejected with an error rather than
silently doing nothing (those settings are not read on the fitting/CLI
path).

`gpurec/config/defaults.toml` lists every field with its default value;
it gives values only, not what each field means. This section says which
`[solver]` fields each step reads; some fields also have an explanatory
comment in `gpurec/api/solver_options.py`. Those are the plain dataclass
defaults (`load_config`'s starting point). Fitting — the command line and
`load_fit_config` — instead starts from `GpurecConfig.genewise_reference()`,
which changes some `[solver]` tolerances and sets the rate box to
`[1e-6, 2.0]` (`defaults.toml`'s own `[rates]` values are not what the
command line fits with).

## How it works

`gpurec` preprocesses the species tree and gene families in Rust, then
solves the reconciliation likelihood, its gradient, and reconciliation
sampling with Triton/CUDA kernels driven from a PyTorch module. A technical
report describing the model and the algorithms will be linked here once it
is written.

## License

gpurec is free software, distributed under the GNU General Public License,
version 3 or (at your option) any later version. See the LICENSE file.

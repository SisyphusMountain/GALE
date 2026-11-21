# GALE

GALE computes gene-tree/species-tree reconciliation under the "undated
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
  GALE compiles two small Rust extension modules).

Build it from a clone (`pip install .` compiles the two Rust extensions and makes the package
importable):

```bash
git clone https://github.com/SisyphusMountain/GALE
cd GALE
pip install .
```

This installs the command `gale`; the Python package it provides is imported as `gpurec`.

Building downloads the `rustree` crate from GitHub as a Cargo dependency
(needed by the Rust extensions), so a network connection is required at
build time.

## Quick start, command line

```bash
cd examples/quickstart
gale -f families.txt -s species.nwk -p output -g 10 --seed 42
```

This fits one shared set of duplication/loss/transfer rates over the example
families and draws 10 sample reconciliations per family into `output/`. See
`examples/quickstart/run_cli.sh`.

Because the command line of GALE accepts
[AleRax](https://github.com/BenoitMorel/AleRax) 1.4.0's arguments (the
subset described below) and writes the same output files, an AleRax command
line works after replacing the program name `alerax` by `gale`.

## The likelihood, and its one difference from AleRax 1.4.0

GALE computes the likelihood of the undated DTL model with one difference
from ALE's undated likelihood and AleRax 1.4.0: a duplication whose two
copies carry the two child clades of a split of the gene tree counts twice.
The two copies can carry the two clades in two ways (the left clade on the
first copy and the right clade on the second, or the other way round), and
these are two distinct histories, each with the full probability. ALE and
AleRax 1.4.0 count such a duplication once, which gives every history half
its weight for each of its duplications that separate two observed groups
of genes. "Duplication, then one copy lost" counts twice in both programs.

GALE 1.0 as first tagged (22 October 2026) followed AleRax 1.4.0 here; it was
corrected on 1 November 2026 (see `CHANGELOG.md`). As a result GALE's
per-family log-likelihoods are higher than AleRax 1.4.0's on families whose
reconciliations hold such duplications (by exactly ln 2 for a family whose
only reconciliation holds one of them), the fitted rates differ, and so do
the sampled reconciliations. When the duplication rate is negligible (as on
the test fixtures fitted at a duplication rate of 1e-10), the two programs
agree to rounding. The extinction probabilities are not affected.

## Supported AleRax options

The command line of GALE accepts a subset of
[AleRax](https://github.com/BenoitMorel/AleRax) 1.4.0's command-line
arguments. Unless stated, argument names, meanings and default values match
AleRax's.

### Implemented

| Argument | AleRax default | GALE behaviour |
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

### Accepted only with the value that describes GALE's model

`-r/--rec-model UndatedDTL`, `--transfer-constraint PARENTS`,
`--gene-tree-rooting UNIFORM`, `--origination UNIFORM`,
`--species-tree-search SKIP`. Any other value is an error.
(`PARENTS` is AleRax's name for "a transfer may not go to the donor or to
one of its ancestors", which is GALE's rule.)

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

GALE has no checkpoints, so it never writes into a folder that already
holds the results of an earlier run: if the `-p/--prefix` folder contains
`per_fam_likelihoods.txt`, `model_parameters/`, `reconciliations/`,
`species_trees/` or `perSpeciesCoverage.txt`, the program stops before any
work (whatever `-g` is) and names what it found. An empty folder, or one
holding only `gale.log` or unrelated files, is used as is (`gale.log`
is appended to).

### GALE-only arguments

`--precision float32|float64`, `--config FILE` (TOML; solver, precision and
rate settings — see "Configuration" below). They have no AleRax
counterpart.

### Families file

Lines before `[FAMILIES]` are ignored; `#` starts a comment; whitespace is
removed from every line (as AleRax does, so paths cannot contain spaces);
`- name` starts a family; keys `gene_tree` or `starting_gene_tree` (a `.ale`
file or a file of Newick trees) and `mapping`; keys `alignment`,
`likelihoods` and `subst_model` are accepted and ignored because they only
concern sequence likelihoods, which GALE does not use; any other key is
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
- `gale.log`: the full command line, GALE version, GPU name, filtering
  messages, optimizer progress, the rates (GLOBAL: one line; PER-FAMILY:
  the smallest and largest value of each rate, and one line per family
  with `--verbose-opt`), Python warnings, timings.

### Reconciliation samples (`-g N`, N > 0)

`<i>` counts from 0, without padding.

The command line samples and writes one batch of families at a time and
keeps only running totals for the whole-dataset files, so its memory use
does not grow with the number of families.

- `reconciliations/all/<family>_sample_<i>.xml`: RecPhyloXML with the same
  header, elements and nesting as AleRax's writer: `<spTree>`, then
  `<recGeneTree>`; gene leaves named by gene, internal gene nodes `NULL`;
  events `leaf`, `speciation` (also for speciation-with-loss),
  `duplication`, `branchingOut` (transfer and transfer-with-loss), `loss`,
  and `transferBack` on the recipient branch.
- `reconciliations/all/<family>_samples.newick`: one sampled gene tree per
  line, internal nodes named by their event (`S`, `D`, `T`). These trees
  have no branch lengths — GALE's parsed clades do not keep the input
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

- `per_fam_likelihoods.txt`, the rates and the sampled reconciliations
  differ from AleRax 1.4.0's on families with duplications: see "The
  likelihood, and its one difference from AleRax 1.4.0" above.
- Numbers are written at full precision, not AleRax's 6 significant digits;
  any parser of AleRax's files reads them.
- Rows of `speciesEventCounts_<i>.txt` are sorted by species name (AleRax's
  order there is unspecified; its summary files are already sorted).
- `<family>_samples.newick` has no branch lengths: GALE's parsed clades do
  not keep the input trees' branch lengths.
- Not written: `ccps/`, `checkpoint/`, `alerax.log` (replaced by
  `gale.log`), `perSpeciesMissing.txt`, `.alerec`, `consensus_50.newick`.

## Simulating gene families

GALE can draw gene families from its own model, with their true history, in
files GALE reads back directly. The simulation runs in Rust
(`crates/gpurec-backtrack/src/simulate.rs`) on the CPU; it needs no GPU.

### The model simulated

The model GALE's likelihood computes:

- **Where a family starts.** One gene copy, on a branch given by
  `origination`: `"uniform"` (command line `UNIFORM`), every branch of the
  species tree equally likely, leaf branches, internal branches and the
  root branch, which is the start GALE's likelihood assumes; or `"root"`
  (`ROOT`), always the root branch, as AleRax's `--origination ROOT`.
  GALE's likelihood does not implement the root start (`--origination`
  takes only `UNIFORM`): `ROOT` is for comparisons with AleRax, and GALE
  cannot fit such families under the start they were made with.
- **What one copy does.** A copy on branch e undergoes one event, with
  probabilities proportional to 1 (speciation), D (duplication), L (loss)
  and T (transfer), the rates of branch e (relative to speciation, as
  GALE's rates). A speciation sends one copy to each child branch; on a
  leaf branch it makes a sampled gene. A duplication leaves two copies on
  e. A transfer leaves one copy on e and sends one to a branch that is
  neither e nor an ancestor of e, each such branch chosen with probability
  proportional to e to the power of its receiver weight (natural-log
  weights; all zero: every such branch equally likely, which is what GALE's
  likelihood assumes). A loss ends the copy. Each new copy then undergoes
  its own event.
- **What is kept.** The gene tree is the history with every lineage that
  leaves no gene removed. GALE's likelihood is the probability of a gene
  tree given that the family leaves at least one gene, so a family that
  leaves none is drawn again; the number of such attempts is reported.
- **Not simulated:** missing genes (GALE's fraction-missing files).

The rates can be one triple for every branch (GALE's global rates), one per
branch (per-species rates) or one per family (per-family rates). Branches
are numbered and named as GALE numbers and names them (AleRax's node names,
`Node_<leaf>_<leaf>_<k>` for an unnamed internal node), so the
`model_parameters.txt` of a GALE run applies directly and the output feeds
straight back into GALE.

**What GALE can fit of it.** GALE fits global or per-family rates, with
every receiver weight equal, the uniform start, and families that leave at
least one gene (a minimum gene count of 1). Families simulated with
per-species rates, unequal receiver weights, the root start or a larger
minimum are not of a model GALE fits. GALE reads every gene tree as
unrooted (`--gene-tree-rooting UNIFORM`), so it fits the simulated gene
trees without their root: the likelihood is then that of the unrooted
trees, still the right one for the model simulated.

**Checks** (`tests/test_simulate.py`). On species trees of 2 and 3 leaves,
200,000 simulated families per case (global rates, high transfer and loss,
per-branch rates with unequal receiver weights, uniform and root starts),
the frequency of each unrooted gene tree is compared with GALE's own
likelihood of it (GPU, float64; per-branch rates, receiver weights and the
root start are set on GALE's model directly, since its fit has none of
them); GALE's global fit on 1,000 simulated families finds the simulated
rates within the standard errors given by the curvature of its
likelihood; the share of attempts that leave no gene equals the extinction
probability; and the mean events per branch and the family sizes agree
with those of an earlier simulator written independently in Python.

### Python

```python
import gpurec

sim = gpurec.simulate_families(
    "species.nwk",
    (0.2, 0.3, 0.1),          # (D, L, T) of every branch; or {branch: (D, L, T)}; or a list, one per family
    "uniform",                # receiver weights; or {branch: natural-log weight} for every branch
    "uniform",                # origination: every branch equally likely
    1000,                     # families kept (each leaves at least one gene)
    1,                        # seed
    1_000_000,                # copy cap (below)
    1,                        # fewest genes a kept family leaves (below)
    8,                        # threads (the output does not depend on them)
)
family = sim.families[0]
family.newick                 # gene tree, genes named <species>_<k>
family.gene_species           # {gene: species}
family.reconciliation         # true reconciliation, a gpurec.ReconciledGeneTree
family.event_counts           # visible events per branch, columns of AleRax's speciesEventCounts
family.history                # every event of the whole history, lost copies included
sim.attempts_without_gene     # attempts drawn again because they left no gene
sim.attempts_too_few_genes    # attempts drawn again because they left fewer genes than the minimum
sim.write("simulated")        # the folder below
```

For many families at once, `gpurec.simulate.simulate_family_arrays(...)`
takes the same arguments plus `event_counts` (`True` or `False`, required)
and returns flat arrays instead of objects, with no Python object per node:
`node_offsets`, `parent`, `children`, `branch`, `event`, `gene`,
`gene_offsets`, `gene_names`, `newick`, `history`, `attempts_without_gene`,
`attempts_over_cap`, `attempts_too_few_genes` and `event_counts`
(`[family, species, 9]`, or None). The layout is in the docstring of
`gpurec.simulate.FamilyArrays`.

Every argument is required. The true reconciliation is the one GALE's
sampler would show: each gene node's event (speciation, duplication,
transfer, leaf) and branch, the receiver of each transfer, and the
speciations and transfers that lost a copy (the steps that stay on their
branch, a duplication or a transfer whose other copy left no gene, are not
shown, as in GALE's samples). `gpurec.reconciliation.files.recphyloxml_text`
writes it. The visible event counts are counted as GALE counts a sampled
reconciliation (`gpurec.reconciliation.files.species_event_counts`).

**The copy cap.** Where a branch's duplication rate exceeds one plus its
loss rate, a family can grow without end. An attempt whose history reaches
the cap is drawn again and counted (`sim.attempts_over_cap`); when that
count is not zero, the families are those of the model under the condition
that they stay below the cap, not exactly those of the model. A family for
which a million attempts in a row leave no gene, reach the cap or leave too
few genes stops the run with an error.

**The minimum gene count.** An attempt that leaves some genes but fewer
than the minimum is drawn again and counted (`sim.attempts_too_few_genes`).
A minimum of 1 keeps every family that leaves a gene: that is the model
GALE's likelihood (and AleRax's) fits. A larger minimum is for programs
that cannot read small families (AleRax stops on families of one or two
genes); the families are then those of the model under the condition that
they leave at least that many genes, which neither likelihood knows, so the
rates fitted to them are biased.

**Reproducibility.** Family i draws from its own random generator, seeded
by the seed and i, so the same seed gives the same families for any number
of threads.

### Command line

Every option is required:

```bash
gale simulate --species-tree species.nwk --rates 0.2 0.3 0.1 --receiver-weights UNIFORM \
    --origination UNIFORM --families 1000 --seed 1 --max-copies 1000000 --min-genes 1 --threads 8 \
    --out simulated
gale -f simulated/families.txt -s simulated/species_tree.newick -p fit --min-covered-species 1
```

`--min-covered-species 1` keeps the simulated families of one, two and
three species, which the default (4, AleRax's) would drop, and fitting
without them biases the rates.

`--rates-file model_parameters/model_parameters.txt` (GALE's output for
GLOBAL rates, header `node D L T`) replaces `--rates`; `--receiver-weights`
also takes a file with header `node weight` and one natural-log weight per
branch. Per-family rates are given in Python. The output folder must be
new or empty and holds:

- `species_tree.newick`: the species tree as given;
- `families.txt`: the families, in AleRax's families-file format (absolute
  paths);
- `gene_trees/<family>.newick`: the true gene tree, rooted (GALE reads it
  as unrooted);
- `mappings/<family>.link`: `gene species` lines;
- `reconciliations/<family>.xml`: the true reconciliation, in RecPhyloXML
  as GALE writes its samples;
- `event_counts/<family>_speciesEventCounts.txt`: its visible events per
  branch, in the format of GALE's per-sample `speciesEventCounts` files;
- `history_counts.tsv`: per family, every event of the whole history
  (speciations on internal branches, duplications, losses, transfers,
  genes), lost copies included;
- `settings.json`: the settings and the seed, the attempts drawn again, and
  GALE's version.

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
  `e_tangent_tol` is rejected with an error: no fit, evaluation or sampling
  step in GALE 0.1 reads it from the config file. `use_hvp_warm_start`
  is read by the per-family fit's exact Hessian (it has no effect on the
  global fit, whose Hessian is finite-difference): set it to `false` when
  batches are small (`[batching] clade_budget` well below the default),
  because the warm start keeps one array per batch and per probe on the
  card. Which of the remaining keys have an effect depends on the step:
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

The GPU memory budget is fixed in GALE 0.1 (85% of total GPU memory, minus
1 GiB of headroom) and cannot be set from the config file; a `[memory]`,
`[newton]` or `[regularizer]` section is rejected with an error rather than
silently doing nothing (those settings are not read on the fitting/CLI
path).

`gpurec/config/defaults.toml` lists every field with its default value;
it gives values only, not what each field means. This section says which
`[solver]` fields each step reads; some fields also have an explanatory
comment in `gpurec/api/solver_options.py`. Those are the plain dataclass
defaults. Fitting — the command line and
`load_fit_config` — instead starts from `GpurecConfig.genewise_reference()`,
which changes some `[solver]` tolerances and sets the rate box to
`[1e-6, 2.0]` (`defaults.toml`'s own `[rates]` values are not what the
command line fits with).

## How it works

GALE preprocesses the species tree and gene families in Rust, then
solves the reconciliation likelihood, its gradient, and reconciliation
sampling with Triton/CUDA kernels driven from a PyTorch module. A technical
report describing the model and the algorithms will be linked here once it
is written.

The scripts that reproduce the experiments of the paper are in `experiments/` (see `experiments/README.md`).

## License

GALE is free software, distributed under the GNU General Public License,
version 3 or (at your option) any later version. See the LICENSE file.

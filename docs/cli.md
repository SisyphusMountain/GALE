# gpurec CLI

Installed as the `gpurec` console script (`pip install .`, or `pip install -e .` from a
clone). `gpurec`'s command line accepts a subset of
[AleRax](https://github.com/BenoitMorel/AleRax) 1.4.0's arguments and writes the same output
files under the folder given by `-p/--prefix`. See the main `README.md` for what gpurec
computes, installation and the output-file list; this page only covers the command-line
arguments and gives one worked example per mode.

## Implemented

| Argument | AleRax default | gpurec behaviour |
|---|---|---|
| `-f/--families FILE` | required | families file (AleRax / GeneRax format) |
| `-s/--species-tree FILE` | required | Newick file; AleRax's named-tree options (`Random`, `MiniNJ`, ...) are rejected |
| `-p/--prefix DIR` | `alerax_output` | output folder; refused, before any work, if it already holds results of an earlier run (see below) |
| `--model-parametrization` | `GLOBAL` | `GLOBAL` or `PER-FAMILY` |
| `--d`, `--l`, `--t` | 0.1 each, must be > 0 | starting rates, or the rates used with `--fix-rates` |
| `--fix-rates` | off | evaluate likelihoods only |
| `-g/--gene-tree-samples N` | 100 | samples per family; 0 = no sampling. Sampled and written one batch of families at a time, so memory does not grow with the number of families |
| `--seed N` | 123 | sampling seed |
| `--fraction-missing-file FILE` | none | lines `species fraction`; must cover every leaf species |
| `--min-covered-species N` | 4 | AleRax filtering |
| `--trim-ratio r` | 0 | AleRax filtering; r must be at least 0 and less than 1 |
| `--max-clade-split-ratio x` | -1 (off) | AleRax filtering |
| `--skip-family-filtering` | off | skip the file-readability checks, as in AleRax |
| `--verbose-opt` | off | log each optimizer step |

## Accepted only with the value that describes gpurec's model

`-r/--rec-model UndatedDTL`, `--transfer-constraint PARENTS`,
`--gene-tree-rooting UNIFORM`, `--origination UNIFORM`,
`--species-tree-search SKIP`. Any other value is an error.
(`PARENTS` is AleRax's name for "a transfer may not go to the donor or to
one of its ancestors", which is gpurec's rule.)

## Rejected

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

## Output folder

gpurec has no checkpoints, so it never writes into a folder that already
holds the results of an earlier run: if the `-p/--prefix` folder contains
`per_fam_likelihoods.txt`, `model_parameters/`, `reconciliations/`,
`species_trees/` or `perSpeciesCoverage.txt`, the program stops before any
work (whatever `-g` is) and names what it found. An empty folder, or one
holding only `gpurec.log` or unrelated files, is used as is (`gpurec.log`
is appended to).

## gpurec-only arguments

| Argument | Meaning |
|---|---|
| `--precision float32\|float64` | floating-point precision (default: from the config file, section `[precision]`) |
| `--config FILE` | TOML file with `[solver]`, `[precision]` and `[rates]` settings; see `gpurec/config/defaults.toml` for every field and the main `README.md` ("Configuration") for which `[solver]` keys each step uses. The GPU memory budget is fixed in gpurec 0.1 and cannot be set here; a `[memory]`, `[newton]` or `[regularizer]` section is rejected, and so is `[rates] init_rate` (fitting starts from `--d`/`--l`/`--t`) and `[solver] e_tangent_tol` / `[solver] use_hvp_warm_start` (neither affects any fit, evaluation or sampling step in gpurec 0.1). |

They have no AleRax counterpart.

## Worked examples

Run from `examples/quickstart/` (`families.txt`, `species.nwk` and the gene-tree files
described in the main `README.md`).

### GLOBAL — one set of rates shared across all families

```bash
gpurec -f families.txt -s species.nwk -p out_global --model-parametrization GLOBAL -g 10 --seed 42
```

Fits one `(D, L, T)` triple over every family, then samples 10 reconciliations per family.
Rates are written to `out_global/model_parameters/model_parameters.txt` (one row per species
node, all rows equal).

### PER-FAMILY — one set of rates per family

```bash
gpurec -f families.txt -s species.nwk -p out_per_family --model-parametrization PER-FAMILY -g 10 --seed 42
```

Fits a separate `(D, L, T)` triple per family. Rates are written to
`out_per_family/model_parameters/<family>_rates.txt`, one file per family.

### `--fix-rates` — evaluate likelihoods at given rates, no fitting

```bash
gpurec -f families.txt -s species.nwk -p out_fixed --fix-rates --d 0.1 --l 0.1 --t 0.05 -g 0
```

Computes `per_fam_likelihoods.txt` at the given `--d`/`--l`/`--t` rates without optimizing
them; `-g 0` skips reconciliation sampling.

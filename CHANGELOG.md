# Changelog

## 0.1.0 — unreleased

First release. Based on the development state of 4 August 2026 (commit 27cdc472).

### Added
- Released under GPL-3.0-or-later.
- Python interface: `load_families`, `fit`, `evaluate`, `FitResult.sample_reconciliations`, `write_alerax`.
- `gpurec` command line taking AleRax 1.4.0's arguments and writing its output files.
- Mapping files (both AleRax formats), fraction-missing files and AleRax's family filtering.
- Reconciliation samples in RecPhyloXML with per-sample and summary count files.

### Changed
- gpurec runs only on a CUDA GPU; the `device` arguments are gone.
- Settings come from the TOML config file only; environment variables are no longer read
  (`GPUREC_WARM_ADJOINT` became `[solver] warm_adjoint`).
- Progress messages go through Python's `logging` module (logger `gpurec`).
- A gene whose species is an internal node of the species tree is now an error (AleRax requires a
  leaf); the gene-to-species mapping must name a leaf of the species tree (was: any node).
- `gpurec.fit` at the top level is now the fitting function; it shadows the `gpurec.fit`
  subpackage as an attribute of the `gpurec` package. Import fitting recipes with
  `from gpurec.fit.genewise_fit import fit_genewise` (not `import gpurec.fit.genewise_fit as m`,
  which would resolve `gpurec.fit` to the function, not the subpackage).
- Fitting uses `load_fit_config` (`GpurecConfig.genewise_reference()`, rate box `[1e-6, 2.0]`);
  the global fit now also uses its E-adjoint tolerance `1e-7`, which in single precision is
  clamped to the float32 floor with one `RuntimeWarning` per run.
- `fit_global` and `fit_genewise` take parsed families (`FamilyData`) and explicit start rates
  instead of file paths.
- A family file whose `key=` has an empty value is now an error.
- `sample_reconciliations(model, family_index, seed)` requires both arguments and raises
  `IndexError` for a family index that is in no batch.
- Python 3.10 is not supported (the config loader needs `tomllib`, Python 3.11+).
- The `GPUREC_MEMORY_POLICY_FRACTION` / `GPUREC_MEMORY_POLICY_RESERVE_GIB` environment variables
  are gone and the GPU memory budget is fixed in 0.1 (85% of total GPU memory, minus 1 GiB of
  headroom); it cannot be set from the config file. `--config`/`load_fit_config` reject a
  `[memory]`, `[newton]` or `[regularizer]` section outright (those settings have no effect on
  fitting) instead of silently ignoring it. They also reject `[rates] init_rate`: fitting starts
  from `--d`/`--l`/`--t` (`start_rates` in Python), never from `init_rate`. README.md
  ("Configuration") now says which `[solver]` keys the final evaluation, sampling, the GLOBAL fit
  and the PER-FAMILY fit each use.
- `--config`/`load_fit_config` also reject `[solver] e_tangent_tol` and `[solver]
  use_hvp_warm_start`: neither one affects any fit, evaluation or sampling step in gpurec 0.1, so
  a config file naming either one is refused instead of silently doing nothing.
- The command line refuses an output folder (`-p`) that already holds results of an earlier run
  (`per_fam_likelihoods.txt`, `model_parameters/`, `reconciliations/`, `species_trees/`,
  `perSpeciesCoverage.txt`), whatever `-g` is, before any work (exit code 2). An empty folder, or
  one holding only `gpurec.log` or unrelated files, is still accepted.
- The command line accepts `-r`, AleRax's short spelling of `--rec-model`.
- A word on the command line that is neither an option nor an option's value (for example a
  stray file name) is now a usage error instead of being ignored. The old AleRax spellings
  `--per-family-rates`, `--per-species-rates`, `--verbose-opt-rates` and
  `--gene-tree-sample-frequency` are still rejected, now with a pointer to what replaced them.
- `--trim-ratio` (and `load_families(trim_ratio=...)`) must be at least 0 and less than 1.
- The command line samples and writes reconciliations one batch of families at a time and keeps
  only running totals for the whole-dataset files, so its memory use no longer grows with the
  number of families. New building blocks: `gpurec.reconciliation.sampling.iter_family_reconciliations`
  and `gpurec.io.alerax_output.ReconciliationWriter`. `FitResult.sample_reconciliations` still
  returns every family's samples in one dictionary.
- `gpurec.log` records the rates (GLOBAL: one line; PER-FAMILY: the range of each rate, and one line
  per family with `--verbose-opt`) and Python warnings (such as the float32 tolerance
  `RuntimeWarning`), which also appear on the screen in the log format.

### Removed
- The `gpurec fit` and `gpurec reconcile` subcommands and `fit_dtl` (replaced by the new interface).
- Research code moved to `research/` (not installed): multi-GPU sharding, baselines, species-wise MAP cross-validation.

### Fixed
- `fit_global` crashed on a removed solver setting; `fit_genewise` crashed with an uncapped rate box.
- Families files using `gene_tree =` (instead of `starting_gene_tree =`) were misread.
- A missing or unreadable input file (for example a mistyped `-s` path), or `-p` naming an existing
  file, printed a Python traceback; the command line now prints a one-line error and exits with 1.
- A malformed line in a mapping file now gives an error naming the file and the line number.
- `fit_specieswise`'s error message and some documentation pointed to `gpurec.fit.map_cv`, which
  moved to `research/map_cv.py` (not installed).

### Known issues
- AleRax 1.4.0 itself aborts on a two-gene family while building its clade probabilities, before
  its `--min-covered-species` filter runs; gpurec drops such a family by the filter as expected
  (a two-gene family covers fewer species than the default `--min-covered-species 4`).
- `reconciliations/totalSpeciesEventCounts.txt` and `totalTransfers.txt` are sums of each family's
  mean, added family by family; the command line adds families in batch order and `write_alerax`
  in `data.names` order, so the two can differ in a value's last digit.

### Tests

Test suite cleanup done while fixing the crashes above and moving research-only code out of the
installed package (`gpurec.fit`/`gpurec.bench`/CLI rewrites made a number of tests obsolete).

Whole files:
- Deleted `tests/test_bench_exports.py` (2 tests) — tested the removed `gpurec.bench` package.
- Deleted `tests/test_bicgstab_tolerance.py` (6 tests) and `tests/test_gmres_e_adjoint.py`
  (7 tests) — tested the removed BiCGSTAB and GMRES E-adjoint solvers (the E-adjoint solve is
  unconditionally a Neumann series now).
- Deleted `tests/test_cli.py` (10 tests) — tested the removed `gpurec fit`/`gpurec reconcile`
  subcommands.
- Moved `tests/test_distributed_sharding.py` (5 tests) to `research/tests/test_distributed_sharding.py`
  along with the multi-GPU sharding code it tests; `research/` is outside the installed package
  and outside the automated test suite's `testpaths`.
- `tests/test_fit_recipes_run.py` (2 tests) was added early in this release (to pin down the
  `fit_global`/`fit_genewise` crash fixes) and removed again later in the same release, once
  `fit_global`/`fit_genewise`'s direct-file-path signature it tested was replaced by the
  `FamilyData`-based one; not a net loss of coverage.

Individual tests removed from files that were otherwise kept:
- `tests/test_neumann_e_adjoint.py`: `test_default_e_adjoint_solver_path_is_gmres`,
  `test_neumann_gradient_matches_gmres`,
  `test_solver_options_accepts_neumann`, `test_solver_options_default_is_gmres`,
  `test_solver_options_rejects_invalid_e_adjoint_solver` — tested the removed GMRES solver or the
  removed `e_adjoint_solver` selection option. `test_neumann_matches_gmres_and_direct_solve` was
  rewritten as `test_neumann_matches_direct_solve` (same solver, minus the GMRES comparison).
- `tests/test_config_wiring.py`: `test_cli_config_model_dtype_is_effective_and_explicit_dtype_wins`,
  `test_config_toml_round_trips_through_cli`, `test_default_solver_options_unchanged_without_config_or_flags`,
  `test_explicit_flag_overrides_config` — tested the removed CLI.
  `test_fit_genewise_config_none_reproduces_today_defaults`,
  `test_fit_genewise_explicit_min_rate_beats_config` — tested `fit_genewise`'s removed
  `min_rate`/`max_rate`/`config=None` keyword defaults (`test_fit_genewise_config_rates_sets_effective_min_rate`
  was rewritten as `test_fit_genewise_config_rates_is_used_as_the_bounds_directly`, since the
  merge step it tested no longer exists but the config-supplies-the-floor behaviour still does).
  `test_map_cv_config_lambdas_threaded`, `test_map_cv_config_none_reproduces_today_defaults`,
  `test_map_cv_config_solver_field_threaded`, `test_map_cv_explicit_lambdas_wins_over_config` —
  tested the removed `gpurec.fit.map_cv` (moved to `research/`).
- `tests/test_reference_defaults.py`: `test_map_cv_cv_so_matches_factory`,
  `test_map_cv_reference_matches_signature` — used the moved `map_cv` module directly.
- `tests/test_centered_pi_option.py`: `test_fit_dtl_routes_solver_options_to_genewise_recipe` —
  tested the removed `fit_dtl` dispatcher.
- `tests/test_config_rates.py`: `test_fit_genewise_signature_defaults_come_from_genewise_preset` —
  tested `fit_genewise`'s removed signature defaults.
- `tests/test_config_memory.py`: `test_memory_policy_env_var_fraction_override_wins` — tested the
  removed `GPUREC_MEMORY_POLICY_FRACTION` environment variable.
- `tests/regression/test_global_recipe.py`: `test_fit_dtl_rejects_unknown_mode` — tested the
  removed `fit_dtl`.
- `tests/regression/test_specieswise_recipe.py`: `test_cli_fit_specieswise_exits_cleanly` (tested
  the removed CLI), `test_fit_dtl_raises_for_specieswise` (tested the removed `fit_dtl`),
  `test_map_cv_smoke_uses_fit_specieswise` (tested the moved `map_cv`).

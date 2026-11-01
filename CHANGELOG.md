# Changelog

## 1.0 — 2026-11-01

First release. Based on the development state of 4 August 2026 (commit 27cdc472). The tag `1.0`
was first placed on 22 October 2026 on code with the modelling error corrected below; it now
points to the corrected code. (This section was headed "0.1.0 — unreleased" until then; the
package version in `pyproject.toml` is still 0.1.0.)

### Corrected (1 November 2026)
- **The likelihood counted once a duplication that it must count twice.** In the undated DTL
  recursion, the term of a duplication whose two copies carry the two child clades of a split of
  the gene tree now counts twice: the two copies can carry the two clades in two ways (left clade
  on the first copy and right clade on the second, or the reverse), which are two distinct
  histories, each with the full probability. GALE 1.0 as first tagged, like ALE's undated
  likelihood and AleRax 1.4.0, counted it once, which gave each history half its weight for every
  duplication separating two observed groups of genes; that weight differs from one history to
  the next, so it is not a constant of the data and it biased the fitted rates. "Duplication,
  then one copy lost" already counted twice and is unchanged; the extinction probabilities are
  unchanged.
- **What changes for users:** per-family log-likelihoods (`per_fam_likelihoods.txt`,
  `FitResult.log_likelihoods`) are higher on families whose reconciliations hold such duplications
  (by exactly ln 2 nats on a family whose only reconciliation holds one); the fitted rates change
  (GLOBAL and PER-FAMILY); the sampled reconciliations change, since the sampler gives these
  duplications twice the weight against the other events of a split. GALE's values therefore no
  longer equal AleRax 1.4.0's on families with duplications. When the duplication rate is
  negligible the two still agree to rounding.
- Where: the forward kernels (`gpurec/core/kernels/pi_forward.py`, which holds the constant
  `LOG2_DUPLICATION_SPLIT_FACTOR`, the base-2 log of 2), the gradient kernel
  (`wave_backward_kernels.py`), the kernels of the Hessian-vector products (`dts_tangent.py`,
  `dts_so.py`) and the native sampler (`crates/gpurec-backtrack/src/lib.rs`). There is no option
  to get the old count back.
- Checks: on the species tree ((X,Y),Z), GALE's double-precision log-likelihood of every named
  gene tree with two or three genes equals the log of its probability obtained by adding up all
  histories with at most 41 gene copies (no recursion involved) to at most 1.6e-8 nats, while the
  once-counted value is off by 0.011 to 1.25 nats (`tests/test_duplication_split_factor.py`, which
  also checks the gradient and Hessian-vector products against central differences, and the
  sampler's draws against the tables it reads). On the 200 HOGENOM families of
  `experiments/data/families/hogenom_200.txt` at the rates (0.2, 0.3, 0.1), GALE's per-family
  log-likelihoods in double precision agree with those of GALE 2 (the development version, an
  independent implementation, with `[model] duplication_split_factor = 2`) to 9.1e-13 nats at
  most (5.4e-16 relative).

### Tests (correction of 1 November 2026)
The tests that compared GALE with AleRax 1.4.0 on families with duplications cannot pass any
more, and their references were replaced by independent ones (they are not deleted):
- `tests/test_fidelity_alerax.py`: `test_trees_1`, `test_trees_2` and `test_trees_200` still
  compare with AleRax's value (duplication rate 1e-10); `test_trees_3` with AleRax's value plus
  ln 2 (its only reconciliation holds one duplication separating two clades); `test_mixed_200`
  with GALE 2's value at the same rates, in double precision.
- `tests/test_fraction_missing_alerax.py`: no longer runs the `alerax` binary; compares with GALE
  2's values at fraction-missing 0 and 0.3 (tolerance 1e-9 nats instead of 5e-3 and 3e-4).
- `tests/test_backtracking_matches_alerax.py`: no longer runs AleRax's sampler; the sampled
  starting species are compared with the exact probabilities computed by GALE 2 (chi-square
  goodness of fit instead of a two-sample test).
- `tests/_fraction_missing_oracle.py` (the CPU reference of `test_fraction_missing_oracle.py`)
  counts the duplication of a split twice too.
- New: `tests/test_duplication_split_factor.py` and its enumeration of histories,
  `tests/_history_enumeration.py`.
When AleRax 1.5.0, which counts these duplications twice, is released, its values will become the
references of the three AleRax tests again (their docstrings say where).

### Added
- Released under GPL-3.0-or-later.
- `gale` command line taking AleRax 1.4.0's arguments and writing its output files.
- Mapping files (both AleRax formats), fraction-missing files and AleRax's family filtering.
- Reconciliation samples in RecPhyloXML with per-sample and summary count files.

### Changed
- GALE runs only on a CUDA GPU; the `device` arguments are gone.
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
  use_hvp_warm_start`: neither one affects any fit, evaluation or sampling step in GALE 0.1, so
  a config file naming either one is refused instead of silently doing nothing.
- The command line refuses an output folder (`-p`) that already holds results of an earlier run
  (`per_fam_likelihoods.txt`, `model_parameters/`, `reconciliations/`, `species_trees/`,
  `perSpeciesCoverage.txt`), whatever `-g` is, before any work (exit code 2). An empty folder, or
  one holding only `gale.log` or unrelated files, is still accepted.
- The command line accepts `-r`, AleRax's short spelling of `--rec-model`.
- A word on the command line that is neither an option nor an option's value (for example a
  stray file name) is now a usage error instead of being ignored. The old AleRax spellings
  `--per-family-rates`, `--per-species-rates`, `--verbose-opt-rates` and
  `--gene-tree-sample-frequency` are still rejected, now with a pointer to what replaced them.
- `--trim-ratio` (and `load_families(trim_ratio=...)`) must be at least 0 and less than 1.
- The command line samples and writes reconciliations one batch of families at a time and keeps
  only running totals for the whole-dataset files, so its memory use no longer grows with the
  number of families. New building blocks: `gpurec.reconciliation.sampling.iter_family_reconciliations`
  and `gpurec.io.alerax_output.ReconciliationWriter`.
- `gale.log` records the rates (GLOBAL: one line; PER-FAMILY: the range of each rate, and one line
  per family with `--verbose-opt`) and Python warnings (such as the float32 tolerance
  `RuntimeWarning`), which also appear on the screen in the log format.
- The Triton kernels take the sizes and offsets that change with every wave or batch (a wave's first
  row and width, its numbers of splits and tiles) as run-time arguments, and Triton no longer
  compiles a separate kernel for each of their values (`do_not_specialize`, see
  `gpurec/core/kernels/README.md`): a per-family fit of 1,000 HOGENOM families compiled 1,421
  kernels from an empty cache (3 min 31 s), now 57 (1 min 15 s); later runs on the same families
  no longer compile kernels for wave sizes they had not met. Losses are identical and gradients
  agree to run-to-run noise.

### Removed
- The `gpurec fit` and `gpurec reconcile` subcommands and `fit_dtl` (replaced by the new interface).
- Research and development material is not part of the release: research code (multi-GPU sharding,
  baselines, species-wise MAP cross-validation), experiments, notebooks, developer checks, cluster
  benchmark scripts, and development plans and reports.
- Package code that the command line, `fit` and `evaluate` never reach: the per-species fit
  (`fit_specieswise`), the general optimiser (`gpurec.fit.optimize`), Newton-CG, the curvature and
  Krylov solvers (`gpurec.solver.curvature`, `gpurec.solver.krylov`), `BatchedLBFGS`, and
  `GpurecConfig.map_cv_reference` / `optimize_reference`; the AleRax output reader moved to the tests,
  its only users. The optional dependency group `experiments` is gone.
- The wave scheduler keeps a single schedule, the greedy one built from the leaves. The three
  schedules it also tried when that one missed its lower bound (a fixed number of waves built
  backwards, a greedy schedule from the roots, Coffman-Graham) are gone, with the
  `nonleaf_schedule_policy` option that chose between them. On the HOGENOM and archaeal data sets
  the waves are unchanged; on the 5,042 families of Davín et al. 2025, one of 291 batches now
  takes one more wave (67 instead of 66).

### Fixed
- `use_adjoint_pruning = false` now turns adjoint pruning off: it kept the threshold
  `adjoint_pruning_threshold` and skipped the same clades as pruning on.
- `fit_global` crashed on a removed solver setting; `fit_genewise` crashed with an uncapped rate box.
- Families files using `gene_tree =` (instead of `starting_gene_tree =`) were misread.
- A missing or unreadable input file (for example a mistyped `-s` path), or `-p` naming an existing
  file, printed a Python traceback; the command line now prints a one-line error and exits with 1.
- A malformed line in a mapping file now gives an error naming the file and the line number.
- With a fraction-missing file, the gradient of every fit ignored the missing fractions while the
  likelihood used them: fitted rates were wrong, and on some data sets the extinction-adjoint
  series diverged ("E-adjoint Neumann series failed to converge").
- A double-precision fit crashed when a family was rejected (a single-precision residual was
  written into a double-precision table).

### Known issues
- AleRax 1.4.0 itself aborts on a two-gene family while building its clade probabilities, before
  its `--min-covered-species` filter runs; gpurec drops such a family by the filter as expected
  (a two-gene family covers fewer species than the default `--min-covered-species 4`).
- `reconciliations/totalSpeciesEventCounts.txt` and `totalTransfers.txt` are sums of each family's
  mean, added family by family; the command line adds families in batch order and `write_alerax`
  in `data.names` order, so the two can differ in a value's last digit.

### Tests

Second cleanup, for the release: tests that checked development details rather than results users
rely on, or that could never run for a user, are removed: checks of the test data themselves, tests
needing an external 3.4 GB capture or the `rustree` module (`tests/regression/`; its five
memory-budget tests moved to `tests/test_memory_gate.py`), near-duplicates, tests pinning earlier
refactors, and the tests of the removed package code. Tests left behind by the configuration
changes now pass every argument explicitly, and the tests of warm starts switch them on themselves.
The fraction-missing-zero curvature test compares to rounding instead of bit for bit (it failed
about one run in three because the GPU may sum in a different order).

Test suite cleanup done while fixing the crashes above and moving research-only code out of the
installed package (`gpurec.fit`/`gpurec.bench`/CLI rewrites made a number of tests obsolete).

Whole files:
- Deleted `tests/test_bench_exports.py` (2 tests) — tested the removed `gpurec.bench` package.
- Deleted `tests/test_bicgstab_tolerance.py` (6 tests) and `tests/test_gmres_e_adjoint.py`
  (7 tests) — tested the removed BiCGSTAB and GMRES E-adjoint solvers (the E-adjoint solve is
  unconditionally a Neumann series now).
- Deleted `tests/test_cli.py` (10 tests) — tested the removed `gpurec fit`/`gpurec reconcile`
  subcommands.
- Removed `tests/test_distributed_sharding.py` (5 tests) with the multi-GPU sharding code it tests,
  which is not part of the release.
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
  tested the removed `gpurec.fit.map_cv` (not part of the release).
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

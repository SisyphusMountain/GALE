# gpurec Package

This package contains the Python-facing reconciliation model and the GPU
execution path. It combines a PyTorch API, Rust preprocessing/backtracking
helpers, and Triton kernels.

## Public Exports

`gpurec/__init__.py` re-exports the full public interface:

- `load_families`: read an AleRax-format families file (or a list of
  gene-tree files / a directory) plus a species tree into a `FamilyData`.
- `FamilyData`: the parsed species tree and retained gene families.
- `fit`: fit duplication/loss/transfer rates (`parametrization="global"` or
  `"per-family"`), returning a `FitResult`.
- `evaluate`: compute likelihoods at given rates, no fitting; same return
  type as `fit`.
- `FitResult`: `.rates`, `.log_likelihoods` (natural log), `.model` (the
  gene-wise `GeneReconModel`), and `.sample_reconciliations(n, seed)`.
- `write_alerax`: write a `FitResult` (and, optionally, its sampled
  reconciliations) to AleRax-format output files.
- `load_fit_config`, `load_config`: read a `GpurecConfig` from an optional
  TOML file (`load_fit_config` starts from the fitting-reference defaults;
  `load_config` starts from the plain dataclass defaults).
- `GpurecConfig`: solver, precision and rate-box settings (it also holds
  memory, Newton and regularizer settings, which have no effect on fitting
  in 0.1; `load_fit_config` rejects them).
- `GeneReconModel`: the PyTorch module that evaluates reconciliation
  likelihoods (and their gradients) on the GPU.
- `SolverOptions`: mutable solver controls for the fixed-point and
  implicit-gradient solves.
- `sample_reconciliations`: low-level, single-family, single-sample
  backtracking helper exposed from the Rust `gpurec-backtrack` extension
  (see `gpurec/core/backtracking/README.md`); `FitResult.sample_reconciliations`
  is the higher-level, many-families/many-samples entry point most callers
  want.
- `ReconciledGeneTree`: one sampled reconciliation, with per-node gene name
  (leaves), species label and event; used by the RecPhyloXML/Newick/counts
  writers.
- `cuda_device`: checks that a CUDA GPU is available and returns it, or
  raises a clear error.
- `log2_rate_bounds`, `project_rate_gradient_`, `clamp_log_rate_`: rate-box
  helpers for a custom PyTorch optimization loop over `GeneReconModel.theta`
  (log-base-2 rates).

## File And Folder Roles

- `__init__.py`: defines the package-level import surface (above).
- `interface.py`: `fit`, `evaluate`, `FitResult` — the top-level Python API.
- `data.py`: `load_families`, `FamilyData`, AleRax-style family filtering.
- `config/`: `GpurecConfig` and its TOML loaders (`load_config`,
  `load_fit_config`); `config/defaults.toml` documents every field.
- `cli/`: the `gpurec` command line (AleRax-compatible arguments).
- `io/`: families-file, mapping-file and AleRax-output-file readers/writers
  (`write_alerax`).
- `reconciliation/`: `ReconciledGeneTree`, the reconciliation-sample writers
  and summaries.
- `optimization.py`: optimizer-agnostic projection helpers for log2-rate
  parameters and gradients.
- `api/`: the PyTorch model (`GeneReconModel`), solver options, and
  implicit-gradient entry points.
- `fit/`: the `fit_global`/`fit_genewise` fitting recipes used by
  `interface.fit`.
- `core/`: implementation modules for preprocessing wrappers, parameter
  extraction, forward inference, backtracking input, memory policy, and
  Triton kernels.

## Execution Boundary

The package keeps the stable interface at the top level (`gpurec`, re-
exported from `interface.py`, `data.py`, `io/`, `config/`) and leaves most
performance-sensitive details in `gpurec.core`. User code should normally
import from `gpurec` rather than reaching directly into kernel or scheduling
modules, unless it is debugging or extending the implementation.

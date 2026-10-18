"""Ablations on the nested random subset of 1,000 HOGENOM families (paper: paragraph "Ablations" and Table
"tab:ablations"): the per-family fit with one design choice switched off at a time (wrappers/gpurec_ablate.py; adjoint
pruning through data/configs/no_pruning.toml). Each configuration runs once untimed, so that Triton's compile cache
holds its kernels, then twice timed (NAME_1, NAME_2); the kernels compiled during each run are counted (compiles.txt).
Outputs go to OUTPUT_ROOT/ablations; analysis/ablations.py summarises them.
Parts (run one at a time, alone on the machine):
  base, nowarm, noevict, bysize, inputorder, alone, storealpha   the modes of wrappers/gpurec_ablate.py
  nopruning    adjoint pruning off (the backward pass computes every clade; data/configs/no_pruning.toml sets both
               use_adjoint_pruning = false and adjoint_pruning_threshold = 0, since the first alone skips the same clades)
  base-again   the release settings once more (base_again_1, _2), in the same session as nopruning and storealpha,
               which the paper compares with it
  grad-check   loss and gradient of the 100-family nested subset computed as released, without adjoint pruning and with
               the split term stored (measure/grad_check.py): how much pruning and storing change the gradient
  pruning-check  measure/pruning_check.py on the 1,000-family subset at the starting rates and at the rates of the run
               base_1: clades skipped, time and gradient with pruning on, "off" by use_adjoint_pruning alone, and truly
               off (run after base)
  storealpha-check  measure/storealpha_check.py on the 1,000-family subset: the size of the store of split terms and
               the peak GPU memory of one loss and gradient with and without it
usage: run_ablations.py PATHS_TOML PART"""
from common import (CONFIGS, gpurec_fit_arguments, hogenom_families_file, hogenom_species_tree, output_dir,
                    run_gpurec, run_part, run_python)

MODES = ("base", "nowarm", "noevict", "bysize", "inputorder", "alone", "storealpha")


def configuration(name, mode, extra):
    def run(paths):
        folder = output_dir(paths, "ablations")
        fam = hogenom_families_file(paths, "nested_1000")
        for k in ("warmup", 1, 2):
            run_gpurec(paths, folder, f"{name}_{k}", "gpurec_ablate.py", [mode],
                       gpurec_fit_arguments(fam, hogenom_species_tree(paths), extra), folder / "compiles.txt")
    return run


def storealpha_check(paths):
    folder = output_dir(paths, "ablations")
    run_python(folder, "storealpha_check", "measure/storealpha_check.py",
               [hogenom_families_file(paths, "nested_1000"), hogenom_species_tree(paths)])


def pruning_check(paths):
    folder = output_dir(paths, "ablations")
    run_python(folder, "pruning_check", "measure/pruning_check.py",
               [hogenom_families_file(paths, "nested_1000"), hogenom_species_tree(paths),
                folder / "out_base_1" / "model_parameters"])


def grad_check(paths):
    folder = output_dir(paths, "ablations")
    run_python(folder, "grad_check", "measure/grad_check.py",
               [hogenom_families_file(paths, "nested_100"), hogenom_species_tree(paths)])


PARTS = {mode: configuration(mode, mode, []) for mode in MODES}
PARTS["nopruning"] = configuration("nopruning", "base", ["--config", CONFIGS / "no_pruning.toml"])
PARTS["base-again"] = configuration("base_again", "base", [])
PARTS["grad-check"] = grad_check
PARTS["pruning-check"] = pruning_check
PARTS["storealpha-check"] = storealpha_check

if __name__ == "__main__":
    run_part(__doc__, PARTS)

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
  session      the whole table in one session (the paper's Table "tab:ablations"): three rounds, each running the eight
               configurations base, alone, noevict, nowarm, bysize, inputorder, storealpha, nopruning one after the
               other; round 0 is untimed (it fills the compile cache for every configuration), rounds 1 and 2 are the
               timed runs. Runs are named CONFIGURATION_ROUND (base_0, ..., nopruning_2) and go to
               OUTPUT_ROOT/ablations/session. Before and after each run, progress.txt records the load of the machine
               and the other processes on the GPU (nvidia-smi), so that a run disturbed by another program can be set
               aside (analysis/ablations.py session OUTPUT_ROOT/ablations/session flags it). About 2.5 hours.
usage: run_ablations.py PATHS_TOML PART"""
import subprocess

from common import (CONFIGS, gpurec_fit_arguments, hogenom_families_file, hogenom_species_tree, log_progress,
                    output_dir, run_gpurec, run_part, run_python)

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


# The configurations of the session, in the order they run in each round: (name of the runs, mode of
# wrappers/gpurec_ablate.py, extra arguments of gpurec).
SESSION = (("base", "base", []), ("alone", "alone", []), ("noevict", "noevict", []), ("nowarm", "nowarm", []),
           ("bysize", "bysize", []), ("inputorder", "inputorder", []), ("storealpha", "storealpha", []),
           ("nopruning", "base", ["--config", CONFIGS / "no_pruning.toml"]))


def other_gpu_processes(paths):
    """The processes that hold the GPU, as nvidia-smi lists them ("pid, name" separated by ";"), leaving out those whose
    name contains one of the strings of gpu_processes_not_counted in paths.toml (programs of the desktop that hold the
    GPU without computing on it)."""
    listing = subprocess.run([paths["tools"]["nvidia_smi"], "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    ignored = paths["machine"]["gpu_processes_not_counted"]
    return ";".join(line.strip() for line in listing.splitlines()
                    if line.strip() and not any(word in line for word in ignored))


def session(paths):
    folder = output_dir(paths, "ablations/session")
    fam = hogenom_families_file(paths, "nested_1000")
    progress = folder / "progress.txt"
    log_progress(progress, f"session start, other GPU processes: [{other_gpu_processes(paths)}]")
    for round_number in (0, 1, 2):
        for name, mode, extra in SESSION:
            run = f"{name}_{round_number}"
            log_progress(progress, f"{run}: before, other GPU processes: [{other_gpu_processes(paths)}]")
            run_gpurec(paths, folder, run, "gpurec_ablate.py", [mode],
                       gpurec_fit_arguments(fam, hogenom_species_tree(paths), extra), folder / "compiles.txt")
            log_progress(progress, f"{run}: after, other GPU processes: [{other_gpu_processes(paths)}]")
    log_progress(progress, "session done")


PARTS = {mode: configuration(mode, mode, []) for mode in MODES}
PARTS["nopruning"] = configuration("nopruning", "base", ["--config", CONFIGS / "no_pruning.toml"])
PARTS["base-again"] = configuration("base_again", "base", [])
PARTS["grad-check"] = grad_check
PARTS["pruning-check"] = pruning_check
PARTS["storealpha-check"] = storealpha_check
PARTS["session"] = session

if __name__ == "__main__":
    run_part(__doc__, PARTS)

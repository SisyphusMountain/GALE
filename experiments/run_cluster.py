"""The fits of all HOGENOM families that the paper ran on a computing cluster with the SLURM job manager (appendix
"Representation of the numbers": double- against single-precision fits). Each part is one job started with srun on one
GPU, with the options of [cluster] in paths.toml; run it from a login node of the cluster, with a copy of paths.toml
whose paths are those of the cluster. Outputs go to OUTPUT_ROOT/cluster, with the same files as the other drivers
(NAME.time, NAME.log, NAME.err, out_NAME).
Parts (the two halves can run at the same time, from two sessions):
  double-half-1   gpurec in double precision (--precision float64) on the first 5,434 families of hogenom_all.txt
  double-half-2   the same on the other 5,435 families. The paper ran these two as interactive jobs 61617580 and
                  61617581 on NVIDIA L40S cards (16 October 2026, partition gpu_l40s_interactive, 1 GPU, 8 cores,
                  64 GB, 3 hours), with the release d48c48f7 plus the double-precision fix of commit bd16458d
                  (both in this branch); the halves were the first 5,434 and the other 5,435 families of the same list
  single-full-1   gpurec in single precision on all 10,869 families
  single-full-2   the same again. The paper ran these on an NVIDIA H100 (partition gpu_h100_interactive, 1 GPU,
                  8 cores, 64 GB) with the release d48c48f7
Compare with: python analysis/compare_fp64_fp32.py OUTPUT_ROOT/cluster/out_single_full_1 OUTPUT_ROOT/cluster/out_single_full_2
OUTPUT_ROOT/cluster/out_double_half_1 OUTPUT_ROOT/cluster/out_double_half_2
The paper started each srun in the foreground of a session kept open from another machine, because the login node of its
cluster stops processes left without a parent (at minute 3 of every hour).
usage: run_cluster.py PATHS_TOML PART"""
import shutil

from common import (FAMILY_LISTS, gpurec_command, gpurec_fit_arguments, hogenom_families_file, hogenom_species_tree,
                    log_progress, output_dir, read_names, run_logged, run_part, write_hogenom_families)


def fit(paths, name, families_file, options_key, extra):
    """One gpurec fit as a SLURM job: srun with the options paths.toml gives under [cluster] OPTIONS_KEY, then the plain
    gpurec command under /usr/bin/time -v (inside the job, so that the time does not include the wait for a GPU)."""
    folder = output_dir(paths, "cluster")
    shutil.rmtree(folder / f"out_{name}", ignore_errors=True)
    log_progress(folder / "progress.txt", f"{name}: start")
    command = ([paths["cluster"]["srun"]] + list(paths["cluster"][options_key])
               + [paths["tools"]["time"], "-v", "-o", folder / f"{name}.time"]
               + gpurec_command("gpurec_plain.py", [], ["-p", f"out_{name}"] + gpurec_fit_arguments(
                   families_file, hogenom_species_tree(paths), extra)))
    code = run_logged(command, folder / f"{name}.log", folder / f"{name}.err", folder)
    log_progress(folder / "progress.txt", f"{name}: exit={code}")


def double_half(k):
    def run(paths):
        names = read_names(FAMILY_LISTS / "hogenom_all.txt")
        first = len(names) // 2   # 5,434 of the 10,869 families
        half = names[:first] if k == 1 else names[first:]
        fam = write_hogenom_families(paths, half, f"hogenom_all_half_{k}.families.txt")
        fit(paths, f"double_half_{k}", fam, "double_precision_job", ["--precision", "float64"])
    return run


def single_full(k):
    def run(paths):
        fit(paths, f"single_full_{k}", hogenom_families_file(paths, "hogenom_all"), "single_precision_job", [])
    return run


PARTS = {"double-half-1": double_half(1), "double-half-2": double_half(2),
         "single-full-1": single_full(1), "single-full-2": single_full(2)}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

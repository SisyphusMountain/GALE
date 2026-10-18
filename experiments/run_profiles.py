"""Profiles of the two programs (paper: section "The computation to be parallelised", paragraph "Where the time goes";
section "Example", paragraph "Speed"; appendix "Details of the experiments", paragraph "Speed"). Outputs go to
OUTPUT_ROOT/profiles.
Parts:
  gpurec-nsys              the full fit of gpurec under NVIDIA Nsight Systems (nsys profile), with the NVTX ranges of
                           wrappers/gpurec_nvtx.py, then the two summary reports as CSV files (full_nvtx_gpu_proj_sum.csv,
                           full_cuda_gpu_kern_sum.csv) read by analysis/nsys_shares.py
  alerax-callgrind         AleRax as released fitting the rates of one median-size HOGENOM family (CLU_005170_8_0_C) on
                           one process: first a plain timed run (alerax_plain), then the same under valgrind's callgrind
                           tool, which counts the instructions executed per function (alerax.callgrind.out, annotated
                           in alerax_callgrind_inclusive.txt and alerax_callgrind_exclusive.txt)
  alerax-scaled-vs-double  callgrind on one evaluation of AleRax at fixed rates (0.1, 0.1, 0.1) for the median family
                           (computed in plain doubles) and for CLU_004317_2_1_C (the family below the double-precision
                           range with the fewest splits, computed with AleRax's scaled numbers):
                           fix_one_family.callgrind.out, fix_scaled_family.callgrind.out
                           (analysis/callgrind_calls.py)
usage: run_profiles.py PATHS_TOML PART"""
from common import (COMMON_OPTIONS, alerax_fit_arguments, gpurec_command, gpurec_fit_arguments, hogenom_families_file,
                    hogenom_species_tree, log_progress, output_dir, run_logged, run_part, run_timed)


def gpurec_nsys(paths):
    folder = output_dir(paths, "profiles")
    nsys = paths["tools"]["nsys"]
    log_progress(folder / "progress.txt", "full: start")
    code = run_timed(paths, [nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "-o", "full",
                             "--force-overwrite", "true"]
                     + gpurec_command("gpurec_nvtx.py", [], ["-p", "out_full"] + gpurec_fit_arguments(
                         hogenom_families_file(paths, "hogenom_all"), hogenom_species_tree(paths), [])),
                     folder / "full.time", folder / "full.log", folder / "full.err", folder)
    log_progress(folder / "progress.txt", f"full: exit={code}")
    code = run_logged([nsys, "stats", "--report", "nvtx_gpu_proj_sum", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                       "--output", "full", "full.nsys-rep"], folder / "nsys_stats.log", folder / "nsys_stats.err", folder)
    log_progress(folder / "progress.txt", f"nsys stats: exit={code}")


def callgrind(paths, folder, name, list_name, arguments):
    """AleRax as released on one process under callgrind; instruction counts in NAME.callgrind.out."""
    command = [paths["tools"]["mpiexec"], "-np", "1", paths["tools"]["valgrind"], "--tool=callgrind",
               f"--callgrind-out-file={folder / (name + '.callgrind.out')}", paths["alerax"]["released"],
               "-f", hogenom_families_file(paths, list_name), "-p", f"out_{name}"] + arguments
    log_progress(folder / "progress.txt", f"{name}: start")
    code = run_logged(command, folder / f"{name}.log", folder / f"{name}.err", folder)
    log_progress(folder / "progress.txt", f"{name}: exit={code}")


def alerax_callgrind(paths):
    folder = output_dir(paths, "profiles")
    fit = alerax_fit_arguments(hogenom_families_file(paths, "profile_median_family"), hogenom_species_tree(paths), [])
    log_progress(folder / "progress.txt", "alerax_plain: start")
    code = run_timed(paths, [paths["tools"]["mpiexec"], "-np", "1", paths["alerax"]["released"], "-p", "out_alerax_plain"] + fit,
                     folder / "alerax_plain.time", folder / "alerax_plain.log", folder / "alerax_plain.err", folder)
    log_progress(folder / "progress.txt", f"alerax_plain: exit={code}")
    callgrind(paths, folder, "alerax", "profile_median_family", fit[2:])
    for option, out in (("--inclusive=yes", "alerax_callgrind_inclusive.txt"), ("--inclusive=no", "alerax_callgrind_exclusive.txt")):
        run_logged([paths["tools"]["callgrind_annotate"], option, "alerax.callgrind.out"], folder / out,
                   folder / (out + ".err"), folder)


def alerax_scaled_vs_double(paths):
    folder = output_dir(paths, "profiles")
    arguments = ["-s", hogenom_species_tree(paths)] + COMMON_OPTIONS + ["--gene-tree-samples", "0", "--fix-rates",
                                                                        "--d", "0.1", "--l", "0.1", "--t", "0.1"]
    for name, list_name in (("fix_one_family", "profile_median_family"), ("fix_scaled_family", "profile_scaled_family")):
        callgrind(paths, folder, name, list_name, arguments)


PARTS = {"gpurec-nsys": gpurec_nsys, "alerax-callgrind": alerax_callgrind, "alerax-scaled-vs-double": alerax_scaled_vs_double}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

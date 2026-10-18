"""Single against double precision (paper: Table "tab:precision", appendix "Representation of the numbers", paragraphs
"Single versus double precision", and the sizes of the likelihoods in section "Numbers: logarithms, offsets and
precision"). Outputs go to OUTPUT_ROOT/precision.
Parts (one GPU job at a time):
  table         measure/precision_test.py on the 200-family set: loss and gradient in both precisions at two rate
                points, and their times (precision_test.log)
  all-families  measure/precision_all.py on all 10,869 families at the rates of the fit full_1 of run_hogenom.py: one
                loss evaluation in each precision (precision_all.tsv, precision_all.log)
  double-fits   gpurec fits all families in double precision (--precision float64), in two halves (the first 5,434
                families of the list and the other 5,435; the paper ran the two halves on two NVIDIA L40S cards at the
                same time); analysis/compare_fp64_fp32.py compares them with the single-precision fits full_1 and full_2
usage: run_precision.py PATHS_TOML PART"""
from common import (CONFIGS, FAMILY_LISTS, gpurec_fit_arguments, hogenom_families_file, hogenom_species_tree,
                    output_dir, read_names, run_gpurec, run_part, run_python, write_hogenom_families)


def table(paths):
    run_python(output_dir(paths, "precision"), "precision_test", "measure/precision_test.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths),
                CONFIGS / "single_precision.toml", CONFIGS / "double_precision.toml"])


def all_families(paths):
    folder = output_dir(paths, "precision")
    run_python(folder, "precision_all", "measure/precision_all.py",
               [hogenom_families_file(paths, "hogenom_all"), hogenom_species_tree(paths),
                output_dir(paths, "hogenom") / "out_full_1" / "model_parameters",
                CONFIGS / "single_precision.toml", CONFIGS / "double_precision.toml", folder / "precision_all.tsv"])


def double_fits(paths):
    folder = output_dir(paths, "precision")
    names = read_names(FAMILY_LISTS / "hogenom_all.txt")
    first = len(names) // 2   # 5,434 of the 10,869 families, as in the paper's first half
    for k, half in ((1, names[:first]), (2, names[first:])):
        fam = write_hogenom_families(paths, half, f"hogenom_all_half_{k}.families.txt")
        run_gpurec(paths, folder, f"double_half_{k}", "gpurec_plain.py", [],
                   gpurec_fit_arguments(fam, hogenom_species_tree(paths), ["--precision", "float64"]), None)


PARTS = {"table": table, "all-families": all_families, "double-fits": double_fits}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

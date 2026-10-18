"""The archaeal gene families of Huang et al. (Zenodo 18701544), a second data set (paper: paragraph "A second data
set" and appendix "Details of the experiments"): the 7,059 families of the deposit's families file
(6_qmds/alerax_families.txt, whose names are copied in data/families/archaea_all.txt), each a sample of 1,000
ultrafast-bootstrap gene trees, a rooted species tree of 257 species, and the fraction of each genome's genes that is
missing. Outputs go to OUTPUT_ROOT/archaea.
Parts:
  gpurec-fit        gpurec fit of all families with the missing fractions, timed (gpurec)
  alerax-fit        AleRax 1.4.0 as released, the same, alerax_ranks MPI processes, timed (alerax)
  score             both fits' rates scored by gpurec in double precision with the missing fractions
                    (scores_gpurec.txt, scores_alerax.txt; analysis/compare_optima.py)
  fixed-rates-200   200 families taken at random (data/families/archaea_200.txt) at fixed rates (0.2, 0.3, 0.1) with
                    the missing fractions: AleRax as released (alerax_released_fixed), AleRax with the four changes and
                    16 passes (alerax_four_16_fixed), gpurec in double precision (gpurec_fixed.txt)
  outliers-no-fm    the 16 of these families where gpurec and the changed AleRax differ by more than 0.001 nats, again
                    without the missing fractions (alerax_four_16_nofm, gpurec_nofm.txt)
usage: run_archaea.py PATHS_TOML PART"""
from common import (CONFIGS, FAMILY_LISTS, alerax_fit_arguments, alerax_fixed_rate_arguments, archaea_families_file,
                    archaea_fraction_missing, archaea_species_tree, gpurec_fit_arguments, output_dir, rates_table, read_names, run_alerax, run_gpurec, run_part, run_python)

RATES = (0.2, 0.3, 0.1)


def all_families(paths):
    return archaea_families_file(paths, read_names(FAMILY_LISTS / "archaea_all.txt"), "archaea_all.families.txt")


def gpurec_fit(paths):
    run_gpurec(paths, output_dir(paths, "archaea"), "gpurec", "gpurec_plain.py", [],
               gpurec_fit_arguments(all_families(paths), archaea_species_tree(paths),
                                    ["--fraction-missing-file", archaea_fraction_missing(paths)]), None)


def alerax_fit(paths):
    run_alerax(paths, output_dir(paths, "archaea"), "alerax", "released", paths["machine"]["alerax_ranks"],
               alerax_fit_arguments(all_families(paths), archaea_species_tree(paths),
                                    ["--fraction-missing-file", archaea_fraction_missing(paths)]))


def score(paths):
    folder = output_dir(paths, "archaea")
    fam = all_families(paths)
    for name in ("gpurec", "alerax"):
        rates_table(folder / f"out_{name}" / "model_parameters", folder / f"rates_{name}.tsv")
        run_python(folder, f"score_{name}", "measure/eval_at_rates.py",
                   [fam, archaea_species_tree(paths), archaea_fraction_missing(paths), CONFIGS / "double_precision.toml",
                    folder / f"rates_{name}.tsv", folder / f"scores_{name}.txt"])


def fixed_rates_200(paths):
    folder = output_dir(paths, "archaea")
    names = read_names(FAMILY_LISTS / "archaea_200.txt")
    fam = archaea_families_file(paths, names, "archaea_200.families.txt")
    fm = ["--fraction-missing-file", archaea_fraction_missing(paths)]
    run_alerax(paths, folder, "alerax_released_fixed", "released", paths["machine"]["alerax_ranks"],
               alerax_fixed_rate_arguments(fam, archaea_species_tree(paths), RATES, fm))
    run_alerax(paths, folder, "alerax_four_16_fixed", "four", paths["machine"]["alerax_ranks"],
               alerax_fixed_rate_arguments(fam, archaea_species_tree(paths), RATES, fm + ["--fixed-point-iterations", 16]))
    (folder / "rates_fixed.tsv").write_text("".join(f"{n} {RATES[0]} {RATES[1]} {RATES[2]}\n" for n in names))
    run_python(folder, "gpurec_fixed", "measure/eval_at_rates.py",
               [fam, archaea_species_tree(paths), archaea_fraction_missing(paths), CONFIGS / "double_precision.toml",
                folder / "rates_fixed.tsv", folder / "gpurec_fixed.txt"])


def outliers_no_fm(paths):
    folder = output_dir(paths, "archaea")
    names = read_names(FAMILY_LISTS / "archaea_outliers16.txt")
    fam = archaea_families_file(paths, names, "archaea_outliers16.families.txt")
    run_alerax(paths, folder, "alerax_four_16_nofm", "four", len(names),
               alerax_fixed_rate_arguments(fam, archaea_species_tree(paths), RATES, ["--fixed-point-iterations", 16]))
    (folder / "rates_outliers16.tsv").write_text("".join(f"{n} {RATES[0]} {RATES[1]} {RATES[2]}\n" for n in names))
    run_python(folder, "gpurec_nofm", "measure/eval_at_rates.py",
               [fam, archaea_species_tree(paths), "none", CONFIGS / "double_precision.toml",
                folder / "rates_outliers16.tsv", folder / "gpurec_nofm.txt"])


PARTS = {"gpurec-fit": gpurec_fit, "alerax-fit": alerax_fit, "score": score, "fixed-rates-200": fixed_rates_200,
         "outliers-no-fm": outliers_no_fm}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

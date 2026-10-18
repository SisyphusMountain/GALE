"""Agreement of the optima, when each program fits its own rates (paper: paragraph "Agreement with AleRax" and appendix
"Details of the experiments", paragraph "Agreement with AleRax at the optimum"): the first 200 families of random batch
1, fitted by gpurec and by three AleRax runs, every fit scored by gpurec in double precision with converged fixed points
(measure/eval_at_rates.py). Outputs go to OUTPUT_ROOT/optima; analysis/compare_optima.py compares the scores.
Parts:
  gpurec-fit    gpurec fit (gpurec), timed
  alerax-fits   AleRax fits on 24 ranks, timed: as released (alerax_released), with the four changes and 4 passes
                (alerax_four_4), and with 16 passes (alerax_four_16)
  score         each fit's rates (rates_NAME.tsv) scored by gpurec in double precision (scores_NAME.txt)
usage: run_optima.py PATHS_TOML PART"""
from common import (CONFIGS, alerax_fit_arguments, gpurec_fit_arguments, hogenom_families_file, hogenom_species_tree,
                    output_dir, rates_table, run_alerax, run_gpurec, run_part, run_python)

FITS = ("gpurec", "alerax_released", "alerax_four_4", "alerax_four_16")


def gpurec_fit(paths):
    run_gpurec(paths, output_dir(paths, "optima"), "gpurec", "gpurec_plain.py", [],
               gpurec_fit_arguments(hogenom_families_file(paths, "opt200"), hogenom_species_tree(paths), []), None)


def alerax_fits(paths):
    folder = output_dir(paths, "optima")
    fam = hogenom_families_file(paths, "opt200")
    for name, build, extra in (("alerax_released", "released", []),
                               ("alerax_four_4", "four", ["--fixed-point-iterations", 4]),
                               ("alerax_four_16", "four", ["--fixed-point-iterations", 16])):
        run_alerax(paths, folder, name, build, paths["machine"]["alerax_ranks"],
                   alerax_fit_arguments(fam, hogenom_species_tree(paths), extra))


def score(paths):
    folder = output_dir(paths, "optima")
    fam = hogenom_families_file(paths, "opt200")
    for name in FITS:
        rates_table(folder / f"out_{name}" / "model_parameters", folder / f"rates_{name}.tsv")
        run_python(folder, f"score_{name}", "measure/eval_at_rates.py",
                   [fam, hogenom_species_tree(paths), "none", CONFIGS / "double_precision.toml",
                    folder / f"rates_{name}.tsv", folder / f"scores_{name}.txt"])


PARTS = {"gpurec-fit": gpurec_fit, "alerax-fits": alerax_fits, "score": score}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

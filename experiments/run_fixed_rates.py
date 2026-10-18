"""Agreement of gpurec and AleRax at the same fixed rates (paper: paragraph "Agreement with AleRax", Figure
"fig_alerax_fixes", Table "tab:agree" and appendix "Numerical accuracy"). gpurec is evaluated with converged settings
(measure/fixedrate_eval.py, single precision: 256 fixed-point iterations); AleRax with each cumulative build of
data/alerax (the four changes added one at a time). Outputs go to OUTPUT_ROOT/fixed_rates; analysis/compare_fixedrate.py
compares each AleRax file with gpurec's.
Parts:
  gpurec-1042        gpurec, the 1,042-family set at (D, L, T) = (0.2, 0.3, 0.1) (gpurec_1042.txt)
  gpurec-cli-1042    the default command of gpurec with --fix-rates at (0.2, 0.3, 0.1), timed (cli_1042)
  alerax-1042        AleRax at (0.2, 0.3, 0.1) on the 1,042-family set, 24 ranks, four builds, timed:
                       passes_4    as released (4 passes)
                       addition_16 16 passes and the addition of scaled values corrected
                       switch_16   and the switch to scaled values below the smallest normal double
                       four_16     and the subtraction of scaled values corrected (all four changes)
  alerax-200-passes  AleRax, build "passes" (arithmetic of the release), the 200-family set at (0.2, 0.3, 0.1) with 4,
                     16 and 64 passes (passes_4_200, passes_16_200, passes_64_200): how far 4 passes are from converged
                     (compare_fixedrate.py with the 64-pass file in place of gpurec's)
Each AleRax run writes its per-family log-likelihoods to out_NAME/per_fam_likelihoods.txt.
usage: run_fixed_rates.py PATHS_TOML PART"""
from common import (CONFIGS, alerax_fixed_rate_arguments, gpurec_fit_arguments, hogenom_families_file,
                    hogenom_species_tree, output_dir, run_alerax, run_gpurec, run_part, run_python)

RATES = (0.2, 0.3, 0.1)


def gpurec_eval(paths, list_name, rates, name):
    folder = output_dir(paths, "fixed_rates")
    run_python(folder, name, "measure/fixedrate_eval.py",
               [hogenom_families_file(paths, list_name), hogenom_species_tree(paths), CONFIGS / "single_precision.toml",
                *rates, folder / f"{name}.txt"])


def gpurec_cli(paths):
    folder = output_dir(paths, "fixed_rates")
    d, l, t = RATES
    run_gpurec(paths, folder, "cli_1042", "gpurec_plain.py", [],
               gpurec_fit_arguments(hogenom_families_file(paths, "hogenom_1042"), hogenom_species_tree(paths),
                                    ["--fix-rates", "--d", d, "--l", l, "--t", t]), None)


def alerax_1042(paths):
    folder = output_dir(paths, "fixed_rates")
    fam = hogenom_families_file(paths, "hogenom_1042")
    for name, build, passes in (("passes_4", "passes", 4), ("addition_16", "addition", 16),
                                ("switch_16", "switch", 16), ("four_16", "four", 16)):
        run_alerax(paths, folder, name, build, paths["machine"]["alerax_ranks"],
                   alerax_fixed_rate_arguments(fam, hogenom_species_tree(paths), RATES, ["--fixed-point-iterations", passes]))


def alerax_200_passes(paths):
    folder = output_dir(paths, "fixed_rates")
    fam = hogenom_families_file(paths, "hogenom_200")
    for passes in (4, 16, 64):
        run_alerax(paths, folder, f"passes_{passes}_200", "passes", paths["machine"]["alerax_ranks"],
                   alerax_fixed_rate_arguments(fam, hogenom_species_tree(paths), RATES, ["--fixed-point-iterations", passes]))


PARTS = {
    "gpurec-1042": lambda paths: gpurec_eval(paths, "hogenom_1042", RATES, "gpurec_1042"),
    "gpurec-cli-1042": gpurec_cli,
    "alerax-1042": alerax_1042,
    "alerax-200-passes": alerax_200_passes,
}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

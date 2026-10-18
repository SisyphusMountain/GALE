"""Wall time against the number of families (paper: paragraph "Scaling", Figure "fig_scaling" and appendix "Details of
the experiments", paragraph "Scaling"), gpurec on the GPU and AleRax 1.4.0 on 24 CPU cores, whole command, reading
included. AleRax runs with the build "passes" (the option
--fixed-point-iterations) at 4 passes, which computes what AleRax 1.4.0 as released computes.
Outputs go to OUTPUT_ROOT/scaling. Parts (run one at a time, alone on the machine):
  gpurec-nested-large  nested random subsets of 100, 300, 1,000 and 3,000 families, two runs each (g_nestedN_1, _2)
  alerax-nested        AleRax once on each of these nested subsets
  gpurec-batches       the 22 random batches of 500 families that partition the full set (369 in the last), one run
                       each (g_bNN)
  alerax-batches       AleRax once on each batch (a_bNN)
  small-draws          20 independent random sets of 1, 3, 10 and 30 families (data/families/small_draws): for each set,
                       gpurec once untimed to compile the kernels for its sizes, gpurec again (timed), then AleRax
                       with one MPI rank per family below 24 families (AleRax refuses more ranks than families) and 24
                       from 24 families on; the kernels compiled during each gpurec run are counted (compiles.txt); in
                       OUTPUT_ROOT/scaling/small_draws
usage: run_scaling.py PATHS_TOML PART"""
from common import (FAMILY_LISTS, alerax_fit_arguments, gpurec_fit_arguments, hogenom_families_file,
                    hogenom_species_tree, output_dir, run_alerax, run_gpurec, run_part)

LARGE = (100, 300, 1000, 3000)
BATCHES = [f"{i:02d}" for i in range(1, 23)]
ALERAX_PASSES = ["--fixed-point-iterations", "4"]  # AleRax 1.4.0's own number of passes


def gpurec_nested_large(paths):
    folder = output_dir(paths, "scaling")
    for n in LARGE:
        fam = hogenom_families_file(paths, f"nested_{n}")
        for k in (1, 2):
            run_gpurec(paths, folder, f"g_nested{n}_{k}", "gpurec_plain.py", [],
                       gpurec_fit_arguments(fam, hogenom_species_tree(paths), []), None)


def alerax_nested(paths):
    folder = output_dir(paths, "scaling")
    for n in LARGE:
        fam = hogenom_families_file(paths, f"nested_{n}")
        run_alerax(paths, folder, f"a_nested{n}", "passes", paths["machine"]["alerax_ranks"],
                   alerax_fit_arguments(fam, hogenom_species_tree(paths), ALERAX_PASSES))


def gpurec_batches(paths):
    folder = output_dir(paths, "scaling")
    for b in BATCHES:
        fam = hogenom_families_file(paths, f"random_batches/batch_{b}")
        run_gpurec(paths, folder, f"g_b{b}", "gpurec_plain.py", [], gpurec_fit_arguments(fam, hogenom_species_tree(paths), []), None)


def alerax_batches(paths):
    folder = output_dir(paths, "scaling")
    for b in BATCHES:
        fam = hogenom_families_file(paths, f"random_batches/batch_{b}")
        run_alerax(paths, folder, f"a_b{b}", "passes", paths["machine"]["alerax_ranks"],
                   alerax_fit_arguments(fam, hogenom_species_tree(paths), ALERAX_PASSES))


def small_draws(paths):
    folder = output_dir(paths, "scaling/small_draws")
    for list_file in sorted((FAMILY_LISTS / "small_draws").glob("*.txt")):
        s = list_file.stem
        n = int(s[1:].split("_")[0])
        fam = hogenom_families_file(paths, f"small_draws/{s}")
        for k in ("warm", "timed"):
            run_gpurec(paths, folder, f"g_{s}_{k}", "gpurec_plain.py", [],
                       gpurec_fit_arguments(fam, hogenom_species_tree(paths), []), folder / "compiles.txt")
        run_alerax(paths, folder, f"a_{s}", "passes", min(n, paths["machine"]["alerax_ranks"]),
                   alerax_fit_arguments(fam, hogenom_species_tree(paths), ALERAX_PASSES))


PARTS = {
    "gpurec-nested-large": gpurec_nested_large,
    "alerax-nested": alerax_nested,
    "gpurec-batches": gpurec_batches,
    "alerax-batches": alerax_batches,
    "small-draws": small_draws,
}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

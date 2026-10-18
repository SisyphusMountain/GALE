"""Measurements on the CPU only, from gpurec's preprocessing and planning: no fit. Outputs go to OUTPUT_ROOT/planning.
Parts:
  waves-table    measure/waves_table.py on all 10,869 HOGENOM families: the waves planned with the release's limits
                 against each family alone (Table "tab:waves"; waves_full.json)
  waves-figure   measure/waves_figure_data.py on the 1,055 HOGENOM families of the appendix "Scheduling the operations
                 on a GPU" (CSV files in waves_1055/, drawn by figures/make_fig_waves.py)
  family-stats   measure/family_stats.py: sizes of the families and of their likelihoods (paragraph "Where the time
                 goes", section "Numbers"); needs the outputs of run_hogenom.py gpurec-full and run_precision.py
                 all-families
usage: run_planning.py PATHS_TOML PART"""
from common import (FAMILY_LISTS, hogenom_families_file, hogenom_species_tree, output_dir, read_names, run_part,
                    run_python)


def waves_table(paths):
    folder = output_dir(paths, "planning")
    run_python(folder, "waves_table", "measure/waves_table.py",
               [hogenom_families_file(paths, "hogenom_all"), hogenom_species_tree(paths), folder / "waves_full.json"])


def waves_figure(paths):
    folder = output_dir(paths, "planning")
    run_python(folder, "waves_1055", "measure/waves_figure_data.py", [paths["data"]["hogenom_1055_dir"], folder / "waves_1055"])


def family_stats(paths):
    folder = output_dir(paths, "planning")
    run_python(folder, "family_stats", "measure/family_stats.py",
               [hogenom_families_file(paths, "hogenom_all"), hogenom_species_tree(paths),
                read_names(FAMILY_LISTS / "profile_median_family.txt")[0],
                output_dir(paths, "precision") / "precision_all.tsv",
                output_dir(paths, "hogenom") / "out_full_1" / "per_fam_likelihoods.txt",
                FAMILY_LISTS / "nested_1000.txt", FAMILY_LISTS / "hogenom_506.txt", FAMILY_LISTS / "hogenom_1042.txt"])


PARTS = {"waves-table": waves_table, "waves-figure": waves_figure, "family-stats": family_stats}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

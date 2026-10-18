"""Warm start of the adjoint and adaptive rebatching (paper: appendix "Details of the experiments", paragraphs "Warm
start of the adjoint" and "Adaptive rebatching", Figure "fig_rebatch"). Every run goes through wrappers/gpurec_trace.py,
which writes a trace of the fit (trace_NAME.json: model builds, loss-and-gradient evaluations, total time, peak GPU
memory). Outputs go to OUTPUT_ROOT/warm_rebatch; analysis/traces.py summarises the traces.
Parts (run one at a time, alone on the machine):
  warm-1042    the 1,042-family set, warm start forced off then left to the memory check, three times in alternation
               (warm_off_1, warm_on_1, ..., warm_on_3)
  warm-full    all families: warm start forced off (full_warm_off_1), left to the memory check (full_warm_on_1), forced
               off again (full_warm_off_2)
  rebatch-1042 the 1,042-family set, with removal of converged families (removal_on_1), without (removal_off), and
               twice more with (removal_on_2, removal_on_3)
usage: run_warm_rebatch.py PATHS_TOML PART"""
from common import gpurec_fit_arguments, hogenom_families_file, hogenom_species_tree, output_dir, run_gpurec, run_part


def traced(paths, list_name, runs):
    """RUNS: (name, warm on|off, removal on|off), run in this order."""
    folder = output_dir(paths, "warm_rebatch")
    fam = hogenom_families_file(paths, list_name)
    for name, warm, rebatch in runs:
        run_gpurec(paths, folder, name, "gpurec_trace.py", [warm, rebatch, folder / f"trace_{name}.json"],
                   gpurec_fit_arguments(fam, hogenom_species_tree(paths), []), None)


PARTS = {
    "warm-1042": lambda paths: traced(paths, "hogenom_1042", [(f"warm_{w}_{i}", w, "on") for i in (1, 2, 3) for w in ("off", "on")]),
    "warm-full": lambda paths: traced(paths, "hogenom_all", [("full_warm_off_1", "off", "on"), ("full_warm_on_1", "on", "on"),
                                                             ("full_warm_off_2", "off", "on")]),
    "rebatch-1042": lambda paths: traced(paths, "hogenom_1042", [("removal_on_1", "on", "on"), ("removal_off", "on", "off"),
                                                                 ("removal_on_2", "on", "on"), ("removal_on_3", "on", "on")]),
}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

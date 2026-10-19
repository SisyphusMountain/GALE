"""Smaller measurements of the solver and of the hardware. Outputs go to OUTPUT_ROOT/measurements, one log per part.
Parts (one GPU job at a time):
  kscan        measure/kscan.py: the 200-family set at the rates of the fit full_1 of run_hogenom.py, against the number
               of fixed-point iterations (appendix "Managing the fixed-point iterations")
  kscan-fitted measure/kscan_per_family.py: the 200-family set at the rates of the fit full_1 of run_hogenom.py, every
               family's log-likelihood for K = 2 to 256 fixed-point iterations (kscan_fitted.csv), then
               analysis/kscan_stats.py against K = 256 (kscan_fitted_stats.csv): panel (a) of Figure "fig:kscan" and
               Table "tab:kscan"
  kscan-fixed  the same at the rates (0.2, 0.3, 0.1) of run_fixed_rates.py (kscan_fixed.csv), the statistics against
               K = 256 and against AleRax with the four changes, out_four_16 of run_fixed_rates.py alerax-1042
               (kscan_fixed_stats.csv): panel (b) of Figure "fig:kscan" and Table "tab:kscan"
  grad-test    measure/grad_test.py: cost of the gradient against the loss alone, and the gradient against finite
               differences (section "The gradient by implicit differentiation")
  grad-resid   measure/grad_resid.py: the first 20 families of the 200-family set in double precision, the gradient
               against central finite differences with adjoint pruning on and truly off (where the difference of 1e-4
               of grad-test comes from)
  launch-cost  measure/launch_cost.py: the cost of one kernel launch (Figure 1 of the paper)
  mem-scan     measure/mem_scan.py: peak GPU memory against the clade limit of a batch, 200-family set
  hardware     measure/hw_bench.py twice (hardware_1.json, hardware_2.json): matrix products and memory copies on the
               CPU and the GPU (Table "tab:hardware")
usage: run_measurements.py PATHS_TOML PART"""
from common import CONFIGS, hogenom_families_file, hogenom_species_tree, output_dir, run_part, run_python
from run_fixed_rates import RATES


def kscan(paths):
    run_python(output_dir(paths, "measurements"), "kscan", "measure/kscan.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths),
                output_dir(paths, "hogenom") / "out_full_1" / "model_parameters", CONFIGS / "single_precision.toml"])


def kscan_fitted(paths):
    folder = output_dir(paths, "measurements")
    run_python(folder, "kscan_fitted", "measure/kscan_per_family.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths), CONFIGS / "single_precision.toml",
                folder / "kscan_fitted.csv", "fitted", output_dir(paths, "hogenom") / "out_full_1" / "model_parameters"])
    run_python(folder, "kscan_fitted_stats", "analysis/kscan_stats.py",
               ["fitted", folder / "kscan_fitted.csv", folder / "kscan_fitted_stats.csv"])


def kscan_fixed(paths):
    folder = output_dir(paths, "measurements")
    alerax = output_dir(paths, "fixed_rates") / "out_four_16" / "per_fam_likelihoods.txt"
    if not alerax.is_file():
        raise SystemExit(f"{alerax} is missing: run run_fixed_rates.py alerax-1042 first")
    run_python(folder, "kscan_fixed", "measure/kscan_per_family.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths), CONFIGS / "single_precision.toml",
                folder / "kscan_fixed.csv", "fixed", *RATES])
    run_python(folder, "kscan_fixed_stats", "analysis/kscan_stats.py",
               ["fixed", folder / "kscan_fixed.csv", alerax, folder / "kscan_fixed_stats.csv"])


def grad_test(paths):
    run_python(output_dir(paths, "measurements"), "grad_test", "measure/grad_test.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths),
                CONFIGS / "single_precision.toml", CONFIGS / "double_precision.toml"])


def grad_resid(paths):
    run_python(output_dir(paths, "measurements"), "grad_resid", "measure/grad_resid.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths), CONFIGS / "double_precision.toml"])


def mem_scan(paths):
    run_python(output_dir(paths, "measurements"), "mem_scan", "measure/mem_scan.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths), CONFIGS / "single_precision.toml"])


def hardware(paths):
    folder = output_dir(paths, "measurements")
    for k in (1, 2):
        run_python(folder, f"hardware_{k}", "measure/hw_bench.py", [folder / f"hardware_{k}.json"])


PARTS = {
    "kscan": kscan,
    "kscan-fitted": kscan_fitted,
    "kscan-fixed": kscan_fixed,
    "grad-test": grad_test,
    "grad-resid": grad_resid,
    "launch-cost": lambda paths: run_python(output_dir(paths, "measurements"), "launch_cost", "measure/launch_cost.py", []),
    "mem-scan": mem_scan,
    "hardware": hardware,
}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

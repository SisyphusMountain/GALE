"""Smaller measurements of the solver and of the hardware. Outputs go to OUTPUT_ROOT/measurements, one log per part.
Parts (one GPU job at a time):
  kscan        measure/kscan.py: the 200-family set at the rates of the fit full_1 of run_hogenom.py, against the number
               of fixed-point iterations (appendix "Managing the fixed-point iterations")
  grad-test    measure/grad_test.py: cost of the gradient against the loss alone, and the gradient against finite
               differences (section "The gradient by implicit differentiation")
  launch-cost  measure/launch_cost.py: the cost of one kernel launch (Figure 1 of the paper)
  mem-scan     measure/mem_scan.py: peak GPU memory against the clade limit of a batch, 200-family set
  hardware     measure/hw_bench.py twice (hardware_1.json, hardware_2.json): matrix products and memory copies on the
               CPU and the GPU (Table "tab:hardware")
usage: run_measurements.py PATHS_TOML PART"""
from common import CONFIGS, hogenom_families_file, hogenom_species_tree, output_dir, run_part, run_python


def kscan(paths):
    run_python(output_dir(paths, "measurements"), "kscan", "measure/kscan.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths),
                output_dir(paths, "hogenom") / "out_full_1" / "model_parameters", CONFIGS / "single_precision.toml"])


def grad_test(paths):
    run_python(output_dir(paths, "measurements"), "grad_test", "measure/grad_test.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths),
                CONFIGS / "single_precision.toml", CONFIGS / "double_precision.toml"])


def mem_scan(paths):
    run_python(output_dir(paths, "measurements"), "mem_scan", "measure/mem_scan.py",
               [hogenom_families_file(paths, "hogenom_200"), hogenom_species_tree(paths), CONFIGS / "single_precision.toml"])


def hardware(paths):
    folder = output_dir(paths, "measurements")
    for k in (1, 2):
        run_python(folder, f"hardware_{k}", "measure/hw_bench.py", [folder / f"hardware_{k}.json"])


PARTS = {
    "kscan": kscan,
    "grad-test": grad_test,
    "launch-cost": lambda paths: run_python(output_dir(paths, "measurements"), "launch_cost", "measure/launch_cost.py", []),
    "mem-scan": mem_scan,
    "hardware": hardware,
}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

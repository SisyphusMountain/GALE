"""Helpers shared by the experiment drivers (the run_*.py scripts of this folder).

They read the configuration file paths.toml, write AleRax-format families files from the family lists of data/families,
run a command under /usr/bin/time -v (which records the wall time and the peak memory of the command), and read the
wall time back. Nothing here changes what gpurec or AleRax compute.
"""
import argparse
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
DATA = EXPERIMENTS / "data"
FAMILY_LISTS = DATA / "families"
CONFIGS = DATA / "configs"

# Layout of the HOGENOM part of the AleRax data archive (Morel et al. 2024): facts of the archive, not settings.
HOGENOM_GENE_TREES = "families/{name}/gene_trees/ufboot1000.MFP.geneTree.newick"
HOGENOM_SPECIES_TREE = ("runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/"
                        "alegenerax/species_trees/starting_species_tree.newick")

# Layout of the folder 3_Reconciliation/This_study of the archaeal deposit (Huang et al., Zenodo 18701544): facts of
# the deposit, not settings. The species tree is the one rooted between the DPANN and the other archaea ("MHH root").
ARCHAEA_GENE_TREES = "3_UFBOOTs/ufboot_for_alerax/{name}.ufboot"
ARCHAEA_SPECIES_TREE = "4_species_tree/others/Undine_C60_MHHroot_short_name.nw"
ARCHAEA_FRACTION_MISSING = "5_reconcilation models/fraction_missing"

# The options that make AleRax and gpurec fit one set of rates per family on a fixed species tree (both programs
# accept them; see the paper, appendix "Details of the experiments").
COMMON_OPTIONS = ["--rec-model", "UndatedDTL", "--model-parametrization", "PER-FAMILY", "--species-tree-search", "SKIP"]


def read_paths(path):
    """The configuration file, as a dictionary. Every key the drivers use must be present (see paths.toml)."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def output_dir(paths, name):
    """The folder of one experiment under the output root; created if needed."""
    folder = Path(paths["output"]["root"]) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_names(list_file):
    """Family names of a list of data/families, one per line, in order."""
    return [line.strip() for line in open(list_file) if line.strip()]


def hogenom_species_tree(paths):
    return Path(paths["data"]["hogenom_root"]) / HOGENOM_SPECIES_TREE


def archaea_species_tree(paths):
    return Path(paths["data"]["archaea_root"]) / ARCHAEA_SPECIES_TREE


def archaea_fraction_missing(paths):
    return Path(paths["data"]["archaea_root"]) / ARCHAEA_FRACTION_MISSING


def hogenom_families_file(paths, list_name):
    """Write, under OUTPUT_ROOT/families, the AleRax families file of the HOGENOM list data/families/LIST_NAME.txt, and
    return its path."""
    return write_hogenom_families(paths, read_names(FAMILY_LISTS / (list_name + ".txt")),
                                  list_name.replace("/", "_") + ".families.txt")


def write_hogenom_families(paths, names, file_name):
    """Write, under OUTPUT_ROOT/families, an AleRax families file for the HOGENOM families NAMES, each given by its
    sample of 1,000 gene trees (the key starting_gene_tree, as in the families files of the paper's runs), and return
    its path."""
    root = Path(paths["data"]["hogenom_root"])
    out = output_dir(paths, "families") / file_name
    lines = ["[FAMILIES]"]
    for name in names:
        lines += [f"- {name}", f"starting_gene_tree = {root / HOGENOM_GENE_TREES.format(name=name)}"]
    out.write_text("\n".join(lines) + "\n")
    return out


def archaea_families_file(paths, names, file_name):
    """Write, under OUTPUT_ROOT/families, an AleRax families file for the archaeal families NAMES, each given by its
    sample of 1,000 ultrafast-bootstrap gene trees (the key gene_tree, as in the deposit's own families file)."""
    root = Path(paths["data"]["archaea_root"])
    out = output_dir(paths, "families") / file_name
    lines = ["[FAMILIES]"]
    for name in names:
        lines += [f"- {name}", f"gene_tree = {root / ARCHAEA_GENE_TREES.format(name=name)}"]
    out.write_text("\n".join(lines) + "\n")
    return out


def log_progress(progress_file, message):
    """Append one line, with the time and the load of the machine, to a progress file."""
    load = " ".join(open("/proc/loadavg").read().split()[:3])
    with open(progress_file, "a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} load {load}: {message}\n")


def run_timed(paths, command, time_file, stdout_file, stderr_file, cwd):
    """Run COMMAND under /usr/bin/time -v (wall time and peak memory written to TIME_FILE), with its output in
    STDOUT_FILE and STDERR_FILE, from the folder CWD. Returns the exit code."""
    full = [paths["tools"]["time"], "-v", "-o", str(time_file)] + [str(part) for part in command]
    with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
        return subprocess.run(full, stdout=out, stderr=err, cwd=cwd).returncode


def run_logged(command, stdout_file, stderr_file, cwd):
    """Run COMMAND without timing it, output in STDOUT_FILE and STDERR_FILE. Returns the exit code."""
    with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
        return subprocess.run([str(part) for part in command], stdout=out, stderr=err, cwd=cwd).returncode


def gpurec_command(wrapper, wrapper_arguments, gpurec_arguments):
    """The command that starts one of the wrappers of experiments/wrappers (the gpurec command line, with or without
    hooks) with the Python that runs this driver, so that the gpurec of this environment is the one measured."""
    return ([sys.executable, "-u", str(EXPERIMENTS / "wrappers" / wrapper)] + [str(a) for a in wrapper_arguments]
            + [str(a) for a in gpurec_arguments])


def alerax_command(paths, build, ranks, arguments):
    """The command that runs one AleRax build (a key of [alerax] in paths.toml) on RANKS MPI processes, inside the
    memory cap of paths.toml (AleRax holds all its families in memory; the cap stops it alone if it grows too large)."""
    return (list(paths["tools"]["alerax_memory_cap"]) + [paths["tools"]["mpiexec"], "-np", str(ranks),
                                                          paths["alerax"][build]] + [str(a) for a in arguments])


def triton_cache_entries(paths):
    """Number of entries of Triton's kernel cache: the difference before and after a run is the number of kernels
    Triton compiled during it (several kernels take wave and batch sizes as compile-time constants)."""
    cache = Path(paths["tools"]["triton_cache"])
    return len(list(cache.iterdir())) if cache.is_dir() else 0


def wall_seconds(time_file):
    """Wall time of a command, in seconds, from the output of /usr/bin/time -v."""
    for line in open(time_file):
        if "Elapsed (wall clock)" in line:
            parts = [float(x) for x in line.split()[-1].split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    raise ValueError(f"no wall time in {time_file}")


def rates_table(model_parameters_dir, out_file):
    """Write one line "family D L T" per family from the folder model_parameters of an AleRax or gpurec output (the
    second line of each FAMILY_rates.txt), as read by measure/eval_at_rates.py."""
    rows = []
    for f in sorted(Path(model_parameters_dir).glob("*_rates.txt")):
        rows.append(f"{f.name[:-len('_rates.txt')]} {f.read_text().splitlines()[1].strip()}")
    Path(out_file).write_text("\n".join(rows) + "\n")


def gpurec_fit_arguments(families_file, species_tree, extra):
    """The arguments of the paper's gpurec command: one set of rates per family, fixed species tree, no sampling of
    reconciliations, plus EXTRA (for instance --fraction-missing-file FILE)."""
    return ["-f", families_file, "-s", species_tree] + COMMON_OPTIONS + ["-g", "0"] + list(extra)


def alerax_fit_arguments(families_file, species_tree, extra):
    """The arguments of the paper's AleRax command: the options of gpurec's, with AleRax's L-BFGS-B optimiser and no
    sampling of reconciliations, plus EXTRA (for instance --fixed-point-iterations 4, or --fix-rates --d 0.2 ...)."""
    return (["-f", families_file, "-s", species_tree] + COMMON_OPTIONS
            + ["--gene-tree-samples", "0", "--rate-optimizer", "LBFGSB"] + list(extra))


def alerax_fixed_rate_arguments(families_file, species_tree, rates, extra):
    """The arguments of an AleRax evaluation at fixed rates RATES = (D, L, T), the same for every family, no sampling of
    reconciliations, plus EXTRA (for instance --fixed-point-iterations 16)."""
    d, l, t = rates
    return (["-f", families_file, "-s", species_tree] + COMMON_OPTIONS
            + ["--gene-tree-samples", "0", "--fix-rates", "--d", str(d), "--l", str(l), "--t", str(t)] + list(extra))


def run_gpurec(paths, folder, name, wrapper, wrapper_arguments, gpurec_arguments, compiles_file):
    """One timed gpurec run in FOLDER: output folder out_NAME (removed first: gpurec refuses a folder that holds
    results), wall time in NAME.time, gpurec's screen output in NAME.log and NAME.err. If COMPILES_FILE is not None, the
    number of kernels Triton compiled during the run is appended to it. Returns the exit code."""
    shutil.rmtree(folder / f"out_{name}", ignore_errors=True)
    log_progress(folder / "progress.txt", f"{name}: start")
    before = triton_cache_entries(paths)
    code = run_timed(paths, gpurec_command(wrapper, wrapper_arguments, ["-p", f"out_{name}"] + list(gpurec_arguments)),
                     folder / f"{name}.time", folder / f"{name}.log", folder / f"{name}.err", folder)
    if compiles_file is not None:
        with open(compiles_file, "a") as handle:
            handle.write(f"{name} new_triton_cache_entries={triton_cache_entries(paths) - before}\n")
    log_progress(folder / "progress.txt", f"{name}: exit={code}")
    return code


def run_alerax(paths, folder, name, build, ranks, arguments):
    """One timed AleRax run in FOLDER (build: a key of [alerax] in paths.toml), on RANKS MPI processes: output folder
    out_NAME (removed first), wall time in NAME.time, screen output in NAME.log and NAME.err. Returns the exit code."""
    shutil.rmtree(folder / f"out_{name}", ignore_errors=True)
    log_progress(folder / "progress.txt", f"{name}: start ({build}, {ranks} ranks)")
    code = run_timed(paths, alerax_command(paths, build, ranks, ["-p", f"out_{name}"] + list(arguments)),
                     folder / f"{name}.time", folder / f"{name}.log", folder / f"{name}.err", folder)
    log_progress(folder / "progress.txt", f"{name}: exit={code}")
    return code


def run_python(folder, name, script, arguments):
    """Run one script of experiments/measure (or analysis) with the Python of this driver, output in NAME.log and
    NAME.err of FOLDER. Returns the exit code."""
    log_progress(folder / "progress.txt", f"{name}: start")
    code = run_logged([sys.executable, "-u", EXPERIMENTS / script] + list(arguments),
                      folder / f"{name}.log", folder / f"{name}.err", folder)
    log_progress(folder / "progress.txt", f"{name}: exit={code}")
    return code


def run_part(doc, parts):
    """Command line of a driver: python run_X.py PATHS_TOML PART, where PART is one of PARTS (a dictionary from part
    names to functions that take the configuration)."""
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", help="the configuration file (a copy of paths.toml with the placeholders replaced)")
    parser.add_argument("part", choices=list(parts), help="the part of the experiment to run")
    args = parser.parse_args()
    parts[args.part](read_paths(args.paths))

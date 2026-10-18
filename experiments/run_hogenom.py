"""Timed fits of the HOGENOM families by gpurec and AleRax (paper: section "Example: rates for 10,869 HOGENOM families",
and appendix "Details of the experiments", paragraphs "Speed" and "Agreement with AleRax at the optimum").
Every run is the whole command, reading included, timed by /usr/bin/time -v; outputs go to OUTPUT_ROOT/hogenom.
Parts (run one at a time, alone on the machine):
  gpurec-full         gpurec on all 10,869 families, twice (full_1, full_2): the full-set time and the agreement of
                      the two runs
  gpurec-full-timers  the same command with timers that wait for the GPU around the gradient, the curvature and the
                      batch builds (full_timers): where the time goes; its per-family likelihoods are compared with
                      AleRax's (analysis/agreement.py)
  gpurec-1042         gpurec on the 1,042-family set, three times (set1042_1..3)
  gpurec-506          gpurec on the 506-family set (set506)
  alerax-full         AleRax 1.4.0 as released on all families, in eight chunks of at most 1,359 families run one
                      after the other, each on alerax_ranks MPI processes (24 in the paper): AleRax holds all its
                      families in memory, more than the machine's 123 GiB for the full set. The per-family fits are
                      independent, so the times add up and the per-family likelihoods are those of one run
usage: run_hogenom.py PATHS_TOML PART"""
import math

from common import (alerax_fit_arguments, hogenom_families_file, hogenom_species_tree, output_dir, read_names,
                    FAMILY_LISTS, gpurec_fit_arguments, run_alerax, run_gpurec, run_part, wall_seconds,
                    write_hogenom_families)

ALERAX_CHUNKS = 8  # the paper's split of the full set for AleRax (memory), not a setting of the measurement


def gpurec_runs(paths, list_name, names, wrapper):
    folder = output_dir(paths, "hogenom")
    fam = hogenom_families_file(paths, list_name)
    for name in names:
        run_gpurec(paths, folder, name, wrapper, [], gpurec_fit_arguments(fam, hogenom_species_tree(paths), []), None)


def alerax_full(paths):
    folder = output_dir(paths, "hogenom")
    names = read_names(FAMILY_LISTS / "hogenom_all.txt")
    per = math.ceil(len(names) / ALERAX_CHUNKS)
    timing, likelihoods = [], []
    for c in range(ALERAX_CHUNKS):
        chunk = names[c * per:(c + 1) * per]
        fam = write_hogenom_families(paths, chunk, f"hogenom_all_chunk_{c:02d}.families.txt")
        name = f"alerax_chunk_{c:02d}"
        code = run_alerax(paths, folder, name, "released", paths["machine"]["alerax_ranks"],
                          alerax_fit_arguments(fam, hogenom_species_tree(paths), []))
        if code != 0:
            raise SystemExit(f"AleRax failed on chunk {c} (exit {code}); see {folder / (name + '.log')}")
        timing.append(f"{name} {len(chunk)} {wall_seconds(folder / (name + '.time')):.1f}")
        likelihoods += [line for line in open(folder / f"out_{name}" / "per_fam_likelihoods.txt") if line.strip()]
    (folder / "alerax_full_timing.txt").write_text("\n".join(timing) + "\n")
    (folder / "alerax_full_per_fam_likelihoods.txt").write_text("".join(likelihoods))
    total = sum(float(line.split()[2]) for line in timing)
    print(f"AleRax, {len(names)} families in {ALERAX_CHUNKS} chunks: {total:.0f} s in total")


PARTS = {
    "gpurec-full": lambda paths: gpurec_runs(paths, "hogenom_all", ["full_1", "full_2"], "gpurec_plain.py"),
    "gpurec-full-timers": lambda paths: gpurec_runs(paths, "hogenom_all", ["full_timers"], "gpurec_timers.py"),
    "gpurec-1042": lambda paths: gpurec_runs(paths, "hogenom_1042", ["set1042_1", "set1042_2", "set1042_3"], "gpurec_plain.py"),
    "gpurec-506": lambda paths: gpurec_runs(paths, "hogenom_506", ["set506"], "gpurec_plain.py"),
    "alerax-full": alerax_full,
}

if __name__ == "__main__":
    run_part(__doc__, PARTS)

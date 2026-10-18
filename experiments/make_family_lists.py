"""Regenerate the random family lists of data/families from the HOGENOM data, and check them against the stored lists.
  random_batches/batch_NN.txt  all 12,408 HOGENOM families, sorted by name, shuffled once with the seed 11; the families
                               covering fewer than 4 species are dropped (AleRax and gpurec drop them too; the species of
                               a gene is the part of its name before the first "_", read from the first tree of the
                               family's sample); the rest is cut into consecutive batches of 500 families (369 in the
                               last). A batch is a random sample of the whole set.
  small_draws/nN_dDD.txt       20 independent random sets of N = 1, 3, 10 and 30 families drawn from hogenom_all.txt
                               with the seed 20261018 (sets drawn in the order N = 1, 3, 10, 30, then DD = 1 to 20)
  nested_N.txt                 hogenom_all.txt shuffled once with the seed 20261014; nested_N is its first N families,
                               N = 100, 300, 1,000 and 3,000 (the paper's draw of 14 October 2026)
  archaea_200.txt              200 families drawn with the seed 20261015 (random.sample) from the archaeal families that
                               gpurec kept (those covering at least 4 species: the 6,724 families of
                               OUTPUT_ROOT/archaea/out_gpurec/per_fam_likelihoods.txt, written by run_archaea.py part
                               gpurec-fit), sorted by name, and written sorted by name (the paper's draw of 15 October
                               2026); skipped if that fit has not been run
The lists are written under OUTPUT_ROOT/family_lists; the script then reports whether each equals the stored list.
The other lists of data/families were not drawn at random (see README.md, "Family lists").
usage: make_family_lists.py PATHS_TOML"""
import os
import random
import sys
from pathlib import Path

from common import FAMILY_LISTS, HOGENOM_GENE_TREES, output_dir, read_names, read_paths

BATCH = 500
BATCH_SEED = 11
DRAW_SEED = 20261018
NESTED_SEED = 20261014
NESTED_SIZES = (100, 300, 1000, 3000)
ARCHAEA_SEED = 20261015
ARCHAEA_SIZE = 200


def write(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(n + "\n" for n in names))


def random_batches(paths, out):
    root = Path(paths["data"]["hogenom_root"])
    names = sorted(os.listdir(root / "families"))
    random.Random(BATCH_SEED).shuffle(names)
    keep = []
    for f in names:
        leaves = open(root / HOGENOM_GENE_TREES.format(name=f)).readline().replace("(", ",").replace(")", ",").replace(";", "").split(",")
        species = {l.split(":")[0].split("_")[0] for l in leaves if l.strip()}
        if len(species) >= 4:
            keep.append(f)
    print("families with at least 4 species:", len(keep), "of", len(names))
    for i in range(0, len(keep), BATCH):
        write(out / "random_batches" / f"batch_{i // BATCH + 1:02d}.txt", keep[i:i + BATCH])


def small_draws(out):
    names = read_names(FAMILY_LISTS / "hogenom_all.txt")
    rng = random.Random(DRAW_SEED)
    for n in (1, 3, 10, 30):
        for d in range(1, 21):
            pick = rng.sample(range(len(names)), n)
            write(out / "small_draws" / f"n{n}_d{d:02d}.txt", [names[k] for k in pick])


def nested_subsets(out):
    names = read_names(FAMILY_LISTS / "hogenom_all.txt")
    random.Random(NESTED_SEED).shuffle(names)
    for n in NESTED_SIZES:
        write(out / f"nested_{n}.txt", names[:n])


def archaea_200(paths, out):
    """True if the list was drawn, False if the archaeal fit it starts from is missing."""
    fit = output_dir(paths, "archaea") / "out_gpurec" / "per_fam_likelihoods.txt"
    if not fit.exists():
        print(f"archaea_200.txt not drawn: {fit} is missing (run run_archaea.py gpurec-fit first)")
        return False
    kept = sorted(line.split()[0] for line in open(fit) if line.strip())
    write(out / "archaea_200.txt", sorted(random.Random(ARCHAEA_SEED).sample(kept, ARCHAEA_SIZE)))
    return True


if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
    raise SystemExit(__doc__)
paths = read_paths(sys.argv[1])
out = output_dir(paths, "family_lists")
small_draws(out)
random_batches(paths, out)
nested_subsets(out)
single = [f"nested_{n}.txt" for n in NESTED_SIZES] + (["archaea_200.txt"] if archaea_200(paths, out) else [])
for sub in ("small_draws", "random_batches"):
    files = sorted((out / sub).glob("*.txt"))
    same = sum(f.read_text() == (FAMILY_LISTS / sub / f.name).read_text() for f in files)
    print(f"{sub}: {same} of {len(files)} lists equal to the stored ones")
for name in single:
    print(f"{name}: {'equal' if (out / name).read_text() == (FAMILY_LISTS / name).read_text() else 'DIFFERENT'} to the stored list")

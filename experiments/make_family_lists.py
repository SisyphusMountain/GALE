"""Regenerate the random family lists of data/families from the HOGENOM data, and check them against the stored lists.
  random_batches/batch_NN.txt  all 12,408 HOGENOM families, sorted by name, shuffled once with the seed 11; the families
                               covering fewer than 4 species are dropped (AleRax and gpurec drop them too; the species of
                               a gene is the part of its name before the first "_", read from the first tree of the
                               family's sample); the rest is cut into consecutive batches of 500 families (369 in the
                               last). A batch is a random sample of the whole set.
  small_draws/nN_dDD.txt       20 independent random sets of N = 1, 3, 10 and 30 families drawn from hogenom_all.txt
                               with the seed 20261018 (sets drawn in the order N = 1, 3, 10, 30, then DD = 1 to 20)
The lists are written under OUTPUT_ROOT/family_lists; the script then reports whether each equals the stored list.
The other lists of data/families were not drawn by a script kept here (see README.md).
usage: make_family_lists.py PATHS_TOML"""
import os
import random
import sys
from pathlib import Path

from common import FAMILY_LISTS, HOGENOM_GENE_TREES, output_dir, read_names, read_paths

BATCH = 500
BATCH_SEED = 11
DRAW_SEED = 20261018


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


if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
    raise SystemExit(__doc__)
paths = read_paths(sys.argv[1])
out = output_dir(paths, "family_lists")
small_draws(out)
random_batches(paths, out)
for sub in ("small_draws", "random_batches"):
    files = sorted((out / sub).glob("*.txt"))
    same = sum(f.read_text() == (FAMILY_LISTS / sub / f.name).read_text() for f in files)
    print(f"{sub}: {same} of {len(files)} lists equal to the stored ones")

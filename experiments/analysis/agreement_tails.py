"""Tails of the per-family differences of every agreement comparison of the paper: the number of families, the largest
|difference|, and the 99th and 95th percentiles of |difference| (numpy, linear interpolation). The largest difference
of each comparison is also a number of the paper, so it checks that the right files were read.
Reads the outputs of the drivers under OUTPUT_ROOT:
  fixed_rates/gpurec_1042.txt against fixed_rates/out_BUILD/per_fam_likelihoods.txt   (run_fixed_rates.py parts
      gpurec-1042 and alerax-1042): Table tab:agree, BUILD = passes_4 (as released: \\fxStockPNine, \\fxStockPFive),
      addition_16 (\\fxPat*), switch_16 (\\fxFix*), four_16 (\\fxAll*)
  archaea/gpurec_fixed.txt against archaea/out_alerax_four_16_fixed and out_alerax_released_fixed (run_archaea.py part
      fixed-rates-200): \\archFxPNine, \\archFxPFive (four changes)
  precision/precision_all.tsv (run_precision.py part all-families): single against double precision, in bits: \\pPNine,
      \\pPFive
  hogenom/out_full_timers/per_fam_likelihoods.txt against hogenom/alerax_full_per_fam_likelihoods.txt (run_hogenom.py
      parts gpurec-full-timers and alerax-full; the same files as analysis/agreement.py): gpurec minus AleRax at their
      own optima, signed: \\dLowPOne (1st percentile), \\dLowMax (lowest)
  precision/out_double_half_1 and _2 against hogenom/out_full_1 and out_full_2 (run_precision.py part double-fits;
      the same folders as analysis/compare_fp64_fp32.py): \\fpFitPNine, \\fpFitPFive (double against single run 1),
      \\fpRunPNine, \\fpRunPFive (single run 1 against single run 2)
  fixed_rates/out_passes_4_200 against out_passes_64_200 (run_fixed_rates.py part alerax-200-passes): \\aIterPNine,
      \\aIterPFive
  the 200 families of data/families/hogenom_200.txt in gpurec_1042.txt and out_four_16 (main text): \\agMaxAbs (largest
      |difference|), \\agMaxRel (largest |difference| divided by |log-likelihood of AleRax|)
A comparison whose files are missing is reported as such and skipped.
usage: agreement_tails.py OUTPUT_ROOT"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
HOGENOM_200 = Path(__file__).resolve().parents[1] / "data" / "families" / "hogenom_200.txt"


def read(path):
    """Per-family values of a file with one line "family value" (gpurec's and AleRax's per_fam_likelihoods.txt)."""
    return {l.split()[0]: float(l.split()[1]) for l in open(path) if l.strip() and not l.startswith("#")}


def present(*files):
    missing = [str(f) for f in files if not os.path.exists(f)]
    if missing:
        print(f"  (skipped: missing {', '.join(missing)})")
    return not missing


def tails(name, differences, unit):
    a = np.abs(np.asarray(differences))
    print(f"{name}: n={len(a)}  max {a.max():.3e}  p99 {np.percentile(a, 99):.3e}  p95 {np.percentile(a, 95):.3e} {unit}")


def pair(name, first, second):
    """Tails of FIRST minus SECOND, two files with the same families."""
    if not present(first, second):
        return
    x, y = read(first), read(second)
    names = sorted(set(x) & set(y))
    assert len(names) == len(x) == len(y), (first, second, len(x), len(y), len(names))
    tails(name, [x[k] - y[k] for k in names], "nats")


F = ROOT / "fixed_rates"
for label, build in [("Table tab:agree row 1, AleRax as released (\\fxStock)", "passes_4"),
                     ("row 2, 16 passes and addition corrected (\\fxPat)", "addition_16"),
                     ("row 3, and switch to scaled values (\\fxFix)", "switch_16"),
                     ("row 4, all four changes (\\fxAll)", "four_16")]:
    pair(label, F / "gpurec_1042.txt", F / f"out_{build}" / "per_fam_likelihoods.txt")

A = ROOT / "archaea"
pair("archaea, 200 families at fixed rates, four changes (\\archFx)", A / "gpurec_fixed.txt",
     A / "out_alerax_four_16_fixed" / "per_fam_likelihoods.txt")
pair("archaea, 200 families at fixed rates, as released", A / "gpurec_fixed.txt",
     A / "out_alerax_released_fixed" / "per_fam_likelihoods.txt")

P = ROOT / "precision" / "precision_all.tsv"
if present(P):
    rows = [line.rstrip("\n").split("\t") for line in open(P)][1:]
    tails("single against double precision, all families at their fitted rates (\\pPNine, \\pPFive)",
          [float(r[2]) - float(r[3]) for r in rows], "bits")

H = ROOT / "hogenom"
G_FULL, A_FULL = H / "out_full_timers" / "per_fam_likelihoods.txt", H / "alerax_full_per_fam_likelihoods.txt"
if present(G_FULL, A_FULL):
    g, a = read(G_FULL), read(A_FULL)
    names = sorted(set(g) & set(a))
    d = np.array([g[k] - a[k] for k in names])
    print(f"full-set optima, gpurec minus AleRax (\\dLowPOne, \\dLowMax): n={len(d)}  lowest {d.min():+.3f}  "
          f"1st percentile {np.percentile(d, 1):+.3e}  5th percentile {np.percentile(d, 5):+.3e}  "
          f"95th {np.percentile(d, 95):+.3f}  99th {np.percentile(d, 99):+.3f}  highest {d.max():+.3f}  nats")

HALVES = [ROOT / "precision" / f"out_double_half_{k}" / "per_fam_likelihoods.txt" for k in (1, 2)]
SINGLE = [H / f"out_full_{k}" / "per_fam_likelihoods.txt" for k in (1, 2)]
if present(*HALVES, *SINGLE):
    double = {**read(HALVES[0]), **read(HALVES[1])}
    s1, s2 = read(SINGLE[0]), read(SINGLE[1])
    for name, x, y in [("double against single run 1, optima (\\fpFit)", double, s1),
                       ("single run 1 against single run 2, optima (\\fpRun)", s1, s2)]:
        names = sorted(set(x) & set(y))
        tails(name, [x[k] - y[k] for k in names], "nats")

pair("AleRax 4 against 64 passes, 200 families (\\aIter)", F / "out_passes_4_200" / "per_fam_likelihoods.txt",
     F / "out_passes_64_200" / "per_fam_likelihoods.txt")

G_FIX, A_FOUR = F / "gpurec_1042.txt", F / "out_four_16" / "per_fam_likelihoods.txt"
if present(G_FIX, A_FOUR):
    names = [line.strip() for line in open(HOGENOM_200) if line.strip()]
    g, a = read(G_FIX), read(A_FOUR)
    d = np.array([g[k] - a[k] for k in names])
    rel = np.abs(d) / np.abs([a[k] for k in names])
    print(f"main text, the 200 families of hogenom_200, four changes (\\agMaxAbs, \\agMaxRel): "
          f"max |diff| {np.abs(d).max():.3e} nats, max relative {rel.max():.3e}")

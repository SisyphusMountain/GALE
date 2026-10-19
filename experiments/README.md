# Reproducing the experiments of the paper

This folder holds the scripts that produce every measured number, table and figure of the paper on gpurec
(`main2.tex`, with its numbers in `numbers_gpu.tex`, where each number is a LaTeX macro such as `\tFull`). The table
in "Which script produces which result" below maps each result to the command that produces it.

The scripts measure the gpurec of this repository (the package `gpurec/`, run unchanged) and AleRax 1.4.0
(Morel et al. 2024) with up to four changes, given as patches in `data/alerax`. Nothing in `gpurec/` is modified:
when a measurement needs to time or switch off a part of gpurec, a wrapper in `wrappers/` replaces a Python function
at run time, before starting gpurec's own command line.

## Contents

| Path | What it is |
|---|---|
| `paths.toml` | The configuration: every path outside this repository (data sets, AleRax builds, tools, output folder) and the number of MPI processes for AleRax. An example with placeholders; copy it and fill it in. |
| `common.py` | Helpers shared by the drivers: reading `paths.toml`, writing AleRax-format families files, running a command under `/usr/bin/time -v`. |
| `run_*.py` | The drivers, one per family of experiments. Each takes the configuration file and the name of a part: `python experiments/run_X.py PATHS_TOML PART`. `--help` lists the parts. |
| `wrappers/` | The gpurec command line, unchanged (`gpurec_plain.py`) or with hooks: timers that wait for the GPU (`gpurec_timers.py`), ranges for NVIDIA Nsight Systems (`gpurec_nvtx.py`), a trace of the fit with the warm start or the removal of converged families switched off (`gpurec_trace.py`), one design choice switched off (`gpurec_ablate.py`). |
| `measure/` | Measurement scripts that use gpurec as a Python library (likelihood at given rates, precision, gradient check, fixed-point iterations, memory, launch cost, hardware throughput, wave planning, family statistics). The drivers call them; each can also be run alone (its docstring gives its arguments). |
| `analysis/` | Scripts that turn the outputs into the paper's numbers. |
| `figures/` | Scripts that draw the paper's figures from the outputs (they need matplotlib and numpy). |
| `make_family_lists.py` | Regenerates the random family lists of `data/families` and checks them against the stored ones. |
| `data/families/` | The family lists of every run, one family name per line (see "Family lists"). |
| `data/alerax/` | The five patches to AleRax 1.4.0 (see "Building AleRax"). |
| `data/configs/` | gpurec configuration files: single precision (the default, written out), double precision, adjoint pruning off. |

## What is needed

**Machine.** The paper's measurements ran on one machine: an NVIDIA GeForce RTX 4090 (24 GB, 128 multiprocessors,
driver 610.57.04) and an Intel Core i9-13900K (8 performance and 16 efficiency cores, 123 GiB of main memory), Linux.
gpurec ran with Python 3.12, PyTorch 2.13.0 (CUDA 13.0) and Triton 3.7.1; AleRax was compiled with GCC 15.2
(`-O3`) and ran with Open MPI. Two results were measured elsewhere: the double-precision fits of all families on two
NVIDIA L40S cards, and the single-precision fits they are compared with on an NVIDIA H100 (see "Differences from the
paper's runs"). Timings are only comparable on an otherwise idle machine: run one part at a time, with nothing else on
the GPU or the CPU.

**Software.**
- gpurec installed from this repository (`pip install .`, see the main README). The drivers start gpurec with the
  Python that runs them, so run them with the Python of that installation.
- For `analysis/` and `figures/`: numpy and matplotlib (`pip install numpy matplotlib`). The analysis scripts need only
  the Python standard library and numpy; the figure scripts read `paths.toml` with the standard library.
- AleRax 1.4.0 built five times (next section), MPI (`mpiexec`), GNU time (`/usr/bin/time`).
- For the profiles: NVIDIA Nsight Systems (`nsys`), valgrind (its tool callgrind counts the instructions executed in
  each function) and `callgrind_annotate`.
- Optional: `systemd-run`, to cap the memory of AleRax runs (`alerax_memory_cap` in `paths.toml`).

**Data sets.** None is copied here; `paths.toml` points to them.
1. **HOGENOM**, from the data archive of the AleRax paper (Morel, Williams, Stamatakis, Szöllősi, *AleRax: a tool for
   gene and species tree co-estimation and reconciliation under a probabilistic model of gene duplication, transfer,
   and loss*, Bioinformatics 40(4), btae162, 2024, doi:10.1093/bioinformatics/btae162): the archive
   `https://cme.h-its.org/exelixis/material/alerax_data.tar.gz` (about 16 GB). Only its folder `hogenom` is used:
   `hogenom/families/FAMILY/gene_trees/ufboot1000.MFP.geneTree.newick` (12,408 families, each a sample of 1,000
   ultrafast-bootstrap gene trees; 7.2 GB) and the species tree
   `hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick`
   (666 species, 1,331 branches). Extract these files and set `hogenom_root` to the folder `hogenom`.
2. **Archaea**, the deposit of Huang et al., *Phylogenetic reconciliation supports a methanogenic ancestor of the
   Archaea and a derived origin for host-associated lineages*, Zenodo 2026, doi:10.5281/zenodo.18701544
   (`https://zenodo.org/records/18701544`): the file `3_Reconciliation.zip` (4.3 GB). Extract it and set
   `archaea_root` to its folder `3_Reconciliation/This_study`. The
   scripts use the families of `6_qmds/alerax_families.txt` (7,059 families; their names are copied in
   `data/families/archaea_all.txt`, and since that file's gene-tree paths point to the authors' machine, the scripts
   write them as `3_UFBOOTs/ufboot_for_alerax/FAMILY.ufboot`), the species tree
   `4_species_tree/others/Undine_C60_MHHroot_short_name.nw` (257 species) and the missing fraction of each genome,
   `5_reconcilation models/fraction_missing`.

## Building AleRax

The patches apply in order to AleRax 1.4.0 (git tag `v1.4.0`, commit f38d4ef, with its submodule `ext/GeneRaxCore` at
commit d0c3da88). Each build adds one patch to the previous one:

| Patch | Change |
|---|---|
| `01-print-likelihoods-with-17-digits.patch` | `per_fam_likelihoods.txt` printed with 17 significant digits instead of 6, so that the comparisons are not limited by printing. Changes no computation. |
| `02-fixed-point-iterations-option.patch` | A command-line option `--fixed-point-iterations N` (default 4) replaces the number of fixed-point passes, written as 4 in three places of the code. |
| `03-scaled-addition-keeps-smaller-term.patch` | AleRax stores a probability too small for a double as a double and a count of factors 2^-256 (a "scaled value"). The sum of two scaled values with different counts returned the larger and dropped the smaller; it now adds both. |
| `04-switch-to-scaled-values-below-smallest-normal-double.patch` | AleRax recomputes a family with scaled values only when its log-likelihood is not a normal number. A likelihood between 4.9e-324 and 2.2e-308 (log-likelihood between -744.44 and -708.40) passed that test with few correct digits; it now switches whenever the log-likelihood is below -708.40. |
| `05-scaled-subtraction-keeps-smaller-term.patch` | The difference of two scaled values with different counts returned the first and ignored the second; it now subtracts both. |

| Build (key in `paths.toml`) | Patches | Used for |
|---|---|---|
| `released` | 01 | everything called "AleRax 1.4.0 as released" that fits rates: the full-set timing, the archaeal fit, the optimum comparison, the profiles |
| `passes` | 01, 02 | the scaling timings and the fixed-rate runs called "as released" (with `--fixed-point-iterations 4`, it computes what the release computes) |
| `addition` | 01-03 | fixed rates, 16 passes, "addition corrected" |
| `switch` | 01-04 | fixed rates, 16 passes, "and switch to scaled values" |
| `four` | 01-05 | fixed rates and fits "with the four changes" |

For each build (here `four`):

```bash
git clone --recursive https://github.com/BenoitMorel/AleRax alerax_four
cd alerax_four
git checkout v1.4.0 && git submodule update --init --recursive
for p in 01 02 03 04 05; do patch -p1 < /PATH/TO/experiments/data/alerax/$p-*.patch; done
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 8
# the program is build/bin/alerax
```

For `released` apply 01 only, for `passes` 01 and 02, for `addition` 01 to 03, for `switch` 01 to 04.
Built this way, the four builds `passes` (4 passes), `addition`, `switch` and `four` (16 passes) give, at the rates
(0.2, 0.3, 0.1), the same 17 digits as the paper's runs for the three families CLU_003291_1_2_C, CLU_005391_0_2_C and
CLU_000445_30_3_C, and the callgrind profile of `released` counts the same calls as the paper's.

## Configuration

Copy `paths.toml`, replace every value in capitals, and pass the copy's path as the first argument of every driver and
figure script. There are no environment variables and no default values: every path is in this file. Every output
goes under `[output] root` (written OUTPUT_ROOT below), one folder per driver: `hogenom`, `scaling`, `ablations`,
`warm_rebatch`, `precision`, `fixed_rates`, `optima`, `archaea`, `measurements`, `profiles`, `planning`, `cluster`,
plus `families` (the families files the drivers write from `data/families`).

A gpurec run named NAME writes, in its driver's folder: `NAME.time` (output of `/usr/bin/time -v`: wall time of the
whole command, peak memory), `NAME.log` and `NAME.err` (gpurec's screen output; the time gpurec reports, "done in",
and the total log-likelihood are in `NAME.err`) and the folder `out_NAME` (gpurec's output, AleRax's format:
`per_fam_likelihoods.txt`, `model_parameters/FAMILY_rates.txt`, `gale.log`; runs made before the command was renamed from `gpurec` to `gale` wrote `gpurec.log`). AleRax runs write the same files.
`progress.txt` in each folder records the start and end of every run with the load of the machine.
`analysis/run_summary.py FOLDER 'PATTERN'` prints, for each run, the wall time, "done in", the number of families
kept, the peak GPU memory and the total log-likelihood.

Below, `E=experiments` and `P=my_paths.toml`; commands are run from the repository's root.

## Which script produces which result

Times in the last column are those of the paper's runs (RTX 4090 and i9-13900K) unless marked "about".

### Section "Example: rates for 10,869 HOGENOM families" and appendix "Details of the experiments"

| Result (macros of `numbers_gpu.tex`) | Command | Output and analysis | Time |
|---|---|---|---|
| gpurec on all families, two runs: `\tFull`, `\tFullMin`, `\tFullRunA`, `\tFullRunB`, `\dRuns`; data set size `\nFamilies`, `\nClades`, `\nBranches` (lines "Remaining families" and "clades x species" of the log) | `python $E/run_hogenom.py $P gpurec-full` | `hogenom/full_1.err`, `full_2.err`; `python $E/analysis/run_summary.py OUTPUT_ROOT/hogenom 'full_?'` | 2 x 12.5 min |
| AleRax on all families (eight chunks): `\tAleRax`, `\tAleRaxHours`; with the above, `\speedup` | `python $E/run_hogenom.py $P alerax-full` | `hogenom/alerax_full_timing.txt` (seconds per chunk; the driver prints the sum), `alerax_full_per_fam_likelihoods.txt` | 4.9 h |
| Timed run, where the time goes: `\tInstr`, `\tGrad`, `\nGradCalls`, `\tHess`, `\nHessCalls`, `\tBuild`, `\nBuilds` | `python $E/run_hogenom.py $P gpurec-full-timers` | the table at the end of `hogenom/full_timers.log`; "done in" of `full_timers.err` | 14.5 min |
| Figure `fig_times_gpu` (Figure 2): (a) the full-set times, (b) the shares of the profiled run | `python $E/figures/make_fig_times.py $P fig_times_gpu.pdf` | needs `gpurec-full`, `alerax-full` and `run_profiles.py gpurec-nsys` (panel b runs `analysis/split_gradient_ranges.py` on the profile) | about 15 s |
| Agreement at the optimum, full set: `\dMedian`, `\dMean`, `\dTotal`, `\nWorse`, `\nBetter`, `\pearson` | (the two parts above) | `python $E/analysis/agreement.py OUTPUT_ROOT/hogenom/out_full_timers/per_fam_likelihoods.txt OUTPUT_ROOT/hogenom/alerax_full_per_fam_likelihoods.txt` | seconds |
| Nsight Systems profile: `\prTRun`, `\prHess`, `\prGrad`, `\prBuild`, `\prBusy`, `\prKOne`, `\prKTwo`, `\prKThree` | `python $E/run_profiles.py $P gpurec-nsys` | `python $E/analysis/nsys_shares.py OUTPUT_ROOT/profiles/full_nvtx_gpu_proj_sum.csv OUTPUT_ROOT/profiles/full_cuda_gpu_kern_sum.csv OUTPUT_ROOT/profiles/full.err` | 12 min, then several minutes for `nsys export` and `nsys stats` |
| Where the time of the profiled run goes (Figure 2b): `\prTLik` (forward pass of the loss-and-gradient evaluations), `\prTBack` (backward pass), `\prTHess`, `\prTBuild`, `\prTTotal`, `\prLik`, `\prBack`, `\prOther` | (the part above) | `python $E/analysis/split_gradient_ranges.py OUTPUT_ROOT/profiles/full.sqlite OUTPUT_ROOT/profiles/full.err`: seconds and share of each part. On the paper's profile it prints forward 135.6 s, backward 161.8 s, hessian 361.3 s, model build 37.5 s, other 34.3 s (4.7 %), total 730.5 s | about 15 s |
| The 506- and 1,042-family sets: `\tFiveHundred`, `\nFiveHundred`, `\nOneK`, and the nine runs `\tOneKRebatchMin`, `\tOneKRebatchMax`, `\tOneKRebatchMed` | `python $E/run_hogenom.py $P gpurec-506`, `... gpurec-1042`; the nine runs are `set1042_1..3` here, `warm_on_1..3` and `removal_on_1..3` of `run_warm_rebatch.py` | `run_summary.py` on `hogenom` and `warm_rebatch` | 3.2 min; 3 x 3.5 min |

### Agreement with AleRax at fixed rates (section "Example", appendix "Numerical accuracy")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| gpurec at (0.2, 0.3, 0.1) on 1,042 families, converged settings | `python $E/run_fixed_rates.py $P gpurec-1042` | `fixed_rates/gpurec_1042.txt` | about 1 min |
| AleRax at (0.2, 0.3, 0.1), four builds; `\tAleFixStock` (`passes_4`), `\tAleFixConv` (`addition_16`) | `python $E/run_fixed_rates.py $P alerax-1042` | `fixed_rates/out_{passes_4,addition_16,switch_16,four_16}/per_fam_likelihoods.txt`; times: `run_summary.py OUTPUT_ROOT/fixed_rates '*_16' passes_4` | 3.2 + 3 x 13.7 min |
| Table `tab:agree` rows 1, 2, 4, 5: `\fxN`, `\fxStock*`, `\fxPat*`, `\fxFix*` (including `\fxFixPlainN`, `\fxFixPlainMax`, `\fxFixScaledN`, `\fxFixScaledLow`: the lines by range of log-likelihood), `\fxAll*`, `\fxBandN` | (the two parts above) | `python $E/analysis/compare_fixedrate.py OUTPUT_ROOT/fixed_rates/gpurec_1042.txt OUTPUT_ROOT/fixed_rates/out_BUILD/per_fam_likelihoods.txt` for BUILD = `passes_4`, `addition_16`, `switch_16`, `four_16` | seconds |
| 4 against 64 passes of AleRax, 200 families: `\aIterMax`, `\aIterMed`, `\aIterN`; 16 passes equal 64 to 1e-12 | `python $E/run_fixed_rates.py $P alerax-200-passes` | `compare_fixedrate.py OUTPUT_ROOT/fixed_rates/out_passes_64_200/per_fam_likelihoods.txt OUTPUT_ROOT/fixed_rates/out_passes_4_200/per_fam_likelihoods.txt` (the 64-pass file takes the place of gpurec's; same with `passes_16_200`) | about 15 min |
| Figure `fig_alerax_fixes` | `python $E/figures/make_fig_alerax_fixes.py $P fig_alerax_fixes.pdf` | needs `gpurec-1042` and `alerax-1042` | seconds |
| gpurec's default command with `--fix-rates`: `\tGpuFix` ("done in"), `\tGpuFixCmd` (wall time) | `python $E/run_fixed_rates.py $P gpurec-cli-1042` | `run_summary.py OUTPUT_ROOT/fixed_rates cli_1042` | 17 s |
| Tails of the agreement comparisons (99th and 95th percentiles of the per-family differences): `\fxStockPNine`, `\fxStockPFive`, `\fxPatPNine`, `\fxPatPFive`, `\fxFixPNine`, `\fxFixPFive`, `\fxAllPNine`, `\fxAllPFive` (Table `tab:agree`), `\archFxPNine`, `\archFxPFive`, `\pPNine`, `\pPFive`, `\fpFitPNine`, `\fpFitPFive`, `\fpRunPNine`, `\fpRunPFive`, `\aIterPNine`, `\aIterPFive`, `\dLowPOne`, `\dLowMax` (full-set optima, gpurec minus AleRax: 1st percentile and lowest), `\agMaxAbs`, `\agMaxRel` (main text: the 200 families of `hogenom_200.txt` at fixed rates against the four changes) | the parts named in the docstring of the script: `run_fixed_rates.py` `gpurec-1042`, `alerax-1042`, `alerax-200-passes`; `run_archaea.py fixed-rates-200`; `run_precision.py all-families`, `double-fits`; `run_hogenom.py gpurec-full`, `gpurec-full-timers`, `alerax-full` | `python $E/analysis/agreement_tails.py OUTPUT_ROOT` (skips a comparison whose files are missing) | seconds |
| The five reference cases of the package (single precision within 1.5e-5 nats of AleRax, double within 1e-12) | the package's own test, `pytest tests/test_fidelity_alerax.py` | the test checks bounds of 1e-3 nats (single) and 1e-8 (double); the paper's values were read from a run of these cases | seconds |

### Agreement of the optima, 200 random families (appendix "Details of the experiments")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| Fit times: `\oN`, `\oTG` (gpurec, "done in"), `\oTOld` (AleRax as released), `\oTNewFour`, `\oTNewSixteen` (four changes, 4 and 16 passes) | `python $E/run_optima.py $P gpurec-fit`, then `... alerax-fits` | `run_summary.py OUTPUT_ROOT/optima '*'` | 20 s; 8 + 12 + 64 min |
| Each fit scored by gpurec in double precision | `python $E/run_optima.py $P score` | `optima/scores_NAME.txt` | minutes |
| `\oOldTot`, `\oOldBetter`, `\oOldMax` (against AleRax as released); `\oNMilli`, `\oNLow`, `\oLowMax`, `\oTwoMax`, `\oNewTot` (against the four changes, 16 passes) | (the parts above) | `python $E/analysis/compare_optima.py OUTPUT_ROOT/optima/scores_gpurec.txt OUTPUT_ROOT/optima/scores_alerax_released.txt OUTPUT_ROOT/optima/rates_alerax_released.tsv`, and the same with `alerax_four_16` (and `alerax_four_4` for the remark on 4 passes). `\oNLow` is the number of families lower by more than 0.001 without the one with different maxima | seconds |

### Scaling and ablations (section "Example", paragraphs "Scaling" and "Ablations")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| Nested subsets of 100 to 3,000 families: `\scGOne` to `\scGFour` (gpurec), `\scAOne` to `\scAFour` (AleRax), `\scRatioMin`, `\scRatioMax` | `python $E/run_scaling.py $P gpurec-nested-large`, `... alerax-nested` | `scaling/g_nestedN_K.time`, `a_nestedN.time`; numbers printed by the figure script | 10 min; 1.8 h |
| 22 random batches (appendix): `\scBatchMed`, `\scBatchMin`, `\scBatchMax` | `python $E/run_scaling.py $P gpurec-batches`, `... alerax-batches` | `python $E/analysis/batch_ratios.py OUTPUT_ROOT/scaling` | 15 min; about 5 h |
| Figure `fig_scaling` | `python $E/figures/make_fig_scaling.py $P fig_scaling.pdf` | needs `small-draws`, `gpurec-nested-large`, `alerax-nested` and `run_hogenom.py` parts `gpurec-full` and `alerax-full`; prints the numbers of the paragraph "Scaling" | seconds |
| 20 random sets each of 1, 3, 10 and 30 families: `\drawROne`, `\drawFastOne`, `\drawRThree`, `\drawFastThree`, `\drawRTen`, `\drawRThirty`, and the ranges of times quoted in the text | `python $E/run_scaling.py $P small-draws` | `python $E/analysis/small_draws.py OUTPUT_ROOT/scaling/small_draws` (also printed by the figure script) | about 1.5 h |
| Table `tab:ablations`, all rows from one session: `\ablBase`, `\ablAlone`, `\ablAloneX`, `\ablNoEvict`, `\ablNoEvictX`, `\ablNoEvictLoss`, `\ablNoWarm`, `\ablNoWarmX`, `\ablBySize`, `\ablBySizeX`, `\ablInput`, `\ablInputX`, `\ablStoreAlpha`, `\ablStoreAlphaX`, `\ablNoPrune`, `\ablNoPruneX` (adjoint pruning truly off: `data/configs/no_pruning.toml` sets `use_adjoint_pruning = false` and `adjoint_pruning_threshold = 0.0`), and the peak memories | `python $E/run_ablations.py $P session`: three rounds over base, alone, noevict, nowarm, bysize, inputorder, storealpha, nopruning, round 0 untimed, round 1 timed, round 2 timed only for the configurations near the release settings (base, bysize, inputorder, storealpha, nopruning); before and after each run `progress.txt` records the load of the machine and the other processes on the GPU | `python $E/analysis/ablations.py session OUTPUT_ROOT/ablations/session`: per configuration the mean of its timed runs (a run with another process on the GPU is left out, and a rerun named NAME_3 made alone after the session takes its place), the ratio to base, the peak GPU memory and total log-likelihood of each timed run, and a warning for any run with another process on the GPU (not to be used) | about 1 h 50 min (rounds 0 and 1 take about 46 min each, round 2 about 6 min) |
| How many terms of the adjoint series the gradient needs: error of the gradient with 2 to 64 terms against 256, at the starting and the fitted rates (`\adjStartEight`, `\adjFitSixteenN`, `\adjFitThirtyTwoN`, `\adjTimeSixteen`, `\adjTimeSixtyFour`); `\prAdjShare` comes from the profile (`run_profiles.py`, the kernel `_apply_reconciliation_self_loop_transpose_kernel`) | `python $E/run_ablations.py $P adjoint-terms` (after `base`) | `ablations/adjoint_terms.log` | about 3 min |
| The same configurations, run one at a time (the earlier protocol): one untimed and two timed runs each; `base-again` is the release settings once more | `for m in base alone noevict nowarm bysize inputorder base-again nopruning storealpha; do python $E/run_ablations.py $P $m; done` | `python $E/analysis/ablations.py separate OUTPUT_ROOT/ablations` | 3 runs per configuration: 3.5, 80, 35, 6, 4, 4, 3.5, 4.5, 3.5 min |
| What pruning skips: clades skipped, time and gradient with pruning on, with `use_adjoint_pruning = false` alone (which skips the same clades) and truly off (`\pruneSkipped`, `\pruneGradRel`) | `python $E/run_ablations.py $P pruning-check` (after `base`) | `ablations/pruning_check.log` | about 1 min |
| Gradient with pruning off and with the stored split term (relative differences quoted in the comments of `numbers_gpu.tex`); size of the store and peak memory: `\alphaStoreMiB`, `\alphaPeakRecomp`, `\alphaPeakStore` | `python $E/run_ablations.py $P grad-check`, `... storealpha-check` | `ablations/grad_check.log`, `ablations/storealpha_check.log` | about 1 min each |

### Waves and family sizes (sections "Waves of clades", "Numbers"; appendix "Scheduling", "Where the time of AleRax goes")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| Table `tab:waves` | `python $E/run_planning.py $P waves-table` | `planning/waves_full.json` | a few minutes (CPU only) |
| Appendix table and figure `hogenom_waves` (all 10,869 families, release batches against each family alone) | `python $E/run_planning.py $P waves-figure`, then `python $E/figures/make_fig_waves.py OUTPUT_ROOT/planning/waves_figure hogenom_waves.pdf` | `planning/waves_figure.log` (the table and the numbers of the text), CSV files in `planning/waves_figure` | about 1 min (CPU only) |
| `\pBitsMed`, `\pBitsMax`, `\nUnderSingle`, `\nUnderDouble`, `\aFamClades`, `\aFamSplits`, `\aScaledSplits`, `\scSetMed` | `python $E/run_planning.py $P family-stats` (after `run_hogenom.py gpurec-full` and `run_precision.py all-families`) | `planning/family_stats.log` | a few minutes (CPU only) |

### AleRax profile (appendix, paragraph "Where the time of AleRax goes")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| `\aFamTime` (one process); the shares of instructions per function (inclusive) of an earlier draft (`\prAOpt`, `\prACLV`, `\prACand`, `\prAAnc`) | `python $E/run_profiles.py $P alerax-callgrind` | wall time in `profiles/alerax_plain.time`; shares in `profiles/alerax_callgrind_inclusive.txt` | 3 s; about 5 min |
| `\csEvals`: likelihood evaluations of the fit | (same) | `python $E/analysis/callgrind_calls.py OUTPUT_ROOT/profiles/alerax.callgrind.out updateCLV recomputeSpeciesProbabilities`: calls of `updateCLV` divided by the clades of the family (`\aFamClades`) | seconds |
| `\aScaledRatio`: instructions with scaled values against plain doubles | `python $E/run_profiles.py $P alerax-scaled-vs-double` | `callgrind_calls.py OUTPUT_ROOT/profiles/fix_scaled_family.callgrind.out updateCLV`: instructions of `UndatedDTLMultiModel<ScaledValue>::updateCLV` divided by those of `UndatedDTLMultiModel<double>::updateCLV` in the same file | about 10 min |

### Precision (sections "Numbers" and appendix "Representation of the numbers")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| Table `tab:precision` (and the factor 29 of double precision) | `python $E/run_precision.py $P table` | `precision/precision_test.log` | about 5 min |
| All families at their fitted rates: `\pNFam`, `\pMax`, `\pMed`, `\pNAbove`, `\pTot`, `\pTimeSingle`, `\pTimeDouble`, `\pRatio`; Figure S1 of the supplement | `python $E/run_precision.py $P all-families` (after `run_hogenom.py gpurec-full`); `python $E/figures/make_fig_precision.py OUTPUT_ROOT/precision/precision_all.tsv fig_precision.pdf` | `precision/precision_all.log`, `precision_all.tsv` | about 15 min |
| Double- against single-precision fits: `\fpFitMed`, `\fpRunMed`, `\fpFitN`, `\fpFitMax`, `\fpFitTotA`, `\fpFitTotB` | `python $E/run_precision.py $P double-fits` | `python $E/analysis/compare_fp64_fp32.py OUTPUT_ROOT/hogenom/out_full_1 OUTPUT_ROOT/hogenom/out_full_2 OUTPUT_ROOT/precision/out_double_half_1 OUTPUT_ROOT/precision/out_double_half_2` | not measured on the RTX 4090 (double precision is 29 to 53 times slower there); hours |
| The same on a SLURM cluster, as the paper ran them (double precision on L40S cards, single precision on an H100) | `python $E/run_cluster.py $P double-half-1`, `... double-half-2` (at the same time, from two sessions), `... single-full-1`, `... single-full-2` | `python $E/analysis/compare_fp64_fp32.py OUTPUT_ROOT/cluster/out_single_full_1 OUTPUT_ROOT/cluster/out_single_full_2 OUTPUT_ROOT/cluster/out_double_half_1 OUTPUT_ROOT/cluster/out_double_half_2` (`analysis/agreement_tails.py` reads the local runs of `run_precision.py double-fits` and `run_hogenom.py gpurec-full`; for the cluster runs, copy or link them to those names) | in the paper: 1 h 49 min and 1 h 34 min for the halves (L40S), 25 and 18 min for the single-precision fits (H100) |

### Solver measurements and hardware

| Result | Command | Output | Time |
|---|---|---|---|
| Gradient cost and check: `\tLossOnly`, `\tLossGrad`, `\fdRatioLow`, `\fdRatioHigh` (computed from these two times: 4 or 6 loss evaluations against one loss and gradient), `\gradFD` | `python $E/run_measurements.py $P grad-test` | `measurements/grad_test.log` | about 3 min |
| Where the difference of `\gradFD` comes from: the gradient against finite differences (20 families, double precision, step 1e-2, converged forward solves) with adjoint pruning on and truly off (`use_adjoint_pruning = false` and threshold 0): `\gradFDNoPruneA` (rates 0.2, 0.3, 0.1), `\gradFDNoPruneB` (rates 0.05, 0.05, 0.05); with pruning on, about 1e-4 | `python $E/run_measurements.py $P grad-resid` | `measurements/grad_resid.log`: the paper's run printed 1.084e-04 and 1.300e-05 at (0.2, 0.3, 0.1), 8.327e-05 and 2.584e-06 at (0.05, 0.05, 0.05), each the same with 64 or 512 iterations and terms | about 6 min |
| Fixed-point iterations: `\kSixteenMax`, `\kSixteenMed`, `\kSixteenN`, `\kThirtyTwoMax` | `python $E/run_measurements.py $P kscan` (after `run_hogenom.py gpurec-full`) | `measurements/kscan.log` | about 1 min |
| Cost of one launch: `\tLaunch` (Figure 1) | `python $E/run_measurements.py $P launch-cost` | `measurements/launch_cost.log` | seconds |
| Peak memory against the clade limit of a batch (`\memEightyK`) | `python $E/run_measurements.py $P mem-scan` | `measurements/mem_scan.log` | about 1 min |
| Table `tab:hardware` (throughput rows) | `python $E/run_measurements.py $P hardware` | `measurements/hardware_1.json`, `hardware_2.json` | a few minutes |

### Warm start of the adjoint and adaptive rebatching (appendix "Details of the experiments")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| `\tWarmOn`, `\tWarmOnMin`, `\tWarmOnMax`, `\tWarmOff`, `\tWarmOffMin`, `\tWarmOffMax`, `\warmGain`, `\memWarmOff`, `\memWarmOn` (and `\warmGainFirst`, an earlier repetition of the same protocol) | `python $E/run_warm_rebatch.py $P warm-1042` | `python $E/analysis/traces.py OUTPUT_ROOT/warm_rebatch/trace_warm_*.json` | 6 x 3.5 to 4.5 min |
| `\tFullWarmOffA`, `\tFullWarmOffB`, `\fullWarmFactor`, `\memFullWarmOff`, `\memFullWarmOn`; the refusal of the warm start at the first build (18.1 GiB) is in `full_warm_on_1.err` | `python $E/run_warm_rebatch.py $P warm-full` | `traces.py OUTPUT_ROOT/warm_rebatch/trace_full_*.json` | 23 + 12.5 + 22 min |
| `\tOneKNoRebatch`, `\tOneKNoRebatchB`, `\rebatchFactorLow`, `\rebatchFactorHigh`, `\tRebatchFirst` (time of the second build) | `python $E/run_warm_rebatch.py $P rebatch-1042` | `traces.py OUTPUT_ROOT/warm_rebatch/trace_removal_*.json` | 36 + 3 x 3.5 min |
| Figure `fig_rebatch` | `python $E/figures/make_fig_rebatch.py OUTPUT_ROOT/warm_rebatch/trace_removal_on_1.json OUTPUT_ROOT/warm_rebatch/trace_removal_off.json fig_rebatch.pdf` | (the paper drew a run with removal of 231 s) | seconds |

### A second data set: archaea (section "Example" and appendix "Details of the experiments")

| Result | Command | Output and analysis | Time |
|---|---|---|---|
| `\archTG` (gpurec, "done in"), `\archTA` (AleRax, wall time), `\archSpeedup`; `\archN` (families of the deposit's list), `\archNkept` (line "Remaining families" of gpurec's log), `\archSpecies` | `python $E/run_archaea.py $P gpurec-fit`, `... alerax-fit` | `run_summary.py OUTPUT_ROOT/archaea gpurec alerax` | 12 min; 2.2 h |
| `\archGain`, `\archStuckN`, `\archStuckGain`, `\archAleN`, `\archAleMax` | `python $E/run_archaea.py $P score` | `python $E/analysis/compare_optima.py OUTPUT_ROOT/archaea/scores_gpurec.txt OUTPUT_ROOT/archaea/scores_alerax.txt OUTPUT_ROOT/archaea/rates_alerax.tsv` | minutes |
| `\archFxMed`, `\archFxMax` (gpurec against the four changes), `\archFxStock` (against AleRax as released) | `python $E/run_archaea.py $P fixed-rates-200` | `compare_fixedrate.py OUTPUT_ROOT/archaea/gpurec_fixed.txt OUTPUT_ROOT/archaea/out_alerax_four_16_fixed/per_fam_likelihoods.txt`, and the same with `out_alerax_released_fixed` | minutes |
| "differences of the same size without the missing fractions" | `python $E/run_archaea.py $P outliers-no-fm` | `compare_fixedrate.py OUTPUT_ROOT/archaea/gpurec_nofm.txt OUTPUT_ROOT/archaea/out_alerax_four_16_nofm/per_fam_likelihoods.txt` | minutes |

## Dependencies between the parts

- `run_hogenom.py gpurec-full` first: its fitted rates (`hogenom/out_full_1/model_parameters`) are used by
  `run_precision.py all-families` and `run_measurements.py kscan`, its likelihoods by `run_planning.py family-stats`.
- `run_planning.py family-stats` needs `run_precision.py all-families`.
- The `score` parts need the fits of the same driver.
- Figures need the parts named in their rows.
- Several kernels of gpurec take the sizes of waves and batches as compile-time constants, so that a run meeting sizes
  not seen before also compiles kernels (Triton keeps them in its cache). The paper's times are those of runs with a
  warm cache: the small subsets and the ablations are preceded by an untimed run for that reason, and the other
  parts were run after earlier runs on the same data had filled the cache.

## Family lists

`data/families` holds the list of families of every run, one name per line; the drivers turn a list into an AleRax
families file under `OUTPUT_ROOT/families` (`- FAMILY` then `starting_gene_tree = PATH`, or `gene_tree = PATH` for the
archaea).

| List | Families | What it is |
|---|---|---|
| `hogenom_all.txt` | 10,869 | the HOGENOM families covering at least 4 species (out of 12,408; the species of a gene is the part of its name before the first "_"), in the order of the families file of the paper's runs |
| `hogenom_1042.txt` | 1,042 | the families of the 1,055-family HOGENOM folder (sorted by name), keeping those that cover at least 4 species |
| `hogenom_506.txt` | 506 | the first 506 of `hogenom_1042.txt` (the first 512 of the 1,055 names, then the same filter) |
| `hogenom_200.txt` | 200 | the first 200 of `hogenom_1042.txt` |
| `nested_N.txt` | 100 to 3,000 | random nested subsets: `hogenom_all.txt` shuffled once (seed 20261014), each the beginning of the next one (`make_family_lists.py`) |
| `random_batches/batch_NN.txt` | 500 (369) | 22 random batches that partition `hogenom_all.txt` (`make_family_lists.py`) |
| `small_draws/nN_dDD.txt` | 1 to 30 | 20 independent random sets per size (`make_family_lists.py`) |
| `opt200.txt` | 200 | the first 200 families of random batch 1 |
| `profile_median_family.txt`, `profile_scaled_family.txt` | 1 each | the two families of the AleRax profiles |
| `archaea_all.txt` | 7,059 | the archaeal families of the deposit's families file, in its order |
| `archaea_200.txt` | 200 | archaeal families taken at random (seed 20261015) among the 6,724 that gpurec keeps, sorted by name (`make_family_lists.py`, after `run_archaea.py gpurec-fit`) |
| `archaea_outliers16.txt` | 16 | the families of `archaea_200.txt` where gpurec and AleRax with the four changes differ by more than 0.001 nats |

`python $E/make_family_lists.py $P` redraws the random batches, the small draws, the nested subsets and (if the
archaeal fit has been run) `archaea_200.txt`, and reports whether they equal the stored lists. On the paper's data it
reports all 22 batches, all 80 small draws, the four nested subsets and `archaea_200.txt` equal to the stored lists.

## Differences from the paper's runs

- **gpurec version.** The paper's runs used the branch `release/0.1-memory` of this repository, at commit d48c48f7
  (full-set timings and the runs of 14-15 October), ecb74b76 (archaea, Nsight profile) and 386d8858 (runs of 16-18
  October), through wrappers that inserted that source tree on Python's path. This folder is on `release/0.1`, which
  continues 386d8858 with commits that only remove code that the command line does not run (custom optimisation loops,
  unused configuration loaders and helpers, the Gauss-Newton product, penalty functions, an old planner) and
  documentation.
  The configuration file `configs/fit_24gb_card.toml` used by some runs of 14 October (80,000 clades per batch, Hessian
  warm start off) holds the values that became the defaults at d48c48f7, so the plain command reproduces it.
- **Run order.** The table of ablations comes from one session that interleaves all the configurations (part
  `session`, the protocol of the paper's script `results_gpu/scaling_ablations/queue5.sh` of 19 October 2026), so that
  every row is measured in the same state of the machine. The parts that run one configuration at a time are the
  earlier protocol: there each configuration runs once untimed and then twice timed, one after the other, whereas the
  paper's earlier queues ran, for the first six configurations, two runs to fill the compile cache and then two timed
  runs, the four separated by other runs, and the "release settings again" (`\ablBaseB`) as two timed runs without an
  untimed one, the cache being already warm. The runs of the 1,042-family
  set were spread over several queues (`set1042`, `warm_on`, `removal_on`); the earlier run without removal of
  converged families (`\tOneKNoRebatch`, `run_norebatch.py`) set the same option as `removal_off` here (`drop_frac`
  above 1).
- **Adjoint pruning off.** The paper's first run without pruning (18 October) set only `use_adjoint_pruning = false`,
  which in release 0.1 skips the same clades as pruning on (`pruning-check` measures it). `data/configs/no_pruning.toml`
  also sets `adjoint_pruning_threshold = 0.0`, as the paper's later runs did, so that the configuration `nopruning` of
  the session (`\ablNoPrune`) and `grad-resid` (`\gradFDNoPruneA`, `\gradFDNoPruneB`) compute every clade. Only the
  relative gradient difference "pruning off" of `grad-check` quoted in the comments of `numbers_gpu.tex` (1.9e-7, 100
  families) came from the earlier setting; `grad-check` here turns pruning truly off, so that number is to be measured
  again.
- **The double-precision fits** ran on two NVIDIA L40S cards of a computing cluster, one half of the families each, and
  were compared with two single-precision fits made on an NVIDIA H100 of the same cluster, with the code of d48c48f7
  (plus, for double precision, the fix of commit bd16458d; both are in this branch). `run_cluster.py` runs the same
  jobs (same srun options, kept in `[cluster]` of `paths.toml`); the paper's jobs were started by hand-written shell
  scripts on the cluster, which held its paths. The halves are the first 5,434 families of `hogenom_all.txt` and the
  other 5,435, the same families as the paper's halves (checked against the families of their outputs).
  `run_precision.py double-fits` runs the same halves on the local card instead, and is compared with `full_1` and
  `full_2` of the RTX 4090.
- **AleRax builds.** The paper's AleRax runs used two copies of AleRax 1.4.0 that implement the same changes in two
  ways: changes 02 to 04 were first made with the number of passes in a global variable and an addition that stops
  bringing the smaller term into the bucket of the larger one after four factors of 2^-256 (beyond which the term
  rounds to zero anyway); the patches here are the later copy (number of passes in AleRax's model settings), which
  produced the runs with all four changes. The full-set AleRax timing used the AleRax 1.4.0 installed on the machine,
  without patch 01, which only changes printing.
- **Figures.** `make_fig_times.py` and `make_fig_scaling.py` read the run outputs instead of `numbers_gpu.tex`.

## Not reproduced or not verified

- The address of the 1,055-family HOGENOM archive (`hogenom.tar.gz`, folder with `hogenom_S.tree`, `hogenom_trees/`,
  `hogenom_mappings/`) was not recorded. What is known: the archive is 45,541,138 bytes (MD5 checksum
  24bcc160951d5eb5160d38467c621255), its copy in the download folder of the paper's machine is dated 8 October 2025,
  it has no file naming its source, and its 1,055 families are all among the 12,408 HOGENOM families of the AleRax data archive.
- The paper's values for the five reference cases of the package were read from a run of them, not written to a file.
- `measure/grad_check.py` now turns pruning off with the threshold 0 as well (see "Adjoint pruning off"); the
  gradient difference "pruning off" of 1.9e-7 in the comments of `numbers_gpu.tex` came from the earlier setting and has
  not been measured again.

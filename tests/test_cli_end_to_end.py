import shutil
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

RELEASE = Path(__file__).parent / "data" / "release"


def _copy_dataset(tmp_path):
    for name in ("species.nwk", "families.txt", "fam1.nwk", "fam2.nwk", "fam3.nwk", "fam3.map", "fam4.nwk"):
        shutil.copy(RELEASE / name, tmp_path / name)


@pytest.mark.gpu
def test_fixed_rates_with_samples(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "--fix-rates", "-g", "2", "--seed", "5"]) == 0
    out = tmp_path / "out"
    names = [line.split()[0] for line in (out / "per_fam_likelihoods.txt").read_text().splitlines()]
    assert sorted(names) == ["fam1", "fam2", "fam3"]
    for family in ("fam1", "fam2", "fam3"):
        for i in (0, 1):
            for kind in ("sample_{}.xml", "eventCounts_{}.txt", "speciesEventCounts_{}.txt", "transfers_{}.txt"):
                assert (out / "reconciliations" / "all" / f"{family}_{kind.format(i)}").exists()
        assert (out / "reconciliations" / "all" / f"{family}_samples.newick").read_text().count(";") == 2
        assert (out / "reconciliations" / "summaries" / f"{family}_meanSpeciesEventCounts.txt").exists()
    assert (out / "reconciliations" / "totalSpeciesEventCounts.txt").exists()
    assert len(list((out / "reconciliations" / "origins").iterdir())) == 15
    log = (out / "gpurec.log").read_text()
    assert "Excluding family fam4" in log and "--fix-rates" in log


@pytest.mark.gpu
def test_global_fit_without_samples(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 0
    assert (tmp_path / "out" / "model_parameters" / "model_parameters.txt").exists()
    assert not (tmp_path / "out" / "reconciliations").exists()


@pytest.mark.gpu
def test_existing_reconciliations_folder_is_refused(tmp_path, monkeypatch, capsys):
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "reconciliations").mkdir(parents=True)
    (tmp_path / "out" / "reconciliations" / "keep.txt").write_text("old")
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "--fix-rates", "-g", "2"]) == 2
    assert "reconciliations" in capsys.readouterr().err
    assert (tmp_path / "out" / "reconciliations" / "keep.txt").read_text() == "old"
    assert not (tmp_path / "out" / "per_fam_likelihoods.txt").exists()


@pytest.mark.gpu
def test_run_error_returns_one(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.cli.main import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "species.nwk").write_text("(A,B)R;\n")
    (tmp_path / "families.txt").write_text("[FAMILIES]\n- f\ngene_tree=missing.nwk\n")
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 1
    assert "no gene family" in (tmp_path / "out" / "gpurec.log").read_text()


def test_no_cuda_still_logs_version_and_command(tmp_path, monkeypatch):
    from gpurec.api import model as model_module
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 1
    log = (tmp_path / "out" / "gpurec.log").read_text()
    assert "-g 0" in log
    assert f"gpurec {version('gpurec')}" in log
    assert "CUDA GPU" in log


@pytest.mark.parametrize("earlier_result", [
    "per_fam_likelihoods.txt", "model_parameters", "reconciliations", "species_trees", "perSpeciesCoverage.txt",
])
def test_folder_with_an_earlier_result_is_refused_whatever_g_is(tmp_path, monkeypatch, capsys, earlier_result):
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    if earlier_result.endswith(".txt"):
        (out / earlier_result).write_text("old")
    else:
        (out / earlier_result).mkdir()
        (out / earlier_result / "keep.txt").write_text("old")
    before = sorted(str(path.relative_to(out)) for path in out.rglob("*"))
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 2
    err = capsys.readouterr().err
    assert earlier_result in err and "Traceback" not in err
    assert sorted(str(path.relative_to(out)) for path in out.rglob("*")) == before


def test_folder_with_only_a_log_or_unrelated_files_is_accepted(tmp_path, monkeypatch):
    from gpurec.api import model as model_module
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "gpurec.log").write_text("earlier log\n")
    (tmp_path / "out" / "notes.txt").write_text("mine")
    # Without a GPU the run stops at the GPU check (exit 1), which is past the folder check.
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 1
    log = (tmp_path / "out" / "gpurec.log").read_text()
    assert log.startswith("earlier log\n") and "CUDA GPU" in log


@pytest.mark.gpu
def test_empty_existing_folder_runs(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "--fix-rates", "-g", "0"]) == 0
    assert (tmp_path / "out" / "per_fam_likelihoods.txt").exists()


def _pretend_gpu(monkeypatch):
    """Let the command line reach the file reading without a GPU (the files are read on the CPU)."""
    import gpurec

    monkeypatch.setattr(gpurec, "cuda_device", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "no GPU (test)")


@pytest.mark.parametrize("option", ["-s", "-f"])
def test_missing_input_file_is_one_line_error(tmp_path, monkeypatch, capsys, option):
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    _pretend_gpu(monkeypatch)
    argv = {"-s": ["-f", "families.txt", "-s", "missing.nwk"], "-f": ["-f", "missing.txt", "-s", "species.nwk"]}
    assert main(argv[option] + ["-p", "out", "-g", "0"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    errors = [line for line in err.splitlines() if "No such file" in line]
    assert len(errors) == 1 and "missing." in errors[0]


def test_prefix_naming_an_existing_file_is_one_line_error(tmp_path, monkeypatch, capsys):
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").write_text("a file")
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert err.startswith("gpurec: ") and err.count("\n") == 1 and "out" in err
    assert (tmp_path / "out").read_text() == "a file"


def test_unreadable_output_folder_is_one_line_error_no_traceback(tmp_path, monkeypatch, capsys):
    """A prefix the process cannot even inspect (e.g. inside a directory it cannot read) must not
    raise an uncaught PermissionError out of the earlier-results check -- it is a one-line error,
    exit code 1, same shape as the mkdir/log-file OSError case just below it."""
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    real_exists = Path.exists

    def fake_exists(self, *args, **kwargs):
        if Path("out") in self.parents:
            raise PermissionError(13, "Permission denied", str(self))
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert err.count("\n") == 1
    assert err.startswith("gpurec: cannot use out as the output folder:") and "Permission denied" in err


def test_python_warnings_go_to_the_log(tmp_path, monkeypatch):
    import warnings

    import gpurec
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    _pretend_gpu(monkeypatch)

    def warn_then_stop(path):
        warnings.warn("tolerance below the float32 floor (test)", RuntimeWarning)
        raise ValueError("stop here (test)")

    monkeypatch.setattr(gpurec, "load_fit_config", warn_then_stop)
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 1
    log = (tmp_path / "out" / "gpurec.log").read_text()
    assert "RuntimeWarning: tolerance below the float32 floor (test)" in log
    warning_line = next(line for line in log.splitlines() if "float32 floor (test)" in line)
    assert warning_line.startswith("[")
    import logging
    assert not logging.getLogger("py.warnings").handlers


@pytest.mark.gpu
def test_command_line_writes_samples_family_by_family(tmp_path, monkeypatch):
    """The command line never holds every family's samples at once; its files match the Python path's."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec import evaluate, load_families, load_fit_config, write_alerax
    from gpurec.cli.main import main
    from gpurec.interface import FitResult

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    def whole_dictionary(self, n, seed):
        raise AssertionError("the command line must not build the dictionary of every family's samples")

    with monkeypatch.context() as patch:
        patch.setattr(FitResult, "sample_reconciliations", whole_dictionary)
        assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "--fix-rates", "-g", "3", "--seed", "5"]) == 0

    data = load_families(
        families=Path("families.txt"), species_tree=Path("species.nwk"), min_covered_species=4,
        trim_ratio=0.0, max_clade_split_ratio=None, fraction_missing=None, skip_family_filtering=False,
    )
    result = evaluate(data, parametrization="global", rates=(0.1, 0.1, 0.1), config=load_fit_config(None))
    write_alerax(result, result.sample_reconciliations(n=3, seed=5), tmp_path / "python")
    cli, python = tmp_path / "out" / "reconciliations", tmp_path / "python" / "reconciliations"
    per_family = sorted(str(p.relative_to(python)) for p in python.rglob("*") if p.is_file() and p.parent != python)
    assert per_family == sorted(str(p.relative_to(cli)) for p in cli.rglob("*") if p.is_file() and p.parent != cli)
    for relative in per_family:
        assert (cli / relative).read_bytes() == (python / relative).read_bytes(), relative
    for total in ("totalSpeciesEventCounts.txt", "totalTransfers.txt"):
        assert (cli / total).exists()


def _fake_fit(parametrization_rows):
    """A stand-in for gpurec.fit that returns fixed rates, so the log can be checked without a GPU."""
    from gpurec.interface import FitResult

    def fake_fit(data, parametrization, start_rates, config):
        rows = torch.tensor(parametrization_rows[: data.n_families], dtype=torch.float64)
        log_likelihoods = {name: -1.0 for name in data.names}
        return FitResult(data, parametrization, torch.log2(rows), log_likelihoods, None, None, 0.0)

    return fake_fit


def test_log_records_fitted_global_rates(tmp_path, monkeypatch):
    import gpurec
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    _pretend_gpu(monkeypatch)
    monkeypatch.setattr(gpurec, "fit", _fake_fit([[0.5, 0.25, 0.125]] * 3))
    assert main(["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0"]) == 0
    log = (tmp_path / "out" / "gpurec.log").read_text()
    assert "rates: D 0.5 L 0.25 T 0.125" in log


@pytest.mark.parametrize("verbose", [False, True])
def test_log_records_fitted_per_family_rates(tmp_path, monkeypatch, verbose):
    import gpurec
    from gpurec.cli.main import main

    _copy_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    _pretend_gpu(monkeypatch)
    monkeypatch.setattr(gpurec, "fit", _fake_fit([[0.5, 0.25, 0.125], [0.25, 0.5, 1.0], [1.0, 1.0, 0.5]]))
    argv = ["-f", "families.txt", "-s", "species.nwk", "-p", "out", "-g", "0", "--model-parametrization", "PER-FAMILY"]
    assert main(argv + (["--verbose-opt"] if verbose else [])) == 0
    log = (tmp_path / "out" / "gpurec.log").read_text()
    assert "per-family rates of 3 families: D from 0.25 to 1, L from 0.25 to 1, T from 0.125 to 1" in log
    assert ("rates of fam2: D 0.25 L 0.5 T 1" in log) == verbose

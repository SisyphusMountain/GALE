import math
import shutil
from pathlib import Path

import pytest
import torch

from gpurec.data import load_families

RELEASE = Path(__file__).parent / "data" / "release"


def _load(**changes):
    arguments = dict(
        families=Path("families.txt"),
        species_tree=Path("species.nwk"),
        min_covered_species=4,
        trim_ratio=0.0,
        max_clade_split_ratio=None,
        fraction_missing=None,
        skip_family_filtering=False,
    )
    arguments.update(changes)
    return load_families(**arguments)


def test_default_filtering_drops_small_family(monkeypatch):
    monkeypatch.chdir(RELEASE)
    data = _load()
    assert data.names == ("fam1", "fam2", "fam3")
    assert data.dropped == (("fam4", "covers 2 species, fewer than 4"),)


def test_mapping_file_gives_same_species_as_prefix_rule(monkeypatch):
    monkeypatch.chdir(RELEASE)
    data = _load()
    fam1, fam3 = data.families[0], data.families[2]
    assert sorted(fam1["leaf_col_index"]) == sorted(fam3["leaf_col_index"])
    assert data.covered_species[0] == data.covered_species[2]


def test_min_covered_species_one_keeps_all(monkeypatch):
    monkeypatch.chdir(RELEASE)
    assert _load(min_covered_species=1).names == ("fam1", "fam2", "fam3", "fam4")


def test_trim_ratio_keeps_smallest_families(monkeypatch):
    monkeypatch.chdir(RELEASE)
    data = _load(min_covered_species=1, trim_ratio=0.5)
    assert data.n_families == 2
    assert "fam4" in data.names


def test_unreadable_file_is_dropped_unless_filtering_is_skipped(tmp_path, monkeypatch):
    shutil.copy(RELEASE / "species.nwk", tmp_path / "species.nwk")
    shutil.copy(RELEASE / "fam1.nwk", tmp_path / "fam1.nwk")
    (tmp_path / "families.txt").write_text("[FAMILIES]\n- fam1\ngene_tree=fam1.nwk\n- lost\ngene_tree=missing.nwk\n")
    monkeypatch.chdir(tmp_path)
    data = _load()
    assert data.names == ("fam1",)
    assert data.dropped[0][0] == "lost" and "cannot open" in data.dropped[0][1]
    with pytest.raises(Exception):
        _load(skip_family_filtering=True)


def test_list_and_directory_inputs(tmp_path):
    shutil.copy(RELEASE / "fam1.nwk", tmp_path / "fam1.nwk")
    shutil.copy(RELEASE / "fam2.nwk", tmp_path / "fam2.nwk")
    from_list = _load(families=[tmp_path / "fam1.nwk"], species_tree=RELEASE / "species.nwk")
    from_dir = _load(families=tmp_path, species_tree=RELEASE / "species.nwk")
    assert from_list.names == ("fam1",)
    assert from_dir.names == ("fam1", "fam2")


def test_duplicate_family_names_are_rejected(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    shutil.copy(RELEASE / "fam1.nwk", tmp_path / "a" / "g.nwk")
    shutil.copy(RELEASE / "fam1.nwk", tmp_path / "b" / "g.nwk")
    with pytest.raises(ValueError, match="same name"):
        _load(families=[tmp_path / "a" / "g.nwk", tmp_path / "b" / "g.nwk"], species_tree=RELEASE / "species.nwk")


def test_fraction_missing_file_must_cover_every_leaf(tmp_path, monkeypatch):
    monkeypatch.chdir(RELEASE)
    partial = tmp_path / "fm.txt"
    partial.write_text("5 0.1\n")
    with pytest.raises(ValueError, match="does not cover"):
        _load(fraction_missing=partial)


def test_cuda_device_message(monkeypatch):
    from gpurec.api import model as model_module

    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        model_module.cuda_device()


@pytest.mark.gpu
def test_build_model_matches_model_built_from_files(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.config import GpurecConfig

    monkeypatch.chdir(RELEASE)
    data = _load()
    config = GpurecConfig()
    built = data.build_model([0], "genewise", SolverOptions(), config, config.batching.clade_budget)
    direct = GeneReconModel("species.nwk", ["fam1.nwk"], mode="genewise", config=config)
    theta = torch.log2(torch.tensor([[0.1, 0.2, 0.05]], device="cuda", dtype=built.theta.dtype))
    a = float(built.genewise_loss_vector(theta=theta)[0])
    b = float(direct.genewise_loss_vector(theta=theta)[0])
    assert math.isfinite(a) and a == pytest.approx(b, rel=1e-6)


@pytest.mark.gpu
def test_build_model_wires_fraction_missing_from_file(tmp_path, monkeypatch):
    """FamilyData.build_model's species_name_to_index=self.species.leaf_index must resolve a
    {species: fraction} dict onto the SAME per-species indices as the direct-construction path
    (GeneReconModel(..., fraction_missing=<dict>)), which instead calls
    gpurec.core.scheduling.batching.species_name_to_index(species_tree_path) on the file. Both
    read the same species.nwk, but through different code paths (preprocess-dataset's species
    dict vs. a fresh Rust parse of the file) -- if those two ever disagree on leaf ordering,
    leaf_fm_log would be wrong for one of the two builds without either one erroring."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.config import GpurecConfig

    monkeypatch.chdir(RELEASE)
    # species.nwk's 8 leaves are 5, 11, 12, 13, 14, 9, 10, 8 -- a different fraction per leaf so a
    # mis-mapped index shows up as a wrong VALUE, not just a wrong shape.
    fm_path = tmp_path / "fraction_missing.txt"
    fm_path.write_text("5 0.05\n11 0.10\n12 0.15\n13 0.20\n14 0.25\n9 0.30\n10 0.35\n8 0.40\n")
    data = _load(fraction_missing=fm_path)
    assert data.fraction_missing == {
        "5": 0.05, "11": 0.10, "12": 0.15, "13": 0.20,
        "14": 0.25, "9": 0.30, "10": 0.35, "8": 0.40,
    }
    config = GpurecConfig()
    built = data.build_model([0], "genewise", SolverOptions(), config, config.batching.clade_budget)
    direct = GeneReconModel(
        "species.nwk", ["fam1.nwk"], mode="genewise", config=config,
        fraction_missing=data.fraction_missing,
    )
    assert built.leaf_fm_log is not None and direct.leaf_fm_log is not None
    assert torch.equal(built.leaf_fm_log, direct.leaf_fm_log), (
        f"built={built.leaf_fm_log}\ndirect={direct.leaf_fm_log}"
    )
    theta = torch.log2(torch.tensor([[0.1, 0.2, 0.05]], device="cuda", dtype=built.theta.dtype))
    a = float(built.genewise_loss_vector(theta=theta)[0])
    b = float(direct.genewise_loss_vector(theta=theta)[0])
    assert math.isfinite(a) and a == pytest.approx(b, rel=1e-6)


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
def test_trim_ratio_outside_zero_one_is_an_error(monkeypatch, ratio):
    monkeypatch.chdir(RELEASE)
    with pytest.raises(ValueError, match="trim_ratio"):
        _load(trim_ratio=ratio)


def test_max_clade_split_ratio_drops_a_family_with_too_many_clades(tmp_path):
    shutil.copy(RELEASE / "species.nwk", tmp_path / "species.nwk")
    shutil.copy(RELEASE / "fam1.nwk", tmp_path / "fam1.nwk")  # one tree, 8 genes: 27 clades
    # Two different trees on the same 8 genes: more distinct clades (33).
    (tmp_path / "two.nwk").write_text(
        "(((5_6,11_12),12_13),((13_14,14_15),(9_10,(10_11,8_9))));\n"
        "(((5_6,13_14),12_13),((11_12,14_15),(9_10,(10_11,8_9))));\n"
    )
    files = [tmp_path / "fam1.nwk", tmp_path / "two.nwk"]
    everything = _load(families=files, species_tree=tmp_path / "species.nwk", min_covered_species=1)
    clades = {name: int(family["C"]) for name, family in zip(everything.names, everything.families)}
    assert clades == {"fam1": 27, "two": 33}
    # 2.3 x (2 x 8 - 3) = 29.9: fam1 (27 clades) is kept, two (33 clades) is dropped.
    data = _load(families=files, species_tree=tmp_path / "species.nwk", min_covered_species=1,
                 max_clade_split_ratio=2.3)
    assert data.names == ("fam1",)
    assert data.dropped == (("two", "has more clades than 2.3 x (2 x genes - 3)"),)

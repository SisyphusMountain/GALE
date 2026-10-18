from pathlib import Path

import pytest
import torch
from gpurec.config import GpurecConfig

pytestmark = pytest.mark.gpu
FIXTURE = Path(__file__).parent / "data" / "alerax" / "test_trees_1"


def test_nodes_carry_clades_and_many_matches_single():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from gpurec.api.model import GeneReconModel
    from gpurec.core.backtracking.input import (
        _load_native_module,
        batch_forward,
        family_sampler_arguments,
    )

    model = GeneReconModel(str(FIXTURE / "sp.nwk"), [str(FIXTURE / "g.nwk")], mode="global", clade_budget=GpurecConfig().batching.clade_budget)
    with torch.no_grad():
        model.theta.copy_(torch.log2(torch.tensor([0.1, 0.1, 0.1], device=model.theta.device)))
    static = model.batch_statics[0]
    arguments = family_sampler_arguments(model, static, batch_forward(model, static), 0)
    native = _load_native_module()
    many = native.sample_reconciliations_many(*arguments, [11, 12])
    assert many[0] == native.sample_reconciliations_torch(*arguments, 11)
    assert many[1] == native.sample_reconciliations_torch(*arguments, 12)

    leaf_rows = set(model.families[0]["leaf_row_index"])
    for event, species, children, clade in many[0]:
        if event == "loss":
            assert clade == -1
        if event == "leaf":
            assert clade in leaf_rows

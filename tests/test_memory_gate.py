"""Forward self-loop memory-gate accounting (Task C).

Regression guard for the ``budget 0.00 GiB`` false-rejection: with warm-adjoint ON, the model's
build-time ``warm_adjoint_fits`` decision already reserves transient per-wave scratch headroom
(``cache + max_batch_scratch <= budget`` BEFORE the ~GiB warm cache is allocated). The forward
``proposal0_memory_gate`` used to RE-read ``cuda_memory_budget_bytes`` -- now depleted by the
resident cache -- and falsely reject a per-wave scratch the build already accounted for. The fix
threads that reservation as ``reserved_scratch_bytes`` so the gate validates the wave scratch against
the reservation (a scratch that GENUINELY exceeds it is still rejected) instead of the moving budget.

These tests are synthetic (W/S/budget arithmetic) -- fast, no CUDA / 500-leaf hardware needed.
"""
import math

import pytest

import torch

from gpurec.core import memory_policy
from gpurec.core.memory_policy import (
    proposal0_memory_gate,
    proposal0_wave_scratch_bytes,
    warm_adjoint_fits,
)

DTYPE = torch.float32


def test_cold_path_rejects_when_current_budget_too_small(monkeypatch):
    """reserved_scratch_bytes=None (cold): the gate reads current free memory. A tiny budget rejects
    -- this is the exact double-gate condition that falsely fired with the resident warm cache."""
    W, S = 8192, 999
    scratch = proposal0_wave_scratch_bytes(W, S, DTYPE)
    # Emulate the depleted post-cache free memory: budget ~ 0.
    monkeypatch.setattr(memory_policy, "cuda_memory_budget_bytes", lambda device=None: 0)
    ok, required, budget = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=None)
    assert not ok
    assert required == scratch
    assert budget == 0


def test_reserved_path_trusts_build_reservation_over_depleted_budget(monkeypatch):
    """The fix: when the warm-adjoint fit already reserved >= this wave's scratch, the gate PASSES
    even though the re-read current budget is ~0 (it must not consult the depleted free memory)."""
    W, S = 8192, 999
    scratch = proposal0_wave_scratch_bytes(W, S, DTYPE)
    reserved = scratch  # build reserved exactly (or more) than this wave needs

    # If the gate wrongly re-read the budget it would reject; make that failure loud if it happens.
    def _boom(device=None):
        raise AssertionError("gate must NOT read cuda_memory_budget_bytes on the reserved path")

    monkeypatch.setattr(memory_policy, "cuda_memory_budget_bytes", _boom)
    ok, required, budget = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=reserved)
    assert ok
    assert budget == reserved


def test_reserved_path_still_rejects_genuinely_oversized_scratch():
    """Safety preserved: a wave scratch that GENUINELY exceeds the reserved headroom is rejected --
    the fix corrects the accounting, it does not disable the gate."""
    W, S = 8192, 999
    scratch = proposal0_wave_scratch_bytes(W, S, DTYPE)
    reserved = scratch - 1  # one byte short of what this wave needs
    ok, required, budget = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=reserved)
    assert not ok
    assert required == scratch
    assert budget == reserved


def test_budget_counts_reclaimable_cache(monkeypatch):
    """cuda_memory_budget_bytes must credit PyTorch's reclaimable caching-allocator pool
    (reserved - allocated) as available. mem_get_info's driver free_b counts that reusable pool as
    'used', so mid-run (e.g. the fp64 final_eval backward holds a large reserved pool) the raw
    free_b collapses to ~0 and the gate falsely rejects a scratch that the allocator could serve
    from its own pool. The budget must be free_b + (reserved - allocated) - reserve, capped by
    total*fraction. Regression for the 500-leaf final_eval 'budget 0.09 GiB' false rejection."""
    GIB = 1024 ** 3
    total = 24 * GIB
    free_b = 1 * GIB          # driver sees only 1 GiB free...
    reserved = 18 * GIB       # ...but torch holds an 18 GiB pool,
    allocated = 8 * GIB       # 8 GiB live -> 10 GiB reclaimable.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx=0: (free_b, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx=0: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda idx=0: allocated)
    # Nothing but PyTorch on this card: the baseline the budget subtracts for other programs is set to 0
    # here, instead of being measured once (from these mocks, or from the real card if an earlier test
    # already asked) and kept for the whole process.
    monkeypatch.setattr(memory_policy, "_NON_PYTORCH_AT_START", {0: 0})
    # fraction=0.85 cap = 20.4 GiB (not binding); reserve=1 GiB.
    budget = memory_policy.cuda_memory_budget_bytes(0, default_fraction=0.85, default_reserve_gib=1.0)
    reclaimable = reserved - allocated
    expected = min(int(total * 0.85), (free_b + reclaimable) - 1 * GIB)
    assert budget == expected
    # Concretely: 1 + 10 - 1 = 10 GiB available, far above a fresh raw-free budget of ~0.
    assert budget == 10 * GIB
    # And a genuinely tight state (no reclaimable pool, low free) still yields a near-zero budget.
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx=0: allocated)  # reclaimable = 0
    tight = memory_policy.cuda_memory_budget_bytes(0, default_fraction=0.85, default_reserve_gib=1.0)
    assert tight == 0  # free_b(1) - reserve(1) = 0, no pool to credit -> genuine OOM still caught


def test_budget_counts_other_programs_once(monkeypatch):
    """Other programs hold 8 GiB of a 24 GiB card, PyTorch holds nothing yet: the driver reports 16 GiB
    free. The budget is the smaller of 85 % of the 16 GiB they leave and the 16 GiB free minus the 1 GiB
    reserve, 13.6 GiB. GALE 1.0 as first released also took the 8 GiB off the free memory, which the
    driver had already done, and gave 7 GiB; with 12 GiB held by others it gave 0 and every backward
    pass was refused."""
    GIB = 1024 ** 3
    total, others = 24 * GIB, 8 * GIB
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx=0: (total - others, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx=0: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda idx=0: 0)
    monkeypatch.setattr(memory_policy, "_NON_PYTORCH_AT_START", {})
    budget = memory_policy.cuda_memory_budget_bytes(0, default_fraction=0.85, default_reserve_gib=1.0)
    assert memory_policy._NON_PYTORCH_AT_START == {0: others}
    assert budget == int((total - others) * 0.85)


def test_wave_scratch_is_subset_of_warm_adjoint_reservation(monkeypatch):
    """Invariant the fix relies on: a single wave's W is a subset of the batch's clades, so its
    scratch never exceeds the ``max_batch_clades`` scratch that ``warm_adjoint_fits`` reserved.
    Thus, whenever warm fit at build, the reserved-path gate provably passes every wave."""
    S = 999
    max_batch_clades = 314_998            # e.g. clade_budget cap
    # Pretend a comfortable build-time budget so warm 'fits'.
    reserved = proposal0_wave_scratch_bytes(max_batch_clades, S, DTYPE)
    monkeypatch.setattr(
        memory_policy, "cuda_memory_budget_bytes", lambda device=None: reserved * 100
    )
    ok_build, cache, scratch_reserved, budget = warm_adjoint_fits(
        1_000_000, S, DTYPE, max_batch_clades=max_batch_clades
    )
    assert ok_build
    assert scratch_reserved == reserved
    # Every physically-realizable wave (W <= max_batch_clades) passes the reserved-path gate, even
    # when the (mocked) current budget is ~0, whereas the cold path would reject it.
    monkeypatch.setattr(memory_policy, "cuda_memory_budget_bytes", lambda device=None: 0)
    for W in (1, 4096, 8192, max_batch_clades):
        ok_reserved, _, _ = proposal0_memory_gate(
            W, S, DTYPE, reserved_scratch_bytes=scratch_reserved
        )
        assert ok_reserved, W
        ok_cold, _, _ = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=None)
        assert not ok_cold, W  # cold path re-reads the depleted budget and (wrongly, pre-fix) rejects

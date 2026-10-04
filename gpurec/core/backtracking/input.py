import torch

from gpurec.core.inference.solver import solve_resident_e_pi
from gpurec.core.native import load_native_module

_NATIVE_MODULE = "gpurec_backtrack"
_NATIVE_LIBRARY = "libgpurec_backtrack.so"


def _load_native_module():
    return load_native_module(_NATIVE_MODULE, _NATIVE_LIBRARY)


def _numpy(value, dtype: torch.dtype) -> object:
    return torch.as_tensor(value).detach().cpu().to(dtype=dtype).contiguous().numpy()


def _family_param(param: torch.Tensor, family_index: int) -> torch.Tensor:
    param = torch.as_tensor(param).detach()
    if param.ndim <= 1:
        return param.reshape(-1)[0 if int(param.numel()) == 1 else family_index]
    return param[0] if int(param.shape[0]) == 1 else param[family_index]


def _species_param(param: torch.Tensor, model, family_index: int, S: int) -> torch.Tensor:
    param = torch.as_tensor(param).detach()
    if param.ndim == 0:
        return param.reshape(1).expand(S)
    if param.ndim == 1:
        if int(param.numel()) == S:
            return param
        return param.reshape(-1)[0 if int(param.numel()) == 1 else family_index].reshape(1).expand(S)
    row = param[0] if int(param.shape[0]) == 1 else param[family_index]
    return row.reshape(-1)[0].reshape(1).expand(S) if int(row.numel()) == 1 else row


def batch_forward(model, static) -> dict:
    """Run the forward solve of one batch once, for sampling every family in it."""
    theta = model._theta_for_static(static, model.theta)
    E, _, _, Ebar, _, pi_wave, pibar_wave, _, log_p_s, log_p_d, *_, receiver_log_probs = solve_resident_e_pi(
        static,
        theta,
        model.receiver_weights,
    )
    pi_state = getattr(static, "pi_forward_state", None)
    if pi_state is None:
        raise RuntimeError("Pi forward did not publish its row-offset state")
    perm = static.wave_layout["perm"]
    return {
        "E": E,
        "Ebar": Ebar,
        "pi": pi_wave.index_select(0, perm),
        "pibar": pibar_wave.index_select(0, perm),
        "pi_offset": pi_state.pi_offset.index_select(0, perm),
        "pibar_offset": pi_state.pibar_offset.index_select(0, perm),
        "log_p_s": log_p_s,
        "log_p_d": log_p_d,
        "receiver_log_probs": receiver_log_probs,
    }


def family_sampler_arguments(model, static, forward: dict, local_family_index: int) -> tuple:
    """The 16 arrays the native sampler takes for one family of a batch, in order."""
    family_index = static.family_indices[local_family_index]
    family = model.families[family_index]
    offset = sum(int(model.families[idx]["C"]) for idx in static.family_indices[:local_family_index])
    C = int(family["C"])
    pi_offset = forward["pi_offset"][offset : offset + C]
    pibar_offset = forward["pibar_offset"][offset : offset + C]
    pi_family = forward["pi"][offset : offset + C].to(pi_offset.dtype) + pi_offset[:, None]
    pibar_family = forward["pibar"][offset : offset + C].to(pibar_offset.dtype) + pibar_offset[:, None]
    species_helpers = model.species_helpers
    S = int(species_helpers["S"])
    leaf_species = torch.full((C,), -1, dtype=torch.int64)
    leaf_species[torch.as_tensor(family["leaf_row_index"], dtype=torch.int64)] = torch.as_tensor(
        family["leaf_col_index"], dtype=torch.int64
    )
    return (
        int(family["root_clade_id"]),
        _numpy(leaf_species, torch.int64),
        _numpy(family["split_parents_sorted"], torch.int64),
        _numpy(family["split_leftrights_sorted"], torch.int64),
        _numpy(family["log_split_probs_sorted"], torch.float64),
        _numpy(pi_family, torch.float64),
        _numpy(pibar_family, torch.float64),
        _numpy(_family_param(forward["E"], local_family_index), torch.float64),
        _numpy(_family_param(forward["Ebar"], local_family_index), torch.float64),
        _numpy(_species_param(forward["log_p_s"], model, local_family_index, S), torch.float64),
        _numpy(_species_param(forward["log_p_d"], model, local_family_index, S), torch.float64),
        _numpy(forward["receiver_log_probs"], torch.float64),
        _numpy(species_helpers["sp_child1"], torch.int64),
        _numpy(species_helpers["sp_child2"], torch.int64),
        _numpy(species_helpers["sp_subtree_start"], torch.int64),
        _numpy(species_helpers["sp_subtree_end"], torch.int64),
    )


def _find_batch(model, family_index: int):
    for static in model.batch_statics:
        if family_index in static.family_indices:
            return static, static.family_indices.index(family_index)
    raise IndexError(f"family index {family_index} is not in any batch")


def sample_reconciliations(model, *, family_index: int, seed: int):
    static, local_family_index = _find_batch(model, family_index)
    arguments = family_sampler_arguments(model, static, batch_forward(model, static), local_family_index)
    return _load_native_module().sample_reconciliations_torch(*arguments, seed)

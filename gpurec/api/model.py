import logging
import math
from dataclasses import replace as _replace_dataclass

import torch

from gpurec.api._batch_state import _BatchStatic, build_batch_static
from gpurec.api._execution import stream_genewise_loss_vector_grad, theta_for_static
from gpurec.api.solver_options import SolverOptions
from gpurec.config import GpurecConfig, PrecisionOptions, torch_dtype_name
from gpurec.config.rates import RateBounds
from gpurec.core.scheduling.batching import plan_batch_wave_layouts, preprocess_dataset

logger = logging.getLogger(__name__)

_MODE_FLAGS = {
    "global": (False, False),
    "specieswise": (False, True),
    "genewise": (True, False),
}


def _mode_flags(mode: str) -> tuple[bool, bool]:
    try:
        return _MODE_FLAGS[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"mode must be one of {sorted(_MODE_FLAGS)}, got {mode!r}") from exc


# Batching layout of GeneReconModel. Named here once so FamilyData.build_model uses the same values.
DEFAULT_FAMILY_CHUNK_SIZE = 300
DEFAULT_BATCH_PACKING = "depth_first_fit"
DEFAULT_MAX_WAVE_SIZE = 8192


def cuda_device() -> torch.device:
    """gpurec's kernels are Triton kernels: they only run on a CUDA GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gpurec needs a CUDA GPU, and none is available (torch.cuda.is_available() is False)"
        )
    return torch.device("cuda")


def _resolve_precision(config: GpurecConfig | None, dtype) -> PrecisionOptions:
    precision_options = _replace_dataclass(config.precision if config is not None else PrecisionOptions())
    if dtype is not None:
        precision_options.model_dtype = torch_dtype_name(dtype)
    precision_options.validate()
    return precision_options


class GeneReconModel(torch.nn.Module):
    def __init__(
        self,
        species_tree,
        gene_trees,
        *,
        mode: str = "global",
        family_chunk_size: int | None = DEFAULT_FAMILY_CHUNK_SIZE,
        clade_budget: int,
        batch_packing: str = DEFAULT_BATCH_PACKING,
        max_wave_size: int = DEFAULT_MAX_WAVE_SIZE,
        solver_options: SolverOptions | dict | None = None,
        config: GpurecConfig | None = None,
        dtype: torch.dtype | str | None = None,
        per_family_origination: bool = False,
        fraction_missing=None,
    ):
        super().__init__()
        precision_options = _resolve_precision(config, dtype)
        gene_trees = list(gene_trees)
        raw = preprocess_dataset(
            str(species_tree),
            gene_trees,
            gene_to_species=[None] * len(gene_trees),
            accumulator_dtype=precision_options.accumulator_torch_dtype,
            family_chunk_size=family_chunk_size,
            clade_budget=clade_budget,
            batch_packing=batch_packing,
            max_wave_size=max_wave_size,
            family_group_assignments=None,
        )
        from gpurec.core.scheduling.batching import species_name_to_index as _species_name_to_index

        self._setup(
            raw["species"], raw["families"], raw.get("batches"), raw.get("batch_wave_layouts"),
            mode=mode, solver_options=solver_options, config=config, precision_options=precision_options,
            per_family_origination=per_family_origination, fraction_missing=fraction_missing,
            species_name_to_index=(
                _species_name_to_index(str(species_tree)) if isinstance(fraction_missing, dict) else None
            ),
            family_chunk_size=family_chunk_size, clade_budget=clade_budget,
            batch_packing=batch_packing, max_wave_size=max_wave_size,
        )

    @classmethod
    def from_preprocessed(
        cls, species_raw, families, *, mode, solver_options, config, fraction_missing,
        species_name_to_index, per_family_origination, family_chunk_size, clade_budget,
        batch_packing, max_wave_size,
    ) -> "GeneReconModel":
        """Build a model from already-parsed families (see ``gpurec.data.FamilyData``)."""
        plan = plan_batch_wave_layouts(
            list(families),
            family_chunk_size=family_chunk_size,
            clade_budget=clade_budget,
            batch_packing=batch_packing,
            max_wave_size=max_wave_size,
            family_group_assignments=None,
        )
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model._setup(
            species_raw, list(families), plan["batches"], plan["batch_wave_layouts"],
            mode=mode, solver_options=solver_options, config=config,
            precision_options=_resolve_precision(config, None),
            per_family_origination=per_family_origination, fraction_missing=fraction_missing,
            species_name_to_index=species_name_to_index, family_chunk_size=family_chunk_size,
            clade_budget=clade_budget, batch_packing=batch_packing, max_wave_size=max_wave_size,
        )
        return model

    def _setup(
        self, species_raw, families, batches, batch_wave_layouts, *, mode, solver_options, config,
        precision_options, per_family_origination, fraction_missing, species_name_to_index,
        family_chunk_size, clade_budget, batch_packing, max_wave_size,
    ) -> None:
        device = cuda_device()
        mode = str(mode).strip().lower()
        genewise, specieswise = _mode_flags(mode)
        if per_family_origination and not genewise:
            raise ValueError("per_family_origination requires mode='genewise'")
        # ``config`` supplies solver and precision defaults here. Explicit
        # ``solver_options`` and ``dtype`` arguments win over their matching
        # config values; the remaining config sections belong to fit/curvature
        # entry points rather than this constructor.
        if config is not None and solver_options is None:
            solver_options = config.solver
        dtype = precision_options.model_torch_dtype
        self.solver_options = solver_options
        self.precision_options = precision_options
        self.accumulator_dtype = precision_options.accumulator_torch_dtype
        self.dtype = dtype
        species_helpers = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in species_raw.items()}
        self.families = families
        batches = batches or [list(range(len(families)))]
        batch_wave_layouts = batch_wave_layouts or [None] * len(batches)
        theta_shape = (len(families), 3) if genewise else ((int(species_helpers["S"]), 3) if specieswise else (3,))
        bounds = RateBounds()
        self.theta = torch.nn.Parameter(
            torch.full(theta_shape, math.log2(bounds.init_rate), dtype=dtype, device=device)
        )
        self.receiver_weights = torch.nn.Parameter(
            torch.zeros((int(species_helpers["S"]),), dtype=dtype, device=device)
        )
        # Per-species origination logits (softmax over the S species nodes); when
        # per_family_origination=True this is instead per-family [G,S] (independent softmax per
        # family). Default all-zeros => the uniform origination prior the likelihood assumes; enters
        # ONLY the NLL aggregation.
        origination_shape = (len(families), int(species_helpers["S"])) if per_family_origination \
            else (int(species_helpers["S"]),)
        self.origination_weights = torch.nn.Parameter(
            torch.zeros(origination_shape, dtype=dtype, device=device)
        )
        self.per_family_origination = per_family_origination
        self.species_helpers = species_helpers
        from gpurec.core.parameters.fraction_missing import build_fraction_missing_tensors
        self.leaf_fm_log = build_fraction_missing_tensors(
            species_helpers,
            fraction_missing=fraction_missing,
            species_name_to_index=species_name_to_index,
            device=device,
            dtype=dtype,
        )
        self.mode = mode
        self.genewise = genewise
        self.specieswise = specieswise
        self.family_chunk_size = family_chunk_size
        self.clade_budget = clade_budget
        self.batch_packing = batch_packing
        self.max_wave_size = max_wave_size
        self.family_batches = [list(batch) for batch in batches]
        self.batch_wave_layouts = batch_wave_layouts
        self.batch_statics = [
            self._make_batch_static(batch, plan, device=device, max_wave_size=max_wave_size)
            for batch, plan in zip(self.family_batches, self.batch_wave_layouts)
        ]
        first = self.batch_statics[0]
        self.wave_layout = first.wave_layout
        self.rate_family_idx = first.rate_family_idx
        self.warm_E = first.warm_E
        self._resolve_warm_adjoint_gate()

    def _resolve_warm_adjoint_gate(self) -> None:
        """Memory-gate the ``solver_options.warm_adjoint`` cache by the clades x species product.

        The warm-adjoint cache (``static.warm_v``) is resident at ~``total_clades * S * dtype`` and on
        large family sets dwarfs both the static base (~0.25 GiB) and one batch's transient scratch
        (~1.8 GiB) -- full Hogenom ~19 GiB. Decide ONCE per (re)build whether it (plus the largest
        batch's backward scratch) fits the free-memory budget; if not, mark every batch static so
        ``_execution`` ignores ``solver_options.warm_adjoint`` and runs cold. Small/medium sets (e.g.
        Hogenom-1055 ~5.5 GiB) keep warm and its speed. This is the right gate (clades x species) vs a
        family count.
        """
        device = self.theta.device
        if device.type != "cuda":
            return
        from gpurec.core.memory_policy import warm_adjoint_fits

        S = int(self.species_helpers["S"])
        batch_clades = [
            sum(int(self.families[i]["C"]) for i in batch) for batch in self.family_batches
        ]
        total_clades = sum(batch_clades)
        ok, cache, scratch, budget = warm_adjoint_fits(
            total_clades, S, self.theta.dtype, device=device, max_batch_clades=max(batch_clades, default=0)
        )
        for static in self.batch_statics:
            static.warm_adjoint_ok = ok
            # When warm fits, the build gate verified ``cache + scratch <= budget`` with ``scratch``
            # covering the largest batch's per-wave scratch (W <= max_batch_clades). Record it so the
            # forward self-loop gate trusts this reservation instead of re-reading the depleted
            # post-cache free memory (the double-gate bug). Cleared to None when warm does not fit.
            static.warm_scratch_reserved_bytes = scratch if ok else None
        self.warm_adjoint_ok = ok
        if not ok:
            gib = 1024 ** 3
            logger.info(
                f"[gpurec] warm-adjoint disabled (running cold): cache {cache / gib:.1f} GiB + scratch "
                f"{scratch / gib:.1f} GiB > budget {(budget or 0) / gib:.1f} GiB "
                f"[{total_clades:,} clades x {S} species, {len(self.family_batches)} batches]"
            )

    def _make_batch_static(self, batch, plan=None, *, device: torch.device, max_wave_size: int) -> _BatchStatic:
        return build_batch_static(
            self.families,
            batch,
            plan,
            species_helpers=self.species_helpers,
            genewise=self.genewise,
            specieswise=self.specieswise,
            solver_options=self.solver_options,
            precision_options=self.precision_options,
            accumulator_dtype=self.accumulator_dtype,
            device=device,
            max_wave_size=max_wave_size,
            leaf_fm_log=self.leaf_fm_log,
        )

    @property
    def solver_options(self) -> SolverOptions:
        return self._solver_options

    @solver_options.setter
    def solver_options(self, options: SolverOptions | dict | None) -> None:
        if options is None:
            options = SolverOptions()
        elif isinstance(options, dict):
            options = SolverOptions(**options)
        elif not isinstance(options, SolverOptions):
            raise TypeError("solver_options must be a SolverOptions instance, dict, or None")

        options.validate()
        self._solver_options = options
        for static in getattr(self, "batch_statics", []):
            static.solver_options = options

    def _theta_for_static(self, static: _BatchStatic, theta: torch.Tensor) -> torch.Tensor:
        return theta_for_static(static, theta, genewise=self.genewise)

    def genewise_loss_vector_and_grad(
        self,
        *,
        theta: torch.Tensor | None = None,
        receiver_weights: torch.Tensor | None = None,
        origination_weights: torch.Tensor | None = None,
        need_grad: bool = True,
        update_warm_starts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return per-family NLLs and optional independent genewise gradients.

        This is the closure helper for row-wise optimizers. It is only valid in genewise
        mode because shared-theta modes do not have independent per-family
        parameter rows. Warm starts are not updated by default so line-search
        probes stay deterministic even when trial points are rejected.
        """
        if not self.genewise:
            raise ValueError("genewise_loss_vector_and_grad() requires genewise mode")
        theta = self.theta if theta is None else theta
        receiver_weights = self.receiver_weights if receiver_weights is None else receiver_weights
        origination_weights = self.origination_weights if origination_weights is None else origination_weights
        loss_vec, grad_theta, grad_receiver, _grad_origination = stream_genewise_loss_vector_grad(
            self.batch_statics,
            theta,
            receiver_weights,
            origination_weights,
            need_grad=need_grad,
            update_warm_starts=update_warm_starts,
        )
        return loss_vec, grad_theta, grad_receiver

    def genewise_loss_vector(
        self,
        *,
        theta: torch.Tensor | None = None,
        receiver_weights: torch.Tensor | None = None,
        update_warm_starts: bool = False,
    ) -> torch.Tensor:
        loss_vec, _grad_theta, _grad_receiver = self.genewise_loss_vector_and_grad(
            theta=theta,
            receiver_weights=receiver_weights,
            need_grad=False,
            update_warm_starts=update_warm_starts,
        )
        return loss_vec

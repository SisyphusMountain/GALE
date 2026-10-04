"""Fitting drivers: first-order / Newton-CG / MAP / CV / genewise recipes."""
from gpurec.fit.genewise_fit import fit_genewise
from gpurec.fit.global_fit import fit_global

__all__ = ["fit_genewise", "fit_global"]

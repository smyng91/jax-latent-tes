"""Differentiable dual-cycle PCM topology optimization."""

__all__ = [
    "BCs",
    "Geom",
    "Sim",
    "liquid_fraction",
    "simulate",
    "two_tube_geom",
    "density_from_logits",
    "filter_density",
    "project_density",
]


def __getattr__(name):
    if name in {"BCs", "Geom", "Sim", "liquid_fraction", "simulate", "two_tube_geom"}:
        from pcm import physics as m
        return getattr(m, name)
    if name in {"density_from_logits", "filter_density", "project_density"}:
        from pcm import design as m
        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
High-level plotting helpers built on Matplotlib, Seaborn, and Plotly.

The mattplotlib package provides a consistent styling layer, a small set of
beginner-friendly plotting functions, and configuration utilities so you can
quickly produce publication-ready graphics across backends.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

from . import matplotlib_plots, seaborn_plots
from .config import MattplotlibConfig, get_config, load_config
from .style import apply_mpl_style, get_style_path

_PLOTLY_MODULE: ModuleType | None = None

__all__ = [
    "MattplotlibConfig",
    "apply_mpl_style",
    "get_config",
    "get_style_path",
    "load_config",
    "matplotlib_plots",
    "plotly_plots",
    "seaborn_plots",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - small helper
    if name != "plotly_plots":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    global _PLOTLY_MODULE

    if _PLOTLY_MODULE is not None:
        return _PLOTLY_MODULE

    try:
        module = import_module(".plotly_plots", __name__)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "The mattplotlib Plotly helpers require the optional 'plotly' dependency. "
            "Install it with 'pip install plotly' to access mattplotlib.plotly_plots."
        ) from exc

    _PLOTLY_MODULE = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - convenience
    members = set(globals()) | set(__all__)
    members.add("plotly_plots")
    return sorted(members)

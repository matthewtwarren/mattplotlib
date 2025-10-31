"""
Shared style helpers for mattplotlib visuals.

The utilities here keep matplotlib, seaborn, and plotly visuals aligned by
ensuring a consistent font, colour cycle, and default style sheet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rcParams

from .config import MattplotlibConfig, get_config


PathLike = Union[str, Path]


def get_style_path(style_path: Optional[PathLike] = None) -> Path:
    """
    Resolve the matplotlib style file to apply.

    Parameters
    ----------
    style_path:
        Optional override pointing to a custom ``.mplstyle`` or ``matplotlibrc`` file.
        When omitted the path stored in :class:`MattplotlibConfig` is used.
    """

    if style_path is not None:
        return Path(style_path).expanduser()
    return Path(get_config().matplotlib_style)


def apply_mpl_style(style_path: Optional[PathLike] = None) -> Path:
    """
    Apply the mattplotlib Matplotlib style globally.

    Returns the resolved style path so callers can log/debug the active style.
    """

    resolved = get_style_path(style_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Matplotlib style file not found at '{resolved}'. "
            "Update your mattplotlib configuration or supply a custom style path."
        )

    plt.style.use(resolved)
    _apply_shared_rc(get_config())
    return resolved


def _apply_shared_rc(config: MattplotlibConfig) -> None:
    """Set core rcParams to keep plots aligned across libraries."""

    rcParams["font.family"] = config.font_family
    rcParams["axes.prop_cycle"] = cycler(color=list(config.color_cycle))
    rcParams.setdefault("axes.titlesize", "x-large")
    rcParams.setdefault("axes.labelsize", "large")

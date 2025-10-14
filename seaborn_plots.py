"""
Wrapper functions around seaborn that keep theming and API consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from .config import get_config
from .style import apply_mpl_style


if TYPE_CHECKING:  # pragma: no cover - typing aid
    from pandas import DataFrame
else:
    DataFrame = Any  # type: ignore


def _prepare_axes(ax: Optional[plt.Axes], use_style: bool) -> plt.Axes:
    if use_style:
        apply_mpl_style()
    cfg = get_config()
    sns.set_theme(**cfg.seaborn_theme)
    return ax if ax is not None else plt.gca()


def scatterplot(
    data: DataFrame,
    *,
    x: str,
    y: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    use_style: bool = True,
    **kwargs,
) -> plt.Axes:
    """Create a seaborn scatterplot with mattplotlib styling."""

    axis = _prepare_axes(ax, use_style)
    sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=axis, **kwargs)
    if title:
        axis.set_title(title)
    return axis


def boxplot(
    data: DataFrame,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    use_style: bool = True,
    **kwargs,
) -> plt.Axes:
    """Create a seaborn boxplot."""

    axis = _prepare_axes(ax, use_style)
    sns.boxplot(data=data, x=x, y=y, hue=hue, ax=axis, **kwargs)
    if title:
        axis.set_title(title)
    return axis


def histogram(
    data: DataFrame,
    *,
    x: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    use_style: bool = True,
    bins: Optional[int] = None,
    **kwargs,
) -> plt.Axes:
    """Create a histogram or KDE using seaborn's displot API."""

    axis = _prepare_axes(ax, use_style)
    sns.histplot(data=data, x=x, hue=hue, bins=bins, ax=axis, **kwargs)
    if title:
        axis.set_title(title)
    return axis

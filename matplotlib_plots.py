"""
Convenience wrappers for common Matplotlib charts.

Each helper applies mattplotlib styling (unless disabled) and returns the
underlying ``Axes`` object for further customisation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_mpl_style


def _prepare_axes(ax: Optional[plt.Axes], use_style: bool) -> plt.Axes:
    if use_style:
        apply_mpl_style()
    return ax if ax is not None else plt.gca()


def line_plot(
    x: Sequence[float],
    y: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    use_style: bool = True,
    **line_kwargs,
) -> plt.Axes:
    """
    Draw a simple line chart.
    """

    axis = _prepare_axes(ax, use_style)
    axis.plot(x, y, label=label, **line_kwargs)

    if label:
        axis.legend()
    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)

    return axis


def scatter_plot(
    x: Sequence[float],
    y: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    use_style: bool = True,
    **scatter_kwargs,
) -> plt.Axes:
    """Draw a scatter plot with optional labels."""

    axis = _prepare_axes(ax, use_style)
    axis.scatter(x, y, label=label, **scatter_kwargs)

    if label:
        axis.legend()
    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)

    return axis


def bar_chart(
    categories: Sequence[str],
    values: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    use_style: bool = True,
    **bar_kwargs,
) -> plt.Axes:
    """Render a categorical bar chart."""

    axis = _prepare_axes(ax, use_style)
    axis.bar(categories, values, **bar_kwargs)

    # Ensure category labels remain legible on narrow figures.
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")

    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)

    return axis


def heatmap(
    matrix: Sequence[Sequence[float]],
    *,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    colorbar: bool = True,
    use_style: bool = True,
    **imshow_kwargs,
) -> plt.Axes:
    """Display a 2D matrix as a heatmap."""

    axis = _prepare_axes(ax, use_style)
    matrix_np = np.asarray(matrix)
    aspect = imshow_kwargs.pop("aspect", "auto")
    image = axis.imshow(matrix_np, aspect=aspect, **imshow_kwargs)

    if colorbar:
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)

    if x_labels:
        axis.set_xticks(range(len(x_labels)), labels=x_labels, rotation=45, ha="right")
    if y_labels:
        axis.set_yticks(range(len(y_labels)), labels=y_labels)

    return axis

"""
Wrappers around Plotly Express with mattplotlib defaults.
"""

from __future__ import annotations

from typing import Optional, Sequence

import plotly.express as px
from plotly.graph_objs import Figure

from .config import get_config


def _base_kwargs(
    template: Optional[str],
    color_sequence: Optional[Sequence[str]],
) -> dict:
    cfg = get_config()
    return {
        "template": template or cfg.plotly_template,
        "color_discrete_sequence": list(color_sequence or cfg.color_cycle),
    }


def _finalise(fig: Figure, title: Optional[str]) -> Figure:
    cfg = get_config()
    fig.update_layout(
        title=title,
        font={"family": cfg.font_family},
    )
    return fig


def line_chart(
    data,
    *,
    x: str,
    y: str,
    color: Optional[str] = None,
    template: Optional[str] = None,
    color_sequence: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Figure:
    """Create a Plotly line chart with mattplotlib defaults."""

    fig = px.line(
        data,
        x=x,
        y=y,
        color=color,
        **_base_kwargs(template, color_sequence),
        **kwargs,
    )
    return _finalise(fig, title)


def scatter_chart(
    data,
    *,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    template: Optional[str] = None,
    color_sequence: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Figure:
    """Create a Plotly scatter chart."""

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        size=size,
        **_base_kwargs(template, color_sequence),
        **kwargs,
    )
    return _finalise(fig, title)


def bar_chart(
    data,
    *,
    x: str,
    y: str,
    color: Optional[str] = None,
    template: Optional[str] = None,
    color_sequence: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    barmode: str = "group",
    **kwargs,
) -> Figure:
    """Create a Plotly bar chart."""

    fig = px.bar(
        data,
        x=x,
        y=y,
        color=color,
        barmode=barmode,
        **_base_kwargs(template, color_sequence),
        **kwargs,
    )
    return _finalise(fig, title)

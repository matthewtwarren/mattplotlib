"""
Convenience wrappers for common Matplotlib charts.

Each helper applies mattplotlib styling (unless disabled) and returns the
underlying ``Axes`` object for further customisation.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_mpl_style


def _prepare_axes(ax: Optional[plt.Axes], use_style: bool) -> plt.Axes:
    if use_style:
        apply_mpl_style()
    return ax if ax is not None else plt.gca()


def nested_donut_chart(
    outer_values: Sequence[Sequence[float]],
    *,
    inner_values: Optional[Sequence[float]] = None,
    inner_labels: Optional[Sequence[str]] = None,
    outer_labels: Optional[Sequence[str]] = None,
    outer_label_formatter: Optional[Callable[[int, float], str]] = None,
    ax: Optional[plt.Axes] = None,
    use_style: bool = True,
    radius: float = 1.0,
    width: float = 0.3,
    startangle: float = 90.0,
    inner_colors: Optional[Sequence[str]] = None,
    outer_colors: Optional[Sequence[str]] = None,
    inner_autopct: Optional[Union[str, Callable[[float], str]]] = "%1.0f%%",
    inner_textprops: Optional[Mapping[str, Any]] = None,
    inner_wedgeprops: Optional[Mapping[str, Any]] = None,
    inner_pie_kwargs: Optional[Mapping[str, Any]] = None,
    outer_wedgeprops: Optional[Mapping[str, Any]] = None,
    outer_pie_kwargs: Optional[Mapping[str, Any]] = None,
    annotate_outer: bool = True,
    annotation_distance: float = 1.3,
    annotation_mid_radius: Optional[float] = None,
    annotation_kwargs: Optional[Mapping[str, Any]] = None,
    center_text: Optional[str] = None,
    center_text_kwargs: Optional[Mapping[str, Any]] = None,
    equal_aspect: bool = True,
    normalize_outer: bool = False,
) -> plt.Axes:
    """Draw a flexible, multi-ring donut chart.

    Parameters
    ----------
    outer_values:
        Nested sequence where each inner sequence represents the segments that
        make up an inner ring category.
    inner_values:
        Aggregate values for each inner ring segment. When omitted, they are
        inferred by summing ``outer_values`` along axis 1.
    inner_labels / outer_labels:
        Labels applied to the inner and outer rings respectively. Provide a
        callable via ``outer_label_formatter`` to dynamically format outer
        labels from their index and value when ``outer_labels`` is omitted.
    outer_label_formatter:
        Callable receiving ``(index, value)`` that returns a label string.
    ax:
        Matplotlib axes to draw on. A new axis is created when ``None``.
    radius / width / startangle:
        Geometric properties for the concentric pies. ``width`` applies to both
        rings; adjust ``radius`` to scale the overall chart.
    inner_colors / outer_colors:
        Optional colour sequences. Defaults fall back to evenly sampled colours
        from Matplotlib's ``tab20c`` colormap.
    inner_autopct / inner_textprops / inner_wedgeprops / inner_pie_kwargs:
        Fine-grained controls for the inner ring appearance and labelling. Set
        ``inner_autopct=None`` to disable percentage labels.
    outer_wedgeprops / outer_pie_kwargs:
        Appearance adjustments for the outer ring.
    annotate_outer / annotation_distance / annotation_mid_radius /
    annotation_kwargs:
        Toggle and configure radial call-out annotations for the outer labels.
        ``annotation_distance`` controls how far the label is rendered from the
        chart centre, while ``annotation_mid_radius`` governs the anchor point
        along the wedge arc.
    center_text / center_text_kwargs:
        Optional text rendered at the donut hole centre.
    equal_aspect:
        Keep axes equal for a circular chart. Disable for manual scaling.
    normalize_outer:
        Automatically scale ``outer_values`` so each row sums to the supplied
        ``inner_values`` (or their inferred totals). Disable to raise a
        ``ValueError`` when the numbers diverge.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the donut chart for further customisation.
    """

    outer_array = np.asarray(outer_values, dtype=float)
    if outer_array.ndim != 2:
        raise ValueError("outer_values must be a 2D sequence of numeric data")

    if inner_values is None:
        inner_array = outer_array.sum(axis=1)
    else:
        inner_array = np.asarray(inner_values, dtype=float)
        if inner_array.ndim != 1:
            raise ValueError("inner_values must be a 1D sequence of numeric data")
        if inner_array.shape[0] != outer_array.shape[0]:
            raise ValueError(
                "inner_values must contain one value for each row in outer_values"
            )
        row_totals = outer_array.sum(axis=1)
        mismatch = ~np.isclose(row_totals, inner_array, rtol=1e-5, atol=1e-8)
        if np.any(mismatch):
            if normalize_outer:
                scale = np.divide(
                    inner_array,
                    np.where(row_totals == 0, 1, row_totals),
                    out=np.ones_like(inner_array),
                    where=row_totals != 0,
                )
                outer_array = outer_array * scale[:, None]
            else:
                mismatch_rows = ", ".join(str(idx) for idx in np.where(mismatch)[0])
                raise ValueError(
                    "outer_values row sums differ from inner_values for rows: "
                    f"{mismatch_rows}. Set normalize_outer=True to adjust automatically."
                )

    axis = _prepare_axes(ax, use_style)

    if inner_labels is not None and len(inner_labels) != inner_array.shape[0]:
        raise ValueError("inner_labels length must match the number of inner segments")

    flat_outer = outer_array.flatten()
    if outer_labels is not None and len(outer_labels) != flat_outer.shape[0]:
        raise ValueError("outer_labels length must match the flattened outer_values")
    if outer_labels is None and outer_label_formatter is not None:
        outer_labels = [outer_label_formatter(idx, value) for idx, value in enumerate(flat_outer)]
    if outer_labels is None:
        outer_labels = [None] * flat_outer.shape[0]

    # Fallback colour palettes inspired by Matplotlib's tab20c cycle.
    if inner_colors is None:
        cmap = plt.get_cmap("tab20c")
        inner_colors = [cmap(i) for i in np.linspace(0.0, 0.9, inner_array.shape[0])]
    if outer_colors is None:
        cmap = plt.get_cmap("tab20c")
        outer_colors = [cmap((i + 0.5) % 1.0) for i in np.linspace(0.0, 0.95, flat_outer.shape[0])]

    inner_radius = max(radius - width, 0)

    default_inner_wedgeprops = {"width": width, "edgecolor": "white"}
    if inner_wedgeprops:
        default_inner_wedgeprops.update(inner_wedgeprops)

    inner_kwargs: dict[str, Any] = {
        "radius": inner_radius,
        "colors": inner_colors,
        "startangle": startangle,
        "wedgeprops": default_inner_wedgeprops,
    }
    if inner_pie_kwargs:
        inner_kwargs.update(inner_pie_kwargs)
    if inner_autopct:
        inner_kwargs["autopct"] = inner_autopct

    default_textprops = {"color": "white", "fontsize": 9}
    if inner_textprops:
        default_textprops.update(inner_textprops)

    if "textprops" in inner_kwargs and isinstance(inner_kwargs["textprops"], dict):
        if inner_textprops:
            merged = {**inner_kwargs["textprops"], **inner_textprops}
            inner_kwargs["textprops"] = merged
    else:
        inner_kwargs.setdefault("textprops", default_textprops)

    inner_kwargs.setdefault("pctdistance", 0.75)

    inner_pie_result = axis.pie(inner_array, labels=inner_labels, **inner_kwargs)
    if inner_autopct:
        _, _, inner_autotexts = inner_pie_result
        for autotext in inner_autotexts:
            autotext.set_va("center")

    default_outer_wedgeprops = {"width": width, "edgecolor": "white"}
    if outer_wedgeprops:
        default_outer_wedgeprops.update(outer_wedgeprops)

    outer_kwargs: dict[str, Any] = {
        "radius": radius,
        "colors": outer_colors,
        "startangle": startangle,
        "wedgeprops": default_outer_wedgeprops,
    }
    if outer_pie_kwargs:
        outer_kwargs.update(outer_pie_kwargs)
    outer_kwargs.setdefault("pctdistance", 0.85)

    wedges, _ = axis.pie(flat_outer, **outer_kwargs)

    if annotate_outer and any(label for label in outer_labels):
        default_annotation_kwargs: dict[str, Any] = {
            "arrowprops": {"arrowstyle": "-", "color": "0.25", "linewidth": 0.8},
            "bbox": {"boxstyle": "round,pad=0.5", "fc": "white", "ec": "0.2", "lw": 0.5},
            "fontsize": 9,
            "va": "center",
        }
        if annotation_kwargs:
            # Allow users to override nested dicts like arrowprops/bbox.
            for key, value in annotation_kwargs.items():
                if key in {"arrowprops", "bbox"} and key in default_annotation_kwargs:
                    merged = {**default_annotation_kwargs[key], **value}
                    default_annotation_kwargs[key] = merged
                else:
                    default_annotation_kwargs[key] = value

        mid_radius = annotation_mid_radius if annotation_mid_radius is not None else radius - (width / 2)

        for idx, (wedge, label) in enumerate(zip(wedges, outer_labels)):
            if not label or wedge.theta1 == wedge.theta2:
                continue

            angle = (wedge.theta2 + wedge.theta1) / 2.0
            angle_rad = np.deg2rad(angle)
            x = np.cos(angle_rad)
            y = np.sin(angle_rad)
            x_mid = mid_radius * x
            y_mid = mid_radius * y

            # Place tezt annotation horizontally offset
            x_text = annotation_distance * np.sign(x)
            y_text = annotation_distance * y

            horizontalalignment = "left" if x >= 0 else "right"
            ha_override = default_annotation_kwargs.get("horizontalalignment") or default_annotation_kwargs.get("ha")
            if ha_override is not None:
                horizontalalignment = ha_override

            # Build a per-wedge copy of annotation kwargs so we can safely tweak
            # nested dicts (like ``arrowprops``) without mutating the defaults.
            ann_kwargs = {}
            for key, value in default_annotation_kwargs.items():
                if key in {"ha", "horizontalalignment"}:
                    continue
                # copy nested dicts to avoid side-effects
                if isinstance(value, dict):
                    ann_kwargs[key] = {**value}
                else:
                    ann_kwargs[key] = value

            # Add a connectionstyle specific to this wedge so the arrow follows
            connectionstyle = f"angle,angleA=0,angleB={angle}"
            if "arrowprops" in ann_kwargs and isinstance(ann_kwargs["arrowprops"], dict):
                ann_kwargs["arrowprops"].update({"connectionstyle": connectionstyle})
            else:
                ann_kwargs["arrowprops"] = {"connectionstyle": connectionstyle}

            axis.annotate(
                label,
                xy=(x_mid, y_mid),
                xytext=(x_text, y_text),
                horizontalalignment=horizontalalignment,
                **ann_kwargs,
            )

    if center_text:
        text_kwargs = {"ha": "center", "va": "center", "fontsize": 10, "fontweight": "bold"}
        if center_text_kwargs:
            text_kwargs.update(center_text_kwargs)
        axis.text(0, 0, center_text, **text_kwargs)

    if equal_aspect:
        axis.set_aspect("equal")

    return axis

"""
High-level plotting helpers built on Matplotlib, Seaborn, and Plotly.

The mattplotlib package provides a consistent styling layer, a small set of
beginner-friendly plotting functions, and configuration utilities so you can
quickly produce publication-ready graphics across backends.
"""

from . import matplotlib_plots, plotly_plots, seaborn_plots
from .config import MattplotlibConfig, get_config, load_config
from .style import apply_mpl_style, get_style_path

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

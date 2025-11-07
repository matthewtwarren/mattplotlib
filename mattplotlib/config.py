"""
Configuration helpers for the mattplotlib plotting toolkit.

The :class:`MattplotlibConfig` dataclass captures the shared styling choices
used across Matplotlib, Seaborn, and Plotly helpers. Configuration can be
loaded from a TOML file (see ``examples/mattplotlib.toml``) or constructed in
memory via :func:`load_config`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import tomllib  # type: ignore[import-not-found]  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        tomllib = None  # type: ignore


_CONFIG: MattplotlibConfig | None = None
_ENV_CONFIG_VAR = "MATTPLIB_CONFIG"


@dataclass(slots=True)
class MattplotlibConfig:
    """Container for style and backend defaults."""

    font_family: str = "Andale Mono"
    color_cycle: tuple[str, ...] = (
        "#1B9E77",
        "#D95F02",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#E6AB02",
        "#A90E0E",
        "#559EE6",
        "#6D4444",
        "#9A9A9A",
    )
    palettes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    seaborn_theme: Optional[Dict[str, Any]] = None
    plotly_template: str = "plotly_white"
    matplotlib_style: str = str(
        Path(__file__).resolve().parent / "styles" / "mattplotlib.mplstyle"
    )

    def __post_init__(self) -> None:
        theme: Any = self.seaborn_theme

        if theme is None:
            theme = {}
        elif isinstance(theme, str):
            theme = {"style": theme}
        elif isinstance(theme, MappingABC):
            theme = dict(theme)
        else:
            raise TypeError(
                "seaborn_theme must be a mapping, string, or None. "
                f"Received {type(theme).__name__}."
            )

        theme.setdefault("style", "whitegrid")
        theme.setdefault("context", "talk")
        theme.setdefault("palette", list(self.color_cycle))
        theme.setdefault("font", self.font_family)

        self.seaborn_theme = theme

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the config."""
        return {
            "font_family": self.font_family,
            "color_cycle": list(self.color_cycle),
            "palettes": {k: list(v) for k, v in self.palettes.items()},
            "seaborn_theme": self.seaborn_theme,
            "plotly_template": self.plotly_template,
            "matplotlib_style": self.matplotlib_style,
        }


def _coerce_value(name: str, value: Any) -> Any:
    if name == "color_cycle" and isinstance(value, list):
        return tuple(value)
    if name == "palettes" and isinstance(value, dict):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in value.items()}
    if name == "matplotlib_style":
        return str(Path(value).expanduser())
    return value


def _build_config(data: Mapping[str, Any]) -> MattplotlibConfig:
    allowed = {field.name for field in fields(MattplotlibConfig)}
    filtered: Dict[str, Any] = {}
    for key, value in data.items():
        if key in allowed:
            filtered[key] = _coerce_value(key, value)
    return MattplotlibConfig(**filtered)


def load_config(
    path: Optional[str] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> MattplotlibConfig:
    """
    Load configuration from a TOML file and cache it globally.

    Parameters
    ----------
    path:
        Path to a TOML configuration file. If omitted, the function will look
        for the ``MATTPLIB_CONFIG`` environment variable and fall back to the
        default configuration when nothing is provided.
    overrides:
        Explicit overrides that should win over file-based settings.
    """

    global _CONFIG

    data: Dict[str, Any] = {}

    config_path = _resolve_config_path(path)
    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at '{config_path}'. "
                "Update your mattplotlib settings or drop the path argument."
            )
        if tomllib is None:
            raise RuntimeError(
                "Loading TOML configuration requires Python 3.11+ or the tomli package."
            )
        with config_path.open("rb") as fh:
            file_data = tomllib.load(fh)
        data.update(file_data.get("mattplotlib", file_data))

    if overrides:
        data.update(dict(overrides))

    _CONFIG = _build_config(data)
    return _CONFIG


def get_config() -> MattplotlibConfig:
    """
    Return the cached configuration, loading it on first use.

    The loader honours the ``MATTPLIB_CONFIG`` environment variable when the
    config has not already been initialised.
    """

    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def get_palette(name: str) -> tuple[str, ...]:
    """Return a named palette from the loaded configuration.

    Raises KeyError if the palette name is unknown.
    """
    cfg = get_config()
    try:
        return cfg.palettes[name]
    except KeyError as exc:  # pragma: no cover - simple passthrough
        available = ", ".join(sorted(cfg.palettes)) or "<none>"
        raise KeyError(f"Palette '{name}' not found. Available: {available}.") from exc


def _env_config_path() -> Optional[Path]:
    env_value = os.getenv(_ENV_CONFIG_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return None


def _resolve_config_path(path: Optional[str]) -> Optional[Path]:
    if path:
        return Path(path).expanduser()
    return _env_config_path()

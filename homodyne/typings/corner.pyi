"""
Type stubs for corner plotting library.
"""

from typing import Any

import numpy as np
from matplotlib.figure import Figure

def corner(
    xs: np.ndarray,
    bins: int | list[int] = ...,
    range: list[tuple[float, float]] | None = ...,
    weights: np.ndarray | None = ...,
    color: str = ...,
    smooth: float | None = ...,
    smooth1d: float | None = ...,
    labels: list[str] | None = ...,
    label_kwargs: dict[str, Any] | None = ...,
    titles: list[str] | None = ...,
    show_titles: bool = ...,
    title_fmt: str = ...,
    title_kwargs: dict[str, Any] | None = ...,
    truths: list[float] | None = ...,
    truth_color: str = ...,
    scale_hist: bool = ...,
    quantiles: list[float] | None = ...,
    verbose: bool = ...,
    fig: Figure | None = ...,
    max_n_ticks: int = ...,
    top_ticks: bool = ...,
    use_math_text: bool = ...,
    reverse: bool = ...,
    labelpad: float = ...,
    hist_kwargs: dict[str, Any] | None = ...,
    **hist2d_kwargs: Any,
) -> Figure: ...

"""
Type stubs for corner plotting library.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def corner(
    xs: np.ndarray,
    bins: Union[int, List[int]] = ...,
    range: Optional[List[Tuple[float, float]]] = ...,
    weights: Optional[np.ndarray] = ...,
    color: str = ...,
    smooth: Optional[float] = ...,
    smooth1d: Optional[float] = ...,
    labels: Optional[List[str]] = ...,
    label_kwargs: Optional[Dict[str, Any]] = ...,
    titles: Optional[List[str]] = ...,
    show_titles: bool = ...,
    title_fmt: str = ...,
    title_kwargs: Optional[Dict[str, Any]] = ...,
    truths: Optional[List[float]] = ...,
    truth_color: str = ...,
    scale_hist: bool = ...,
    quantiles: Optional[List[float]] = ...,
    verbose: bool = ...,
    fig: Optional[Figure] = ...,
    max_n_ticks: int = ...,
    top_ticks: bool = ...,
    use_math_text: bool = ...,
    reverse: bool = ...,
    labelpad: float = ...,
    hist_kwargs: Optional[Dict[str, Any]] = ...,
    **hist2d_kwargs: Any,
) -> Figure: ...

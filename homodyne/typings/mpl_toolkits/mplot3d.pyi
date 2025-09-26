"""
Type stubs for mpl_toolkits.mplot3d 3D plotting.
"""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class Axes3D(Axes):
    def __init__(
        self, fig: Figure, rect: Any = ..., *args: Any, **kwargs: Any
    ) -> None: ...
    def plot_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        rstride: int = ...,
        cstride: int = ...,
        color: str | None = ...,
        cmap: str | None = ...,
        facecolors: np.ndarray | None = ...,
        norm: Any | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        shade: bool = ...,
        alpha: float | None = ...,
        **kwargs: Any,
    ) -> Any: ...
    def plot_wireframe(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        rstride: int = ...,
        cstride: int = ...,
        color: str | None = ...,
        **kwargs: Any,
    ) -> Any: ...
    def scatter(  # type: ignore[override]
        self,
        xs: np.ndarray | list[float],
        ys: np.ndarray | list[float],
        zs: np.ndarray | list[float],
        zdir: str = ...,
        s: float | np.ndarray = ...,
        c: str | np.ndarray = ...,
        depthshade: bool = ...,
        **kwargs: Any,
    ) -> Any: ...
    def set_xlabel(self, xlabel: str, **kwargs: Any) -> None: ...  # type: ignore[override]
    def set_ylabel(self, ylabel: str, **kwargs: Any) -> None: ...  # type: ignore[override]
    def set_zlabel(self, zlabel: str, **kwargs: Any) -> None: ...

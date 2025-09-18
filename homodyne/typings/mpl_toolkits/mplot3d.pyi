"""
Type stubs for mpl_toolkits.mplot3d 3D plotting.
"""

from typing import Any, List, Optional, Union

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
        color: Optional[str] = ...,
        cmap: Optional[str] = ...,
        facecolors: Optional[np.ndarray] = ...,
        norm: Optional[Any] = ...,
        vmin: Optional[float] = ...,
        vmax: Optional[float] = ...,
        shade: bool = ...,
        alpha: Optional[float] = ...,
        **kwargs: Any,
    ) -> Any: ...
    def plot_wireframe(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        rstride: int = ...,
        cstride: int = ...,
        color: Optional[str] = ...,
        **kwargs: Any,
    ) -> Any: ...
    def scatter(
        self,
        xs: Union[np.ndarray, List[float]],
        ys: Union[np.ndarray, List[float]],
        zs: Union[np.ndarray, List[float]],
        zdir: str = ...,
        s: Union[float, np.ndarray] = ...,
        c: Union[str, np.ndarray] = ...,
        depthshade: bool = ...,
        **kwargs: Any,
    ) -> Any: ...
    def set_xlabel(self, xlabel: str, **kwargs: Any) -> None: ...
    def set_ylabel(self, ylabel: str, **kwargs: Any) -> None: ...
    def set_zlabel(self, zlabel: str, **kwargs: Any) -> None: ...

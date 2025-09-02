# sklearn.utils type stubs
from typing import Any

from numpy.typing import NDArray

def resample(
    *arrays: NDArray[Any],
    replace: bool = True,
    n_samples: int | None = None,
    random_state: int | None = None,
    stratify: NDArray[Any] | None = None
) -> tuple[NDArray[Any], ...]: ...

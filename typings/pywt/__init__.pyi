"""Type stub for PyWavelets, which ships none.

It declares only the names that mosaic uses: ``ContinuousWavelet``,
``central_frequency`` and ``cwt``. The signatures follow PyWavelets 1.8.
"""

import numpy as np
import numpy.typing as npt

class ContinuousWavelet:
    def __init__(self, name: str, dtype: npt.DTypeLike = ...) -> None: ...

def central_frequency(
    wavelet: ContinuousWavelet | str, precision: int = 8
) -> float: ...
def cwt(
    data: npt.ArrayLike,
    scales: npt.ArrayLike,
    wavelet: ContinuousWavelet | str,
    sampling_period: float = 1.0,
    method: str = "conv",
    axis: int = -1,
    *,
    precision: int = 12,
) -> tuple[npt.NDArray[np.float64 | np.complex128], npt.NDArray[np.float64]]: ...

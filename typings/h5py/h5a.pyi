"""The attributes of h5py's low-level API that mosaic uses."""

import numpy as np

class AttrID:
    dtype: np.dtype[np.generic]
    shape: tuple[int, ...] | None

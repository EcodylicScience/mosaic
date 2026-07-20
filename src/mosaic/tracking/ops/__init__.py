"""Built-in tracking ops. Importing this package registers them.

Kept import-light: op modules import only their ``Params`` and light deps at
module top; heavy backends (ultralytics / torch / POLO / TREx) load lazily inside
each op's ``run()``. The ``extract-frames`` op registers via
``frame_extraction/dataset_runs.py`` (imported by ``mosaic.tracking``); this
package registers the training, inference, and TREx ops.
"""

from mosaic.tracking.ops import convert, infer, train, trex  # noqa: F401

__all__ = ["convert", "infer", "train", "trex"]

"""Field prose shared by every feature taking an optional nearest-neighbor filter.

``ExtractTemplates``, ``GlobalKMeansClustering``, ``GlobalWardClustering`` and
``TemporalStackingFeature`` each accept a ``pair_filter: NNResult | None`` field
that narrows the loaded input to nearest-neighbor pairs the same way. The
text lives in one place rather than four.
"""

from __future__ import annotations

PAIR_FILTER_DESCRIPTION = (
    "Unset, every row is read. A nearest-neighbor result narrows the "
    "input, while it loads, to rows where one individual in the pair "
    "is the other's nearest neighbor. On an input without id1/id2 "
    "columns, the filter has no effect."
)

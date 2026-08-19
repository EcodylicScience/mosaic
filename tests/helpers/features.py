"""Template and per-sequence frames for the global fit-then-apply features.

The four global-feature suites (`global-scaler`, `global-tsne`, `global-kmeans`,
`global-ward`) each fit on a templates table and then apply to a per-sequence
frame, so they each wrote the same two builders. The scaler's pair carried
`offset` / `scale` so its assertions could name the mean and standard deviation
they expect back; the other three drew a plain standard normal, which is the
same builder at `offset=0.0, scale=1.0`.

**The seeds are load-bearing.** A clustering or embedding assertion is made
against the numbers these frames hold, not against their shape, so changing a
seed, a draw order or a column order silently changes what the suites test. The
templates seed (42) and the per-sequence seed (99) are fixed here for that
reason, and each feature column is one draw taken in column order, so widening
`n_features` leaves the earlier columns untouched.

`n_rows` and `n_features` are required rather than defaulted: the four suites
disagreed on both defaults, so any default here would read as authoritative
while matching only some of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_templates(
    n_rows: int,
    n_features: int,
    *,
    offset: float = 0.0,
    scale: float = 1.0,
) -> pd.DataFrame:
    """A templates table of `feat_0 .. feat_<n_features-1>`.

    Args:
        n_rows: How many template vectors to draw.
        n_features: How many feature columns each vector holds.
        offset: Added to every drawn value, so a test can assert on a known
            mean.
        scale: Multiplies every drawn value, so a test can assert on a known
            standard deviation.

    Returns:
        The templates, one column per feature and no metadata columns.
    """
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows) * scale + offset
    return pd.DataFrame(data)


def make_sequence_df(
    n_rows: int,
    n_features: int,
    *,
    offset: float = 0.0,
    scale: float = 1.0,
    sequence: str = "seq_a",
    group: str = "grp_a",
) -> pd.DataFrame:
    """One sequence's frame: the metadata columns, then the feature columns.

    Args:
        n_rows: How many frames the sequence holds.
        n_features: How many feature columns each frame holds.
        offset: Added to every drawn feature value.
        scale: Multiplies every drawn feature value.
        sequence: The sequence name written into every row.
        group: The group name written into every row.

    Returns:
        The frame, with `frame`, `time`, `id`, `group` and `sequence` ahead of
        the feature columns -- the order a feature's `apply` is asserted to
        preserve.
    """
    rng = np.random.default_rng(99)
    data: dict[str, object] = {
        "frame": np.arange(n_rows),
        "time": np.arange(n_rows, dtype=float) / 30.0,
        "id": np.zeros(n_rows, dtype=int),
        "group": [group] * n_rows,
        "sequence": [sequence] * n_rows,
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows) * scale + offset
    return pd.DataFrame(data)


def write_templates(tmp_path: Path, templates: pd.DataFrame) -> Path:
    """Write *templates* where a feature's templates artifact resolves it.

    The directory stands in for the producing feature's run root, which is what
    makes the `templates.parquet` pattern the artifact declares resolvable.

    A per-entry sibling is written beside it, because a real run root holds one
    output parquet per sequence and a directory holding only the named artifact
    is the one arrangement in which resolving by glob cannot go wrong. A fixture
    that cannot reproduce the failure cannot notice it either.

    Args:
        tmp_path: The test's temporary directory. Must not already hold a
            `templates_run` directory.
        templates: The table to write.

    Returns:
        The path of the written parquet.
    """
    template_dir = tmp_path / "templates_run"
    template_dir.mkdir()
    path = template_dir / "templates.parquet"
    templates.to_parquet(path, index=False)
    make_sequence_df(n_rows=4, n_features=templates.shape[1]).to_parquet(
        template_dir / "seq_a.parquet", index=False
    )
    return path

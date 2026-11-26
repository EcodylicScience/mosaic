"""The ``lmt_sqlite`` ``id_tags`` converter: a database's animals as individual tags.

A Live Mouse Tracker database is one recording, and the ``lmt_sqlite`` format has
two label converters, this one and the event converter that writes ``behavior``
labels. Both read the file through
:func:`~mosaic.core.track_library.lmt.open_lmt_database`, which opens it
``mode=ro&immutable=1``, so a conversion never changes the database, and a file with
a journal beside it is refused.

The ``id_tags`` converter returns one entry per database: the ``ANIMAL`` table as
the ``ids`` array, the ascending ``ANIMAL.ID``, and one text array per column present
among ``RFID``, ``NAME``, ``GENOTYPE``, ``AGE``, ``SEX``, ``STRAIN``, ``SETUP``,
``TREATMENT`` and ``IND``, named in lower case. Every cell is the ``str`` of its
value whatever SQLite stored, so an RFID keeps its leading zeros, and a NULL is the
empty string. The arrays load without unpickling. The entry is the raw row's group
under the conversion's ``group_from`` policy and the raw row's sequence, the entry
that the tracks of the same file convert to.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np

from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
    entry_group,
)
from mosaic.core.track_library.lmt import open_lmt_database

__all__ = ["LmtAnimalsConverter"]

_FIELD_COLUMNS: Final = (
    "RFID",
    "NAME",
    "GENOTYPE",
    "AGE",
    "SEX",
    "STRAIN",
    "SETUP",
    "TREATMENT",
    "IND",
)


class LmtAnimalsConverter(LabelConverter[LabelConvertParams]):
    """A Live Mouse Tracker database -> its animals as individual tags."""

    src_format = "lmt_sqlite"
    label_kind = "id_tags"
    label_format = "id_tags_v1"
    version = "0.1"
    Params = LabelConvertParams
    refuses_shared_entry_keys = True

    def convert(
        self,
        src_path: Path,
        params: LabelConvertParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read ``ANIMAL`` into one :class:`LabelEntry`.

        Raises:
            ValueError: The database has no ``ANIMAL`` table or no ``ID`` column, or
                a journal sits beside it. The message names the file.
        """
        connection = open_lmt_database(src_path)
        try:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(ANIMAL)")
            }
            if not columns:
                raise ValueError(f"{src_path}: the database has no ANIMAL table")
            if "ID" not in columns:
                raise ValueError(f"{src_path}: the ANIMAL table has no ID column")
            present = [name for name in _FIELD_COLUMNS if name in columns]
            selected = ", ".join(["ID", *present])
            rows = connection.execute(
                f"SELECT {selected} FROM ANIMAL ORDER BY ID"
            ).fetchall()
        finally:
            connection.close()

        ids = np.array([row[0] for row in rows], dtype=np.int64)
        payload: dict[str, object] = {"ids": ids}
        for position, name in enumerate(present, start=1):
            payload[name.lower()] = np.array(
                [("" if row[position] is None else str(row[position])) for row in rows],
                dtype=np.str_,
            )
        return [
            LabelEntry(
                group=entry_group(params.group_from, raw_row, ""),
                sequence=str(raw_row.get("sequence", "") or ""),
                payload=payload,
                n_frames=len(rows),
                label_ids=tuple(int(value) for value in ids),
                label_names=tuple(sorted(key for key in payload if key != "ids")),
            )
        ]

"""Hash a dataset's outputs and index rows, to compare two runs of the bench.

Usage: fingerprint.py DATASET_DIR

Prints one line per file: the SHA-256 of every parquet file under ``tracks/``
and ``features/``, and of every ``index.csv`` there with its rows sorted and
its ``started_at`` and ``finished_at`` columns dropped. The last line is the
hash of all the lines before it. Two bench runs over the same work directory
compare equal when their outputs and index rows are byte-identical.
"""

import csv
import hashlib
import sys
from pathlib import Path

TIME_COLUMNS = frozenset({"started_at", "finished_at"})


def _index_digest(path: Path) -> str:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return hashlib.sha256(b"").hexdigest()
    header = rows[0]
    keep = [i for i, name in enumerate(header) if name not in TIME_COLUMNS]
    lines = sorted(",".join(row[i] for i in keep if i < len(row)) for row in rows[1:])
    text = ",".join(header[i] for i in keep) + "\n" + "\n".join(lines)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    root = Path(sys.argv[1])
    lines: list[str] = []
    for top in ("tracks", "features"):
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.suffix == ".parquet":
                lines.append(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}"
                )
            elif path.name == "index.csv":
                lines.append(f"{_index_digest(path)}  {relative} (rows)")
    total = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    print("\n".join(lines))
    print(f"{total}  total over {len(lines)} files")


if __name__ == "__main__":
    main()

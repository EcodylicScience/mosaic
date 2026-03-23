from __future__ import annotations

import gc
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._utils import ChunkedPayload, DataPayload, FeatureMeta, StreamPayload

FeatureOutput = StreamPayload | ChunkedPayload | DataPayload | pd.DataFrame | None


# --- Trimming ---


def trim_feature_output(
    df_feat: FeatureOutput,
    core_start: int,
    core_end: int,
) -> FeatureOutput:
    """Trim feature output to original segment bounds (removing overlap regions)."""
    if df_feat is None:
        return df_feat

    if isinstance(df_feat, ChunkedPayload):
        if core_start == 0 and core_end >= df_feat.parquet_data.shape[0]:
            return df_feat
        df_feat.parquet_data = df_feat.parquet_data[core_start:core_end]
        return df_feat
    if isinstance(df_feat, DataPayload):
        if core_start == 0 and core_end >= df_feat.data.shape[0]:
            return df_feat
        df_feat.data = df_feat.data[core_start:core_end]
        return df_feat
    if isinstance(df_feat, StreamPayload):
        print(
            "warning: overlap trimming not supported for StreamPayload outputs",
            file=sys.stderr,
        )
        return df_feat

    # pd.DataFrame
    if core_start == 0 and core_end >= len(df_feat):
        return df_feat
    return df_feat.iloc[core_start:core_end].reset_index(drop=True)


# --- Parquet writing ---


def _write_parquet_chunks(meta: FeatureMeta, payload: ChunkedPayload) -> int:
    """Write a ChunkedPayload as a parquet file in row-chunks. Returns n_rows."""
    data = payload.parquet_data
    columns = payload.columns
    sequence = payload.sequence
    group = payload.group
    chunk_size = max(1, payload.chunk_size)
    n_rows = data.shape[0]
    schema_fields: list[tuple[str, object]] = [("frame", pa.int32())]
    schema_fields.extend([(name, pa.float32()) for name in columns])
    schema_fields.append(("sequence", pa.string()))
    if group:
        schema_fields.append(("group", pa.string()))
    schema = pa.schema(schema_fields)
    writer = pq.ParquetWriter(meta.out_path, schema, compression="snappy")
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        arrays = {"frame": pa.array(np.arange(start, end, dtype=np.int32))}
        for idx, name in enumerate(columns):
            arrays[name] = pa.array(data[start:end, idx])
        arrays["sequence"] = pa.array([sequence] * (end - start))
        if group:
            arrays["group"] = pa.array([str(group)] * (end - start))
        table = pa.Table.from_pydict(arrays, schema=schema)
        writer.write_table(table)
    writer.close()
    gc.collect()
    return n_rows


def _write_parquet_stream(meta: FeatureMeta, payload: StreamPayload) -> int:
    """Write a StreamPayload as a streaming parquet file. Returns total_rows."""
    columns = payload.columns
    sequence = payload.sequence
    group = payload.group
    chunk_iter = payload.parquet_chunk_iter
    pair_ids = payload.pair_ids
    schema_fields: list[tuple[str, object]] = [("frame", pa.int32())]
    schema_fields.extend([(name, pa.float32()) for name in columns])
    if pair_ids is not None:
        schema_fields.append(("id1", pa.int32()))
        schema_fields.append(("id2", pa.int32()))
    schema_fields.append(("sequence", pa.string()))
    if group:
        schema_fields.append(("group", pa.string()))
    schema = pa.schema(schema_fields)
    writer = pq.ParquetWriter(meta.out_path, schema, compression="snappy")
    total_rows = 0
    source_frame_indices = payload.frame_indices
    for start, chunk in chunk_iter:
        chunk_len = chunk.shape[0]
        if source_frame_indices is not None:
            frame_arr = source_frame_indices[start : start + chunk_len]
        else:
            frame_arr = np.arange(start, start + chunk_len, dtype=np.int32)
        arrays = {"frame": pa.array(frame_arr)}
        for idx, name in enumerate(columns):
            arrays[name] = pa.array(chunk[:, idx])
        if pair_ids is not None:
            arrays["id1"] = pa.array(np.full(chunk_len, pair_ids[0], dtype=np.int32))
            arrays["id2"] = pa.array(np.full(chunk_len, pair_ids[1], dtype=np.int32))
        arrays["sequence"] = pa.array([sequence] * chunk_len)
        if group:
            arrays["group"] = pa.array([str(group)] * chunk_len)
        table = pa.Table.from_pydict(arrays, schema=schema)
        writer.write_table(table)
        total_rows = max(total_rows, start + chunk_len)
    writer.close()
    gc.collect()
    return total_rows


def write_output(
    meta: FeatureMeta,
    df_feat: FeatureOutput,
) -> int:
    """Write feature output to parquet. Returns n_rows written."""
    out_path = meta.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df_feat, StreamPayload):
        return _write_parquet_stream(meta, df_feat)

    if isinstance(df_feat, ChunkedPayload):
        return _write_parquet_chunks(meta, df_feat)

    df_out: pd.DataFrame
    if isinstance(df_feat, DataPayload):
        df_out = pd.DataFrame(df_feat.data, columns=df_feat.columns)
        fi = df_feat.frame_indices
        if fi is not None and len(fi) == df_out.shape[0]:
            df_out.insert(0, "frame", fi.astype(np.int32))
        else:
            df_out.insert(0, "frame", np.arange(df_out.shape[0], dtype=np.int32))
        ppr = df_feat.pair_ids_per_row
        if ppr is not None and len(ppr) == df_out.shape[0]:
            df_out["id1"] = ppr[:, 0].astype(np.int32)
            df_out["id2"] = ppr[:, 1].astype(np.int32)
        if df_feat.sequence is not None and "sequence" not in df_out.columns:
            df_out["sequence"] = df_feat.sequence
        if df_feat.group is not None and "group" not in df_out.columns:
            df_out["group"] = df_feat.group
    elif isinstance(df_feat, pd.DataFrame):
        df_out = df_feat
    else:
        df_out = pd.DataFrame()

    n_rows = len(df_out)
    df_out.to_parquet(out_path, index=False)
    gc.collect()
    return n_rows

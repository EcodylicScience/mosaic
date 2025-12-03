from urllib.parse import quote, unquote
import numpy as np
import pandas as pd

def to_safe_name(s: str) -> str:
    # encode EVERYTHING that could be problematic on any OS
    return quote(s, safe="")   # e.g. "task1/train/m1" -> "task1%2Ftrain%2Fm1"

def from_safe_name(safe: str) -> str:
    return unquote(safe)


def chunk_sequence(df: pd.DataFrame,
                   time_chunk_sec: float | None = None,
                   frame_chunk: int | None = None):
    """
    Yield (chunk_id, df_chunk, meta) from a per-sequence DataFrame.
    If time_chunk_sec is provided and 'time' exists, chunk by time.
    Else if frame_chunk is provided and 'frame' exists, chunk by frame.
    Else yield the whole sequence as a single chunk.
    meta contains start/end frame/time if available.
    """
    frame_key = "frame" if "frame" in df.columns else None
    time_key = "time" if "time" in df.columns else None

    if time_chunk_sec and time_key in df.columns:
        starts = np.arange(df[time_key].min(), df[time_key].max() + time_chunk_sec, time_chunk_sec)
        for idx, start in enumerate(starts):
            end = start + time_chunk_sec
            mask = (df[time_key] >= start) & (df[time_key] < end)
            sub = df[mask]
            if sub.empty:
                continue
            yield idx, sub, {
                "start_time": float(start),
                "end_time": float(end),
                "start_frame": int(sub[frame_key].iloc[0]) if frame_key else None,
                "end_frame": int(sub[frame_key].iloc[-1]) if frame_key else None,
            }
    elif frame_chunk and frame_key in df.columns:
        frames = df[frame_key].to_numpy()
        start_frame = frames.min()
        end_frame = frames.max()
        for idx, start in enumerate(range(start_frame, end_frame + 1, int(frame_chunk))):
            end = start + int(frame_chunk)
            mask = (df[frame_key] >= start) & (df[frame_key] < end)
            sub = df[mask]
            if sub.empty:
                continue
            yield idx, sub, {
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time": float(sub[time_key].iloc[0]) if time_key else None,
                "end_time": float(sub[time_key].iloc[-1]) if time_key else None,
            }
    else:
        meta = {}
        if frame_key:
            meta["start_frame"] = int(df[frame_key].iloc[0])
            meta["end_frame"] = int(df[frame_key].iloc[-1])
        if time_key:
            meta["start_time"] = float(df[time_key].iloc[0])
            meta["end_time"] = float(df[time_key].iloc[-1])
        yield 0, df, meta

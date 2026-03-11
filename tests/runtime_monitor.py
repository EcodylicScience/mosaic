"""
Runtime monitor for mosaic parallel feature extraction.

Usage:
    from runtime_monitor import ResourceMonitor, benchmark_workers

    with ResourceMonitor() as mon:
        ds.run_feature(feat, parallel_workers=4, parallel_mode="process")

    mon.summary()
    mon.plot()

    # Or benchmark different configurations:
    results = benchmark_workers(ds, feat, worker_counts=[1,2,4,8], modes=["process","thread"])

Requires: pip install psutil
"""

from __future__ import annotations

import gc
import os
import threading
import time
from typing import Optional, Sequence

try:
    import psutil
except ImportError:
    raise ImportError(
        "psutil is required for runtime_monitor. Install it with: pip install psutil"
    )

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ResourceMonitor:
    """Context manager that samples CPU, memory, I/O, and worker metrics
    in a background thread during a parallel workload."""

    def __init__(self, sample_interval: float = 0.5, track_children: bool = True):
        self._interval = sample_interval
        self._track_children = track_children
        self._samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: float = 0.0
        self._wall_time: Optional[float] = None
        self._n_cores = psutil.cpu_count(logical=True) or 1

    # -- context manager --

    def __enter__(self):
        self._samples.clear()
        self._wall_time = None
        self._t0 = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sampler, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._wall_time = time.monotonic() - self._t0
        return False

    # -- sampler --

    def _sampler(self):
        proc = psutil.Process(os.getpid())
        # Prime cpu_percent counters (first call always returns 0)
        proc.cpu_percent()
        if self._track_children:
            for c in proc.children(recursive=True):
                try:
                    c.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        psutil.cpu_percent()  # prime system-wide counter
        self._stop_event.wait(self._interval)

        while not self._stop_event.is_set():
            sample: dict = {}
            sample["elapsed"] = time.monotonic() - self._t0

            # Parent process
            try:
                sample["parent_cpu"] = proc.cpu_percent()
                mem = proc.memory_info()
                sample["parent_rss_mb"] = mem.rss / (1024 * 1024)
                sample["parent_threads"] = proc.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                sample["parent_cpu"] = 0.0
                sample["parent_rss_mb"] = 0.0
                sample["parent_threads"] = 0

            # Children (subprocess workers)
            child_cpu = 0.0
            child_rss = 0.0
            n_children = 0
            if self._track_children:
                try:
                    children = proc.children(recursive=True)
                    n_children = len(children)
                    for c in children:
                        try:
                            child_cpu += c.cpu_percent()
                            child_rss += c.memory_info().rss / (1024 * 1024)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            sample["n_children"] = n_children
            sample["children_cpu"] = child_cpu
            sample["children_rss_mb"] = child_rss
            sample["total_cpu"] = sample["parent_cpu"] + child_cpu
            sample["total_rss_mb"] = sample["parent_rss_mb"] + child_rss

            # System-wide
            sample["system_cpu"] = psutil.cpu_percent()
            sample["system_mem_pct"] = psutil.virtual_memory().percent

            # Disk I/O (system-wide, works on macOS)
            try:
                dio = psutil.disk_io_counters()
                if dio is not None:
                    sample["disk_read_mb"] = dio.read_bytes / (1024 * 1024)
                    sample["disk_write_mb"] = dio.write_bytes / (1024 * 1024)
                else:
                    sample["disk_read_mb"] = None
                    sample["disk_write_mb"] = None
            except Exception:
                sample["disk_read_mb"] = None
                sample["disk_write_mb"] = None

            self._samples.append(sample)
            self._stop_event.wait(self._interval)

    # -- data access --

    @property
    def df(self) -> pd.DataFrame:
        """Raw samples as a DataFrame."""
        if not self._samples:
            return pd.DataFrame()
        return pd.DataFrame(self._samples)

    @property
    def wall_time(self) -> Optional[float]:
        return self._wall_time

    # -- summary --

    def summary(self):
        """Print a concise summary with bottleneck hint."""
        df = self.df
        if df.empty:
            print("No samples collected.")
            return

        wt = self._wall_time or df["elapsed"].iloc[-1]
        peak_parent_rss = df["parent_rss_mb"].max()
        peak_children_rss = df["children_rss_mb"].max()
        peak_total_rss = df["total_rss_mb"].max()
        max_children = df["n_children"].max()
        avg_total_cpu = df["total_cpu"].mean()
        max_possible_cpu = self._n_cores * 100
        efficiency = avg_total_cpu / max_possible_cpu * 100
        max_parent_threads = df["parent_threads"].max()

        print(f"Wall time:          {wt:.1f}s")
        print(
            f"Peak total RSS:     {peak_total_rss:,.0f} MB  "
            f"(parent: {peak_parent_rss:,.0f} MB + children: {peak_children_rss:,.0f} MB)"
        )
        print(f"Max children:       {max_children}")
        print(f"Max parent threads: {max_parent_threads}")
        print(
            f"Avg total CPU:      {avg_total_cpu:.0f}%  "
            f"(of {max_possible_cpu}% max on {self._n_cores} cores)"
        )
        print(f"CPU efficiency:     {efficiency:.1f}%")

        # Bottleneck heuristic
        hint = self._bottleneck_hint(df, max_children)
        print(f"Bottleneck hint:    {hint}")

    def _bottleneck_hint(self, df: pd.DataFrame, max_children: int) -> str:
        avg_total_cpu = df["total_cpu"].mean()
        max_parent_threads = df["parent_threads"].max()

        # Compute disk I/O rate if available
        io_rate_mb_s = 0.0
        if df["disk_read_mb"].notna().any() and len(df) > 1:
            dr = df["disk_read_mb"].dropna()
            dw = df["disk_write_mb"].dropna()
            if len(dr) > 1:
                elapsed_span = df["elapsed"].iloc[-1] - df["elapsed"].iloc[0]
                if elapsed_span > 0:
                    io_rate_mb_s = (
                        (dr.iloc[-1] - dr.iloc[0]) + (dw.iloc[-1] - dw.iloc[0])
                    ) / elapsed_span

        # Determine expected parallelism
        if max_children > 0:
            expected_max = max_children * 100
            mode = "process"
        else:
            expected_max = max_parent_threads * 100
            mode = "thread"

        cpu_ratio = avg_total_cpu / max(expected_max, 1)

        if mode == "thread" and avg_total_cpu < 150 and max_parent_threads > 2:
            return "GIL-contended — threads can't use multiple cores; try parallel_mode='process'"

        if cpu_ratio > 0.6 and io_rate_mb_s < 50:
            return "CPU-bound (high CPU utilization, low disk I/O)"

        if cpu_ratio < 0.3 and io_rate_mb_s > 100:
            return "I/O-bound (low CPU, high disk I/O)"

        if cpu_ratio < 0.3 and io_rate_mb_s < 50:
            return (
                "Underutilized — low CPU and low I/O; "
                "possible serialization overhead, worker starvation, or too few tasks"
            )

        return "Mixed — moderate CPU and I/O usage"

    # -- plotting --

    def plot(self, figsize: tuple[int, int] = (12, 10)):
        """Plot CPU, memory, workers, and I/O over time."""
        df = self.df
        if df.empty:
            print("No samples to plot.")
            return

        has_io = df["disk_read_mb"].notna().any()
        n_panels = 4 if has_io else 3
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(figsize[0], figsize[1]), sharex=True
        )
        t = df["elapsed"]

        # Panel 1: CPU
        ax = axes[0]
        ax.plot(t, df["total_cpu"], label="Total CPU (parent+children)", linewidth=1.5)
        ax.plot(
            t, df["parent_cpu"], label="Parent CPU", linewidth=1, alpha=0.7, linestyle="--"
        )
        ax.plot(
            t, df["children_cpu"], label="Children CPU", linewidth=1, alpha=0.7, linestyle=":"
        )
        ax.axhline(
            self._n_cores * 100,
            color="red",
            linestyle="--",
            alpha=0.4,
            label=f"Max ({self._n_cores} cores × 100%)",
        )
        ax.set_ylabel("CPU %")
        ax.set_title("CPU Usage")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Panel 2: Memory
        ax = axes[1]
        ax.fill_between(
            t, 0, df["parent_rss_mb"], alpha=0.4, label="Parent RSS"
        )
        ax.fill_between(
            t,
            df["parent_rss_mb"],
            df["total_rss_mb"],
            alpha=0.4,
            label="Children RSS",
        )
        ax.plot(t, df["total_rss_mb"], color="black", linewidth=1, label="Total RSS")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage (RSS)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Panel 3: Workers
        ax = axes[2]
        ax.plot(t, df["n_children"], label="Child processes", linewidth=1.5)
        ax.plot(
            t, df["parent_threads"], label="Parent threads", linewidth=1, linestyle="--"
        )
        ax.set_ylabel("Count")
        ax.set_title("Active Workers")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Panel 4: Disk I/O rate (if available)
        if has_io:
            ax = axes[3]
            dr = df["disk_read_mb"].values
            dw = df["disk_write_mb"].values
            dt = np.diff(t.values, prepend=t.values[0])
            dt[dt == 0] = self._interval  # avoid division by zero

            read_rate = np.diff(dr, prepend=dr[0]) / dt
            write_rate = np.diff(dw, prepend=dw[0]) / dt

            # Clip negative rates (can happen on counter wrap or first sample)
            read_rate = np.clip(read_rate, 0, None)
            write_rate = np.clip(write_rate, 0, None)

            ax.plot(t, read_rate, label="Disk read MB/s", linewidth=1)
            ax.plot(t, write_rate, label="Disk write MB/s", linewidth=1)
            ax.set_ylabel("MB/s")
            ax.set_title("Disk I/O Rate (system-wide)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Elapsed time (s)")
        fig.tight_layout()
        plt.show()


def benchmark_workers(
    ds,
    feature,
    worker_counts: Sequence[int] = (1, 2, 4, 8),
    modes: Sequence[str] = ("process", "thread"),
    plot: bool = True,
    **run_feature_kwargs,
) -> pd.DataFrame:
    """Run the same feature with different parallel configs and compare.

    Args:
        ds: mosaic Dataset instance
        feature: Feature instance to benchmark
        worker_counts: list of worker counts to test
        modes: list of parallel modes ("process", "thread")
        plot: whether to show a comparison bar chart
        **run_feature_kwargs: extra kwargs passed to ds.run_feature()

    Returns:
        DataFrame with columns: mode, workers, wall_time_s, peak_rss_mb, avg_cpu_pct, speedup
    """
    results = []
    baseline_time = None

    for mode in modes:
        for n_workers in sorted(worker_counts):
            print(f"\n{'='*60}")
            print(f"Benchmarking: mode={mode}, workers={n_workers}")
            print(f"{'='*60}")

            gc.collect()
            with ResourceMonitor() as mon:
                ds.run_feature(
                    feature,
                    parallel_workers=n_workers if n_workers > 1 else None,
                    parallel_mode=mode if n_workers > 1 else None,
                    overwrite=True,
                    **run_feature_kwargs,
                )

            df = mon.df
            wt = mon.wall_time or 0.0
            peak_rss = df["total_rss_mb"].max() if not df.empty else 0.0
            avg_cpu = df["total_cpu"].mean() if not df.empty else 0.0

            if baseline_time is None:
                baseline_time = wt

            results.append(
                {
                    "mode": mode if n_workers > 1 else "sequential",
                    "workers": n_workers,
                    "wall_time_s": round(wt, 1),
                    "peak_rss_mb": round(peak_rss, 0),
                    "avg_cpu_pct": round(avg_cpu, 0),
                    "speedup": round(baseline_time / wt, 2) if wt > 0 else 0.0,
                }
            )

            mon.summary()

    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    if plot and len(results_df) > 1:
        _plot_benchmark(results_df)

    return results_df


def _plot_benchmark(results_df: pd.DataFrame):
    """Bar chart comparing wall times across configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    labels = [f"{r['mode']}\n{r['workers']}w" for _, r in results_df.iterrows()]
    x = np.arange(len(labels))

    # Wall time
    ax = axes[0]
    bars = ax.bar(x, results_df["wall_time_s"], color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall Time by Configuration")
    for bar, val in zip(bars, results_df["wall_time_s"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(True, alpha=0.3, axis="y")

    # Speedup
    ax = axes[1]
    colors = ["green" if s >= 1 else "red" for s in results_df["speedup"]]
    bars = ax.bar(x, results_df["speedup"], color=colors, alpha=0.7)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Speedup (vs baseline)")
    ax.set_title("Speedup by Configuration")
    for bar, val in zip(bars, results_df["speedup"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plt.show()

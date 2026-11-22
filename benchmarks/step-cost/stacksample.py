"""Run ``python -m mosaic ARGS`` in this process while sampling every thread's stack.

Usage: stacksample.py OUT_PREFIX -- MOSAIC_ARGS...

Every 10 ms it records the stack of each thread. It writes OUT_PREFIX.stacks.txt:
for each thread role (main, sampler, pool, other), the inclusive share of samples
of each frame and the share of the innermost frames. The sampler runs only while
it holds the GIL, so its seconds undercount; read its percentages.
"""

import runpy
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from types import CodeType, FrameType

INTERVAL_SECONDS = 0.01


def _label(code: CodeType) -> str:
    filename = code.co_filename
    for marker in ("/src/mosaic/", "site-packages/"):
        if marker in filename:
            filename = filename.split(marker, 1)[1]
            break
    else:
        filename = filename.rsplit("/", 1)[-1]
    return f"{filename}:{code.co_name}"


def main() -> int:
    prefix = sys.argv[1]
    args = sys.argv[sys.argv.index("--") + 1 :]
    main_thread = threading.get_ident()
    inclusive: defaultdict[str, Counter[str]] = defaultdict(Counter)
    innermost: defaultdict[str, Counter[str]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    stop = threading.Event()
    # The standard library offers no public way to read another thread's stack.
    current_frames: Callable[[], dict[int, FrameType]] = getattr(sys, "_current_frames")

    def role(thread_id: int, names: dict[int | None, str]) -> str:
        if thread_id == main_thread:
            return "main"
        name = names.get(thread_id, "")
        if "sampl" in name.lower():
            return "sampler"
        if "ThreadPoolExecutor" in name or "pool" in name.lower():
            return "pool"
        return "other:" + name.split("-")[0]

    def sample() -> None:
        me = threading.get_ident()
        while not stop.wait(INTERVAL_SECONDS):
            names = {thread.ident: thread.name for thread in threading.enumerate()}
            for thread_id, frame in current_frames().items():
                if thread_id == me:
                    continue
                thread_role = role(thread_id, names)
                counts[thread_role] += 1
                seen: set[str] = set()
                innermost[thread_role][_label(frame.f_code)] += 1
                current = frame
                while current is not None:
                    label = _label(current.f_code)
                    if label not in seen:
                        inclusive[thread_role][label] += 1
                        seen.add(label)
                    current = current.f_back

    sampler = threading.Thread(target=sample, daemon=True, name="stack-sampler")
    sampler.start()
    started = time.perf_counter()
    sys.argv = ["mosaic", *args]
    status = 0
    try:
        runpy.run_module("mosaic", run_name="__main__")
    except SystemExit as exit_:
        status = exit_.code if isinstance(exit_.code, int) else 0
    stop.set()
    sampler.join()
    wall = time.perf_counter() - started
    with open(f"{prefix}.stacks.txt", "w") as out:
        out.write(
            f"wall {wall:.2f}s exit {status}; samples per thread role: {dict(counts)}\n"
        )
        for thread_role in sorted(inclusive, key=lambda name: -counts[name]):
            total = counts[thread_role]
            out.write(f"\n## {thread_role}: {total} samples\n### inclusive\n")
            for label, count in inclusive[thread_role].most_common(90):
                out.write(
                    f"{count * INTERVAL_SECONDS:7.2f}s {100 * count / total:5.1f}%  {label}\n"
                )
            out.write("### innermost\n")
            for label, count in innermost[thread_role].most_common(30):
                out.write(
                    f"{count * INTERVAL_SECONDS:7.2f}s {100 * count / total:5.1f}%  {label}\n"
                )
    return status


if __name__ == "__main__":
    sys.exit(main())

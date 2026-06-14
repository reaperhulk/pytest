"""Benchmark collection time and memory usage for many parametrized tests.

This benchmark generates a synthetic test module containing a number of test
functions, each decorated with ``@pytest.mark.parametrize`` to expand into many
parametrized test items.  It then runs pytest in *collection only* mode
in-process and reports:

* wall-clock time spent collecting,
* the number of collected items,
* the peak Python heap allocation during collection (via ``tracemalloc``),
* the amount of Python heap still retained once collection has finished
  (i.e. the memory cost of the collection tree itself),
* the process max RSS (via ``resource``).

Usage::

    python bench/bench_collect_parametrized.py [NUM_FUNCS] [NUM_PARAMS]

Defaults to 100 functions x 500 params = 50000 items.

The ``--profile`` flag runs the collection under ``cProfile`` and prints the
top cumulative-time callers, which is handy for spotting hot spots.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import pytest


TEST_MODULE_TEMPLATE = '''\
import pytest


{functions}
'''

FUNCTION_TEMPLATE = '''\
@pytest.mark.parametrize("a, b", [(i, i + 1) for i in range({num_params})])
def test_func_{idx}(a, b):
    assert a + 1 == b
'''


def write_test_module(directory: Path, num_funcs: int, num_params: int) -> Path:
    functions = "\n\n".join(
        FUNCTION_TEMPLATE.format(idx=i, num_params=num_params)
        for i in range(num_funcs)
    )
    source = TEST_MODULE_TEMPLATE.format(functions=functions)
    path = directory / "test_generated_parametrized.py"
    path.write_text(source)
    return path


class _CollectStats:
    """Plugin that snapshots memory while the session is still alive."""

    def __init__(self) -> None:
        self.num_items = 0
        self.retained_kib = 0.0
        self.peak_kib = 0.0

    @pytest.hookimpl
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.num_items = len(session.items)
        # Force any pending garbage to be collected so the retained figure
        # reflects live objects only.
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        self.retained_kib = current / 1024
        self.peak_kib = peak / 1024


@contextlib.contextmanager
def suppress_stdout():
    """Redirect stdout to /dev/null so the per-item collect listing (which can
    be tens of thousands of lines) doesn't pollute timings or the terminal."""
    devnull = open(os.devnull, "w")
    saved = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = saved
        devnull.close()


def _collect(test_path: Path, stats: _CollectStats) -> int:
    with suppress_stdout():
        return pytest.main(
            [
                str(test_path),
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            plugins=[stats],
        )


def run_timed(test_path: Path) -> tuple[float, _CollectStats]:
    """Time a collection run *without* tracemalloc (which inflates timings)."""
    stats = _CollectStats()
    start = time.perf_counter()
    ret = _collect(test_path, stats)
    elapsed = time.perf_counter() - start
    assert ret == 0, f"pytest exited with {ret}"
    return elapsed, stats


def run_memory(test_path: Path) -> _CollectStats:
    """Measure heap usage of a collection run with tracemalloc enabled."""
    stats = _CollectStats()
    tracemalloc.start()
    ret = _collect(test_path, stats)
    tracemalloc.stop()
    assert ret == 0, f"pytest exited with {ret}"
    return stats


def run_profile(test_path: Path) -> None:
    import cProfile
    import pstats

    stats = _CollectStats()
    profiler = cProfile.Profile()
    with suppress_stdout():
        profiler.enable()
        pytest.main(
            [str(test_path), "--collect-only", "-q", "-p", "no:cacheprovider"],
            plugins=[stats],
        )
        profiler.disable()
    p = pstats.Stats(profiler)
    p.strip_dirs()
    p.sort_stats("cumulative")
    p.print_stats(40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_funcs", nargs="?", type=int, default=100)
    parser.add_argument("num_params", nargs="?", type=int, default=500)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the timed collection this many times and report the best.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        test_path = write_test_module(directory, args.num_funcs, args.num_params)

        if args.profile:
            run_profile(test_path)
            return

        best_time = float("inf")
        last_stats: _CollectStats | None = None
        for _ in range(args.repeat):
            elapsed, last_stats = run_timed(test_path)
            best_time = min(best_time, elapsed)

        # Separate run with tracemalloc for accurate heap figures.
        mem_stats = run_memory(test_path)

        assert last_stats is not None
        maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is in KiB; on macOS it is in bytes.
        if sys.platform == "darwin":
            maxrss_kib /= 1024

        items = last_stats.num_items
        print(f"functions:        {args.num_funcs}")
        print(f"params each:      {args.num_params}")
        print(f"collected items:  {items}")
        print(f"collection time:  {best_time * 1000:.1f} ms (best of {args.repeat})")
        if items:
            print(f"  per item:       {best_time / items * 1e6:.2f} us")
        print(f"peak heap:        {mem_stats.peak_kib / 1024:.1f} MiB")
        print(f"retained heap:    {mem_stats.retained_kib / 1024:.1f} MiB")
        if items:
            print(
                f"  per item:       {mem_stats.retained_kib * 1024 / items:.0f} bytes"
            )
        print(f"process max RSS:  {maxrss_kib / 1024:.1f} MiB")


if __name__ == "__main__":
    main()

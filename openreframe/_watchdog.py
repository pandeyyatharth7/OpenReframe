"""Time-based OOM detection.

Why this exists: on the target machine WDDM CUDA sysmem fallback is ACTIVE
(measured in docs/benchmarks.md, `oom_probe` row). Under that condition a real
out-of-memory condition does NOT raise `torch.cuda.OutOfMemoryError` — the
driver spills into system RAM and the kernel crawls instead. Every
`except OutOfMemoryError` around a generation call is therefore dead code.

The only observable signal is time, so detection is time-based: if a stage runs
far past its measured warm p95, treat it as a stall, interrupt it, and recover.
Keep the exception handlers too — they resume working once "CUDA System Memory
Fallback" is disabled in the NVIDIA Control Panel.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from .config import OOM_ABSOLUTE_FLOOR_S, OOM_SUSPECT_FACTOR

POLL_S = 0.5  # matches the sampling cadence of prototypes/_bench.py


def limit_for(expected_p95_s: float,
              factor: float = OOM_SUSPECT_FACTOR,
              floor_s: float = OOM_ABSOLUTE_FLOOR_S) -> float:
    """Wall-clock budget for a stage whose warm p95 is `expected_p95_s`.

    The floor keeps the limit sane for very fast stages (a 100 ms preview must
    not be aborted at 400 ms because of ordinary scheduler noise).
    """
    return max(expected_p95_s * factor, floor_s)


class StageWatchdog:
    """Context manager that trips `on_exceed` if the block overruns its budget.

    `on_exceed` should be cheap and idempotent; for diffusion it sets
    `pipe._interrupt = True` so the loop stops at the next step boundary rather
    than running to completion at ~50x the normal speed.

    Not thread-safe with respect to `on_exceed`; the caller owns the object it
    mutates.
    """

    def __init__(self, expected_p95_s: float,
                 on_exceed: Callable[[], None] | None = None,
                 factor: float = OOM_SUSPECT_FACTOR,
                 floor_s: float = OOM_ABSOLUTE_FLOOR_S,
                 enabled: bool = True) -> None:
        self.limit_s = limit_for(expected_p95_s, factor, floor_s)
        self.expected_p95_s = expected_p95_s
        self.on_exceed = on_exceed
        self.enabled = enabled
        self.tripped = False
        self.elapsed_s = 0.0
        self._t0 = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(POLL_S):
            if time.perf_counter() - self._t0 > self.limit_s:
                self.tripped = True
                if self.on_exceed is not None:
                    try:
                        self.on_exceed()
                    except Exception:
                        # Never let the watchdog itself kill the caller.
                        pass
                return

    def __enter__(self) -> "StageWatchdog":
        self._t0 = time.perf_counter()
        self._stop.clear()
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="stage-watchdog")
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        self.elapsed_s = time.perf_counter() - self._t0
        if self._thread is not None:
            self._thread.join(timeout=POLL_S * 2)
        return False

    def summary(self) -> dict:
        return {"tripped": self.tripped, "elapsed_s": round(self.elapsed_s, 3),
                "limit_s": round(self.limit_s, 1)}
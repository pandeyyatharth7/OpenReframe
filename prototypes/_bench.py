"""Shared Sprint-0 benchmark harness: stage-scoped VRAM accounting, nvidia-smi
polling (500 ms), wall-clock timing, markdown row output to docs/benchmarks.md.

Importing this module also points HF_HUB_CACHE into <repo>/models so all
downloaded weights stay inside the repo and out of git.
"""
from __future__ import annotations

import atexit
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_MD = REPO_ROOT / "docs" / "benchmarks.md"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

import os

os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hub"))
os.environ.setdefault("HF_HOME", str(MODELS_DIR))

VRAM_CEIL_MB = 6.8 * 1024  # pre-committed ceiling from docs/decisions.md D0


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None


def driver_version() -> str:
    return _nvidia_smi("driver_version") or "unknown"


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return _nvidia_smi("name") or "no-gpu"


def smi_used_mb() -> float:
    v = _nvidia_smi("memory.used")
    return float(v) if v is not None else -1.0


class _SmiPoller(threading.Thread):
    """Daemon thread sampling nvidia-smi memory.used every 500 ms."""

    def __init__(self, interval_s: float = 0.5):
        super().__init__(daemon=True, name="smi-poller")
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            used = smi_used_mb()
            if used >= 0:
                self.samples.append((time.perf_counter(), used))
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_evt.set()


_poller: _SmiPoller | None = None


def _get_poller() -> _SmiPoller:
    global _poller
    if _poller is None or not _poller.is_alive():
        _poller = _SmiPoller()
        _poller.start()
        atexit.register(_poller.stop)
    return _poller


class Bench:
    def __init__(self, script: str, model: str = "", precision: str = "",
                 resolution: str = "", steps: str = "", scheduler: str = ""):
        self.script = script
        self.meta = dict(model=model, precision=precision, resolution=resolution,
                         steps=steps, scheduler=scheduler)
        self.rows: list[dict] = []
        self.poller = _get_poller()
        print(f"[bench] {script} | gpu={gpu_name()} | driver={driver_version()} "
              f"| torch={torch.__version__} | cuda={torch.version.cuda}")

    @contextmanager
    def stage(self, name: str, **meta_extra):
        for k, v in meta_extra.items():
            self.meta[k] = v
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        mark_t = time.perf_counter()
        mark_idx = len(self.poller.samples)
        info: dict = {}
        try:
            yield info
        finally:
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            peak_alloc = torch.cuda.max_memory_allocated() / 2**20
            peak_res = torch.cuda.max_memory_reserved() / 2**20
            win = [s for t, s in self.poller.samples[mark_idx:] if t >= mark_t]
            smi_peak = max(win) if win else smi_used_mb()
            row = dict(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       script=self.script, stage=name,
                       peak_alloc_mb=round(peak_alloc, 1),
                       peak_reserved_mb=round(peak_res, 1),
                       smi_peak_mb=round(smi_peak, 1), wall_s=round(wall, 3),
                       **{k: self.meta.get(k, "") for k in
                          ("model", "precision", "resolution", "steps", "scheduler")},
                       extra="; ".join(f"{k}={v}" for k, v in meta_extra.items()))
            self.rows.append(row)
            print(f"[bench]   {name:<22s} wall={wall:8.3f}s "
                  f"torch_peak={peak_alloc:8.1f}/{peak_res:8.1f}MB "
                  f"smi_peak={smi_peak:7.1f}MB")

    def finish(self, notes: str = "") -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        lines = []
        if not BENCHMARKS_MD.exists():
            lines.append("# Benchmarks — OpenReframe Sprint 0\n")
            lines.append("All values measured on the target machine. `torch_peak` = "
                         "max_memory_allocated / max_memory_reserved (MiB); `smi_peak` = "
                         "driver-level memory.used max during the stage (MiB), sampled at 500 ms.\n")
            lines.append("| timestamp | script | stage | model | precision | resolution | steps | scheduler "
                         "| torch_peak MB (alloc/res) | smi_peak MB | wall s | notes |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in self.rows:
            cells = [r["ts"], r["script"], r["stage"], r["model"], r["precision"],
                     r["resolution"], str(r["steps"]), r["scheduler"],
                     f"{r['peak_alloc_mb']}/{r['peak_reserved_mb']}",
                     str(r["smi_peak_mb"]), str(r["wall_s"]),
                     "; ".join(x for x in (r["extra"], notes) if x)]
            lines.append("| " + " | ".join(cells) + " |")
        with open(BENCHMARKS_MD, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[bench] appended {len(self.rows)} row(s) -> {BENCHMARKS_MD}")


def deliberate_oom(bench: Bench) -> dict:
    """Deliberately trigger CUDA OOM, observe it, verify recovery.

    Proves the failure mode Sprint 4 must handle gracefully.
    """
    result: dict = {}
    with bench.stage("oom_probe") as st:
        free_b, total_b = torch.cuda.mem_get_info()
        total_mb = total_b / 2**20
        result["total_vram_mb"] = round(total_mb, 1)
        result["free_before_mb"] = round(free_b / 2**20, 1)

        # WDDM sysmem-fallback check first: try to allocate ~1.5x TOTAL dedicated VRAM.
        # If this *succeeds*, the driver is spilling into system RAM instead of OOMing.
        request_b = int(total_b * 1.5)
        oom_error_msg = ""
        try:
            t = torch.empty(request_b, dtype=torch.uint8, device="cuda")
            got = t.numel()
            del t
            torch.cuda.synchronize()
            result["oom_triggered"] = False
            result["fallback_note"] = (
                f"allocated {got/2**20:.0f}MB > dedicated VRAM {total_mb:.0f}MB: "
                "WDDM CUDA sysmem fallback is ACTIVE — real OOM will surface as extreme slowdown, not an exception")
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            result["oom_triggered"] = True
            oom_error_msg = str(e).splitlines()[0][:200]
            result["oom_error"] = oom_error_msg

        # Recovery path (what FastAPI must do): empty cache, retry a small op.
        t1 = time.perf_counter()
        torch.cuda.empty_cache()
        probe = torch.randn(64, 64, device="cuda")
        y = probe @ probe
        torch.cuda.synchronize()
        del probe, y
        result["recovery_ok"] = True
        result["recovery_s"] = round(time.perf_counter() - t1, 4)
        st.update(oom=result["oom_triggered"], recovery_ok=result["recovery_ok"])
    print("[bench] OOM probe:", result)
    return result

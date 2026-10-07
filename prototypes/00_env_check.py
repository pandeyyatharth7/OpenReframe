"""00_env_check.py — Sprint 0 environment validation for OpenReframe.

Verifies Python/PyTorch/CUDA/driver/GPU against project requirements and fails
loudly (exit code 1) on hard failures, with WARN items surfaced for docs.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import shutil
import subprocess
import sys

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", required: bool = True) -> bool:
    status = PASS if ok else (FAIL if required else WARN)
    results.append((status, name, detail))
    print(f"[{status:^4}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 else None
    except Exception:
        return None


def system_ram_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    st = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullTotalPhys / 2**30


def main() -> int:
    print("=" * 72)
    print("OpenReframe environment check")
    print("=" * 72)

    v = sys.version_info
    check("python >= 3.10", v >= (3, 10), f"Python {v.major}.{v.minor}.{v.micro}")

    try:
        import torch
        check("torch importable", True, torch.__version__)
    except Exception as e:
        check("torch importable", False, f"{e}")
        return finish()

    check("CUDA build present", torch.backends.cuda.is_built(),
          f"torch CUDA compiled: {torch.backends.cuda.is_built()}")
    cuda_ok = torch.cuda.is_available()
    check("CUDA available at runtime", cuda_ok,
          "" if cuda_ok else "torch.cuda.is_available()=False — install CUDA wheels "
          "(pip install torch --index-url https://download.pytorch.org/whl/cu124), "
          "verify driver with nvidia-smi")
    if not cuda_ok:
        return finish()

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
    free_b, _ = torch.cuda.mem_get_info()
    check("GPU detected", "RTX" in name or "NVIDIA" in name.upper(), name)
    check("Ada/Ampere+ (sm_86+)", cap >= (8, 6), f"compute capability {cap[0]}.{cap[1]}")
    check("VRAM ~8GB", total_mb >= 7500, f"{total_mb:.0f} MiB dedicated "
          f"(desktop baseline using {total_mb - free_b/2**20:.0f} MiB)")

    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = (a @ b)
    torch.cuda.synchronize()
    check("fp32 matmul", bool(c.isfinite().all().item()),
          f"max|c|={c.abs().max().item():.1f}")
    del a, b, c
    torch.cuda.empty_cache()

    try:
        a16 = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        y16 = a16 @ a16
        torch.cuda.synchronize()
        check("fp16 matmul", bool(y16.isfinite().all().item()))
        del a16, y16
    except Exception as e:
        check("fp16 matmul", False, str(e))

    try:
        abf = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        ybf = abf @ abf
        torch.cuda.synchronize()
        del abf, ybf
        check("bf16 supported", True, "(optional precision)")
    except Exception:
        check("bf16 supported", False, "(fallback to fp16)", required=False)

    check("cuDNN enabled", torch.backends.cudnn.enabled,
          f"cudnn {torch.backends.cudnn.version()}", required=False)
    print(f"[INFO] torch.version.cuda = {torch.version.cuda}")

    smi_name = nvidia_smi("name")
    drv = nvidia_smi("driver_version")
    check("nvidia-smi reachable", smi_name is not None,
          f"name={smi_name} driver={drv}")
    used_smi = nvidia_smi("memory.used")
    total_smi = nvidia_smi("memory.total")
    if total_smi and used_smi:
        baseline = float(total_smi) - float(used_smi)
        print(f"[INFO] nvidia-smi: {used_smi} MiB used / {total_smi} MiB total "
              f"(non-torch desktop baseline {baseline:.0f} MiB)")

    ram = system_ram_gb()
    check("system RAM >= 15GB (cpu-offload needs ~16)", ram >= 15, f"{ram:.1f} GB")

    disk = shutil.disk_usage("C:\\")
    free_disk_gb = disk.free / 2**30
    check("free disk >= 25GB (model weights)", free_disk_gb >= 25,
          f"{free_disk_gb:.0f} GB free")

    try:
        import huggingface_hub
        check("huggingface_hub importable", True, huggingface_hub.__version__)
    except ImportError:
        check("huggingface_hub importable", False,
              "needed for model downloads", required=False)

    return finish()


def finish() -> int:
    fails = sum(1 for s, *_ in results if s == FAIL)
    warns = sum(1 for s, *_ in results if s == WARN)
    print("-" * 72)
    print(f"SUMMARY: {len(results) - fails - warns} pass, {warns} warn, {fails} fail")
    if fails:
        print("ENVIRONMENT INVALID — fix FAIL items before running profiling.")
    else:
        print("ENVIRONMENT OK — proceed with Sprint 0 profiling.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

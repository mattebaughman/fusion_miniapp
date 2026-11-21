"""Report the PyTorch environment and forward-pass result inside a worker allocation as JSON.

Standalone (torch and the standard library only), so it runs before the package is installed.
It uses the mini-app's device rule (cuda if torch.cuda.is_available(), else cpu; ROCm builds answer
through torch.cuda), runs one forward pass whose output stays on the device as the next stage's
input, and never sets a visible-device variable: the launcher decides what the worker sees.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import torch

VISIBLE = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
SLURM = ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")


def gpu_management() -> dict | None:
    """Driver-reported inventory. Vendor tools can see more GPUs than this Slurm task can use."""
    if not torch.cuda.is_available():
        return None
    if torch.version.hip:
        cmd = ["amd-smi", "static", "--driver", "--json"]
    else:
        cmd = ["nvidia-smi", "--query-gpu=uuid,driver_version,compute_mode", "--format=csv,noheader"]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        if torch.version.hip:
            return {"tool": cmd[0], "devices": json.loads(raw)}
        rows = csv.DictReader(raw.splitlines(), fieldnames=["uuid", "driver_version", "compute_mode"],
                              skipinitialspace=True)
        return {"tool": cmd[0], "devices": list(rows)}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"tool": cmd[0], "error": type(exc).__name__}


def main() -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    stage_a = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.GELU(), torch.nn.Linear(64, 16)).to(dev).eval()
    stage_b = torch.nn.Linear(16, 4).to(dev).eval()
    conv = torch.nn.Conv2d(1, 2, kernel_size=3).to(dev).eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = stage_a(torch.randn(32, 8, device=dev))
        z = stage_b(y)                                  # stage A's output, still on the device
        grid = conv(z.reshape(2, 1, 8, 8))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "torch_version_cuda": torch.version.cuda,
        "torch_version_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(), "device": str(dev),
        "gpu_management": gpu_management(),
        "threads": {"torch": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                    "env": {v: os.environ.get(v) for v in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "visible_devices": {v: os.environ.get(v) for v in VISIBLE},
        "slurm": {v: os.environ.get(v) for v in SLURM},
        "cpu_count": os.cpu_count(), "cores_usable": len(affinity) if affinity is not None else None,
        "forward": {"stage_a_out": [list(y.shape), str(y.device)], "stage_b_out": [list(z.shape), str(z.device)],
                    "conv_out": [list(grid.shape), str(grid.device)],
                    "finite": bool(torch.isfinite(z).all() and torch.isfinite(grid).all()),
                    "seconds": round(seconds, 4)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="fail if no GPU or the forward is nonfinite")
    parser.add_argument("--out", type=argparse.FileType("w"), default=sys.stdout, help="JSON file (default stdout)")
    args = parser.parse_args()
    probe = main()
    print(json.dumps(probe, indent=1), file=args.out, flush=True)
    if args.require_gpu and not (probe["cuda_available"] and probe["forward"]["finite"]):
        sys.exit("fusion-miniapp requires a visible GPU and a finite forward pass")

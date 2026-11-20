"""The first job on a machine: what PyTorch and devices a worker gets in this allocation. Prints JSON.

Standalone (torch and the standard library only), so it runs before the package is installed.
It uses the mini-app's device rule (cuda if torch.cuda.is_available(), else cpu; ROCm builds answer
through torch.cuda), runs one forward pass whose output stays on the device as the next stage's
input, and never sets a visible-device variable: the launcher decides what the worker sees.
"""
import json
import os
import platform
import time

import torch

VISIBLE = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
SLURM = ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")


def main() -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    stage_a = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.GELU(), torch.nn.Linear(64, 16)).to(dev).eval()
    stage_b = torch.nn.Linear(16, 4).to(dev).eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = stage_a(torch.randn(32, 8, device=dev))
        z = stage_b(y)                                  # stage A's output, still on the device
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "torch_version_cuda": torch.version.cuda,
        "torch_version_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(), "device": str(dev),
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "visible_devices": {v: os.environ.get(v) for v in VISIBLE},
        "slurm": {v: os.environ.get(v) for v in SLURM},
        "cpu_count": os.cpu_count(), "cores_usable": len(affinity) if affinity is not None else None,
        "forward": {"stage_a_out": [list(y.shape), str(y.device)], "stage_b_out": [list(z.shape), str(z.device)],
                    "finite": bool(torch.isfinite(z).all()), "seconds": round(seconds, 4)},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))

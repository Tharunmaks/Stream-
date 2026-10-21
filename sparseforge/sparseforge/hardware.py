import os, platform, shutil

def detect_hardware():
    flags = set()
    model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if (line.startswith("flags") or line.startswith("Features")) and not flags:
                    flags = set(line.split(":", 1)[1].split())
                elif line.startswith("model name") and not model:
                    model = line.split(":", 1)[1].strip()
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_gb = round(int(line.split()[1]) / 1024**2, 2)
    except OSError:
        pass
    numa = len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]) if os.path.isdir("/sys/devices/system/node") else 1
    gpu = bool(shutil.which("nvidia-smi")) or os.path.exists("/dev/nvidia0")  # no torch import (Termux)
    return {
        "cpu_model": model or platform.processor(),
        "logical_cores": os.cpu_count(),
        "numa_nodes": numa,
        "ram_gb": mem_gb,
        "disk_free_gb": round(shutil.disk_usage("/").free / 1024**3, 1),
        "amx_bf16": "amx_bf16" in flags, "amx_int8": "amx_int8" in flags,
        "avx512f": "avx512f" in flags, "avx512_bf16": "avx512_bf16" in flags,
        "avx2": "avx2" in flags, "neon": "asimd" in flags or "neon" in flags,
        "arm_dotprod": "asimddp" in flags, "arm_i8mm": "i8mm" in flags, "arch": platform.machine(), "gpu": gpu,
    }

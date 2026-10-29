"""Read-only CUDA hardware capability report, independent from scientific admission."""
import json
import torch


def capture_hardware():
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index, "name": gpu.name,
                "total_memory_bytes": gpu.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            })
    return {
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_devices": devices,
    }


def main():
    print(json.dumps(capture_hardware()))


if __name__ == "__main__":
    main()

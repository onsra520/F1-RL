"""Optional runtime hardware probe, outside the dependency-free simulator core."""
import json
import torch


if __name__ == "__main__":
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": gpu.name, "total_memory_bytes": gpu.total_memory,
                            "capability": list(torch.cuda.get_device_capability(index))})
    print(json.dumps({"cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                      "gpu_devices": devices}))

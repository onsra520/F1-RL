"""Read-only CUDA hardware capability report, independent from scientific admission."""
import json


def capture_hardware(torch_runtime):
    devices = []
    if torch_runtime.cuda.is_available():
        for index in range(torch_runtime.cuda.device_count()):
            gpu = torch_runtime.cuda.get_device_properties(index)
            devices.append({
                "index": index, "name": gpu.name,
                "total_memory_bytes": gpu.total_memory,
                "capability": list(torch_runtime.cuda.get_device_capability(index)),
            })
    return {
        "cuda_runtime": torch_runtime.version.cuda,
        "cuda_available": torch_runtime.cuda.is_available(),
        "gpu_devices": devices,
    }


def main(torch_runtime):
    print(json.dumps(capture_hardware(torch_runtime)))


if __name__ == "__main__":
    main()

# Development-only diagnostics and comparison

This is the current guide for **TRAIN-only** performance and driving troubleshooting. It does not authorize new registered evaluation, retraining, or opening HELD-OUT assets.

## Script-to-module boundaries

| Versionless workflow | Package entrypoint | Implementation module | Effect |
| --- | --- | --- | --- |
| `evaluation compare-driving` | `scripts/evaluation/compare_driving.py` | `src/evaluation/driving_comparison.py` | Paired development evaluation on fixed tuning/reserved seed roster |
| `diagnostics diagnose-driving` | `scripts/diagnostics/diagnose_driving.py` | `src/evaluation/driving_diagnostics.py` | Scripted-vs-model traces and runtime profiling |
| `diagnostics probe-gpu` | `scripts/diagnostics/probe_gpu.py` | `src/interfaces/hardware_probe.py` | Read-only local hardware capabilities |
| `diagnostics benchmark-environments` | `scripts/diagnostics/benchmark_environments.py` | `src/training/vector_benchmark.py` | Development throughput benchmark |
| `evidence verify-frozen-assets` | `scripts/evidence/verify_frozen_assets.py` | `src/validation/frozen_asset_integrity.py` | Read-only frozen asset hash validation |

## Safe commands

From the repository root, with an interpreter containing the declared RL dependencies:

```powershell
python -m scripts.workflows list
python -m scripts.workflows diagnostics probe-gpu
python -m scripts.workflows evaluation compare-driving --help
python -m scripts.workflows diagnostics diagnose-driving --help
python -m scripts.workflows diagnostics benchmark-environments --help
```

Comparisons and traces can run simulation episodes and generate outputs. Do not invoke execution commands solely to test their availability. Their `--help` paths do not run PPO or allocate evidence.

The comparison's eight registered **development-only** seeds and prespecified improvement thresholds remain unchanged during implementation extraction. These are historical development choices, **not a scientific acceptance gate**. The models and traces remain outside git when provided through user paths.

## Acceptance and limits

- Use `tests/scripts/test_development_tools.py` to check pure metrics, diagnostic actions, help and workflow/module parity without training.
- The Physics/PPO `src/` implementation and sealed evaluation protocols were not modified by this script refactor.
- CLI names are stable and versionless; historical records may retain old script filenames as provenance, while retired demo aliases are absent from the active registry.
- The authoritative backend↔frontend connection does not yet exist.

For broader scripts and documentation status see [workflow guide](WORKFLOWS.md), [import audit](SCRIPT-IMPORT-AUDIT.md) and [change report](SCRIPTS-DOCS-CHANGE-REPORT.md).

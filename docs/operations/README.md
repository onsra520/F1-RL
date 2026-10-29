# Operations and current limits

## Present engineering state

- Python source is organised under `src/` and current model inference works in the accepted migration baseline.
- The React/Three.js racing editor builds independently. **No UI ↔ Python backend ↔ PPO runtime bridge is implemented.** FastAPI/WebSocket are planned adapters, not a running integrated service.
- Twelve frozen PPO model checkpoints and FULL-06 (96 cells/7 reports, 5,760 TRAIN-development episodes) are preserved.
- Historical scientific admission remains **BLOCKED**; engineering regression is not registered evaluation signoff.
- External raw telemetry is not in Git. Historical preflight needs two hash-pinned immutable ZIPs.
- The frontend build has a large chunk warning and the existing dev/test dependency audit has six findings; production-only npm audit previously reported zero.

## Documentation and provenance

For hash manifests, run-record review and scientific fail-closed behavior, use [Evidence verification](EVIDENCE-VERIFICATION.md).

See [independent historical recovery](HISTORICAL-RECOVERY.md), [engineering validation](../architecture/REFACTOR-VALIDATION.md) and [historical/current attestation boundary](../architecture/HISTORICAL-PROVENANCE.md).

For CLI operations use [script workflows](../development/WORKFLOWS.md), not proposed or obsolete `train.py`/`serve.py` commands.

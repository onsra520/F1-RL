# F1-RL current engineering state

The codebase is Python-first with the active import root src/. The simulation
and tracks drive Gymnasium and PPO training/inference; web/ contains a separate
React/Three.js editor and replay UI. A working UI-to-Python simulation/PPO
service bridge is **not implemented** and is not implied by the frontend build.

Training permission is independent of general engineering tests. The original
scientific admission checks stay fail-closed and HELD-OUT stays sealed.
The fixed frozen development bundle contains 12 PPO checkpoints, 7 reports and
96 cell logs. Verification proves bytes and roster, not autonomous race finishes
or registered scientific acceptance.

## Relevant live commands

~~~powershell
python -m scripts.workflows list --json
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows inference run-policy --help
python -m pytest -q tests/scripts
~~~

Commands marked training-explicit, restricted-execution or development-execution
must only be run under the original consent, input and evaluation boundaries.
Do not use historical demonstration success metrics as proof of race completion.

## Implementation map

- src/interfaces/: orchestration, demo/package, runtime selection, inference
- src/training/: vector environment and throughput benchmark
- src/evaluation/: driving comparison and non-scientific diagnostics
- src/validation/: evidence, original scientific gates and read-only audits
- scripts/: grouped functional entrypoints and versionless router
- docs/research/: historical contracts, frozen evidence and provenance records

For per-command ownership use SCRIPT-STATUS.json and ENTRYPOINT-ARCHITECTURE.md.
For original model/evidence provenance use ../research/README.md.

# Final Engineering Acceptance — 2026-11-10

Trace ID: `8397d00f7e9b41218485826378c6e4dd`
Candidate branch: `fix/engineering-auth-source-verification-20261110`
Source HEAD tested: `9b6a45d4cd50289a2ab5ce0d3ff078c6a161a2eb`
Base local `main`: `169f9ef4682f3bd60c80c63b581bebdfe3db80e1`
Remote `origin/main` before integration: `74a5147bfc8f907670b45f5fd02315505262813d`

## Acceptance decision

Engineering acceptance is **PASS with the approved, limited `BLOCKED_EXTERNAL_EVIDENCE` exception**. This is not a full regression pass or scientific acceptance. The exception applies only to the seven node IDs in [the exception manifest](ENGINEERING-ACCEPTANCE-EXCEPTIONS-20261110.json). No other failure was relabeled.

## Regression results

- Candidate collection: **3,348 nodes**.
- Policy filter: **101 `BLOCKED_BY_POLICY`**, matched against all 3,348 collected IDs; zero policy IDs missing.
- External evidence exception: **7 `BLOCKED_EXTERNAL_EVIDENCE`**; focused execution reproduced `FileNotFoundError` for the missing root `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`.
- Resource group: the exact five Auto Drive/T05 nodes ran sequentially at the unchanged 120-second per-session wait: **5/5 PASS**, 177.67 seconds total. Physics/collision/runtime source hashes are recorded in the manifest.
- Policy-filtered full regression, one run with four pytest workers and `loadgroup`: **3,232 PASS / 0 FAIL / 3 SKIP**, 557.20 seconds. The 101 policy nodes, seven external-evidence nodes, and five separately verified serial nodes were excluded from this invocation and are accounted for separately above.

The 146 focused authorization/source-binding/guard tests yielded **139 PASS** and exactly the seven accepted external-evidence failures; there were no unexpected focused failures. The previous 14 source-binding failures are resolved except for these seven nodes whose required external witnesses are absent.

JUnit records (SHA-256 verified):

- Full regression: `D:\Projects\.archives\F1-RL-engineering-acceptance-20261110-9b6a45d\final-policy-regression.junit.xml` — `e2166b351c4e42c82444ff7ca6f724e0a865de456c1e4fb4a32e11879780c7fa`
- Focused authorization/source-binding: `D:\Projects\.archives\F1-RL-engineering-acceptance-20261110-9b6a45d\focused-auth-source-binding.junit.xml` — `89a39d0a880635af37a6ee81330bf81a5b44ea15169c48ac2bb9f3c921a08352`
- Serial Auto Drive/T05: `D:\Projects\.archives\F1-RL-engineering-acceptance-20261110-9b6a45d\auto-drive-t05-serial.junit.xml` — `1e1bd60bc54c29f16d67a8e35bf18e2a91373a38b1e0e817dd8f58cc0b9161f7`

The full run is intentionally reported as policy-filtered engineering acceptance; it is **not** `FULL_REGRESSION_PASS`.

## Remediation and provenance

Six remediation commits on top of the existing local-main integration:

- `4f3e2e81f1d33338bab490b5c3400d91fe80d77b` — explicit engineering-only PPO mode.
- `8ad09fa723a97828d64df87d36e957a3a20017d3` — portable WP7 source revision remains draft.
- `0ff479c986a8be5ee68686f8aa9e6ef5f1e1e293` — verify registered frozen parent source in preview.
- `9f1b4b49161afaf33f2e4f0ccc33c6c8df7a7519` — limit EOL normalization to bridged text sources.
- `e78435334cdcf0dd947238806fead0c037e7fccb` — portable verification while preserving historical evidence.
- `9b6a45d4cd50289a2ab5ce0d3ff078c6a161a2eb` — explicit engineering-mode denial regression.

Changed source/test scope: `f1rl/exploratory/cli.py`, `permit.py`, `pilot.py`, `f1rl/research/wp7_train5_v2_protocol.py`, `scripts/wp7_train5_handoff_preflight.py`, affected `tests/research` authorization/source-binding tests, `pytest.ini`, `pyproject.toml`, and a new portable **draft, unauthorized** WP7 source-revision record. Source verifiers were revised for portable binding and line-ending correctness; the historical registration `configs/research/wp7-train5-paired-v2-registration.json` and its hashes are unchanged. Physics, reward, observation/action, and PPO algorithm code are unchanged. Guard tests passed, and no admission permission or owner approval was opened.

Engineering mode remains separate from scientific authorization. Historical WP7 admission remains `draft_not_authorized`; training/smoke/held-out permissions are false and owner approval is null. The new portable registration is also draft-only. No live PPO learning, checkpoint generation, or HELD-OUT access occurred.

## Exception closure

The exact four witnesses and registered hashes are in the exception manifest. Close the exception only after recovering those original files from an authorized source and verifying all four hashes, then rerun only the seven listed nodes. Do not regenerate witnesses or edit the historical registration.

## Integration status

This report and manifest are committed separately after the source regression. The regression ran at source HEAD `9b6a45d4…`; only these documentation/manifest files follow that tested source commit. Final local and remote refs are recorded in the delivery summary.

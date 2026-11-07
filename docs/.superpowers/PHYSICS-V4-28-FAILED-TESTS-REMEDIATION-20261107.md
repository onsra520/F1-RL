# Physics v4 handoff — root-cause remediation of 28 prior FAIL cases

Date: 2026-11-07. Branch: `handoff/physics-v4-khanh-bang-20261106`.
Previous committed source anchor: `dbaa7aacf5e5b03ce05e2f69a95c8f4913b528fa`.
This is an **engineering regression fix**, not PPO, FULL-06, held-out scoring, scientific admission, or physical lab calibration.

## Root causes and exact corrections

| Original failure | Count | Correction | Scientific boundary |
|---|---:|---|---|
| `test_wp3_crash_terminal_registry.py` | 20 | Restored the **original exact-byte** archived three-case replay, technical registry and 352-PASS regression receipt from two separately checked-out historical Git commits; adapt test expectation to prove current branch ancestry FAILS CLOSED; add test-only ancestry stub exclusively for independent payload, seed, roster, model and forged-receipt verification | The original WP3 authoring commit `ce0e55d` is **not an ancestor** of this handoff branch, so its scientific registry **cannot** authorize current source. No production guard changed |
| `test_wp3_physics_release_gate.py` | 7 | Restored the three authentic preexisting WP3 V4L/V4N/V4M blocked physical replay JSON fixtures | The original replay outcomes remain invalid/uncertified; scientific release stays **BLOCKED** |
| `test_wp7_slalom_dynamic_route_reconstruction.py` | 1 | Use `load_track_manifest(..., validate_asset_ids=("slalom-v1",))` instead of loading all HELD-OUT assets in the shared manifest | Still checks genuine registered Slalom TRAIN asset, its content/world SHA, control result and all checkpoint waypoints; no HELD-OUT geometry is read |

## Original artifact provenance and byte integrity

Six files restored under `docs/research` using **Git object contents** from two independent historical local worktrees:

- `D:\F1-RL-RUNTIME-SIMULATOR-BANG`, commit `a4b66f714d70bd85264c35213d3ff10e3cab061e`
- `D:\F1-RL-AUTO-DRIVE-LANE-RENDER-20261102`, commit `175e32d18ee0cc3a79a586d272d43857f1175d12`

Both source Git objects were byte-identical for all six files. The narrow
`docs/research/.gitattributes` marks those six files `-text` so
Git/Windows cannot normalize their line endings or change the SHA-256
during later checkouts. The actual Git blob IDs are verified against
the archived source before commit.

| Original immutable file | SHA-256 |
|---|---|
| `WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json` | `80f7db86f4e12a07521e70b7725630b66ca709ae7fe50ff9f52f84bf45b7f3a5` |
| `WP3-CRASH-TERMINAL-V1-REGRESSION-20261024.txt` | `f550d9a53e264fe4a8cc9ffc15dc932f61bdd1cc5048406b86c9f7c6d0e60142` |
| `WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json` | `086f27aef269d2cb999f121b008b3e168168fce274907a513585dfbf0641729d` |
| `WP3-V4L-RIGHT-TURN-PATCH-BRACKET-20261024.json` | `517705a28a743cd5ce6791cb2cb827263df91f9bad2db5fdadebed02b2010b9f` |
| `WP3-V4N-RAMP-BUMP-TRACE-20261024.json` | `0392e1d9f2e049a281eaf7ffb2ff28ce8ed388f5bcbb86163b8139b79c0a0286` |
| `WP3-V4M-SLALOM-GROUNDED-CONTACT-20261024.json` | `4ad0336bf992f4ea443998b815763da218efd01a78ff96f64ad8e7c11c4bb104` |

Read-only external provenance:
`D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\PHYSICS-V4-WP3-ORIGINAL-SIX-FIXTURES-RESTORED-20261107.json`.

## Test-first result: original failures, then identical extended selection

- Before changes: **1,176 PASS, 28 FAIL** in 1,204 completed nonoverlapping cases, 0 ERROR, 0 SKIP.
- Initial WP3 fixture restore and Slalom edit: **31/32 PASS**, one
  legitimate unrelated Git ancestry denial.
- After correcting the TEST EXPECTATION while preserving the real ancestry
  guard: **32/32 PASS**. Then an extra, explicit unit test verifies
  authentic WP3 payload validation with *only the ancestry check stubbed in
  the test*, while actual runtime authorizations remain blocked:
  **33/33 PASS**.
- Full previously sampled 1,204 cases rerun together, with one additional
  historical payload test and five portable source-integrity tests:
  **1,210/1,210 PASS**, **0 FAIL / 0 ERROR / 0 SKIP**.
- Portable Slalom plus WP3 source-only tests: **10/10 PASS**.
- JUnit (original reproduction):
  `HANDOFF-DBAA7AA-DEEP-AUDIT-WP3-RESEARCH-20261107.xml` and
  `HANDOFF-DBAA7AA-DEEP-AUDIT-INTEGRATION-RUNTIME-20261107.xml`.
- JUnit (corrected):
  `HANDOFF-28-FAILS-FIXED-EXTENDED-REGRESSION-PRECOMMIT-20261107.xml`.

The original extended WP7 arc-search run that was cancelled after
long-running CPU is still **NOT RETESTED** here. This is the **same
previously failing 1,204-case selection with six new tests**, not a
full-project claim or a rerun of PPO smoke.

The GitHub source-only CI is extended with **portable** historical byte-SHA
and Slalom TRAIN integrity tests, with no dependence on private/legacy
`D:\F1-RL-WP3-EVIDENCE` folder on the Ubuntu runner.

## Immutable scientific boundaries

- **No code** in `f1rl/research/wp3_crash_contract_registration.py`,
  `f1rl/research/wp3_physics_release_gate.py`, Gym/Physics engine,
  `TrainingRunner` or PPO backend changed.
- Current handoff is **not a descendant** of historical WP3 registry
  authoring commit; authentic source registry admission remains denied.
- Original 3 WP3 invalid replay rows remain **scientific_evidence=false**.
- All 18 proposed research training seeds unused; HELD-OUT, FULL-06,
  training and scientific release remain **BLOCKED**.
- Restoring original files does not create independent physical validation
  or change a single original checkpoint.

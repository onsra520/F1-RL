# WP3 Issue #51 — Official Evaluation readiness and reviewer request

Status: **PENDING independent Physics v4 signoff. NO official Eval episodes started.**

Source: \`0886a68dd1a8e17c0592f2399380665fef3d6734\` (\`fix/wp3-eval-physical-completion\`).
Original TRAIN code: \`fe83f517479f3b990028d6409304fae83b8199a9\`; original 12/12 model ZIP SHA-256: PASS.
Original seed roster: 96 cells / 5,760 episodes; SHA-256 \`f2f9d15954cea48345a8932773ed945f6ed250ad1435b4e675b97e4a6b94f319\`.
Contract: \`spatial_road_edge_partial_lifecycle_v4\` + \`wp3_physical_crash_terminal_v1\`.
Race: \`wp7_directed_lifecycle_crash_contract_v4\` (original 4m boundary).
Original model/seed replay: 3/3 engineering completed; 16/16 coordinator engineering episodes completed, not scientific evidence.

## Why Full Evaluation remains blocked

Official \`scripts/wp3_eval_only_coordinator.py full\` returned **exit code 3**, no episode started.
The saved historical Physics v4 technical receipt does not match the current source SHA in:
- f1rl/learning/evaluation/evaluator.py
- f1rl/learning/gym_environment.py
- f1rl/learning/wp7_race.py

Historic receipt bundle SHA-256: \`eb0c03e7f422d4a9cf5d4faccc988fb5831dfa4c8891aa3e7a2e0d8188e94856\`.
Current recalculated bundle SHA-256: \`72221a7904c2669ac70d0ceb345fc9c10a5eaffac4b66bae25b774b94e68ea03\`.
The approved independent physics review and scored crash approval are **missing**; no unsigned
record has been upgraded to a signed scientific acceptance.

## Reviewer tasks

1. Independently review and certify current source-bound road-edge/partial-support, body/road, bump-stop and impact impulse/energy methods.
2. Record reviewer identity/role, reviewed source \`0886a68dd1a8e17c0592f2399380665fef3d6734\`, signed acceptance and effective domain/tolerances.
3. Authorize scientific crash scoring and register an immutable source/physics/evaluator protocol.
4. Validate the official 96 x 60 paired-seed roster then activate a NEW, separate run ID/root with no HELD-OUT or training.
5. If any episode fails, retain failed row and abort affected cell instead of silently reusing/replacing attempts.

User has approved proceeding, but that approval is **not labeled as an independent scientific reviewer signature**.

Machine-readable bound manifest: \`WP3-OFFICIAL-EVAL-PREFLIGHT.json\` with checksum \`a10b795e1241e3097c8c7bc68ca45e270e5903279b4ccee6539aa774b15b69b7\`.

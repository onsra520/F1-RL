# Scripts architecture and compatibility

The public versionless CLI is:
~~~powershell
python -m scripts.workflows list
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows inference run-policy --help
~~~

Its registry maps 48 stable (domain, action) names to 48 existing executable paths.
An alias is never proof that the target was physically moved, approved for research, or rewritten.

## Layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| CLI/router | scripts/workflows/ | Inventory and explicit subprocess dispatch |
| Legacy compatibility | scripts/*.py | Preserve handoffs, subprocess callers, historical source paths, exit codes |
| Domain/application | src/interfaces/, src/evaluation/, src/training/, src/validation/ | Reusable implementation, explicit repository root where needed |
| Research source records | docs/research/ and docs/research/evidence/ | Historic scientific identity, manifests, immutable receipts |

## Audited disposition

The authoritative per-command classification is SCRIPT-STATUS.json.
The first category has substantive implementation in src. The second uses a
shared module without moving the permit/restore implementation. The third
was already a small delegating CLI. The fourth is intentionally preserved:
scientific source-bound, trusted-pipe, existing replay/physics audit protocol or
owner-consent launcher. It must not be mechanically rewritten.

No historical identifiers, registered evidence contract versions, schema fields,
checkpoint bytes or evidence identities were renamed.

## Source compatibility rules

- The script is resolved from its existing checkout, not an arbitrary installed package.
- The new extracted modules use the explicit checkout root; historical scripts retain their original root logic.
- The workflow router passes arguments unchanged, does not grant consent and does not run when merely listed.
- Inference fallback relaunches scripts/run_inference.py, not a package implementation path.
- The internal trusted-pipe PPO fixture worker stays inaccessible via the public workflow router.
- Imported helper names used by existing tests remain available from legacy wrappers.

## Remaining conscious exceptions

The guarded/historical entries are not called a finished physical rename.
They can only be moved after a complete call-site, prior-source identity, protocol
and external handoff inventory, plus identical behavioral tests. Their current
placement is a documented compatibility boundary, not a missing CLI command.

For code migration, use SCRIPT-MIGRATION-MAP.md and SCRIPT-IMPORT-AUDIT.md.
For current status of the project see CURRENT-STATE.md.
